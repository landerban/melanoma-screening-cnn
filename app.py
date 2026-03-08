"""
app.py — Skin Lesion Classifier UI
Gradio app with:
  • Training tab  — configure, start/stop, live loss & metric plots
  • Test tab      — upload image, get prediction + Grad-CAM overlay

Run:
    python app.py
Then open http://127.0.0.1:7860 in your browser.
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import gradio as gr

from efficientnet_b0 import EfficientNetB0Classifier, get_transforms, CFG
from trainer import TrainingState, Trainer, GradCAM

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_DIR   = Path(__file__).parent
TRAIN_DATA    = PROJECT_DIR / "training_data"
IMAGES_DIR    = TRAIN_DATA / "images"
DEFAULT_CKPT  = str(PROJECT_DIR / "best_model.pth")

# ---------------------------------------------------------------------------
# Global training state  (one instance for the lifetime of the app)
# ---------------------------------------------------------------------------

state = TrainingState()

# ---------------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------------

_DARK_BG  = "#1a1a1a"
_PANEL_BG = "#252525"
_TEXT     = "#e0e0e0"
_GRID     = "#333333"

def _style_ax(ax):
    ax.set_facecolor(_PANEL_BG)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color(_TEXT)
    for spine in ax.spines.values():
        spine.set_color(_GRID)
    ax.grid(color=_GRID, linewidth=0.5, linestyle="--")


def build_plots():
    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.patch.set_facecolor(_DARK_BG)

    ax_loss, ax_met = axes

    with state._lock:
        tl     = list(state.train_losses)
        vl     = list(state.val_losses)
        auc    = list(state.val_aucs)
        pr_auc = list(state.val_pr_aucs)
        rec    = list(state.val_recalls)
        spc    = list(state.val_specs)
        f1s    = list(state.val_f1s)

    ep = range(1, len(tl) + 1)

    # --- Loss panel ---
    _style_ax(ax_loss)
    if tl:
        ax_loss.plot(ep, tl, color="#4fc3f7", lw=1.8, marker="o", ms=3, label="Train")
        ax_loss.plot(ep, vl, color="#ef5350", lw=1.8, marker="o", ms=3, label="Val")
        # mark best AUC epoch
        if auc and state.best_auc > 0:
            best_ep = int(np.argmax(auc)) + 1
            ax_loss.axvline(best_ep, color="#aed581", lw=1.2, linestyle=":", label=f"Best AUC ep{best_ep}")
    ax_loss.set_title("Loss", fontsize=11, fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend(fontsize=8, facecolor=_PANEL_BG, labelcolor=_TEXT)

    # --- Metrics panel ---
    _style_ax(ax_met)
    if auc:
        ep_m = range(1, len(auc) + 1)
        ax_met.plot(ep_m, auc,    color="#ab47bc", lw=1.8, marker="o", ms=3, label="ROC-AUC")
        ax_met.plot(ep_m, pr_auc, color="#ff7043", lw=1.8, marker="s", ms=3, label="PR-AUC")
        ax_met.plot(ep_m, rec,    color="#ffa726", lw=1.8, marker="o", ms=3, label="Recall")
        ax_met.plot(ep_m, spc,    color="#26c6da", lw=1.8, marker="o", ms=3, label="Specificity")
        ax_met.plot(ep_m, f1s,    color="#9ccc65", lw=1.8, marker="o", ms=3, label="F1")
    ax_met.set_title("Validation Metrics", fontsize=11, fontweight="bold")
    ax_met.set_xlabel("Epoch")
    ax_met.set_ylim(-0.05, 1.05)
    ax_met.legend(fontsize=7, facecolor=_PANEL_BG, labelcolor=_TEXT, ncol=2)

    plt.tight_layout(pad=1.2)
    return fig


def build_placeholder_plot():
    plt.close("all")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_PANEL_BG)
    ax.text(0.5, 0.5, "Waiting for training to start …",
            ha="center", va="center", transform=ax.transAxes,
            fontsize=13, color="#666666")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(_GRID)
    return fig


# ---------------------------------------------------------------------------
# UI callbacks
# ---------------------------------------------------------------------------

def get_status_html():
    if state.error:
        return f'<span style="color:#ef5350">ERROR: {state.error}</span>'
    if state.done:
        thresh = state.optimal_threshold
        return (
            f'<span style="color:#9ccc65"><b>Training complete</b></span> &nbsp;|&nbsp; '
            f'Best AUC: <b>{state.best_auc:.4f}</b> &nbsp;|&nbsp; '
            f'Best val loss: {state.best_val_loss:.4f}<br>'
            f'<span style="color:#ffa726">Optimal threshold: <b>{thresh:.3f}</b> '
            f'(recall≥0.80 strategy — use this in the Test Image tab)</span>'
        )
    if not state.running:
        return '<span style="color:#888">Idle — press Start Training</span>'

    stage_label = {1: "Stage 1 (head only)", 2: "Stage 2 (ES on AUC)"}.get(state.stage, "")
    bar_pct = int(100 * state.epoch / max(state.total_epochs, 1))
    return (
        f'<b>{stage_label}</b> &nbsp;|&nbsp; '
        f'Epoch {state.epoch}/{state.total_epochs} &nbsp;|&nbsp; '
        f'ES patience: {state.es_counter}/{CFG["es_patience"]} &nbsp;|&nbsp; '
        f'Best AUC: {state.best_auc:.4f}<br>'
        f'<progress value="{bar_pct}" max="100" '
        f'style="width:100%;height:10px;accent-color:#4fc3f7"></progress>'
    )


def refresh_training():
    plot = build_plots() if state.train_losses else build_placeholder_plot()
    return plot, get_status_html(), state.snapshot_log()


def cb_start(csv_dir, images_dir, loss_fn, batch_size, s1_ep, s2_ep):
    if state.running:
        return get_status_html(), state.snapshot_log()

    state.reset()
    state.running = True

    cfg = {**CFG,
           "loss"          : loss_fn,
           "batch_size"    : int(batch_size),
           "stage1_epochs" : int(s1_ep),
           "stage2_epochs" : int(s2_ep)}

    Trainer(
        training_data_dir=csv_dir,
        images_dir=images_dir,
        cfg=cfg,
        state=state,
    ).launch()

    return get_status_html(), "Training started …"


def cb_stop():
    state.stop_requested = True
    return '<span style="color:#ffa726">Stop requested — finishing current epoch …</span>'


# ---------------------------------------------------------------------------
# Inference callback
# ---------------------------------------------------------------------------

def cb_predict(img_np, ckpt_path, threshold):
    if img_np is None:
        return None, "No image uploaded."
    if not os.path.exists(ckpt_path):
        return None, f"Checkpoint not found:\n{ckpt_path}\n\nTrain the model first."

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EfficientNetB0Classifier(freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    transform  = get_transforms(CFG, "val")
    img_pil    = Image.fromarray(img_np).convert("RGB")
    img_tensor = transform(img_pil).to(device)

    # Grad-CAM (requires grad, so don't wrap in no_grad)
    gcam = GradCAM(model)
    cam  = gcam.generate(img_tensor)

    with torch.no_grad():
        prob = torch.sigmoid(model(img_tensor.unsqueeze(0))).item()

    label      = "MALIGNANT" if prob >= threshold else "BENIGN"
    confidence = prob if label == "MALIGNANT" else 1.0 - prob
    color      = "#ef5350" if label == "MALIGNANT" else "#66bb6a"

    # Build overlay figure
    size       = CFG["input_size"]
    img_vis    = img_pil.resize((size, size))
    img_arr    = np.array(img_vis) / 255.0
    heatmap    = plt.cm.jet(cam)[:, :, :3]
    overlay    = np.clip(0.55 * img_arr + 0.45 * heatmap, 0, 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4.2))
    fig.patch.set_facecolor(_DARK_BG)

    ax1.imshow(img_vis)
    ax1.set_title("Input image", color=_TEXT, fontsize=10)
    ax1.axis("off")

    ax2.imshow(overlay)
    ax2.set_title("Grad-CAM attention", color=_TEXT, fontsize=10)
    ax2.axis("off")

    fig.suptitle(
        f"{label}  ({confidence:.1%} confidence)",
        color=color, fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    result_text = (
        f"Prediction  : {label}\n"
        f"Malignant p : {prob:.4f}\n"
        f"Confidence  : {confidence:.2%}\n"
        f"Threshold   : {threshold}\n\n"
        f"{'⚠  High malignancy probability — clinical review recommended.'   if label == 'MALIGNANT' else '✓  Low malignancy probability.'}\n\n"
        f"NOTE: This is a research tool only.\n"
        f"Do not use for clinical diagnosis."
    )

    return fig, result_text


# ---------------------------------------------------------------------------
# Build Gradio layout
# ---------------------------------------------------------------------------

_CSS = """
body, .gradio-container { background:#1a1a1a !important; color:#e0e0e0 !important; }
.tab-nav button { color:#aaa !important; }
.tab-nav button.selected { color:#4fc3f7 !important; border-bottom:2px solid #4fc3f7 !important; }
label { color:#bbb !important; font-size:0.85rem !important; }
textarea, input[type=text], input[type=number] { background:#2a2a2a !important; color:#e0e0e0 !important; border:1px solid #444 !important; }
"""

with gr.Blocks(title="Skin Lesion Classifier") as app:

    gr.Markdown(
        "# Skin Lesion Classifier\n"
        "**EfficientNet-B0** · ImageNet pretrained · Benign vs Malignant · ISIC datasets"
    )

    # =========================================================
    # TAB 1 — TRAINING
    # =========================================================
    with gr.Tab("Training"):

        with gr.Row():

            # ---- Left column: config + controls ----
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Data")
                csv_dir_in  = gr.Textbox(label="Training data folder (CSVs)",
                                         value=str(TRAIN_DATA))
                imgs_dir_in = gr.Textbox(label="Images folder",
                                         value=str(IMAGES_DIR))

                gr.Markdown("### Training config")
                loss_dd   = gr.Dropdown(
                    ["focal", "weighted_bce"], value="focal",
                    label="Loss  (focal recommended — imbalance is ~1000:1)",
                )
                batch_dd  = gr.Dropdown([16, 32], value=32, label="Batch size")
                s1_slider = gr.Slider(1, 10, value=5,  step=1, label="Stage 1 epochs (head only)")
                s2_slider = gr.Slider(5, 40, value=30, step=1, label="Stage 2 epochs (full fine-tune)")

                with gr.Row():
                    start_btn = gr.Button("Start Training", variant="primary")
                    stop_btn  = gr.Button("Stop",           variant="stop")

            # ---- Right column: live plots + status ----
            with gr.Column(scale=2):
                status_html = gr.HTML(
                    '<span style="color:#888">Idle — press Start Training</span>'
                )
                plot_out = gr.Plot(value=build_placeholder_plot, label="Loss & Metrics")
                log_box  = gr.Textbox(label="Training log", lines=12, interactive=False,
                                      autoscroll=True)

        # Timer — refreshes every 3 s while the tab is open
        timer = gr.Timer(value=3)
        timer.tick(fn=refresh_training, outputs=[plot_out, status_html, log_box])

        start_btn.click(
            fn=cb_start,
            inputs=[csv_dir_in, imgs_dir_in, loss_dd, batch_dd, s1_slider, s2_slider],
            outputs=[status_html, log_box],
        )
        stop_btn.click(fn=cb_stop, outputs=status_html)

    # =========================================================
    # TAB 2 — TEST IMAGE
    # =========================================================
    with gr.Tab("Test Image"):

        gr.Markdown(
            "Upload a skin lesion image. The model returns a benign / malignant "
            "prediction and a **Grad-CAM** heatmap showing which regions drove the decision.\n\n"
            "> If the heatmap focuses on rulers, borders, or markers rather than the lesion "
            "itself, this indicates shortcut learning — inspect your training data."
        )

        with gr.Row():

            with gr.Column(scale=1, min_width=280):
                img_in      = gr.Image(label="Lesion image", type="numpy", height=280)
                ckpt_in     = gr.Textbox(label="Model checkpoint (.pth)", value=DEFAULT_CKPT)
                thresh_sl   = gr.Slider(
                    0.05, 0.95, value=0.5, step=0.01,
                    label="Decision threshold  (lower → more sensitive / more FP)",
                )
                use_opt_btn = gr.Button("Use optimal threshold from training", variant="secondary")
                predict_btn = gr.Button("Run Prediction", variant="primary")
                result_box  = gr.Textbox(label="Result", lines=10, interactive=False)

            with gr.Column(scale=2):
                cam_out = gr.Plot(label="Input image  +  Grad-CAM attention")

        use_opt_btn.click(
            fn=lambda: state.optimal_threshold,
            outputs=thresh_sl,
        )
        predict_btn.click(
            fn=cb_predict,
            inputs=[img_in, ckpt_in, thresh_sl],
            outputs=[cam_out, result_box],
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=True,
        share=False,
        css=_CSS,
    )
