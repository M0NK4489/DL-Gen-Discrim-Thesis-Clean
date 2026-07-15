from __future__ import annotations

import json
import math
import re
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


METRIC_KEYS = [
    "KL_hist",
    "DTW_mean",
    "FTSD",
    "MAE_median",
    "RMSE_mean",
    "PeakMAE",
    "CRPS",
    "QuantileLoss",
    "Winkler",
    "Coverage",
    "ACF_error",
    "CompositeScore",
    "corr_real",
    "corr_fake",
]

CONFIG_KEYS = [
    "FREQ",
    "PRIMARY_HORIZON",
    "SEQ_LEN",
    "CONTEXT_LEN",
    "WINDOW_STRIDE",
    "EVAL_POOL",
    "PLOT_N",
    "EVAL_SEED",
    "SAMPLES",
    "selected_train_windows",
    "selected_val_windows",
    "DTW_N",
    "BINS",
    "CLIP_QUANTILE",
    "ALPHA_PI",
    "SEED",
    "BATCH_SIZE",
    "LR",
    "epochs_vae",
    "epochs_ae",
    "epochs_sup",
    "epochs_gan",
    "epochs_diff",
    "epochs_ar",
    "cond_dim",
    "hidden_dim",
    "latent_dim",
    "time_emb_dim",
    "num_res_blocks",
    "transformer_layers",
    "transformer_heads",
    "transformer_ff_dim",
    "transformer_dropout",
    "diffusion_steps",
    "beta_start",
    "beta_end",
    "generator_hidden",
    "discriminator_hidden",
    "attr_noise_dim",
    "feature_noise_dim",
    "beta",
    "kl_warmup_epochs",
    "alpha_peak",
    "peak_power",
    "lambda_x0",
    "lambda_mom",
    "lambda_diff",
    "lambda_sup_gan",
    "lambda_div",
    "cond_noise_std",
    "min_log_scale",
    "max_log_scale",
    "sample_temperature",
    "g_steps_per_batch",
    "d_steps_per_batch",
    "grad_clip",
    "d_thresh",
]


def _safe_stem(name: object) -> str:
    stem = Path(str(name)).name
    if stem.endswith(".pt"):
        stem = stem[:-3]
    stem = re.sub(r"[^A-Za-z0-9._=-]+", "_", stem)
    return stem or "evaluation"


def _as_excel_value(value):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return json.dumps(np.asarray(value).tolist())
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _col_name(idx: int) -> str:
    name = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def _cell_xml(row_idx: int, col_idx: int, value) -> str:
    ref = f"{_col_name(col_idx)}{row_idx}"
    value = _as_excel_value(value)
    if isinstance(value, bool):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
        return f'<c r="{ref}"><v>{value}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def write_metrics_xlsx(path: Path, rows: list[list[object]]) -> None:
    """Write a tiny .xlsx using only the standard library."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sheet_rows = []
    for r_idx, row in enumerate(rows, start=1):
        cells = "".join(_cell_xml(r_idx, c_idx, value) for c_idx, value in enumerate(row, start=1))
        sheet_rows.append(f'<row r="{r_idx}">{cells}</row>')

    worksheet = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        '<sheetData>'
        + "".join(sheet_rows)
        + '</sheetData></worksheet>'
    )

    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="metrics" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        '</Types>'
    )

    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
        '</Relationships>'
    )

    workbook_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        '</Relationships>'
    )

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        zf.writestr("xl/worksheets/sheet1.xml", worksheet)


def _finite_flat(arr) -> np.ndarray:
    flat = np.asarray(arr, dtype=float).reshape(-1)
    return flat[np.isfinite(flat)]


def _autocorr(x, max_lag=96):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    denom = np.dot(x, x) + 1e-12
    acf = [1.0]
    for lag in range(1, max_lag + 1):
        acf.append(np.dot(x[:-lag], x[lag:]) / denom)
    return np.array(acf)


def _power_spectrum(x):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    fft = np.fft.rfft(x)
    ps = np.abs(fft) ** 2
    return ps / (ps.sum() + 1e-12)


def _save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return str(path)


def save_common_plots(
    out_dir: Path,
    model_label: str,
    y_real_kwh,
    y_fake_kwh_point,
    y_fake_kwh_samples=None,
    bins=80,
    clip_quantile=None,
    interval_alpha=0.10,
):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    plot_paths = []

    real_arr = np.asarray(y_real_kwh, dtype=float)
    fake_arr = np.asarray(y_fake_kwh_point, dtype=float)
    sample_arr = None
    if y_fake_kwh_samples is not None:
        sample_arr = np.asarray(y_fake_kwh_samples, dtype=float)
        if sample_arr.ndim != 4:
            sample_arr = None

    real = _finite_flat(real_arr)
    fake = _finite_flat(fake_arr)

    if clip_quantile is not None:
        cap = np.quantile(real, clip_quantile)
        real = np.clip(real, 0, cap)
        fake = np.clip(fake, 0, cap)

    lo, hi = float(real.min()), float(real.max())
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, int(bins) + 1)

    real_y = real_arr[:, :, 0]
    fake_y = fake_arr[:, :, 0]
    t = np.arange(real_y.shape[1])

    real_mean = np.nanmean(real_y, axis=0)
    real_lo, real_hi = np.nanquantile(real_y, [0.10, 0.90], axis=0)
    if sample_arr is not None:
        fake_profile_pool = sample_arr[:, :, :, 0].reshape(-1, sample_arr.shape[2])
    else:
        fake_profile_pool = fake_y
    fake_mean = np.nanmean(fake_profile_pool, axis=0)
    fake_lo, fake_hi = np.nanquantile(fake_profile_pool, [0.10, 0.90], axis=0)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t, real_mean, label="Real mean", linewidth=2)
    ax.fill_between(t, real_lo, real_hi, alpha=0.18, label="Real 10-90% band")
    ax.plot(t, fake_mean, label="Generated mean", linewidth=2)
    ax.fill_between(t, fake_lo, fake_hi, alpha=0.18, label="Generated 10-90% band")
    ax.set_title(f"Multi-window Real vs Generated Profile | {model_label}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("kWh")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plot_paths.append(_save_fig(fig, out_dir / "01_multi_window_profile_bands.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(real, bins=edges, density=True, alpha=0.6, label="Real")
    ax.hist(fake, bins=edges, density=True, alpha=0.6, label="Generated")
    ax.set_title(f"All-window kWh Distribution | {model_label}")
    ax.set_xlabel("kWh")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    plot_paths.append(_save_fig(fig, out_dir / "02_probability_distribution.png"))
    plt.close(fig)

    q = np.linspace(0.01, 0.99, 300)
    real_q = np.quantile(real, q)
    fake_q = np.quantile(fake, q)
    mx = float(max(real_q.max(), fake_q.max(), 1e-6))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(real_q, fake_q, marker=".", linestyle="none", alpha=0.6)
    ax.plot([0, mx], [0, mx], linewidth=1)
    ax.set_title(f"All-window QQ Plot | {model_label}")
    ax.set_xlabel("Real quantiles")
    ax.set_ylabel("Generated quantiles")
    fig.tight_layout()
    plot_paths.append(_save_fig(fig, out_dir / "03_qq_plot.png"))
    plt.close(fig)

    max_lag = min(200, real_y.shape[1] - 1)
    acf_real = np.mean([_autocorr(real_y[i], max_lag) for i in range(real_y.shape[0])], axis=0)
    acf_fake = np.mean([_autocorr(fake_y[i], max_lag) for i in range(fake_y.shape[0])], axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(acf_real, label="Real ACF")
    ax.plot(acf_fake, label="Generated ACF")
    ax.set_title(f"Autocorrelation | {model_label}")
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.legend()
    fig.tight_layout()
    plot_paths.append(_save_fig(fig, out_dir / "04_autocorrelation.png"))
    plt.close(fig)

    ps_real = np.mean([_power_spectrum(real_y[i]) for i in range(real_y.shape[0])], axis=0)
    ps_fake = np.mean([_power_spectrum(fake_y[i]) for i in range(fake_y.shape[0])], axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ps_real, label="Real power spectrum")
    ax.plot(ps_fake, label="Generated power spectrum")
    ax.set_title(f"Power Spectrum | {model_label}")
    ax.set_xlabel("Frequency bin")
    ax.set_ylabel("Normalized power")
    ax.legend()
    fig.tight_layout()
    plot_paths.append(_save_fig(fig, out_dir / "05_power_spectrum.png"))
    plt.close(fig)

    resid = _finite_flat(fake_arr - real_arr)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(resid, bins=100, density=True, alpha=0.8)
    ax.set_title(f"Residual Distribution: Generated - Real | {model_label}")
    ax.set_xlabel("kWh error")
    ax.set_ylabel("Density")
    fig.tight_layout()
    plot_paths.append(_save_fig(fig, out_dir / "06_residual_distribution.png"))
    plt.close(fig)

    true_vals = _finite_flat(real_arr)
    err_abs = _finite_flat(np.abs(fake_arr - real_arr))
    n = min(true_vals.shape[0], err_abs.shape[0])
    true_vals = true_vals[:n]
    err_abs = err_abs[:n]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(true_vals[::50], err_abs[::50], alpha=0.3, s=5)
    ax.set_title(f"|Error| vs True kWh | {model_label}")
    ax.set_xlabel("True kWh")
    ax.set_ylabel("|Generated - Real|")
    fig.tight_layout()
    plot_paths.append(_save_fig(fig, out_dir / "07_error_vs_true.png"))
    plt.close(fig)

    abs_err = np.abs(fake_y - real_y)
    window_mae = np.nanmean(abs_err, axis=1)
    window_rmse = np.sqrt(np.nanmean((fake_y - real_y) ** 2, axis=1))
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(window_mae[np.isfinite(window_mae)], bins=60, alpha=0.7, label="Window MAE")
    ax.hist(window_rmse[np.isfinite(window_rmse)], bins=60, alpha=0.5, label="Window RMSE")
    ax.set_title(f"Window-level Error Distribution | {model_label}")
    ax.set_xlabel("kWh error")
    ax.set_ylabel("Window count")
    ax.legend()
    fig.tight_layout()
    plot_paths.append(_save_fig(fig, out_dir / "08_window_error_distribution.png"))
    plt.close(fig)

    real_mean_w = np.nanmean(real_y, axis=1)
    fake_mean_w = np.nanmean(fake_y, axis=1)
    real_peak_w = np.nanmax(real_y, axis=1)
    fake_peak_w = np.nanmax(fake_y, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, x_vals, y_vals, label in [
        (axes[0], real_mean_w, fake_mean_w, "Window mean kWh"),
        (axes[1], real_peak_w, fake_peak_w, "Window peak kWh"),
    ]:
        ax.scatter(x_vals, y_vals, alpha=0.35, s=12)
        finite = np.isfinite(x_vals) & np.isfinite(y_vals)
        if np.any(finite):
            lim = float(max(np.nanmax(x_vals[finite]), np.nanmax(y_vals[finite]), 1e-6))
            ax.plot([0, lim], [0, lim], linewidth=1)
        ax.set_title(label)
        ax.set_xlabel("Real")
        ax.set_ylabel("Generated")
    fig.suptitle(f"Window Summary Agreement | {model_label}")
    fig.tight_layout()
    plot_paths.append(_save_fig(fig, out_dir / "09_window_summary_scatter.png"))
    plt.close(fig)

    med_ae = np.nanmedian(abs_err, axis=0)
    lo_ae, hi_ae = np.nanquantile(abs_err, [0.10, 0.90], axis=0)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, med_ae, label="Median absolute error")
    ax.fill_between(t, lo_ae, hi_ae, alpha=0.2, label="10-90% absolute error band")
    ax.set_title(f"Error by Timestep Across Windows | {model_label}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("kWh absolute error")
    ax.legend()
    fig.tight_layout()
    plot_paths.append(_save_fig(fig, out_dir / "10_error_by_timestep.png"))
    plt.close(fig)

    resid_matrix = fake_y - real_y
    order = np.argsort(real_mean_w)
    vmax = np.nanquantile(np.abs(resid_matrix), 0.99)
    vmax = float(vmax if np.isfinite(vmax) and vmax > 0 else 1.0)
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(resid_matrix[order], aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title(f"Residual Heatmap Across Windows | {model_label}")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Validation windows sorted by real mean kWh")
    fig.colorbar(im, ax=ax, label="Generated - Real kWh")
    fig.tight_layout()
    plot_paths.append(_save_fig(fig, out_dir / "11_residual_heatmap.png"))
    plt.close(fig)

    if sample_arr is not None:
        alpha = float(interval_alpha)
        lo_pi = np.nanquantile(sample_arr[:, :, :, 0], alpha / 2, axis=0)
        hi_pi = np.nanquantile(sample_arr[:, :, :, 0], 1 - alpha / 2, axis=0)
        coverage_t = np.nanmean((real_y >= lo_pi) & (real_y <= hi_pi), axis=0)
        width_t = np.nanmean(hi_pi - lo_pi, axis=0)
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(t, coverage_t)
        axes[0].axhline(1 - alpha, linewidth=1, linestyle="--")
        axes[0].set_title(f"90% Prediction Interval Coverage by Timestep | {model_label}")
        axes[0].set_ylabel("Coverage")
        axes[1].plot(t, width_t)
        axes[1].set_title("Mean Prediction Interval Width")
        axes[1].set_xlabel("Timestep")
        axes[1].set_ylabel("kWh")
        fig.tight_layout()
        plot_paths.append(_save_fig(fig, out_dir / "12_interval_coverage_by_timestep.png"))
        plt.close(fig)

    return plot_paths


def collect_common_metrics(namespace: dict, model_label: str, checkpoint_name: str):
    rows = [
        ["field", "value"],
        ["model", model_label],
        ["checkpoint_name", checkpoint_name],
        ["checkpoint_path", namespace.get("CHECKPOINT_PATH", "")],
        ["exported_at", datetime.now().isoformat(timespec="seconds")],
    ]

    for key in CONFIG_KEYS:
        if key in namespace:
            rows.append([key, _as_excel_value(namespace[key])])

    if "quantiles" in namespace:
        rows.append(["quantiles", _as_excel_value(namespace["quantiles"])])
    if "ALPHA_PI" in namespace:
        rows.append(["interval_percent", (1.0 - float(namespace["ALPHA_PI"])) * 100.0])

    rows.append(["", ""])
    rows.append(["metric", "value"])
    for key in METRIC_KEYS:
        if key in namespace:
            rows.append([key, _as_excel_value(namespace[key])])

    if "corr_real" in namespace and "corr_fake" in namespace:
        rows.append(["corr_abs_gap", abs(float(namespace["corr_real"]) - float(namespace["corr_fake"]))])

    return rows


def export_common_evaluation(
    namespace: dict,
    model_label: str,
    checkpoint_name: str | None = None,
    output_root: str | Path = "evaluation_exports",
):
    checkpoint_name = checkpoint_name or namespace.get("CHECKPOINT_NAME") or model_label
    out_dir = Path(output_root) / _safe_stem(checkpoint_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_common_metrics(namespace, model_label=model_label, checkpoint_name=str(checkpoint_name))
    metrics_path = out_dir / "metrics.xlsx"
    write_metrics_xlsx(metrics_path, rows)

    plots_dir = out_dir / "plots"
    bins = namespace.get("BINS", 80)
    clip_quantile = namespace.get("CLIP_QUANTILE", None)
    plot_paths = save_common_plots(
        plots_dir,
        model_label,
        namespace["y_real_kwh"],
        namespace["y_fake_kwh_point"],
        y_fake_kwh_samples=namespace.get("y_fake_kwh_samples"),
        bins=bins,
        clip_quantile=clip_quantile,
        interval_alpha=namespace.get("ALPHA_PI", 0.10),
    )

    return out_dir, metrics_path, plot_paths
