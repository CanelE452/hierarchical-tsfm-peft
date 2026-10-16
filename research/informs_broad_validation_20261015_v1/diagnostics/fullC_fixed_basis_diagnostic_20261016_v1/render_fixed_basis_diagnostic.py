"""Render the explicit saved-V diagnostic without loading a forecasting model."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def bound(path, digest):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Explicit small regular input required")
    data = path.read_bytes()
    if sha(data) != digest:
        raise ValueError("Explicit input SHA changed")
    return data


def execute(args):
    paths = [Path(args.scores), Path(args.paired), Path(args.diagnostic)]
    hashes = [args.scores_sha256, args.paired_sha256, args.diagnostic_sha256]
    blobs = [bound(path, digest) for path, digest in zip(paths, hashes)]
    scores = list(csv.DictReader(io.StringIO(blobs[0].decode("utf-8"), newline="")))
    pairs = list(csv.DictReader(io.StringIO(blobs[1].decode("utf-8"), newline="")))
    result = json.loads(blobs[2])
    if result["status"] != "SAVED_V_STATISTICS_AND_FIXED_BASIS_DIAGNOSTIC_COMPLETE":
        raise ValueError("Completed explicit fixed-basis diagnostic required")
    if len(scores) != 6 or len(pairs) != 270:
        raise ValueError("Expected six routes and 270 descriptive paired groups")
    by_label = {row["label"]: row for row in scores}
    if set(by_label) != {"RAW", "DIRECT_G", "RAW_RIDGE", "PCA_K3", "PCA_K6", "PCA_K11"}:
        raise ValueError("Unexpected diagnostic identities")
    for row in scores:
        if [int(row[k]) for k in ("n_origins", "horizon", "original_channels")] != [43, 48, 11]:
            raise ValueError("This figure is limited to Beijing V43/H48/C11")
        saved = next(item for item in result["scores"] if item["label"] == row["label"])
        if any(float(row[k]) != saved[k] for k in ("mse", "mae")):
            raise ValueError("Diagnostic and score CSV disagree")
    selected_pairs = [row for row in pairs if row["left"] == "RAW" and row["right"] == "PCA_K11"]
    slices = {(row["group_kind"], row["group"]): row for row in selected_pairs}
    if len(slices) != 18:
        raise ValueError("Expected all 18 Raw/full-C descriptive groups")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Fresh rendering directory required")
    output.mkdir(parents=True, exist_ok=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10, "svg.fonttype": "none",
        "svg.hashsalt": "fullC-fixed-basis-diagnostic-root-v1"})
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    raw = by_label["RAW"]
    for column, metric in enumerate(("mse", "mae")):
        ax = axes[0, column]
        values = [float(by_label["PCA_K" + str(k)][metric]) for k in (3, 6, 11)]
        ax.plot([3, 6, 11], values, "o-", color="#0072B2", linewidth=2, label="Fixed nested PCA + Bolt")
        ax.axhline(float(raw[metric]), color="#333333", linestyle="--", label="Raw Bolt (11 channels)")
        for k, value in zip((3, 6, 11), values):
            ax.annotate(f"{value:.4f}", (k, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
        low, high = min(values + [float(raw[metric])]), max(values + [float(raw[metric])])
        ax.set_ylim(low - .15 * (high - low), high + .35 * (high - low))
        ax.set_xticks([3, 6, 11], ["3", "6", "11 = C"])
        ax.set_xlabel("Actual latent channels K")
        ax.set_ylabel("Normalized channel-macro " + metric.upper())
        ax.set_title(f"({'AB'[column]}) Saved V: {metric.upper()} (lower is better)", loc="left")
        ax.legend(fontsize=8.5, loc="best", frameon=False)

    for row_index, kind, groups, names, title in (
        (1, "horizon_band", ["0:12", "12:24", "24:48"], ["1–12", "13–24", "25–48"], "Forecast horizon"),
        (2, "origin_block", ["1", "2", "3"], ["First 15", "Next 14", "Last 14"], "Contiguous V-origin blocks"),
    ):
        for column, metric in enumerate(("mse", "mae")):
            ax = axes[row_index, column]
            values = [-float(slices[(kind, group)]["delta_" + metric]) for group in groups]
            colors = ["#0072B2" if value < 0 else "#D55E00" for value in values]
            ax.bar(range(3), values, color=colors, width=.55)
            ax.axhline(0, color="#333333", linewidth=.8)
            magnitude = max(abs(v) for v in values)
            for x, value in enumerate(values):
                ax.annotate(f"{value:+.4f}", (x, value), xytext=(0, 5 if value >= 0 else -12),
                    textcoords="offset points", ha="center", fontsize=9)
            ax.set_ylim(min(values + [0]) - .3 * magnitude, max(values + [0]) + .3 * magnitude)
            ax.set_xticks(range(3), names)
            ax.set_xlabel(title)
            ax.set_ylabel("PCA K=C − Raw: " + metric.upper())
            panel = "CDEF"[(row_index - 1) * 2 + column]
            ax.set_title(f"({panel}) Full-C change; negative favors PCA", loc="left")

    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_locator(MaxNLocator(5))
    basis = result["basis"]
    full = next(row for row in basis["prefix_diagnostics"] if row["k"] == 11)
    fig.suptitle("Full-C fixed PCA: invertible coordinates can change frozen-Bolt forecasts", fontsize=14, y=.98)
    fig.text(.5, .943, f"Beijing C=11, L=512, H=48, V=43 | basis rank {basis['full_basis_rank']} | "
        f"condition {basis['condition_number_2']:.9f} | max |EEᵀ−I| {full['projection_identity_max_abs']:.2e}", ha="center", fontsize=10)
    fig.text(.5, .028, "Single-seed repeated developmental V; descriptive slices, no independence or significance claim.\n"
        "No new forecast/model calls. Operator diagnostics are not a reconstruction test of actual history.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.09, right=.98, top=.89, bottom=.10, hspace=.58, wspace=.30)
    svg, png = output / "FIXED_FULLC_COORDINATES.svg", output / "FIXED_FULLC_COORDINATES.png"
    fig.savefig(svg, metadata={"Date": None})
    fig.savefig(png, dpi=300)
    plt.close(fig)
    for path, digest in zip(paths, hashes):
        bound(path, digest)
    binding = dict(status="RENDERED_FROM_EXPLICIT_DIAGNOSTIC_OUTPUTS", source_sha256=sha(Path(__file__).read_bytes()),
        inputs=[dict(file=path.name, sha256=digest, bytes=len(blob)) for path, digest, blob in zip(paths, hashes, blobs)],
        outputs=[dict(file=path.name, sha256=sha(path.read_bytes()), bytes=path.stat().st_size) for path in (svg, png)],
        actual_forecasting_or_training_or_cost_operations=0)
    (output / "FIGURE_BINDING.json").write_text(json.dumps(binding, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    for name in ("scores", "paired", "diagnostic"):
        parser.add_argument("--" + name)
        parser.add_argument("--" + name + "-sha256")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(status="METADATA_ONLY_RENDERER",input_reads=0,matplotlib_imported=False)))
        return
    if not all((args.scores, args.scores_sha256, args.paired, args.paired_sha256,
                args.diagnostic, args.diagnostic_sha256, args.output)):
        parser.error("Explicit inputs, SHA256 bindings and fresh output are required")
    binding = execute(args)
    print(json.dumps(dict(status=binding["status"], outputs=binding["outputs"])))


if __name__ == "__main__":
    main()
