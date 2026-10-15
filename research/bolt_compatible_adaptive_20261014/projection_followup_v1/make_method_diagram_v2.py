"""Scientific method-spec diagram, with separate training and deployment inputs."""
from pathlib import Path
import argparse


def generate(destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    output = Path(destination).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    fig, ax = plt.subplots(figsize=(17, 8))
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 8)
    ax.axis("off")
    frozen, learned, text = "#DCEAF5", "#F8D9BE", "#172B3A"
    def box(x, y, width, height, title, body, color=frozen):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.04,rounding_size=.12", facecolor=color, edgecolor="#456174", linewidth=1.4))
        ax.text(x+width/2, y+height-.26, title, ha="center", va="center", fontsize=11, weight="bold", color=text)
        ax.text(x+width/2, y+height/2-.16, body, ha="center", va="center", fontsize=10, color=text, linespacing=1.55)
    def arrow(start, end, color="#456174", curve=0, dashed=False):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, linewidth=1.5, color=color, connectionstyle="arc3,rad="+str(curve), linestyle="--" if dashed else "-"))
    ax.text(.25, 7.65, "Method candidate: frozen temporal forecasting + small forecast-error readout", fontsize=17, weight="bold", color=text)
    ax.text(.25, 7.2, "Inference uses past observations and model predictions. Future labels appear only in FIT training.", fontsize=12, color=text)
    y = 4.55
    box(.25, y, 2.1, 1.7, "Observed past", "$X_{t-511:t}$\n512 × C\nFIT-only normalization")
    box(2.9, y, 2.0, 1.7, "Fixed encoder E", "$Z=(X-\\mu)E$\n512 × K\nPCA / fixed state")
    box(5.45, y, 2.2, 1.7, "Frozen Chronos-Bolt", "$U=F(Z)$\n48 × K\nK separate streams")
    box(8.2, y, 2.15, 1.7, "Causal lag features", "$\\Phi_h \\in \\mathbb{R}^{7K}$\nLags: 0,1,2,4,8,16,32\nKnown past + U")
    box(10.9, y, 2.15, 1.7, "Fixed old D", "$b_h=x_t+\\Phi_hD$\n48 × C baseline\nPast reconstruction fit")
    box(13.6, y, 2.9, 1.7, "Forecast correction", "$\\hat y_h=b_h+a_h\\Phi_h\\Delta D$\nStatic: $a_h=1$\nRamp: $a_h=(h-1)/47$", learned)
    for x1, x2 in ((2.35, 2.9), (4.9, 5.45), (7.65, 8.2), (10.35, 10.9), (13.05, 13.6)):
        arrow((x1, y+.85), (x2, y+.85))
    arrow((3.9, y), (9.25, y), curve=.32)
    ax.text(6.45, 3.25, "same K past buffer; lag-specific origin reference", ha="center", fontsize=10, color="#456174", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    arrow((1.3, y+1.7), (11.9, y+1.7), curve=-.12)
    ax.text(6.6, 6.68, "current C anchor remains available (not strict K-only compression)", ha="center", fontsize=10, color="#456174", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    box(.55, .95, 3.0, 1.6, "FIT labels only", "Actual future $Y_{t+1:t+48}$\nTrue-observed target mask\nNo future inference input", "#EBEDF0")
    box(4.3, .95, 4.2, 1.6, "Fit only the additional slope", "Observed channel-macro MSE + ridge\nNo intercept; shared FIT feature RMS\nAdditional coefficients: 7KC", learned)
    box(9.3, .95, 3.0, 1.6, "TUNE selection", "Same alpha menu + ZERO\nFreeze all 24 selections\nThen exposed DEV evaluation", "#EBEDF0")
    arrow((3.55, 1.75), (4.3, 1.75), color="#A65A28")
    arrow((8.5, 1.75), (9.3, 1.75), color="#A65A28")
    arrow((12.3, 1.75), (15.05, y), color="#A65A28", curve=.1, dashed=True)
    ax.text(13.9, 2.75, "selected $\\Delta D$ only", rotation=34, ha="center", fontsize=10, color="#A65A28", bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
    ax.text(.55, .3, "Required controls: Raw C with the same correction opportunity; F vs FLAT; STATIC vs RAMP; ZERO correction.", fontsize=11, color=text)
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / ("method_spec."+extension), dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(str(output / "method_spec.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    generate(parser.parse_args().output)
