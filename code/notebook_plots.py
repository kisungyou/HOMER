"""Draw the manuscript examples inline from loaded result tables and arrays.

These helpers return Matplotlib figures for notebook display. They neither read
nor write files, choose a backend, invoke TeX, or rerun statistical calculations.
The same functions accept the archived paper results and fresh smoke/full runs.
"""

import matplotlib.pyplot as plt
import numpy as np


STYLE = {
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 100,
}
COLORS = ["#136f91", "#c64b33", "#4b8b45", "#8b5e9c"]


def _errors(table, value, lower="mc_low", upper="mc_high"):
    """Express the already computed interval endpoints as error-bar lengths."""
    return np.array([table[value] - table[lower], table[upper] - table[value]])


def _coverage_limits(ax, table):
    """Include every displayed interval and the nominal level, even in smoke runs."""
    lower = min(float(table.mc_low.min()), 0.95)
    upper = max(float(table.mc_high.max()), 0.95)
    padding = max(0.015, 0.1 * (upper - lower))
    ax.set_ylim(max(0, lower - padding), min(1.01, upper + padding))
    ax.axhline(0.95, color="0.45", lw=0.8, ls=":")


def scalar_coverage(summary):
    """Plot scalar coverage and stored Monte Carlo intervals (Figure 1, top)."""
    metrics = [
        ("plugin_mean_covered", "Estimated / mean", "o", "-", COLORS[1]),
        ("plugin_target_covered", "Estimated / target", "s", "-", COLORS[0]),
        ("oracle_mean_covered", "Population / mean", "o", "--", COLORS[2]),
        ("oracle_target_covered", "Population / target", "s", "--", COLORS[3]),
    ]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        fig.subplots_adjust(left=0.065, right=0.985, bottom=0.32, top=0.88, wspace=0.3)
        for ax, schedule, title in zip(
            axes, ["slow", "boundary", "many"],
            ["Slow block growth", "Boundary schedule", "Too many blocks"],
        ):
            selected = summary[summary.schedule == schedule]
            for metric, label, marker, linestyle, color in metrics:
                q = selected[selected.metric == metric].sort_values("a")
                ax.errorbar(q.a, q.coverage, yerr=_errors(q, "coverage"),
                            label=label, marker=marker, ls=linestyle, color=color,
                            lw=1.2, ms=4, capsize=2)
            ax.set_xticks(sorted(selected.a.unique()))
            ax.set_xlabel(r"$a$, with $n=2^{3a}$")
            ax.set_title(title)
            _coverage_limits(ax, selected[selected.metric.isin([m[0] for m in metrics])])
        axes[0].set_ylabel("Coverage")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.525, 0.01), ncol=2)
        return fig


def functional_coverage(summary):
    """Plot the two functional contrasts and stored intervals (Figure 2, left)."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        fig.subplots_adjust(left=0.085, right=0.985, bottom=0.32, top=0.88, wspace=0.3)
        for ax, contrast, title in zip(
            axes, ["interval_average", "evening_minus_morning"],
            ["Interval average", "Evening minus morning"],
        ):
            selected = summary[summary.contrast == contrast]
            blocks = sorted(selected.k.unique())
            positions = {k: i for i, k in enumerate(blocks)}
            for spectrum, marker in [("concentrated", "o"), ("diffuse", "s")]:
                for distribution, color, linestyle in [
                    ("gaussian", COLORS[0], "-"), ("exponential", COLORS[1], "--"),
                ]:
                    q = selected[(selected.spectrum == spectrum) &
                                 (selected.distribution == distribution)].sort_values("k")
                    offset = (-0.12 if spectrum == "concentrated" else 0.12)
                    offset += -0.035 if distribution == "gaussian" else 0.035
                    x = np.array([positions[k] for k in q.k]) + offset
                    ax.errorbar(x, q.coverage, yerr=_errors(q, "coverage"),
                                marker=marker, color=color, ls=linestyle,
                                capsize=2, ms=4, lw=1.2,
                                label=f"{spectrum}, {distribution}")
            ax.set_xticks(range(len(blocks)), [str(k) for k in blocks])
            ax.set_xlabel("Number of blocks")
            ax.set_ylabel("Coverage")
            ax.set_title(title)
            _coverage_limits(ax, selected[selected.distribution.isin(["gaussian", "exponential"])])
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.535, 0.01), ncol=2)
        return fig


def functional_stability(point_summary, efficiency_concentrated, efficiency_diffuse):
    """Plot efficiency and contamination using stored summaries (Figure 1, bottom).

    The efficiency tables contain the paired bootstrap intervals generated by
    the experiment; plotting does not recompute them from marginal summaries.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(12, 4.0))
        fig.subplots_adjust(left=0.07, right=0.985, bottom=0.36, top=0.89, wspace=0.4)
        order = ["MOM", "PH(0.25)", "PH(1)", "PH(4)", "PH(16)", "Mean"]
        for table, spectrum, color, marker in [
            (efficiency_concentrated, "concentrated", COLORS[1], "o"),
            (efficiency_diffuse, "diffuse", COLORS[0], "s"),
        ]:
            q = table.set_index("method").loc[order]
            axes[0].errorbar(range(len(order)), q.mse_ratio, yerr=_errors(q, "mse_ratio"),
                             marker=marker, ms=4, color=color, label=spectrum, capsize=2, lw=1.2)
        axes[0].axhline(1, color="0.5", lw=0.8, ls=":")
        axes[0].set_xticks(range(len(order)), ["MOM", ".25", "1", "4", "16", "Mean"],
                          rotation=35, ha="right")
        axes[0].set_xlabel(r"HOMER threshold $\lambda$")
        axes[0].set_ylabel("MSE / mean MSE")
        axes[0].set_title("Gaussian efficiency")
        axes[0].legend(loc="best")

        order = ["MOM", "PH(1)", "Adaptive(2)", "PH(16)", "Mean"]
        for ax, mechanism, title in zip(
            axes[1:], ["whole_block", "raw_dispersed"],
            ["Block contamination", "Raw contamination"],
        ):
            for magnitude, color, marker in [(10, COLORS[0], "o"), (100, COLORS[1], "s")]:
                q = point_summary[(point_summary.spectrum == "concentrated") &
                                  (point_summary.distribution == "gaussian") &
                                  (point_summary.mechanism == mechanism) &
                                  (point_summary.magnitude == magnitude)].set_index("method").loc[order]
                ax.errorbar(range(len(order)), q.q95,
                            yerr=_errors(q, "q95", "q95_mc_low", "q95_mc_high"),
                            marker=marker, color=color, ms=4, lw=1.2, capsize=2,
                            label=f"Shift {magnitude}")
            ax.set_xticks(range(len(order)), ["MOM", "HOMER(1)", "Adaptive\nHOMER", "HOMER(16)", "Mean"],
                          rotation=40, ha="right")
            ax.set_yscale("log")
            ax.set_ylabel("95th-percentile norm error")
            ax.set_title(title)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.68, 0.005), ncol=2)
        return fig


def covariance_heatmaps(saved):
    """Plot the prespecified covariance replicate using one unclipped color scale.

    ``saved`` is a mapping loaded from covariance_display.npz (paper mode) or
    covariance_estimates.npz (fresh runs). Selection follows the experiment's
    recorded display_replicate and display_delta, fixed to zero and 32.
    """
    replication = int(saved["display_replicate"])
    delta = float(saved["display_delta"])
    scenario = int(np.flatnonzero(saved["delta"] == delta)[0])
    clean_scenario = int(np.flatnonzero(saved["delta"] == 0)[0])
    methods = list(saved["methods"])
    fitted = saved["estimates"]
    matrices = [saved["truth"], fitted[replication, clean_scenario, methods.index("Mean")]]
    matrices.extend(fitted[replication, scenario, methods.index(name)]
                    for name in ["Mean", "GMed", "Adaptive PH"])
    bound = float(np.ceil(4 * max(np.abs(a).max() for a in matrices)) / 4)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 5, figsize=(12, 2.8))
        fig.subplots_adjust(left=0.035, right=0.985, bottom=0.28, top=0.87, wspace=0.2)
        for number, (ax, matrix, title) in enumerate(zip(
            axes, matrices, ["Truth", "Clean mean", "Stressed mean", "Geometric median", "Adaptive HOMER"],
        )):
            size = matrix.shape[0]
            mesh = ax.pcolormesh(np.arange(size + 1), np.arange(size + 1), matrix,
                                 cmap="RdBu_r", vmin=-bound, vmax=bound, shading="flat", linewidth=0)
            ax.set_aspect("equal")
            ax.set_xlim(0, size)
            ax.set_ylim(size, 0)
            ticks = [0.5, size // 2 - 0.5, size - 0.5]
            labels = ["1", str(size // 2), str(size)]
            ax.set_xticks(ticks, labels)
            ax.set_yticks(ticks, labels if number == 0 else [])
            ax.tick_params(length=2, pad=2)
            ax.set_title(title)
            for spine in ax.spines.values():
                spine.set_visible(False)
        color_ax = fig.add_axes([0.35, 0.105, 0.3, 0.045])
        fig.colorbar(mesh, cax=color_ax, orientation="horizontal", ticks=[-bound, 0, bound])
        return fig


def wearable_stability(results):
    """Plot activity sensitivity under the specified energy stress (Figure 2, right)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        fig.subplots_adjust(left=0.12, right=0.98, bottom=0.25, top=0.88)
        for method, color, marker in [
            ("Mean", COLORS[2], "^"), ("MOM", COLORS[0], "s"), ("Adaptive(2)", COLORS[1], "o"),
        ]:
            q = results[(results.mechanism == "one_block") &
                        (results.amplitude_multiplier == 10) &
                        (results.method == method)].sort_values("activity")
            label = "Adaptive HOMER" if method == "Adaptive(2)" else method
            ax.plot(q.activity, q.shift_relative_clean_mean, marker=marker,
                    color=color, label=label, ms=5, lw=1.2)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_ylim(bottom=-0.002)
        ax.set_xticks(range(1, 7))
        ax.set_xlabel("Activity identifier")
        ax.set_ylabel("Shift / clean subject mean")
        ax.set_title("Energy stress")
        fig.legend(*ax.get_legend_handles_labels(), loc="lower center",
                   bbox_to_anchor=(0.55, 0.01), ncol=3)
        return fig
