import numpy as np
import pandas as pd
from pyroads.merge import merge as merge
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


class cn:
    road = "road"
    slk_from = "slk_from"
    slk_to = "slk_to"
    value = "value"


def plot_dist(ax: Axes, df, title=None, height=None):
    if height is None:
        draw_heights = [20 for item in df.index]
    else:
        draw_heights = height

    if title is not None:
        ax.set_title(title)

    ax.bar(
        x=df[cn.slk_from],
        height=draw_heights,
        width=df[cn.slk_to] - df[cn.slk_from],
        align="edge",
        linewidth=1,
        edgecolor="black",
        alpha=0.6,
    )

    ax.xaxis.set_tick_params("both", True)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    ax.yaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_minor_locator(AutoMinorLocator(10))
    ax.set_ymargin(0.1)

    ax.set_axisbelow(True)
    ax.grid(True, which="minor", axis="both", color="#DDDDDD", linestyle="dotted")
    ax.grid(True, which="major", axis="both")

    if height is not None:
        for patch, draw_height in zip(ax.patches, draw_heights):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_y() + patch.get_height(),
                "nan" if np.isnan(draw_height) else str(round(draw_height * 100) / 100),
                va="bottom",
                ha="center",
                fontdict={"size": 8},
            )


def plot_seg_vs_merged(seg, dat, test_name):
    mer = merge.on_slk_intervals(
        target=seg,
        data=dat,
        join_left=[cn.road],
        column_actions=[
            merge.Action(cn.value, merge.Aggregation.KeepLongest(), rename="longest"),
            merge.Action(
                cn.value,
                merge.Aggregation.LengthWeightedAverage(),
                rename="l w average",
            ),
            merge.Action(
                cn.value,
                merge.Aggregation.LengthWeightedPercentile(0.75),
                rename="75th percentile",
            ),
            merge.Action(
                cn.value,
                merge.Aggregation.LengthWeightedPercentile(0.5),
                rename="50th percentile",
            ),
            merge.Action(cn.value, merge.Aggregation.Average(), rename="Average"),
            merge.Action(cn.value, merge.Aggregation.First(), rename="First"),
            merge.Action(
                cn.value, merge.Aggregation.SumProportionOfData(), rename="PropSum"
            ),
        ],
        from_to=("slk_from", "slk_to"),
    )

    fig, axs = plt.subplots(2, 4, sharex="all", sharey="all")

    fig.set_size_inches(w=16.5, h=11.7)

    plot_dist(axs[0, 0], dat, title="Original", height=dat[cn.value])
    plot_dist(axs[0, 1], mer, title="Keep Longest", height=mer["longest"])
    plot_dist(
        axs[0, 2], mer, title="Length Weighted Average", height=mer["l w average"]
    )
    plot_dist(
        axs[1, 0],
        mer,
        title="75th Percentile (Length Weighted)",
        height=mer["75th percentile"],
    )
    plot_dist(
        axs[1, 1],
        mer,
        title="50th Percentile (Length Weighted)",
        height=mer["50th percentile"],
    )
    plot_dist(axs[1, 2], mer, title="Average", height=mer["Average"])
    plot_dist(axs[0, 3], mer, title="First", height=mer["First"])
    plot_dist(axs[1, 3], mer, title="ProportionalSum", height=mer["PropSum"])

    plt.tight_layout()
    plt.savefig(
        f"./tests_manual/test_plots/{test_name}.pdf",
        orientation="landscape",
        format="pdf",
    )
    print(f"Saved plot to ./tests_manual/test_plots/{test_name}.pdf")


def test_pytest():
    assert True


def test_plot_1():
    seg = pd.DataFrame(
        [[1, 10, 20], [1, 20, 50], [1, 50, 100]],
        columns=[cn.road, cn.slk_from, cn.slk_to],
    )

    dat = pd.DataFrame(
        [
            [1, 0, 15, 500],
            [1, 15, 30, 600],
            [1, 30, 40, 450],
            [1, 40, 45, 420],
            [1, 45, 60, 400],
            [1, 60, 80, 600],
            [1, 80, 85, 540],
            [1, 90, 105, 470],
        ],
        columns=[cn.road, cn.slk_from, cn.slk_to, cn.value],
    )

    plot_seg_vs_merged(seg, dat, "test_plot_1")


def test_plot_2():
    seg = pd.DataFrame(
        [
            [1, 10, 20],
            [1, 20, 30],
            [1, 30, 40],
            [1, 40, 50],
            [1, 50, 60],
            [1, 60, 70],
            [1, 70, 80],
            [1, 80, 90],
            [1, 90, 100],
            [1, 100, 110],
            [1, 110, 120],
        ],
        columns=[cn.road, cn.slk_from, cn.slk_to],
    )

    dat = pd.DataFrame(
        [
            [1, 0, 15, 500],
            [1, 15, 30, 600],
            [1, 30, 40, 450],
            [1, 40, 45, 420],
            [1, 45, 60, 400],
            [1, 60, 80, 600],
            [1, 80, 85, 540],
            [1, 90, 105, 470],
        ],
        columns=[cn.road, cn.slk_from, cn.slk_to, cn.value],
    )

    plot_seg_vs_merged(seg, dat, "test_plot_2")


def test_plot_3():
    seg = pd.DataFrame(
        [[1, 10, 40], [1, 40, 70], [1, 70, 100], [1, 100, 130], [1, 130, 160]],
        columns=[cn.road, cn.slk_from, cn.slk_to],
    )

    dat = pd.DataFrame(
        [
            [1, 10, 20, 500],
            [1, 20, 30, 510],
            [1, 30, 40, 540],
            [1, 40, 50, 320],
            [1, 50, 60, 530],
            [1, 60, 70, 530],
            [1, 70, 80, 520],
            [1, 80, 90, np.nan],
            [1, 90, 100, 540],
            [1, 100, 110, 520],
            [1, 110, 120, 490],
            [1, 120, 130, 450],
        ],
        columns=[cn.road, cn.slk_from, cn.slk_to, cn.value],
    )

    plot_seg_vs_merged(seg, dat, "test_plot_3")


def test_plot_4():
    seg = pd.DataFrame(
        [
            [1, 10, 20],
            [1, 20, 30],
            [1, 30, 40],
        ],
        columns=[cn.road, cn.slk_from, cn.slk_to],
    )

    dat = pd.DataFrame(
        [
            [1, 0, 15, 300],
            [1, 15, 35, 500],
            [1, 35, 50, 400],
        ],
        columns=[cn.road, cn.slk_from, cn.slk_to, cn.value],
    )

    plot_seg_vs_merged(seg, dat, "test_plot_4")


def test_plot_5():
    seg = pd.DataFrame(
        [
            [1, 10, 90],
        ],
        columns=[cn.road, cn.slk_from, cn.slk_to],
    )

    dat = pd.DataFrame(
        [
            [1, 5, 40, 400],
            [1, 40, 50, 500],
            [1, 50, 95, 300],
        ],
        columns=[cn.road, cn.slk_from, cn.slk_to, cn.value],
    )

    plot_seg_vs_merged(seg, dat, "test_plot_5")
