import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from deepvoxnet2.analysis.data import Series, GroupedSeries


color_dict = {
    "r": (1, 0, 0),
    "g": (0, 1, 0),
    "b": (0, 0, 1),
    "c": (0, 1, 1),
    "m": (1, 0, 1),
    "y": (1, 1, 0),
    "k": (0, 0, 0)
}


class Figure(object):
    def __init__(self,
                 xalim, yalim,
                 awidthininches=5, aheightininches=5,
                 dxininches=0.25, dyininches=0.25,
                 lmwidthininches=1.5, rmwidthininches=0.5,
                 bmheightininches=1, tmheightininches=0.5,
                 top_extent=0, right_extent=0,
                 fs=20, lw=2, ms=10,
                 use_tex=True, **kwargs):

        if use_tex:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "Helvetica",
                "text.latex.preamble": r'\usepackage{textcomp}\usepackage{pifont}\usepackage{booktabs}\usepackage{amssymb,amsthm}\usepackage{amsmath}'
            })

        self.xalim, self.yalim = xalim, yalim
        self.awidthininches, self.aheightininches = awidthininches, aheightininches
        self.dxininches, self.dyininches = dxininches, dyininches
        self.lmwidthininches, self.rmwidthininches = lmwidthininches, rmwidthininches
        self.bmheightininches, self.tmheightininches = bmheightininches, tmheightininches
        self.top_extent, self.right_extent = top_extent, right_extent
        # calculate x dimensions
        self.xamin, self.xamax = self.xalim[0], self.xalim[1]
        self.awidth = self.xamax - self.xamin
        self.dx = self.awidth * self.dxininches / self.awidthininches
        self.xmin, self.xmax = self.xamin - self.dx, self.xamax + self.dx + self.awidth * self.right_extent
        self.width = self.xmax - self.xmin
        self.widthininches = self.awidthininches * self.width / self.awidth
        self.fwidthininches = self.lmwidthininches + self.widthininches + self.rmwidthininches
        # calculate y dimensions
        self.yamin, self.yamax = self.yalim[0], self.yalim[1]
        self.aheight = self.yamax - self.yamin
        self.dy = self.aheight * self.dyininches / self.aheightininches
        self.ymin, self.ymax = self.yamin - self.dy, self.yamax + self.dy + self.aheight * self.top_extent
        self.height = self.ymax - self.ymin
        self.heightininches = self.aheightininches * self.height / self.aheight
        self.fheightininches = self.tmheightininches + self.heightininches + self.bmheightininches
        # make figure
        self.fig = plt.figure(figsize=(self.fwidthininches, self.fheightininches))
        self.ax = self.fig.add_axes((self.lmwidthininches / self.fwidthininches, self.bmheightininches / self.fheightininches, self.widthininches / self.fwidthininches, self.heightininches / self.fheightininches))
        # setup figure
        self.lw = lw
        self.fs = fs
        self.ms = ms
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_linewidth(self.lw)
        self.ax.spines['bottom'].set_linewidth(self.lw)
        self.ax.spines['left'].set_bounds(self.yamin, self.yamax)
        self.ax.spines['bottom'].set_bounds(self.xamin, self.xamax)
        self.ax.tick_params(axis='both', which='major', labelsize=self.fs, width=self.lw)
        self.ax.set_xlim([self.xmin, self.xmax])
        self.ax.set_ylim([self.ymin, self.ymax])
        # add whitespace in between x- and y-axis in the dx and dy regions
        self.lwic = self.lw / self.fig.get_dpi() * self.awidth / self.awidthininches
        self.ax.add_patch(Rectangle((self.xmin + self.lwic / 2, self.yamax + self.lwic / 2), self.width - self.lwic, self.ymax - self.yamax - self.lwic, fc='w', ec='w', linewidth=0, zorder=2.001))
        self.ax.add_patch(Rectangle((self.xmin + self.lwic / 2, self.ymin + self.lwic / 2), self.width - self.lwic, self.dy - self.lwic, fc='w', ec='w', linewidth=0, zorder=2.001))
        self.ax.add_patch(Rectangle((self.xmin + self.lwic / 2, self.ymin + self.lwic / 2), self.dx - self.lwic, self.height - self.lwic, fc='w', ec='w', linewidth=0, zorder=2.001))
        self.ax.add_patch(Rectangle((self.xamax + self.lwic / 2, self.ymin + self.lwic / 2), self.xmax - self.xamax - self.lwic, self.height - self.lwic, fc='w', ec='w', linewidth=0, zorder=2.001))
        # optionally set title, labels, ticks and ticklabels
        if "xlabel" in kwargs and kwargs["xlabel"] is not None:
            self.set_xlabel(kwargs["xlabel"])

        if "ylabel" in kwargs and kwargs["ylabel"] is not None:
            self.set_ylabel(kwargs["ylabel"])

        if "xticks" in kwargs and kwargs["xticks"] is not None:
            if kwargs["xticks"] == "auto":
                assert "xticklabels" in kwargs and kwargs["xticklabels"] == "auto"
                self.set_xticks(np.linspace(self.xamin, self.xamax, 5))

            else:
                self.set_xticks(kwargs["xticks"] if kwargs["xticks"] else [])

        if "xticklabels" in kwargs and kwargs["xticklabels"] is not None and kwargs["xticklabels"] != "auto":
            assert "xticks" in kwargs and kwargs["xticks"] is not None and len(kwargs["xticks"]) == len(kwargs["xticklabels"])
            self.set_xticklabels(kwargs["xticklabels"])

        if "yticks" in kwargs and kwargs["yticks"] is not None:
            if kwargs["yticks"] == "auto":
                assert "yticklabels" in kwargs and kwargs["yticklabels"] == "auto"
                self.set_yticks(np.linspace(self.yamin, self.yamax, 5))

            else:
                self.set_yticks(kwargs["yticks"] if kwargs["yticks"] else [])

        if "yticklabels" in kwargs and kwargs["yticklabels"] is not None and kwargs["yticklabels"] != "auto":
            assert "yticks" in kwargs and kwargs["yticks"] is not None and len(kwargs["yticks"]) == len(kwargs["yticklabels"])
            self.set_yticklabels(kwargs["yticklabels"])

        if "title" in kwargs and kwargs["title"] is not None:
            self.set_title(kwargs["title"])

    def __call__(self):
        return self.fig, self.ax

    def set_title(self, title):
        self.fig.suptitle(title, fontsize=self.fs, x=0.5, y=1 - (self.rmwidthininches / 5) / self.fheightininches, horizontalalignment='center', verticalalignment='top')

    def set_xlabel(self, xlabel):
        self.ax.set_xlabel(xlabel, fontsize=self.fs, horizontalalignment='center', verticalalignment='bottom')
        self.ax.xaxis.set_label_coords(0.5, -(self.bmheightininches - self.rmwidthininches / 5) / self.heightininches)

    def set_ylabel(self, ylabel):
        self.ax.set_ylabel(ylabel, fontsize=self.fs, horizontalalignment='center', verticalalignment='top')
        self.ax.yaxis.set_label_coords(-(self.lmwidthininches - self.rmwidthininches / 5) / self.widthininches, 0.5)

    def set_xticks(self, xticks):
        self.ax.set_xticks(xticks)

    def set_xticklabels(self, xticklabels):
        self.ax.set_xticklabels(xticklabels)

    def set_yticks(self, yticks):
        self.ax.set_yticks(yticks)

    def set_yticklabels(self, yticklabels):
        self.ax.set_yticklabels(yticklabels)

    def plot(self, *args, **kwargs):
        self.ax.plot(*args, **kwargs)

    def boxplot(self, series, pos, direction="vertical", width=0.8, fc=(0, 0, 1, 0.5), ec=None, project_stats=False, plot_violin=False, violin_color=None, print_mean=True, different_from=None, **kwargs):
        series = Series(series)
        width2 = width / 2
        if ec is None:
            ec = (fc[0], fc[1], fc[2], 1)

        text = ""
        if print_mean:
            text += "$"
            text += "{0:.3g}".format(series.mean)
            text += "^{" + f"{series.nnans if series.nnans > 0 else ''}" + "}"
            if different_from is not None:
                p_value = series.different_from(different_from, **kwargs)
                p_value_ = min(p_value, 1 - p_value)
                if p_value_ < 0.05:
                    text += "_{>" if p_value > 0.95 else "_{<"
                    text += "*}" if 0.01 < p_value_ < 0.05 else ("**}" if 0.001 < p_value_ < 0.01 else "***}")

            text += "$"

        if plot_violin:
            violin_parts = self.ax.violinplot(series.series_, positions=[pos], vert=direction == "vertical", widths=width, showmeans=False, showmedians=False, showextrema=False, points=1000)
            for pc in violin_parts['bodies']:
                pc.set_color(self.get_color(fc if violin_color is None else violin_color))
                pc.set_linewidth(0)

            width = width / 2
            width2 = width2 / 2

        if direction == "vertical":
            self.add_patch(Rectangle((pos - width2, series.p25), width, series.iqr, fc=fc, ec=ec, linewidth=self.lw))
            self.plot([pos, pos], [series.p75, series.pmax], color=ec, linewidth=self.lw)
            self.plot([pos, pos], [series.pmin, series.p25], color=ec, linewidth=self.lw)
            self.plot([pos] * len(series.outliers), series.outliers, color=fc, linestyle="None", marker=".", markersize=self.ms)
            self.plot([pos - width2, pos + width2], [series.p50, series.p50], color=ec, linewidth=self.lw)
            self.plot([pos - width2, pos + width2], [series.pm, series.pm], color=fc, linestyle="dashed", linewidth=self.lw)
            self.plot([pos - width2, pos + width2], [series.pmax, series.pmax], color=ec, linewidth=self.lw)
            self.plot([pos - width2, pos + width2], [series.pmin, series.pmin], color=ec, linewidth=self.lw)
            if project_stats:
                self.plot([self.xmin, pos - width2], [series.p50, series.p50], color=ec, linewidth=self.lw, zorder=2.01)
                self.plot([self.xmin, pos - width2], [series.pm, series.pm], color=ec, linestyle="dashed", linewidth=self.lw, zorder=2.01)
                self.plot([self.xmin, pos - width2], [series.p75, series.p75], color=ec, linewidth=self.lw, zorder=2.01)
                self.plot([self.xmin, pos - width2], [series.p25, series.p25], color=ec, linewidth=self.lw, zorder=2.01)

            self.text(pos, self.yamax + self.dy, text, rotation=0, ha="center", va="bottom", fontsize=self.fs)
            if different_from:
                self.plot([pos - 0.5, pos + 0.5], [different_from, different_from], "k--", linewidth=self.lw, zorder=2.01)

        else:
            assert direction == "horizontal"
            self.add_patch(Rectangle((series.p25, pos - width2), series.iqr, width, fc=fc, ec=ec, linewidth=self.lw))
            self.plot([series.p75, series.pmax], [pos, pos], color=ec, linewidth=self.lw)
            self.plot([series.pmin, series.p25], [pos, pos], color=ec, linewidth=self.lw)
            self.plot(series.outliers, [pos] * len(series.outliers), color=fc, linestyle="None", marker=".", markersize=self.ms)
            self.plot([series.p50, series.p50], [pos - width2, pos + width2], color=ec, linewidth=self.lw)
            self.plot([series.pm, series.pm], [pos - width2, pos + width2], color=fc, linestyle="dashed", linewidth=self.lw)
            self.plot([series.pmax, series.pmax], [pos - width2, pos + width2], color=ec, linewidth=self.lw)
            self.plot([series.pmin, series.pmin], [pos - width2, pos + width2], color=ec, linewidth=self.lw)
            if project_stats:
                self.plot([series.p50, series.p50], [self.ymin, pos - width2], color=ec, linewidth=self.lw, zorder=2.01)
                self.plot([series.pm, series.pm], [self.ymin, pos - width2], color=ec, linestyle="dashed", linewidth=self.lw, zorder=2.01)
                self.plot([series.p75, series.p75], [self.ymin, pos - width2], color=ec, linewidth=self.lw, zorder=2.01)
                self.plot([series.p25, series.p25], [self.ymin, pos - width2], color=ec, linewidth=self.lw, zorder=2.01)

            self.text(self.xamax + self.dx, pos, text, rotation=270, ha="left", va="center", fontsize=self.fs)
            if different_from:
                self.plot([different_from, different_from], [pos - 0.5, pos + 0.5], "k--", linewidth=self.lw, zorder=2.01)

    def add_patch(self, patch):
        self.ax.add_patch(patch)

    def text(self, x, y, s, fontdict=None, **kwargs):
        self.ax.text(x, y, s, fontdict=fontdict, **kwargs)

    def show(self):
        self.fig.show()

    def savefig(self, file_path, **kwargs):
        self.fig.savefig(file_path, **kwargs)

    @staticmethod
    def get_color(color, alpha=None):
        color_tuple = color_dict[color] if isinstance(color, str) else color
        if len(color_tuple) == 3:
            return color_tuple + (1 if alpha is None else alpha,)

        if len(color_tuple) == 4:
            assert alpha is None or alpha == color_tuple[3]
            return color_tuple


class Boxplot(Figure):
    def __init__(self, grouped_series, labels=None, xalim=None, yalim=None, inchesperposition=None, colors=None, alpha=0.5, direction="vertical", l0_stats=False, l1_stats=False, **kwargs):
        grouped_series = GroupedSeries(grouped_series)
        position, positions = 0, []
        for series_group in grouped_series:
            position += 1
            positions_ = []
            for series in series_group:
                positions_.append(position)
                position += 1

            positions.append(positions_)

        if colors is None:
            colors = [[list(color_dict.keys())[i] for i, series in enumerate(series_group)] for series_group in grouped_series]

        if inchesperposition is not None:
            if direction == "vertical":
                kwargs["awidthininches"] = position * inchesperposition

            else:
                kwargs["aheightininches"] = position * inchesperposition

        if xalim is None:
            xalim = [None, None]

        if xalim[0] is None:
            xalim[0] = 0

        if xalim[1] is None:
            xalim[1] = position

        if yalim is None:
            yalim = [None, None]

        if yalim[0] is None:
            yalim[0] = grouped_series.series.min

        if yalim[1] is None:
            yalim[1] = grouped_series.series.max

        ticks = [np.mean(positions_) for positions_ in positions]
        super(Boxplot, self).__init__(
            xalim=xalim if direction == "vertical" else yalim,
            yalim=yalim if direction == "vertical" else xalim,
            xticks=ticks if direction == "vertical" else "auto",
            xticklabels=labels if direction == "vertical" else "auto",
            yticks="auto" if direction == "vertical" else ticks,
            yticklabels="auto" if direction == "vertical" else labels,
            **kwargs)

        for i, series_group in enumerate(grouped_series):
            for j, series in enumerate(series_group):
                self.boxplot(series, pos=positions[i][j], fc=self.get_color(colors[i][j], alpha=alpha), direction=direction, **kwargs)

        locs = np.zeros((int((position - 1) * position / 2), position + 1))
        loc = (self.yamax + 3 * self.dy) if direction == "vertical" else (self.xamax + 3 * self.dx)
        if l0_stats:
            for i, (series_group, positions_) in enumerate(zip(grouped_series, positions)):
                for j, (series0, position0) in enumerate(zip(series_group, positions_)):
                    for k, (series1, position1) in enumerate(zip(series_group, positions_)):
                        if k > j:
                            p_value = series0.different_from(series1.series, **kwargs)
                            p = min([p_value, 1 - p_value])
                            if p < 0.05:
                                min_loc_pos = np.nonzero(np.sum(locs[:, position0:position1], axis=1) == 0)[0][0]
                                loc_ = loc + min_loc_pos * self.dy
                                locs[min_loc_pos, position0:position1] = 1
                                position01 = (position0 + position1) / 2
                                self.plot(*[[position0 + self.lwic, position01], [loc_, loc_]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][j]), linewidth=self.lw, zorder=2.01)
                                self.plot(*[[position0 + self.lwic, position0 + self.lwic], [loc_, loc_ - self.dy / 4]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][j]), linewidth=self.lw, zorder=2.01)
                                self.plot(*[[position01, position1 - self.lwic], [loc_, loc_]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][k]), linewidth=self.lw, zorder=2.01)
                                self.plot(*[[position1 - self.lwic, position1 - self.lwic], [loc_, loc_ - self.dy / 4]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][k]), linewidth=self.lw, zorder=2.01)
                                self.text(*[position01, loc_][::1 if direction == "vertical" else -1], "${} {}$".format(">" if p_value > 0.95 else "<", "***" if p < 0.001 else ("**" if p < 0.01 else "*")),
                                          rotation=0 if direction == "vertical" else 270,
                                          ha="center" if direction == "vertical" else "left",
                                          va="bottom" if direction == "vertical" else "center",
                                          fontsize=self.ms)

        if l1_stats:
            for i, (series_group_a, positions_a) in enumerate(zip(grouped_series, positions)):
                for j, (series0, position0) in enumerate(zip(series_group_a, positions_a)):
                    for k, (series_group_b, positions_b) in enumerate(zip(grouped_series, positions)):
                        if k > i:
                            for l, (series1, position1) in enumerate(zip(series_group_b, positions_b)):
                                p_value = series0.different_from(series1.series, **kwargs)
                                p = min([p_value, abs(1 - p_value)])
                                if p < 0.05:
                                    min_loc_pos = np.nonzero(np.sum(locs[:, position0:position1], axis=1) == 0)[0][0]
                                    loc_ = loc + min_loc_pos * self.dy
                                    locs[min_loc_pos, position0:position1] = 1
                                    position01 = (position0 + position1) / 2
                                    self.plot(*[[position0 + self.lwic, position01], [loc_, loc_]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][j]), linewidth=self.lw, zorder=2.01)
                                    self.plot(*[[position0 + self.lwic, position0 + self.lwic], [loc_, loc_ - self.dy / 4]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][j]), linewidth=self.lw, zorder=2.01)
                                    self.plot(*[[position01, position1 - self.lwic], [loc_, loc_]][::1 if direction == "vertical" else -1], color=self.get_color(colors[k][l]), linewidth=self.lw, zorder=2.01)
                                    self.plot(*[[position1 - self.lwic, position1 - self.lwic], [loc_, loc_ - self.dy / 4]][::1 if direction == "vertical" else -1], color=self.get_color(colors[k][l]), linewidth=self.lw, zorder=2.01)
                                    self.text(*[position01, loc_][::1 if direction == "vertical" else -1], "${} {}$".format(">" if p_value > 0.95 else "<", "***" if p < 0.001 else ("**" if p < 0.01 else "*")),
                                              rotation=0 if direction == "vertical" else 270,
                                              ha="center" if direction == "vertical" else "left",
                                              va="bottom" if direction == "vertical" else "center",
                                              fontsize=self.ms)


if __name__ == "__main__":
    data = [
        [
            2 * np.random.rand(250) + 2
        ],
        [
            5 * np.random.rand(150),
            5 * np.random.rand(100) + 1
        ],
        [
            2 * np.random.rand(150) + 5,
            2 * np.random.rand(100) + 3,
            3 * np.random.rand(150) + 7,
            np.random.rand(100) + 2,
            2 * np.random.rand(100) + 3,
            5 * np.random.rand(150) + 7,
            7 * np.random.rand(100) + 3
        ],
        [
            np.random.rand(150),
            2 * np.random.rand(100) + 1,
            np.random.rand(100) + 10
        ]
    ]
    jb = Boxplot(data, yalim=[0, 13], project_stats=False, plot_violin=True, direction="vertical", different_from=1.5, labels=["group1", "group2", "group3", "group4"], l0_stats=True, l1_stats=True, top_extent=2.5, right_extent=0, inchesperposition=1, pairwise=False, use_tex=False)
    jb.show()
    jb.savefig("/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/data/phd/pictures/bootstrap_maps/test1.pdf")