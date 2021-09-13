import numpy as np
from collections import Iterable
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon


color_dict = {
    "r": (1, 0, 0),
    "g": (0, 1, 0),
    "b": (0, 0, 1)
}


class Series(object):
    def __init__(self, series):
        if not isinstance(series, Iterable):
            series = [series]

        self.series = np.array(series)
        self.series_, self.nnans, self.min, self.p25, self.p50, self.pm, self.p75, self.max, self.iqr, self.pmin, self.pmax, self.outliers = self.get_stats()
        self.median, self.mean = self.p50, self.pm

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        return self.series[idx]

    def __iter__(self):
        return iter(self.series)

    def different_from(self, value=0, **kwargs):
        return self.basic_test(self.series, value if isinstance(value, Iterable) else [value] * len(self.series), **kwargs)

    def get_stats(self):
        return self.calculate_stats(self.series)

    @staticmethod
    def calculate_stats(series):
        series_ = np.array([value for value in series if not np.isnan(value) and not np.isinf(value)])
        nnans = len([value for value in series if np.isnan(value) or np.isinf(value)])
        min = np.min(series_)
        p25 = np.percentile(series_, 25)
        p50 = np.percentile(series_, 50)
        pm = np.mean(series_)
        p75 = np.percentile(series_, 75)
        max = np.max(series_)
        iqr = p75 - p25
        pmin = np.min([value for value in series_ if value >= p25 - 1.5 * iqr])
        pmax = np.max([value for value in series_ if value <= p75 + 1.5 * iqr])
        outliers = np.array([value for value in series_ if (value > pmax or value < pmin)])
        return series_, nnans, min, p25, p50, pm, p75, max, iqr, pmin, pmax, outliers

    @staticmethod
    def basic_test(series0, series1=None, n=10000, skipnan=True, skipinf=True, pairwise=True, **kwargs):
        series0 = np.array(series0)
        if series1 is None:
            series1 = np.zeros_like(series0)

        else:
            series1 = np.array(series1)

        if pairwise:
            assert len(series0) == len(series1), "For a pairwise test the original series must be of equal length."
            series0 = series0 - series1
            series1 = np.zeros_like(series0)

        if skipnan:
            series0, series1 = zip(*[(s0, s1) for s0, s1 in zip(series0, series1) if not np.isnan(s0) and not np.isnan(s1)])

        if skipinf:
            series0, series1 = zip(*[(s0, s1) for s0, s1 in zip(series0, series1) if not np.isinf(s0) and not np.isinf(s1)])

        if len(series0) > 0 and len(series1) > 0 and np.sum(np.isnan(series0)) == 0 and np.sum(np.isinf(series0)) == 0 and np.sum(np.isnan(series1)) == 0 and np.sum(np.isinf(series1)) == 0:
            if pairwise and np.all(series0 == 0):
                return 0.5

            else:
                count = 0
                for i in range(n):
                    series0_ = np.random.choice(series0, len(series0))
                    series1_ = np.random.choice(series1, len(series1))
                    if np.mean(series0_) - np.mean(series1_) > 0:
                        count += 1

                return count / n

        else:
            return np.nan


class SeriesGroup(object):
    def __init__(self, series_group):
        if not isinstance(series_group, Iterable):
            series_group = [series_group]

        for i, series in enumerate(series_group):
            series_group[i] = Series(series)

        self.series_group = series_group
        self.series = Series([value for series in series_group for value in series])

    def __len__(self):
        return len(self.series_group)

    def __getitem__(self, idx):
        return self.series_group[idx]

    def __iter__(self):
        return iter(self.series_group)


class GroupedSeries(object):
    def __init__(self, grouped_series):
        if not isinstance(grouped_series, Iterable):
            grouped_series = [grouped_series]

        for i, series_group in enumerate(grouped_series):
            grouped_series[i] = SeriesGroup(series_group)

        self.grouped_series = grouped_series
        self.series = Series([value for series_group in grouped_series for value in series_group.series])

    def __len__(self):
        return len(self.grouped_series)

    def __getitem__(self, idx):
        return self.grouped_series[idx]

    def __iter__(self):
        return iter(self.grouped_series)


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
        e = self.lw / 2 / self.fig.get_dpi() * self.awidth / self.awidthininches
        self.ax.add_patch(Rectangle((self.xmin + e, self.yamax + e), self.width - 2 * e, self.ymax - self.yamax - 2 * e, fc='w', ec='w', linewidth=0, zorder=2.001))
        self.ax.add_patch(Rectangle((self.xmin + e, self.ymin + e), self.width - 2 * e, self.dy - 2 * e, fc='w', ec='w', linewidth=0, zorder=2.001))
        self.ax.add_patch(Rectangle((self.xmin + e, self.ymin + e), self.dx - 2 * e, self.height - 2 * e, fc='w', ec='w', linewidth=0, zorder=2.001))
        self.ax.add_patch(Rectangle((self.xamax + e, self.ymin + e), self.xmax - self.xamax - 2 * e, self.height - 2 * e, fc='w', ec='w', linewidth=0, zorder=2.001))
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

            self.text(pos, self.yamax + self.dy, text, rotation=0, ha="center", va="bottom", fontsize=self.ms)
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

            self.text(self.xamax + self.dx, pos, text, rotation=270, ha="left", va="center", fontsize=self.ms)
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

        loc = (self.yamax + 3 * self.dy) if direction == "vertical" else (self.xamax + 3 * self.dx)
        if l0_stats:
            for i, (series_group, positions_) in enumerate(zip(grouped_series, positions)):
                loc_ = (self.yamax + 3 * self.dy) if direction == "vertical" else (self.xamax + 3 * self.dx)
                for j, (series0, position0) in enumerate(zip(series_group, positions_)):
                    for k, (series1, position1) in enumerate(zip(series_group, positions_)):
                        if k > j:
                            p_value = series0.different_from(series1, **kwargs)
                            p = min([p_value, 1 - p_value])
                            if p < 0.05:
                                position01 = (position0 + position1) / 2
                                self.plot(*[[position0, position01], [loc_, loc_]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][j]), linewidth=self.lw, zorder=2.01)
                                self.plot(*[[position0, position0], [loc_, loc_ - self.dy / 4]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][j]), linewidth=self.lw, zorder=2.01)
                                self.plot(*[[position01, position1], [loc_, loc_]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][k]), linewidth=self.lw, zorder=2.01)
                                self.plot(*[[position1, position1], [loc_, loc_ - self.dy / 4]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][k]), linewidth=self.lw, zorder=2.01)
                                self.text(*[(position0 + position1) / 2, loc_][::1 if direction == "vertical" else -1], "${} {}$".format(">" if p_value > 0.95 else "<", "***" if p < 0.001 else ("**" if p < 0.01 else "*")),
                                          rotation=0 if direction == "vertical" else 270,
                                          ha="center" if direction == "vertical" else "left",
                                          va="bottom" if direction == "vertical" else "center",
                                          fontsize=self.ms)
                                loc_ += self.dy if direction == "vertical" else self.dx

                if loc_ > loc:
                    loc = loc_

        if l1_stats:
            loc_ = loc + 2 * (self.dy if direction == "vertical" else self.dx)
            for i, (series_group_a, positions_a) in enumerate(zip(grouped_series, positions)):
                for j, (series0, position0) in enumerate(zip(series_group_a, positions_a)):
                    for k, (series_group_b, positions_b) in enumerate(zip(grouped_series, positions)):
                        if k > i:
                            for l, (series1, position1) in enumerate(zip(series_group_b, positions_b)):
                                p_value = series0.different_from(series1, **kwargs)
                                p = min([p_value, abs(1 - p_value)])
                                if p < 0.05:
                                    position01 = (position0 + position1) / 2
                                    self.plot(*[[position0, position01], [loc_, loc_]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][j]), linewidth=self.lw, zorder=2.01)
                                    self.plot(*[[position0, position0], [loc_, loc_ - self.dy / 4]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][j]), linewidth=self.lw, zorder=2.01)
                                    self.plot(*[[position01, position1], [loc_, loc_]][::1 if direction == "vertical" else -1], color=self.get_color(colors[k][l]), linewidth=self.lw, zorder=2.01)
                                    self.plot(*[[position1, position1], [loc_, loc_ - self.dy / 4]][::1 if direction == "vertical" else -1], color=self.get_color(colors[k][l]), linewidth=self.lw, zorder=2.01)
                                    self.text(*[(position0 + position1) / 2, loc_][::1 if direction == "vertical" else -1], "${} {}$".format(">" if p_value > 0.95 else "<", "***" if p < 0.001 else ("**" if p < 0.01 else "*")),
                                              rotation=0 if direction == "vertical" else 270,
                                              ha="center" if direction == "vertical" else "left",
                                              va="bottom" if direction == "vertical" else "center",
                                              fontsize=self.ms)
                                    loc_ += self.dy if direction == "vertical" else self.dx


if __name__ == "__main__":
    data = [[np.random.rand(250) + 2], [np.random.rand(150), np.random.rand(100) + 1]]
    jb = Boxplot(data, yalim=[0, 3], project_stats=True, plot_violin=True, direction="vertical", different_from=1.5, labels=["group1", "group2"], l0_stats=True, l1_stats=True, top_extent=0.75, right_extent=0, inchesperposition=1, pairwise=False)
    jb.show()
    jb.savefig("/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/data/phd/pictures/bootstrap_maps/test1.pdf")
