import numpy as np
import seaborn as sb
from copy import deepcopy
from collections import Iterable
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


color_dict = {
    "r": (1, 0, 0),
    "g": (0, 1, 0),
    "b": (0, 0, 1),
    "c": (0, 1, 1),
    "m": (1, 0, 1),
    "tab:orange": (1.0, 0.4980392156862745, 0.054901960784313725),  # to find rgb(a) values use: from matplotlib import colors; colors.to_rgba(color_name)
    "tab:brown": (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
    "y": (1, 1, 0),
    "k": (0, 0, 0),
    "w": (1, 1, 1)
}

class Series(object):
    def __init__(self, series):
        series = deepcopy(series)
        if not isinstance(series, Iterable):
            series = [series]

        self.series = np.array(series)
        self.series_, self.nnan, self.ninf, self.nnaninf, self.min, self.p25, self.p50, self.pm, self.p75, self.max, self.iqr, self.pmin, self.pmax, self.outliers, self.std, self.ste = self.get_stats()
        self.median, self.mean = self.p50, self.pm

    def __len__(self):
        return len(self.series)

    def __getitem__(self, idx):
        return self.series[idx]

    def __iter__(self):
        return iter(self.series)

    def different_from(self, value=0, **kwargs):
        return self.basic_test(self.series, value if isinstance(value, Iterable) else [value] * len(self.series), **kwargs)

    def correlate_with(self, series):
        return self.pearson_correlation(self, series)

    def get_stats(self):
        return self.calculate_stats(self.series)

    @staticmethod
    def calculate_stats(series):
        series_ = np.array([value for value in series if not np.isnan(value) and not np.isinf(value)])
        nnan = len([value for value in series if np.isnan(value)])
        ninf = len([value for value in series if np.isinf(value)])
        nnaninf = nnan + ninf
        if len(series_) > 0:
            min = np.min(series_)
            p25 = np.percentile(series_, 25)
            p50 = np.percentile(series_, 50)
            pm = np.mean(series_)
            p75 = np.percentile(series_, 75)
            max = np.max(series_)
            iqr = p75 - p25
            pmin = np.min([value for value in series_ if value >= p25 - 1.5 * iqr])
            pmax = np.max([value for value in series_ if value <= p75 + 1.5 * iqr])
            std = np.std(series_)
            ste = std / np.sqrt(len(series_))

        else:
            min, p25, p50, pm, p75, max, iqr, pmin, pmax, std, ste = [np.nan] * 11

        outliers = np.array([value for value in series_ if (value > pmax or value < pmin)])
        return series_, nnan, ninf, nnaninf, min, p25, p50, pm, p75, max, iqr, pmin, pmax, outliers, std, ste

    @staticmethod
    def basic_test(series0, series1=None, n=10000, skipnan=True, skipinf=True, pairwise=True, **kwargs):
        series0 = Series(series0).series
        series1 = np.zeros_like(series0) if series1 is None else Series(series1).series
        if pairwise:
            assert len(series0) == len(series1), "For a pairwise test the original series must be of equal length."
            series0 = series0 - series1
            series1 = np.array([s0 if np.isnan(s0) or np.isinf(s0) else 0 for s0 in series0])

        if skipnan:
            series0 = series0[~np.isnan(series0)]
            series1 = series1[~np.isnan(series1)]

        if skipinf:
            series0 = series0[~np.isinf(series0)]
            series1 = series1[~np.isinf(series1)]

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

    @staticmethod
    def rank_series(series_group, ranking_mode="any", p_value_threshold=0.05, value_mode="max", **kwargs):
        series_group = SeriesGroup(series_group)
        series_group = [Series((-1 if value_mode == "min" else 1) * series.series) for series in series_group]
        mean_sort_idx = np.argsort([series.mean for series in series_group])[::-1]
        series_group = [series_group[i] for i in mean_sort_idx]
        p_values = np.full((len(series_group), len(series_group)), 0.5)
        for i, series_i in enumerate(series_group):
            for j, series_j in enumerate(series_group):
                p_value = Series.basic_test(series_i.series, series_j.series, **kwargs)
                p_values[i, j] = p_value

        ranking = list(range(len(series_group)))
        ranking_ = [None] * len(series_group)
        prev_rank = 0
        prev_leads = []
        for i in ranking:
            tests = []
            for prev_lead in prev_leads:
                if p_values[ranking.index(prev_lead), i] > 1 - p_value_threshold:
                    tests.append(True)

                else:
                    tests.append(False)

            if len(tests) > 0 and (all(tests) if ranking_mode == "all" else any(tests)):
                prev_rank += 1
                prev_leads = []

            prev_leads.append(i)
            ranking_[mean_sort_idx[i]] = prev_rank

        return ranking_

    @staticmethod
    def pearson_correlation(series0, series1):
        return np.corrcoef(Series(series0).series_, Series(series1).series_)[0, 1]


class SeriesGroup(object):
    def __init__(self, series_group):
        series_group = deepcopy(series_group)
        if not isinstance(series_group, SeriesGroup):
            if not isinstance(series_group, Iterable):
                series_group = [series_group]

            elif isinstance(series_group, tuple):
                series_group = list(series_group)

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

    def rank(self, threshold=0.05, mode="max", **kwargs):
        return Series.rank_series(self, threshold=threshold, mode=mode, **kwargs)


class GroupedSeries(object):
    def __init__(self, grouped_series):
        grouped_series = deepcopy(grouped_series)
        if not isinstance(grouped_series, GroupedSeries):
            if not isinstance(grouped_series, Iterable):
                grouped_series = [grouped_series]

            elif isinstance(grouped_series, tuple):
                grouped_series = list(grouped_series)

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
                 lmwidthininches=1, rmwidthininches=0,
                 bmheightininches=1, tmheightininches=0,
                 top_extent=0, right_extent=0,
                 fs=20, lw=2, ms=10,
                 mask_inner_region=True, mask_outer_region=True,
                 plot_xaxis=True, plot_yaxis=True,
                 use_tex=True, **kwargs):

        if use_tex:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "Helvetica",
                "text.latex.preamble": r'\usepackage{amssymb,amsthm}\usepackage{amsmath}'
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
        self.lw = self.lh = lw
        self.fs = fs
        self.ms = ms
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_visible(plot_yaxis)
        self.ax.spines['bottom'].set_visible(plot_xaxis)
        self.ax.spines['left'].set_linewidth(self.lw)
        self.ax.spines['bottom'].set_linewidth(self.lw)
        self.ax.spines['left'].set_bounds(self.yamin, self.yamax)
        self.ax.spines['bottom'].set_bounds(self.xamin, self.xamax)
        self.ax.tick_params(axis='both', which='major', labelsize=self.fs, width=self.lw)
        self.ax.set_xlim([self.xmin, self.xmax])
        self.ax.set_ylim([self.ymin, self.ymax])
        # add whitespace in between x- and y-axis in the dx and dy regions
        self.lwic = self.lw / self.fig.get_dpi() * self.awidth / self.awidthininches
        self.lhic = self.lh / self.fig.get_dpi() * self.aheight / self.aheightininches
        if mask_outer_region:
            self.ax.add_patch(Rectangle((self.xmin - self.lwic / 2, self.yamax + self.lhic / 2), self.width + self.lwic, self.ymax - self.yamax, fc='w', ec='w', linewidth=0, zorder=2.001))  # top horizontal patch
            self.ax.add_patch(Rectangle((self.xamax + self.lwic / 2, self.ymin - self.lhic / 2), self.xmax - self.xamax, self.height + self.lhic, fc='w', ec='w', linewidth=0, zorder=2.001))  # right vertical patch

        if mask_inner_region:
            self.ax.add_patch(Rectangle((self.xmin - self.lwic / 2, self.ymin + self.lhic / 2), self.width + self.lwic, self.dy - self.lhic, fc='w', ec='w', linewidth=0, zorder=2.001))  # bottom horizontal patch
            self.ax.add_patch(Rectangle((self.xmin + self.lwic / 2, self.ymin - self.lhic / 2), self.dx - self.lwic, self.height + self.lhic, fc='w', ec='w', linewidth=0, zorder=2.001))  # left vertical patch
            self.ax.add_patch(Rectangle((self.xmin - self.lwic / 2, self.ymin - self.lhic / 2), self.dx, self.dy, fc='w', ec='w', linewidth=0, zorder=2.001))  # small square patch at bottom left
        # optionally set title, labels, ticks and ticklabels
        if plot_xaxis:
            if "xlabel" in kwargs and kwargs["xlabel"] is not None:
                self.set_xlabel(kwargs["xlabel"])

            if "xticks" in kwargs and kwargs["xticks"] is not None:
                if kwargs["xticks"] == "auto":
                    assert "xticklabels" in kwargs and kwargs["xticklabels"] == "auto"
                    self.set_xticks(np.linspace(self.xamin, self.xamax, 5))

                else:
                    self.set_xticks(kwargs["xticks"] if kwargs["xticks"] is not None else [])

            else:
                self.set_xticks([xtick for xtick in self.ax.get_xticks() if self.xamin <= xtick <= self.xamax])

            if "xticklabels" in kwargs and kwargs["xticklabels"] is not None and kwargs["xticklabels"] != "auto":
                assert "xticks" in kwargs and kwargs["xticks"] is not None and len(kwargs["xticks"]) == len(kwargs["xticklabels"])
                self.set_xticklabels(kwargs["xticklabels"])

        if plot_yaxis:
            if "ylabel" in kwargs and kwargs["ylabel"] is not None:
                self.set_ylabel(kwargs["ylabel"])

            if "yticks" in kwargs and kwargs["yticks"] is not None:
                if kwargs["yticks"] == "auto":
                    assert "yticklabels" in kwargs and kwargs["yticklabels"] == "auto"
                    self.set_yticks(np.linspace(self.yamin, self.yamax, 5))

                else:
                    self.set_yticks(kwargs["yticks"] if kwargs["yticks"] is not None else [])

            else:
                self.set_yticks([ytick for ytick in self.ax.get_yticks() if self.yamin <= ytick <= self.yamax])

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
        self.ax.xaxis.set_label_coords((self.dxininches + self.awidthininches / 2) / self.widthininches, -self.bmheightininches / self.heightininches)

    def set_ylabel(self, ylabel):
        self.ax.set_ylabel(ylabel, fontsize=self.fs, horizontalalignment='center', verticalalignment='top')
        self.ax.yaxis.set_label_coords(-self.lmwidthininches / self.widthininches, (self.dyininches + self.aheightininches / 2) / self.heightininches)

    def set_xticks(self, xticks):
        self.ax.set_xticks(xticks)

    def set_xticklabels(self, xticklabels):
        self.ax.set_xticklabels(xticklabels)

    def set_yticks(self, yticks):
        self.ax.set_yticks(yticks)

    def set_yticklabels(self, yticklabels):
        self.ax.set_yticklabels(yticklabels)

    def add_patch(self, patch):
        self.ax.add_patch(patch)

    def text(self, x, y, s, fontdict=None, **kwargs):
        self.ax.text(x, y, s, fontdict=fontdict, **kwargs)

    def show(self):
        self.fig.show()

    def savefig(self, file_path, **kwargs):
        self.fig.savefig(file_path, **kwargs)

    @staticmethod
    def prepare(mode, x_and_y_data, xalim=None, yalim=None, colors=None, color_mode="series", **kwargs):
        if mode == "series":
            x_and_y_data = [Series(data) if data is not None else None for data in x_and_y_data]
            colors = list(color_dict.keys())[0] if colors is None else colors
            position, positions, ticks = None, None, None

        elif mode == "series_group":
            x_and_y_data = [SeriesGroup(data) if data is not None else None for data in x_and_y_data]
            if color_mode == "series":
                colors = [list(color_dict.keys())[i] for i, series in enumerate(x_and_y_data[0])] if colors is None else colors

            else:
                colors = list(color_dict.keys())[0] if colors is None else colors

            position, positions, ticks = None, None, None

        elif mode == "grouped_series":
            x_and_y_data = [GroupedSeries(data) if data is not None else None for data in x_and_y_data]
            if color_mode == "series":
                colors = [[list(color_dict.keys())[i] for i, series in enumerate(series_group)] for series_group in x_and_y_data[0]] if colors is None else colors

            else:
                colors = [list(color_dict.keys())[i] for i, series_group in enumerate(x_and_y_data[0])] if colors is None else colors

            position, positions, ticks = None, kwargs.get("positions", None), None
            if "positions" in kwargs:
                if positions is None:
                    position, positions = 0, []
                    for series_group in x_and_y_data[0]:
                        position += 1
                        positions_ = []
                        for series in series_group:
                            positions_.append(position)
                            position += 1

                        positions.append(positions_)

                    kwargs["positions"] = positions

                ticks = [np.mean(positions_) for positions_ in kwargs["positions"]]

        else:
            raise ValueError("Unknown data prepare mode.")

        if xalim is None:
            xalim = [None, None]

        if xalim[0] is None:
            xalim[0] = 0 if positions is not None else x_and_y_data[0].series.min

        if xalim[1] is None:
            xalim[1] = positions[-1][-1] + 1 if positions is not None else x_and_y_data[0].series.max

        if yalim is None:
            yalim = [None, None]

        if yalim[0] is None:
            yalim[0] = x_and_y_data[0].series.min if positions is not None else x_and_y_data[1].series.min

        if yalim[1] is None:
            yalim[1] = x_and_y_data[0].series.max if positions is not None else x_and_y_data[1].series.max

        inchesperposition = kwargs.get("inchesperposition", None)
        direction = kwargs.get("direction", None)
        if direction == "vertical":
            kwargs["xticks"] = kwargs.get("xticks", ticks)
            kwargs["xticklabels"] = kwargs.get("xticklabels", kwargs.get("labels", None))
            if inchesperposition is not None:
                kwargs["awidthininches"] = position * inchesperposition

        elif direction == "horizontal":
            kwargs["yticks"] = kwargs.get("yticks", ticks)
            kwargs["yticklabels"] = kwargs.get("yticklabels", kwargs.get("labels", None))
            xalim, yalim = yalim, xalim
            if inchesperposition is not None:
                kwargs["aheightininches"] = position * inchesperposition

        return x_and_y_data, xalim, yalim, colors, kwargs

    @staticmethod
    def get_color(color, alpha=None):
        color_tuple = color_dict[color] if isinstance(color, str) else color
        if len(color_tuple) == 3:
            return color_tuple + (1 if alpha is None else alpha,)

        if len(color_tuple) == 4:
            assert alpha is None or alpha == color_tuple[3]
            return color_tuple

    def plot(self, *args, **kwargs):
        self.ax.plot(*args, **kwargs)

    def lineplot(self, series_x, series_y, color=(0, 0, 1, 1), alpha=1, marker=".", linestyle="-", linewidth=None, markersize=None, **kwargs):
        series_x, series_y = Series(series_x), Series(series_y)
        color = self.get_color(color, alpha)
        self.plot(series_x, series_y, color=color, linewidth=self.lw if linewidth is None else linewidth, zorder=1.9999, linestyle=linestyle)
        if marker is not None:
            self.plot(series_x, series_y, color=color, linestyle="None", marker=marker, markersize=self.ms if markersize is None else markersize)

    def lineplotwithstats(self, series_group_x, series_group_y, color=(0, 0, 1, 1), alpha=1, marker=".", linestyle="-", alpha_stats=0.5, linestyle_stats=None, plot_std=False, plot_ste=True, plot_iqr=False, **kwargs):
        series_group_x, series_group_y = SeriesGroup(series_group_x), SeriesGroup(series_group_y)
        series_x = [Series([series[i] for series in series_group_x]) for i in range(len(series_group_x[0]))]
        series_y = [Series([series[i] for series in series_group_y]) for i in range(len(series_group_y[0]))]
        self.lineplot([series.mean for series in series_x], [series.mean for series in series_y], color=color, alpha=alpha, marker=marker, linestyle=linestyle, **kwargs)
        if plot_iqr:
            self.lineplot([series.p25 for series in series_x], [series.p25 for series in series_y], color=color, alpha=alpha if alpha_stats is None else alpha_stats, marker=marker, linestyle=linestyle if linestyle_stats is None else linestyle_stats, **kwargs)
            self.lineplot([series.p75 for series in series_x], [series.p75 for series in series_y], color=color, alpha=alpha if alpha_stats is None else alpha_stats, marker=marker, linestyle=linestyle if linestyle_stats is None else linestyle_stats, **kwargs)

        if plot_std:
            self.lineplot([series.mean - series.std for series in series_x], [series.mean - series.std for series in series_y], color=color, alpha=alpha if alpha_stats is None else alpha_stats, marker=marker, linestyle=linestyle if linestyle_stats is None else linestyle_stats, **kwargs)
            self.lineplot([series.mean + series.std for series in series_x], [series.mean + series.std for series in series_y], color=color, alpha=alpha if alpha_stats is None else alpha_stats, marker=marker, linestyle=linestyle if linestyle_stats is None else linestyle_stats, **kwargs)

        if plot_ste:
            self.lineplot([series.mean - series.ste for series in series_x], [series.mean - series.ste for series in series_y], color=color, alpha=alpha if alpha_stats is None else alpha_stats, marker=marker, linestyle=linestyle if linestyle_stats is None else linestyle_stats, **kwargs)
            self.lineplot([series.mean + series.ste for series in series_x], [series.mean + series.ste for series in series_y], color=color, alpha=alpha if alpha_stats is None else alpha_stats, marker=marker, linestyle=linestyle if linestyle_stats is None else linestyle_stats, **kwargs)

    def scatterplot(self, series_x, series_y, color=(0, 0, 1, 1), alpha=1, marker=".", markerfill="full", markersize=None, plot_scatter=True, plot_unity=False, plot_mean=False, plot_kde=False, nbins=0, groupn=0, linestyle="-", ncontours=10, fillcontours=False, markeredgewidth=0, print_correlation=False, **kwargs):
        series_x, series_y = Series(series_x), Series(series_y)
        ms = self.ms if markersize is None else markersize
        if print_correlation:
            text = "Pearson correlation: {:.2f}".format(Series.pearson_correlation(series_x, series_y))
            self.text((self.xamin + self.xamax) / 2, self.yamax + self.dy, text, rotation=0, ha="center", va="bottom", fontsize=self.fs)

        if plot_unity:
            self.plot([self.xamin, self.xamax], [self.xamin, self.xamax], "k:", linewidth=self.lw, zorder=1.9)

        color = self.get_color(color, alpha)
        if plot_scatter:
            self.plot(series_x, series_y, color=color, linestyle="None", marker=marker, fillstyle=markerfill, markersize=ms, markeredgewidth=markeredgewidth, zorder=1.99)

        if plot_mean:
            self.plot(series_x.mean, series_y.mean, color=color[:3], alpha=1, linestyle="None", fillstyle=markerfill, marker=marker, markeredgewidth=markeredgewidth, markersize=ms * 3, zorder=1.999)

        if nbins > 0:
            assert groupn == 0
            color = self.get_color(color[:3], alpha * 0.25)
            sort_idx = np.argsort(series_x)
            series_x_for_binning = [x for x in series_x[sort_idx] for _ in range(nbins)]
            series_y_for_binning = [y for y in series_y[sort_idx] for _ in range(nbins)]
            series_x_means = [np.percentile(series_x, 100 * b / nbins) for b in range(nbins + 1)]
            series_y_means = [np.mean(series_y_for_binning[len(series_x) * b:len(series_x) * (b + 1)]) for b in range(nbins)]
            for b in range(nbins):
                self.plot([series_x_means[b], series_x_means[b + 1]], [series_y_means[b], series_y_means[b]], color=color[:3], alpha=1, linewidth=self.lw, linestyle=linestyle, zorder=1.9999)
                if b < nbins - 1:
                    self.plot([series_x_means[b + 1], series_x_means[b + 1]], [series_y_means[b], series_y_means[b + 1]], color=color[:3], alpha=1, linewidth=self.lw, linestyle=linestyle, zorder=1.9999)

        elif groupn > 0:
            color = self.get_color(color[:3], alpha * 0.25)
            sort_idx = np.argsort(series_x)
            series_x_for_grouping = series_x[sort_idx]
            series_y_for_grouping = series_y[sort_idx]
            ngroups = max(int(np.floor(len(series_x) / groupn)), 1)
            series_x_means = [series_x_for_grouping[0] if g == 0 else (series_x_for_grouping[-1] if g == ngroups else np.mean(series_x_for_grouping[g * groupn - 1:g * groupn + 1])) for g in range(ngroups + 1)]
            series_y_means = [np.mean(series_y_for_grouping[g * groupn:(g + 1) * groupn if g < ngroups - 1 else None]) for g in range(ngroups)]
            for g in range(ngroups):
                self.plot([series_x_means[g], series_x_means[g + 1]], [series_y_means[g], series_y_means[g]], color=color[:3], alpha=1, linewidth=self.lw, linestyle=linestyle, zorder=1.9999)
                if g < ngroups - 1:
                    self.plot([series_x_means[g + 1], series_x_means[g + 1]], [series_y_means[g], series_y_means[g + 1]], color=color[:3], alpha=1, linewidth=self.lw, linestyle=linestyle, zorder=1.9999)

        if plot_kde:
            sb.kdeplot(ax=self.ax, x=series_x, y=series_y, colors=[color[:3]], alpha=alpha, linewidths=self.lw, zorder=1.9999, linestyles=linestyle, levels=ncontours, fill=fillcontours)

    def blandaltmanplot(self, series_x, series_y, color=(0, 0, 1, 1), alpha=1, marker=".", plot_unity=True, nbins=0, alpha_stats=None, **kwargs):
        series_x, series_y = Series(series_x), Series(series_y)
        series_mean = Series(np.mean([series_x.series, series_y.series], axis=0))
        series_diff = Series(series_y.series - series_x.series)
        if plot_unity:
            self.plot([self.xamin, self.xamax], [0, 0], "k", linewidth=self.lw, zorder=1.99)

        if nbins > 0:
            color = self.get_color(color[:3], alpha * 0.5)
            sort_idx = np.argsort(series_mean)
            series_mean_for_binning = [x for x in series_mean[sort_idx] for _ in range(nbins)]
            series_diff_for_binning = [y for y in series_diff[sort_idx] for _ in range(nbins)]
            series_mean_means = [np.percentile(series_mean, 100 * b / nbins) for b in range(nbins + 1)]
            series_diff_means = [np.mean(series_diff_for_binning[len(series_mean) * b:len(series_mean) * (b + 1)]) for b in range(nbins)]
            for b in range(nbins):
                self.plot([series_mean_means[b], series_mean_means[b + 1]], [series_diff_means[b], series_diff_means[b]], color=color[:3], alpha=alpha, linewidth=self.lw, zorder=1.999)
                if b < nbins - 1:
                    self.plot([series_mean_means[b + 1], series_mean_means[b + 1]], [series_diff_means[b], series_diff_means[b + 1]], color=color[:3], alpha=alpha, linewidth=self.lw, zorder=1.999)

        else:
            color = self.get_color(color, alpha)

        self.plot([self.xamin, self.xamax], [series_diff.mean, series_diff.mean], color=color, alpha=alpha if alpha_stats is None else alpha_stats, linewidth=self.lw, zorder=1.9999)
        self.plot([self.xamin, self.xamax], [series_diff.mean + 1.96 * series_diff.std, series_diff.mean + 1.96 * series_diff.std], linestyle=":", color=color, alpha=alpha if alpha_stats is None else alpha_stats, linewidth=self.lw, zorder=1.9999)
        self.plot([self.xamin, self.xamax], [series_diff.mean - 1.96 * series_diff.std, series_diff.mean - 1.96 * series_diff.std], linestyle=":", color=color, alpha=alpha if alpha_stats is None else alpha_stats, linewidth=self.lw, zorder=1.9999)
        self.plot(series_mean, series_diff, color=color, linestyle="None", marker=marker, markersize=self.ms)

    def boxplot(self, series, pos, direction="vertical", width=0.8, fc=(0, 0, 1, 0.5), ec=None, project_stats=False, plot_violin=False, violin_color=None, print_mean=True, different_from=None, mean_formatting="{0:.3g}", **kwargs):
        series = Series(series)
        width2 = width / 2
        if ec is None:
            ec = (fc[0], fc[1], fc[2], 1)

        text = ""
        if print_mean:
            text += "$"
            text += mean_formatting.format(series.mean)
            text += "^{" + f"{series.nnaninf if series.nnaninf > 0 else ''}" + "}"
            text += "$"

        sig_text = ""
        if different_from is not None:
            p_value = series.different_from(different_from, **kwargs)
            p_value_ = min(p_value, 1 - p_value)
            if p_value_ < 0.05:
                sig_text += "$"
                sig_text += ">" if p_value > 0.95 else "<"
                sig_text += "*" if 0.01 < p_value_ < 0.05 else ("**" if 0.001 < p_value_ < 0.01 else "***")
                sig_text += "$"

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
            self.plot([pos] * len(series.outliers), series.outliers, color=ec, linestyle="None", marker=".", markersize=self.ms)
            self.plot([pos - width2, pos + width2], [series.p50, series.p50], color=ec, linewidth=self.lw)
            self.plot([pos - width2, pos + width2], [series.pm, series.pm], color=ec, linestyle="dashed", linewidth=self.lw)
            self.plot([pos - width2, pos + width2], [series.pmax, series.pmax], color=ec, linewidth=self.lw)
            self.plot([pos - width2, pos + width2], [series.pmin, series.pmin], color=ec, linewidth=self.lw)
            if project_stats:
                self.plot([self.xmin, pos - width2], [series.p50, series.p50], color=ec, linewidth=self.lw, zorder=2.01)
                self.plot([self.xmin, pos - width2], [series.pm, series.pm], color=ec, linestyle="dashed", linewidth=self.lw, zorder=2.01)
                self.plot([self.xmin, pos - width2], [series.p75, series.p75], color=ec, linewidth=self.lw, zorder=2.01)
                self.plot([self.xmin, pos - width2], [series.p25, series.p25], color=ec, linewidth=self.lw, zorder=2.01)

            self.text(pos, self.yamax + self.dy, text, rotation=0, ha="center", va="bottom", fontsize=self.fs)
            self.text(pos, self.yamax + self.dy - (self.fs / 2 * self.lhic / self.lw), sig_text, rotation=0, ha="center", va="bottom", fontsize=self.fs / 2)
            if different_from is not None:
                self.plot([pos - 0.5, pos + 0.5], [different_from, different_from], "k:", linewidth=self.lw, zorder=1.99)

        else:
            assert direction == "horizontal"
            self.add_patch(Rectangle((series.p25, pos - width2), series.iqr, width, fc=fc, ec=ec, linewidth=self.lw))
            self.plot([series.p75, series.pmax], [pos, pos], color=ec, linewidth=self.lw)
            self.plot([series.pmin, series.p25], [pos, pos], color=ec, linewidth=self.lw)
            self.plot(series.outliers, [pos] * len(series.outliers), color=ec, linestyle="None", marker=".", markersize=self.ms)
            self.plot([series.p50, series.p50], [pos - width2, pos + width2], color=ec, linewidth=self.lw)
            self.plot([series.pm, series.pm], [pos - width2, pos + width2], color=ec, linestyle="dashed", linewidth=self.lw)
            self.plot([series.pmax, series.pmax], [pos - width2, pos + width2], color=ec, linewidth=self.lw)
            self.plot([series.pmin, series.pmin], [pos - width2, pos + width2], color=ec, linewidth=self.lw)
            if project_stats:
                self.plot([series.p50, series.p50], [self.ymin, pos - width2], color=ec, linewidth=self.lw, zorder=2.01)
                self.plot([series.pm, series.pm], [self.ymin, pos - width2], color=ec, linestyle="dashed", linewidth=self.lw, zorder=2.01)
                self.plot([series.p75, series.p75], [self.ymin, pos - width2], color=ec, linewidth=self.lw, zorder=2.01)
                self.plot([series.p25, series.p25], [self.ymin, pos - width2], color=ec, linewidth=self.lw, zorder=2.01)

            self.text(self.xamax + self.dx, pos, text, rotation=270, ha="left", va="center", fontsize=self.fs)
            self.text(self.xamax + self.dx - (self.fs / 2 * self.lwic / self.lw), pos, sig_text, rotation=270, ha="left", va="bottom", fontsize=self.fs / 2)
            if different_from is not None:
                self.plot([different_from, different_from], [pos - 0.5, pos + 0.5], "k:", linewidth=self.lw, zorder=1.99)

    def barplot(self, series, pos, offset=0, direction="vertical", width=0.8, fc=(0, 0, 1, 0.5), ec=None, print_mean=False, plot_error_bar=False, **kwargs):
        series = Series(series)
        width2 = width / 2
        if ec is None:
            ec = (fc[0], fc[1], fc[2], 1)

        elif ec is False:
            ec = None

        text = "{0:.3g}".format(series.mean) if print_mean else ""
        if direction == "vertical":
            self.add_patch(Rectangle((pos - width2, offset), width, series.mean, fc=fc, ec=ec, linewidth=self.lw))
            self.text(pos, self.yamax + self.dy, text, rotation=0, ha="center", va="bottom", fontsize=self.fs)
            if plot_error_bar:
                self.plot([pos, pos], [offset + series.mean - series.ste / 2, offset + series.mean + series.ste / 2], color=ec, linewidth=self.lw)


class Boxplot(Figure):
    def __init__(self, grouped_series, labels=None, xalim=None, yalim=None, positions=None, inchesperposition=None, colors=None, alpha=0.5, direction="vertical", l0_stats=False, l1_stats=False, p_value_threshold=None, different_from=None, incremental_stats=False, **kwargs):
        [grouped_series], xalim, yalim, self.colors, self.kwargs = Figure.prepare("grouped_series", [grouped_series], xalim, yalim, colors, positions=positions, labels=labels, inchesperposition=inchesperposition, direction=direction, **kwargs)
        self.positions, self.labels, self.inchesperposition, self.direction, self.alpha, self.different_from, self.l0_stats, self.l1_stats, self.p_value_threshold, self.incremental_stats = self.kwargs.pop("positions"), self.kwargs.pop("labels"), self.kwargs.pop("inchesperposition"), self.kwargs.pop("direction"), alpha, different_from, l0_stats, l1_stats, p_value_threshold, incremental_stats
        super(Boxplot, self).__init__(xalim, yalim, **self.kwargs)
        self.plot_boxplot(
            self,
            grouped_series=grouped_series,
            positions=self.positions,
            colors=self.colors,
            alpha=self.alpha,
            direction=self.direction,
            different_from=self.different_from,
            l0_stats=self.l0_stats,
            l1_stats=self.l1_stats,
            p_value_threshold=self.p_value_threshold,
            incremental_stats=incremental_stats,
            **self.kwargs
        )

    @staticmethod
    def plot_boxplot(figure, grouped_series, positions, colors, alpha=0.5, direction="vertical", l0_stats=False, l1_stats=False, p_value_threshold=None, different_from=None, incremental_stats=False, **kwargs):
        grouped_series = GroupedSeries(grouped_series)
        for i, series_group in enumerate(grouped_series):
            for j, series in enumerate(series_group):
                figure.boxplot(series, pos=positions[i][j], fc=Figure.get_color(colors[i][j], alpha=alpha[i][j] if isinstance(alpha, (tuple, list)) else alpha), direction=direction, different_from=different_from[i][j] if isinstance(different_from, (tuple, list)) else different_from, **kwargs)

        positions_count, positions_indices = 0, []
        for positions_ in positions:
            positions_indices_ = []
            for position in positions_:
                positions_indices_.append(positions_count)
                positions_count += 1

            positions_indices.append(positions_indices_)

        locs = np.zeros((int(positions_count ** 2 / 2), positions_count))
        loc = (figure.yamax + 3 * figure.dy) if direction == "vertical" else (figure.xamax + 3 * figure.dx)
        if l0_stats:
            for i, (series_group, positions_) in enumerate(zip(grouped_series, positions)):
                for j, (series0, position0) in enumerate(zip(series_group, positions_)):
                    for k, (series1, position1) in enumerate(zip(series_group, positions_)):
                        if k > j:
                            if incremental_stats and k > j + 1:
                                continue

                            p_value = series0.different_from(series1.series, **kwargs)
                            p = min([p_value, 1 - p_value])
                            if (p_value_threshold is None and p < 0.05) or (p_value_threshold is not None and p < p_value_threshold):
                                lwic = figure.lwic if direction == "vertical" else figure.lhic
                                dy = figure.dy if direction == "vertical" else figure.dx
                                min_loc_pos = np.nonzero(np.sum(locs[:, positions_indices[i][j]:positions_indices[i][k]], axis=1) == 0)[0][0]
                                loc_ = loc + min_loc_pos * dy
                                locs[min_loc_pos, positions_indices[i][j]:positions_indices[i][k]] = 1
                                position01 = (position0 + position1) / 2
                                figure.plot(*[[position0 + lwic, position01], [loc_, loc_]][::1 if direction == "vertical" else -1], color=figure.get_color(colors[i][j])[:3], linewidth=figure.lw, zorder=2.01)
                                figure.plot(*[[position0 + lwic, position0 + lwic], [loc_, loc_ - dy / 4]][::1 if direction == "vertical" else -1], color=figure.get_color(colors[i][j])[:3], linewidth=figure.lw, zorder=2.01)
                                figure.plot(*[[position01, position1 - lwic], [loc_, loc_]][::1 if direction == "vertical" else -1], color=figure.get_color(colors[i][k])[:3], linewidth=figure.lw, zorder=2.01)
                                figure.plot(*[[position1 - lwic, position1 - lwic], [loc_, loc_ - dy / 4]][::1 if direction == "vertical" else -1], color=figure.get_color(colors[i][k])[:3], linewidth=figure.lw, zorder=2.01)
                                if p_value_threshold is None:
                                    figure.text(*[position01, loc_][::1 if direction == "vertical" else -1], "${} {}$".format(">" if p_value > 0.95 else "<", "***" if p < 0.001 else ("**" if p < 0.01 else "*")),
                                              rotation=0 if direction == "vertical" else 270,
                                              ha="center" if direction == "vertical" else "left",
                                              va="bottom" if direction == "vertical" else "center",
                                              fontsize=figure.fs / 2)

        if l1_stats:
            for i, (series_group_a, positions_a) in enumerate(zip(grouped_series, positions)):
                for j, (series0, position0) in enumerate(zip(series_group_a, positions_a)):
                    for k, (series_group_b, positions_b) in enumerate(zip(grouped_series, positions)):
                        if k > i:
                            if incremental_stats and k > i + 1:
                                continue

                            for l, (series1, position1) in enumerate(zip(series_group_b, positions_b)):
                                p_value = series0.different_from(series1.series, **kwargs)
                                p = min([p_value, abs(1 - p_value)])
                                if (p_value_threshold is None and p < 0.05) or (p_value_threshold is not None and p < p_value_threshold):
                                    lwic = figure.lwic if direction == "vertical" else figure.lhic
                                    dy = figure.dy if direction == "vertical" else figure.dx
                                    min_loc_pos = np.nonzero(np.sum(locs[:, positions_indices[i][j]:positions_indices[k][l]], axis=1) == 0)[0][0]
                                    loc_ = loc + min_loc_pos * dy
                                    locs[min_loc_pos, positions_indices[i][j]:positions_indices[k][l]] = 1
                                    position01 = (position0 + position1) / 2
                                    figure.plot(*[[position0 + lwic, position01], [loc_, loc_]][::1 if direction == "vertical" else -1], color=figure.get_color(colors[i][j])[:3], linewidth=figure.lw, zorder=2.01)
                                    figure.plot(*[[position0 + lwic, position0 + lwic], [loc_, loc_ - dy / 4]][::1 if direction == "vertical" else -1], color=figure.get_color(colors[i][j])[:3], linewidth=figure.lw, zorder=2.01)
                                    figure.plot(*[[position01, position1 - lwic], [loc_, loc_]][::1 if direction == "vertical" else -1], color=figure.get_color(colors[k][l])[:3], linewidth=figure.lw, zorder=2.01)
                                    figure.plot(*[[position1 - lwic, position1 - lwic], [loc_, loc_ - dy / 4]][::1 if direction == "vertical" else -1], color=figure.get_color(colors[k][l])[:3], linewidth=figure.lw, zorder=2.01)
                                    if p_value_threshold is None:
                                        figure.text(*[position01, loc_][::1 if direction == "vertical" else -1], "${} {}$".format(">" if p_value > 0.95 else "<", "***" if p < 0.001 else ("**" if p < 0.01 else "*")),
                                                  rotation=0 if direction == "vertical" else 270,
                                                  ha="center" if direction == "vertical" else "left",
                                                  va="bottom" if direction == "vertical" else "center",
                                                  fontsize=figure.fs / 2)


class Barplot(Figure):
    def __init__(self, grouped_series, labels=None, xalim=None, yalim=None, positions=None, inchesperposition=None, colors=None, alpha=0.5, direction="vertical", grouped_offsets=None, **kwargs):
        [grouped_series, self.grouped_offsets], xalim, yalim, self.colors, self.kwargs = Figure.prepare("grouped_series", [grouped_series, grouped_offsets], xalim, yalim, colors, positions=positions, labels=labels, inchesperposition=inchesperposition, direction=direction, **kwargs)
        self.positions, self.labels, self.inchesperposition, self.direction, self.alpha = self.kwargs.pop("positions"), self.kwargs.pop("labels"), self.kwargs.pop("inchesperposition"), self.kwargs.pop("direction"), alpha
        super(Barplot, self).__init__(xalim, yalim, **self.kwargs)
        self.plot_barplot(self, grouped_series, self.positions, self.colors, self.alpha, self.direction, self.grouped_offsets, **self.kwargs)

    @staticmethod
    def plot_barplot(figure, grouped_series, positions, colors, alpha=0.5, direction="vertical", grouped_offsets=None, **kwargs):
        for i, series_group in enumerate(grouped_series):
            for j, series in enumerate(series_group):
                figure.barplot(series, pos=positions[i][j], fc=Figure.get_color(colors[i][j], alpha=alpha[i][j] if isinstance(alpha, (tuple, list)) else alpha), direction=direction, offset=grouped_offsets[i][j] if grouped_offsets is not None else 0, **kwargs)


class Lineplot(Figure):
    def __init__(self, series_group_x, series_group_y, xalim=None, yalim=None, colors=None, alpha=1, **kwargs):
        [series_group_x, series_group_y], xalim, yalim, self.colors, self.kwargs = Figure.prepare("series_group", [series_group_x, series_group_y], xalim, yalim, colors, **kwargs)
        self.alpha = alpha
        super(Lineplot, self).__init__(xalim, yalim, **self.kwargs)
        self.plot_lineplot(self, series_group_x, series_group_y, self.colors, self.alpha, **self.kwargs)

    @staticmethod
    def plot_lineplot(figure, series_group_x, series_group_y, colors, alpha, **kwargs):
        for i, (series_x, series_y) in enumerate(zip(series_group_x, series_group_y)):
            figure.lineplot(series_x, series_y, color=colors[i], alpha=alpha, **kwargs)


class Lineplotwithstats(Figure):
    def __init__(self, grouped_series_x, grouped_series_y, xalim=None, yalim=None, colors=None, alpha=1, **kwargs):
        [grouped_series_x, grouped_series_y], xalim, yalim, self.colors, self.kwargs = Figure.prepare("grouped_series", [grouped_series_x, grouped_series_y], xalim, yalim, colors, color_mode="series_group", **kwargs)
        self.alpha = alpha
        super(Lineplotwithstats, self).__init__(xalim, yalim, **self.kwargs)
        self.plot_lineplotwithstats(self, grouped_series_x, grouped_series_y, self.colors, self.alpha, **self.kwargs)

    @staticmethod
    def plot_lineplotwithstats(figure, grouped_series_x, grouped_series_y, colors, alpha, **kwargs):
        for i, (series_group_x, series_group_y) in enumerate(zip(grouped_series_x, grouped_series_y)):
            figure.lineplotwithstats(series_group_x, series_group_y, color=colors[i], alpha=alpha, **kwargs)


class Scatterplot(Figure):
    def __init__(self, series_group_x, series_group_y, xalim=None, yalim=None, colors=None, alpha=1, **kwargs):
        [series_group_x, series_group_y], xalim, yalim, self.colors, self.kwargs = Figure.prepare("series_group", [series_group_x, series_group_y], xalim, yalim, colors, **kwargs)
        self.alpha = alpha
        super(Scatterplot, self).__init__(xalim, yalim, **self.kwargs)
        self.plot_scatterplot(self, series_group_x, series_group_y, self.colors, self.alpha, **self.kwargs)

    @staticmethod
    def plot_scatterplot(figure, series_group_x, series_group_y, colors, alpha, **kwargs):
        for i, (series_x, series_y) in enumerate(zip(series_group_x, series_group_y)):
            if len(series_x) > 0:
                figure.scatterplot(series_x, series_y, color=colors[i], alpha=alpha, **kwargs)


class Blandaltmanplot(Figure):
    def __init__(self, series_group_x, series_group_y, xalim=None, yalim=None, colors=None, alpha=1, **kwargs):
        series_group_x, series_group_y = SeriesGroup(series_group_x), SeriesGroup(series_group_y)
        series_group_mean = SeriesGroup([np.mean([series_x.series, series_y.series], axis=0) for series_x, series_y in zip(series_group_x, series_group_y)])
        series_group_diff = SeriesGroup([series_y.series - series_x.series for series_x, series_y in zip(series_group_x, series_group_y)])
        [series_group_mean, series_group_diff], xalim, yalim, self.colors, self.kwargs = Figure.prepare("series_group", [series_group_mean, series_group_diff], xalim, yalim, colors, **kwargs)
        self.alpha = alpha
        super(Blandaltmanplot, self).__init__(xalim, yalim, **self.kwargs)
        self.plot_blandaltmanplot(self, series_group_x, series_group_y, self.colors, self.alpha, **self.kwargs)

    @staticmethod
    def plot_blandaltmanplot(figure, series_group_x, series_group_y, colors, alpha, **kwargs):
        for i, (series_x, series_y) in enumerate(zip(series_group_x, series_group_y)):
            figure.blandaltmanplot(series_x, series_y, color=colors[i], alpha=alpha, **kwargs)


if __name__ == "__main__":
    # Example data
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
    scatter_x = [np.random.rand(250), np.random.rand(250)]
    scatter_y = [np.random.rand(250), np.random.rand(250)]
    line_group = np.array(list(zip(*[np.random.rand(100) + i for i in range(6)])))
    # Example plots
    # jb = Boxplot(data, yalim=[0, 13], project_stats=False, plot_violin=True, direction="vertical", different_from=1.5, labels=["group1", "group2", "group3", "group4"], l0_stats=True, l1_stats=True, top_extent=2.5, right_extent=0, inchesperposition=0.2, pairwise=False, use_tex=True)
    # jb.show()
    # jb = Barplot(data, yalim=[0, 13], direction="vertical", labels=["group1", "group2", "group3", "group4"], inchesperposition=0.2, print_mean=True, use_tex=True, plot_error_bar=True)
    # jb.show()
    jb = Figure(xalim=[0, 1], yalim=[-1, 1])
    jb.scatterplot(scatter_x[0], scatter_y[0], nbins=10)
    jb.show()
    jb = Scatterplot(scatter_x, scatter_y, yalim=[0, 1], yticks=[0, 1], plot_unity=True, plot_mean=True, nbins=10)
    jb.show()
    jb = Figure(xalim=[0, 1], yalim=[-1, 1])
    jb.blandaltmanplot(scatter_x[0], scatter_y[1], nbins=10)
    jb.show()
    jb = Blandaltmanplot(scatter_x, scatter_y, nbins=10)
    jb.show()
    jb = Figure([-1, 11], [-1, 11])
    jb.lineplot(line_group[0], line_group[1])
    jb.show()
    jb = Lineplot(line_group[:6], line_group[-6:])
    jb.show()
    jb = Figure(xalim=[-1, 6], yalim=[-1, 6])
    jb.lineplotwithstats(line_group, line_group)
    jb.show()
    jb = Lineplotwithstats([line_group, line_group], [line_group, line_group + 2])
    jb.show()
    # jb.savefig("/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/data/phd/pictures/bootstrap_maps/test1.pdf")
