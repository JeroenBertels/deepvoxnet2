import numpy as np
from collections import Iterable
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon


color_dict = {
    "r": (1, 0, 0),
    "g": (0, 1, 0),
    "b": (0, 0, 1)
}


class Figure(object):
    def __init__(self,
                 xalim, yalim,
                 awidthininches=5, aheightininches=5,
                 dxininches=0.25, dyininches=0.25,
                 lmwidthininches=1.5, rmwidthininches=0.5,
                 bmheightininches=1, tmheightininches=0.5,
                 top_extent=0, right_extent=0,
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
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['left'].set_linewidth(2)
        self.ax.spines['bottom'].set_linewidth(2)
        self.ax.spines['left'].set_bounds(self.yamin, self.yamax)
        self.ax.spines['bottom'].set_bounds(self.xamin, self.xamax)
        self.ax.tick_params(axis='both', which='major', labelsize=20, width=2)
        self.ax.set_xlim([self.xmin, self.xmax])
        self.ax.set_ylim([self.ymin, self.ymax])
        # add whitespace in between x- and y-axis in the dx and dy regions
        self.ax.add_patch(Rectangle((self.xmin, self.yamax), self.width, self.ymax - self.yamax, fc='w', ec='w', linewidth=2, zorder=2.001))
        self.ax.add_patch(Rectangle((self.xmin, self.ymin), self.width, self.dy, fc='w', ec='w', linewidth=2, zorder=2.001))
        self.ax.add_patch(Rectangle((self.xmin, self.ymin), self.dx, self.height, fc='w', ec='w', linewidth=2, zorder=2.001))
        self.ax.add_patch(Rectangle((self.xamax, self.ymin), self.xmax - self.xamax, self.height, fc='w', ec='w', linewidth=2, zorder=2.001))
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
        self.fig.suptitle(title, fontsize=20, x=0.5, y=1 - (self.rmwidthininches / 5) / self.fheightininches, horizontalalignment='center', verticalalignment='top')

    def set_xlabel(self, xlabel):
        self.ax.set_xlabel(xlabel, fontsize=20, horizontalalignment='center', verticalalignment='bottom')
        self.ax.xaxis.set_label_coords(0.5, -(self.bmheightininches - self.rmwidthininches / 5) / self.heightininches)

    def set_ylabel(self, ylabel):
        self.ax.set_ylabel(ylabel, fontsize=20, horizontalalignment='center', verticalalignment='top')
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

    def boxplot(self, data, pos, direction="vertical", width=0.8, fc=(0, 0, 1, 0.5), ec=None, lw=2, ms=8, outliers_only=True, project_stats=False, plot_violin=False, violin_color=None, print_mean=True, different_from=None, **kwargs):
        nnans = len([d for d in data if np.isnan(d) or np.isinf(d)])
        data = np.array([d for d in data if not np.isnan(d) and not np.isinf(d)])
        p50 = np.percentile(data, 50)
        pm = np.mean(data)
        p25 = np.percentile(data, 25)
        p75 = np.percentile(data, 75)
        iqr = p75 - p25
        pmin = np.min([d for d in data if d >= p25 - 1.5 * iqr])
        pmax = np.max([d for d in data if d <= p75 + 1.5 * iqr])
        points = np.array([d for d in data if (d > pmax or d < pmin) or not outliers_only])
        width2 = width / 2
        if ec is None:
            ec = (fc[0], fc[1], fc[2], 1)

        text = ""
        if print_mean:
            mark = False
            if different_from is not None:
                p_value = self.basic_test(data, [different_from] * len(data))
                p_value_ = min(p_value, 1 - p_value)
                if p_value_ < 0.05:
                    mark = "{" + (">" if p_value > 0.95 else "<") + f" {different_from} " + ("***" if p_value_ < 0.001 else ("**" if p_value_ < 0.01 else "*")) + "}"

            text = "{0:.3g}".format(pm) + ("$^{} {}{}$".format("{", nnans, "}") if nnans > 0 else "") + ("$_{} {}{}$".format("{", mark, "}") if mark else "")

        if plot_violin:
            violin_parts = self.ax.violinplot(data, positions=[pos], vert=direction == "vertical", widths=width, showmeans=False, showmedians=False, showextrema=False, points=1000)
            for pc in violin_parts['bodies']:
                pc.set_color(self.get_color(fc if violin_color is None else violin_color))
                pc.set_linewidth(0)

            width = width / 2
            width2 = width2 / 2

        if direction == "vertical":
            self.add_patch(Rectangle((pos - width2, p25), width, iqr, fc=fc, ec=ec, linewidth=lw))
            self.plot([pos, pos], [p75, pmax], color=ec, linewidth=lw)
            self.plot([pos, pos], [pmin, p25], color=ec, linewidth=lw)
            self.plot([pos] * len(points), points, color=fc, linestyle="None", marker=".", markersize=ms)
            self.plot([pos - width2, pos + width2], [p50, p50], color=ec, linewidth=lw)
            self.plot([pos - width2, pos + width2], [pm, pm], color=fc, linestyle="dashed", linewidth=lw)
            self.plot([pos - width2, pos + width2], [pmax, pmax], color=ec, linewidth=lw)
            self.plot([pos - width2, pos + width2], [pmin, pmin], color=ec, linewidth=lw)
            if project_stats:
                self.plot([self.xmin, pos - width2], [p50, p50], color=ec, linewidth=lw, zorder=2.01)
                self.plot([self.xmin, pos - width2], [pm, pm], color=ec, linestyle="dashed", linewidth=lw, zorder=2.01)
                self.plot([self.xmin, pos - width2], [p75, p75], color=ec, linewidth=lw, zorder=2.01)
                self.plot([self.xmin, pos - width2], [p25, p25], color=ec, linewidth=lw, zorder=2.01)

            self.text(pos, self.yamax + self.dy, text, rotation=0, ha="center", va="bottom", fontsize=20)
            if different_from:
                self.plot([pos - 0.5, pos + 0.5], [different_from, different_from], "k--", linewidth=lw, zorder=2.01)

        else:
            assert direction == "horizontal"
            self.add_patch(Rectangle((p25, pos - width2), iqr, width, fc=fc, ec=ec, linewidth=lw))
            self.plot([p75, pmax], [pos, pos], color=ec, linewidth=lw)
            self.plot([pmin, p25], [pos, pos], color=ec, linewidth=lw)
            self.plot(points, [pos] * len(points), color=fc, linestyle="None", marker=".", markersize=ms)
            self.plot([p50, p50], [pos - width2, pos + width2], color=ec, linewidth=lw)
            self.plot([pm, pm], [pos - width2, pos + width2], color=fc, linestyle="dashed", linewidth=lw)
            self.plot([pmax, pmax], [pos - width2, pos + width2], color=ec, linewidth=lw)
            self.plot([pmin, pmin], [pos - width2, pos + width2], color=ec, linewidth=lw)
            if project_stats:
                self.plot([p50, p50], [self.ymin, pos - width2], color=ec, linewidth=lw, zorder=2.01)
                self.plot([pm, pm], [self.ymin, pos - width2], color=ec, linestyle="dashed", linewidth=lw, zorder=2.01)
                self.plot([p75, p75], [self.ymin, pos - width2], color=ec, linewidth=lw, zorder=2.01)
                self.plot([p25, p25], [self.ymin, pos - width2], color=ec, linewidth=lw, zorder=2.01)

            self.text(self.xamax + self.dx, pos, text, rotation=270, ha="left", va="center", fontsize=20)
            if different_from:
                self.plot([different_from, different_from], [pos - 0.5, pos + 0.5], "k--", linewidth=lw, zorder=2.01)

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

    @staticmethod
    def basic_test(x0, x1=None, n=10000, skipna=True):
        x0 = np.array(x0)
        x1 = np.array(x1) if x1 is not None else np.zeros_like(x0)
        if skipna:
            nan_locs = np.any([np.isnan(x0), np.isnan(x1)], axis=0)
            x0 = x0[~nan_locs]
            x1 = x1[~nan_locs]

        else:
            assert np.sum(np.any([np.isnan(x0), np.isnan(x1)], axis=0)) == 0

        N = len(x0)
        if N > 0:
            differences = x0 - x1
            if np.all(differences == 0):
                return 0.5

            else:
                count = 0
                for i in range(n):
                    samples = np.random.choice(differences, N)
                    if np.mean(samples) > 0:
                        count += 1

                return count / n

        else:
            return np.nan


class Boxplot(Figure):
    def __init__(self, data, labels=None, xalim=None, yalim=None, inchesperposition=None, colors=None, alpha=0.5, lw=2, direction="vertical", l0_stats=False, l1_stats=False, **kwargs):
        data = [[data__ for data__ in (data_ if isinstance(data_[0], Iterable) else [data_])] for data_ in (data if isinstance(data[0], Iterable) else [data])]
        position, positions = 0, []
        for data_ in data:
            position += 1
            positions_ = []
            for _ in data_:
                positions_.append(position)
                position += 1

            positions.append(positions_)

        if colors is None:
            colors = [[list(color_dict.keys())[i] for i, _ in enumerate(data_)] for data_ in data]

        if inchesperposition is not None:
            if direction == "vertical":
                kwargs["awidthininches"] = position * inchesperposition

            else:
                kwargs["aheightininches"] = position * inchesperposition

        if xalim is None:
            xalim = [0, position]

        if yalim is None:
            yalim = [np.min([d for data_ in data for data__ in data_ for d in data__ if not np.isnan(d) and not np.isinf(d)]), np.max([d for data_ in data for data__ in data_ for d in data__ if not np.isnan(d) and not np.isinf(d)])]

        ticks = [np.mean(positions_) for positions_ in positions]
        super(Boxplot, self).__init__(
            xalim=xalim if direction == "vertical" else yalim,
            yalim=yalim if direction == "vertical" else xalim,
            xticks=ticks if direction == "vertical" else "auto",
            xticklabels=labels if direction == "vertical" else "auto",
            yticks="auto" if direction == "vertical" else ticks,
            yticklabels="auto" if direction == "vertical" else labels,
            **kwargs)

        for i, data_ in enumerate(data):
            for j, data__ in enumerate(data_):
                self.boxplot(data__, pos=positions[i][j], fc=self.get_color(colors[i][j], alpha=alpha), direction=direction, lw=lw, **kwargs)

        loc = (self.yamax + 3 * self.dy) if direction == "vertical" else (self.xamax + 3 * self.dx)
        if l0_stats:
            for i, (data_, positions_) in enumerate(zip(data, positions)):
                loc_ = (self.yamax + 3 * self.dy) if direction == "vertical" else (self.xamax + 3 * self.dx)
                for j, (data__0, position0) in enumerate(zip(data_, positions_)):
                    for k, (data__1, position1) in enumerate(zip(data_, positions_)):
                        if k > j:
                            p_value = self.basic_test(*zip(*[(d0, d1) for d0, d1 in zip(data__0, data__1) if not np.isnan(d0) and not np.isnan(d1) and not np.isinf(d0) and not np.isinf(d1)]))
                            p = min([p_value, 1 - p_value])
                            if p < 0.05:
                                position01 = (position0 + position1) / 2
                                self.plot(*[[position0, position01], [loc_, loc_]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][j]), linewidth=lw, zorder=2.01)
                                self.plot(*[[position0, position0], [loc_, loc_ - self.dy / 4]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][j]), linewidth=lw, zorder=2.01)
                                self.plot(*[[position01, position1], [loc_, loc_]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][k]), linewidth=lw, zorder=2.01)
                                self.plot(*[[position1, position1], [loc_, loc_ - self.dy / 4]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][k]), linewidth=lw, zorder=2.01)
                                self.text(*[(position0 + position1) / 2, loc_][::1 if direction == "vertical" else -1], "${} {}$".format(">" if p_value > 0.95 else "<", "***" if p < 0.001 else ("**" if p < 0.01 else "*")),
                                          rotation=0 if direction == "vertical" else 270,
                                          ha="center" if direction == "vertical" else "left",
                                          va="bottom" if direction == "vertical" else "center",
                                          fontsize=12)
                                loc_ += self.dy if direction == "vertical" else self.dx

                if loc_ > loc:
                    loc = loc_

        if l1_stats:
            loc_ = loc + 2 * (self.dy if direction == "vertical" else self.dx)
            for i, (data_a, positions_a) in enumerate(zip(data, positions)):
                for j, (data__0, position0) in enumerate(zip(data_a, positions_a)):
                    for k, (data_b, positions_b) in enumerate(zip(data, positions)):
                        if k > i:
                            for l, (data__1, position1) in enumerate(zip(data_b, positions_b)):
                                p_value = self.basic_test(*zip(*[(d0, d1) for d0, d1 in zip(data__0, data__1) if not np.isnan(d0) and not np.isnan(d1) and not np.isinf(d0) and not np.isinf(d1)]))
                                p = min([p_value, abs(1 - p_value)])
                                if p < 0.05:
                                    position01 = (position0 + position1) / 2
                                    self.plot(*[[position0, position01], [loc_, loc_]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][j]), linewidth=lw, zorder=2.01)
                                    self.plot(*[[position0, position0], [loc_, loc_ - self.dy / 4]][::1 if direction == "vertical" else -1], color=self.get_color(colors[i][j]), linewidth=lw, zorder=2.01)
                                    self.plot(*[[position01, position1], [loc_, loc_]][::1 if direction == "vertical" else -1], color=self.get_color(colors[k][l]), linewidth=lw, zorder=2.01)
                                    self.plot(*[[position1, position1], [loc_, loc_ - self.dy / 4]][::1 if direction == "vertical" else -1], color=self.get_color(colors[k][l]), linewidth=lw, zorder=2.01)
                                    self.text(*[(position0 + position1) / 2, loc_][::1 if direction == "vertical" else -1], "${} {}$".format(">" if p_value > 0.95 else "<", "***" if p < 0.001 else ("**" if p < 0.01 else "*")),
                                              rotation=0 if direction == "vertical" else 270,
                                              ha="center" if direction == "vertical" else "left",
                                              va="bottom" if direction == "vertical" else "center",
                                              fontsize=12)
                                    loc_ += self.dy if direction == "vertical" else self.dx


if __name__ == "__main__":
    data = [[np.random.rand(100) + 2], [np.random.rand(50), np.random.rand(20) + 1]]
    jb = Boxplot(data, yalim=[0, 3.2], project_stats=False, plot_violin=False, direction="horizontal", different_from=None, labels=["group1", "group2"], l0_stats=True, l1_stats=True, top_extent=0.75, right_extent=0, inchesperposition=1)
    jb.show()
    jb.savefig("/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/data/phd/pictures/bootstrap_maps/test1.pdf")
