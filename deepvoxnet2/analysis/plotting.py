"""The plotting.py module provides:
- a collection of classes and functions for creating and customizing different types of plots
- a color dictionary (color_dict) for defining custom color palettes
- three key classes: Series, SeriesGroup, and GroupedSeries, which represent data organized in different ways for plotting or analysis (see analysis.py and data.py modules)
- the Figure class provides a base for creating different types of plots and offers a range of customization options for labels, titles, legends, and other plot elements
"""

import numpy as np
import seaborn as sb
from copy import deepcopy
from collections.abc import Iterable
from scipy import stats
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
    "w": (1, 1, 1),
    "cm": (0.5, 0.5, 1),
    "rg": (0.5, 0.5, 0),
    "grey": (0.5, 0.5, 0.5),
}


class Series(object):
    """A class for performing statistical analysis on one-dimensional series data.

    The Series class provides a set of methods for handling a list of numerical values.
    The class attributes include basic statistics such as the number of non-numeric values, minimum, maximum, and percentiles.
    The Series class also contains static methods that can be used to perform a range of statistical tests on the values in the series, including basic hypothesis testing, rank correlation, and Pearson correlation.

    Parameters
    ----------
    series : array_like
        The input series to be analyzed.

    Attributes
    ----------
    series : ndarray
        The input series as a NumPy array, without NaN and Inf values removed.
    series_ : ndarray
        The input series as a NumPy array, with NaN and Inf values removed.
    nnan : int
        The number of NaN values in the input series.
    ninf : int
        The number of Inf values in the input series.
    nnaninf : int
        The number of NaN and Inf values in the input series.
    min : float
        The minimum value in the input series.
    p25 : float
        The 25th percentile value in the input series.
    p50 : float
        The median value in the input series.
    pm : float
        The mean value in the input series.
    p75 : float
        The 75th percentile value in the input series.
    max : float
        The maximum value in the input series.
    iqr : float
        The interquartile range of the input series.
    pmin : float
        The minimum value in the input series that is not considered an outlier.
    pmax : float
        The maximum value in the input series that is not considered an outlier.
    outliers : ndarray
        The outlier values in the input series.
    std : float
        The standard deviation of the input series.
    ste : float
        The standard error of the mean of the input series.
    median : float
        The same value as p50.
    mean : float
        The same value as pm.
    n : int
        The number of non-NaN and non-Inf values in the input series.

    Methods
    -------
    different_from(value=0, **kwargs)
        Returns a p-value indicating the significance of the difference between the input series and a specified value.
    correlate_with(series)
        Computes the Pearson correlation coefficient between the input series and another series.
    get_stats()
        Computes and returns the statistical attributes of the input series.
    basic_test(series0, series1=None, n=10000, skipnan=True, skipinf=True, pairwise=True, confidences=False, **kwargs)
        Computes a statistical test for the null hypothesis that two samples of data are drawn from the same population.
    rank_series(series_group, ranking_mode="any", p_value_threshold=0.1, value_mode="max", **kwargs)
        Computes the rankings of a group of Series objects based on their means and statistical differences between them.
    pearson_correlation(series0, series1)
        Computes the Pearson correlation coefficient between two series.
    """

    def __init__(self, series):
        """Constructs a new Series object with the given series.

        Parameters
        ----------
        series : array_like
            The series to be analyzed.

        Returns
        -------
        None
        """

        series = deepcopy(series)
        if not isinstance(series, Iterable):
            series = [series]

        self.series = np.array(series)
        self.series_, self.nnan, self.ninf, self.nnaninf, self.min, self.p25, self.p50, self.pm, self.p75, self.max, self.iqr, self.pmin, self.pmax, self.outliers, self.std, self.ste = self.get_stats()
        self.median, self.mean = self.p50, self.pm
        self.n = len(self.series_)

    def __len__(self):
        """Returns the number of elements in the series.

        Returns
        -------
        int
            The number of elements in the series.
        """

        return len(self.series)

    def __getitem__(self, idx):
        """Returns the element at the given index in the series.

        Parameters
        ----------
        idx : int
            The index of the element to return.

        Returns
        -------
        float
            The element at the given index in the series.
        """

        return self.series[idx]

    def __iter__(self):
        """Returns an iterator for the series.

        Returns
        -------
        iterator
            An iterator for the series.
        """

        return iter(self.series)

    def different_from(self, value=0, **kwargs):
        """Tests whether the values of the series are different from the specified value(s) using a permutation test.

        Parameters
        ----------
        value : float or iterable of floats, optional
            The value(s) to test against. Default is 0.
        **kwargs : dict, optional
            Additional arguments to pass to the `basic_test` method.

        Returns
        -------
        float
            The proportion of permutations in which the observed difference in mean between the series and the specified value(s)
            is greater than the difference in the permuted samples.

        """

        return self.basic_test(self.series, value if isinstance(value, Iterable) else [value] * len(self.series), **kwargs)

    def correlate_with(self, series):
        """Computes the Pearson correlation coefficient between two series.

        Parameters
        ----------
        series : iterable of floats or a Series object
            The other series to compute the correlation with.

        Returns
        -------
        float
            The Pearson correlation coefficient between the two series.

        """

        return self.pearson_correlation(self, series)

    def get_stats(self):
        """Computes descriptive statistics of the series, including count, mean, standard deviation, median, quartiles, min, and max.

        Returns
        -------
        tuple
            A tuple of the computed statistics in the following order: the non-missing values in the series, the number of NaN values,
            the number of infinite values, the number of NaN and infinite values combined, the minimum value, the 25th percentile,
            the median, the mean, the 75th percentile, the maximum value, the interquartile range, the minimum value that is not an outlier,
            the maximum value that is not an outlier, an array of outlier values, the standard deviation, and the standard error of the mean.

        """

        return self.calculate_stats(self.series)

    @staticmethod
    def calculate_stats(series):
        """Calculates descriptive statistics for a series, including count, mean, standard deviation, median, quartiles, min, and max.

        Parameters
        ----------
        series : iterable of floats
            The series for which to calculate statistics.

        Returns
        -------
        tuple
            A tuple of the computed statistics in the following order: the non-missing values in the series, the number of NaN values,
            the number of infinite values, the number of NaN and infinite values combined, the minimum value, the 25th percentile,
            the median, the mean, the 75th percentile, the maximum value, the interquartile range, the minimum value that is not an outlier,
            the maximum value that is not an outlier, an array of outlier values, the standard deviation, and the standard error of the mean.

        """
        
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
    def basic_test(series0, series1=None, n=10000, skipnan=True, skipinf=True, pairwise=True, confidences=False, **kwargs):
        """Performs a basic statistical test to determine if there is a statistically significant difference between two series.
        
        Returns a p-value which can be interpreted as follows: a small p-value (typically ≤ 0.05) indicates strong evidence
        against the null hypothesis, so you reject the null hypothesis, while a large p-value (> 0.05) indicates weak evidence
        against the null hypothesis, so you fail to reject the null hypothesis.

        Parameters:
        -----------
        series0: array-like
            The first series to compare.
        series1: array-like, optional
            The second series to compare. If None, the method will perform a pairwise test using series0.
        n: int, default 10000
            The number of iterations to run for the Monte Carlo simulation.
        skipnan: bool, default True
            If True, NaN values will be excluded from the series before the test is performed.
        skipinf: bool, default True
            If True, infinity values will be excluded from the series before the test is performed.
        pairwise: bool, default True
            If True, the method will perform a pairwise test using series0 and series1. If False, it will compare the two series as is.
        confidences: bool, default False
            If True, the method will calculate and return a confidence interval for the mean difference between the two series.

        Returns:
        --------
        p_value: float
            The p-value obtained from the basic test.
        """

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

            elif confidences:  # https://courses.lumenlearning.com/boundless-statistics/chapter/hypothesis-testing-two-samples/
                x0, s0, n0 = np.mean(series0), np.std(series0), len(series0)
                x1, s1, n1 = np.mean(series1), np.std(series1), len(series1)
                t = (x0 - x1) / np.sqrt(s0 ** 2 + s1 ** 2)
                df = (s0 ** 2 + s1 ** 2) ** 2 / (1 / (n0 - 1) * s0 ** 4 + 1 / (n1 - 1) * s1 ** 4)
                p = 2 * stats.t.sf(np.abs(t), df)  # https://www.statology.org/p-value-from-t-score-python/
                if t > 0:
                    return 1 - (p / 2)

                else:
                    return p / 2

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
    def rank_series(series_group, ranking_mode="any", p_value_threshold=0.1, value_mode="max", **kwargs):
        """Ranks a group of series based on their mean values, using a basic statistical test to determine if there is a statistically significant difference between any pair of series.
        
        The ranking will be done based on the mean values of the series in descending order. Then, for each pair of adjacent series in the ranking, the method will perform a statistical 
        test to determine if the two series are significantly different from one another. If the p-value for the test is greater 
        than the given threshold, the two series will be considered tied and will receive the same rank. The method will then 
        move on to the next pair of adjacent series and repeat the process until all series have been ranked.

        Parameters:
        -----------
        series_group: list of array-like
            A list of series to be ranked.
        ranking_mode: str, default "any"
            The ranking mode to use when determining if two adjacent series are significantly different. Can be "any" or "all".
        p_value_threshold: float, default 0.1
            The threshold to use when comparing p-values from the statistical test.
        value_mode: str, default "max"
            The value mode to use when sorting the series. Can be "max" or "min".

        Returns:
        --------
        ranking_: list
            A list containing the rank of each series in the input list.
        """
        
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
        """Calculates the Pearson correlation coefficient between two series.

        Parameters:
        -----------
        series0: array-like
            The first series to compare.
        series1: array-like
            The second series to compare.

        Returns:
        --------
        correlation: float
            The Pearson correlation coefficient between series0 and series1.
        """

        return np.corrcoef(Series(series0).series_, Series(series1).series_)[0, 1]


class SeriesGroup(object):
    """Represents a group of `Series` objects.

    Parameters
    ----------
    series_group : list-like
        List of `Series` objects or values to be converted to `Series` objects.

    Attributes
    ----------
    series_group : list
        The list of `Series` objects in this `SeriesGroup` instance.
    series : `Series`
        A `Series` object that combines all `Series` objects in `series_group`.

    Methods
    -------
    __len__()
        Returns the length of the `series_group` attribute.
    __getitem__(idx)
        Returns the `Series` object at index `idx` in `series_group`.
    __iter__()
        Returns an iterator over the `series_group` attribute.
    rank(threshold=0.05, mode="max", **kwargs)
        Returns the ranks of the `Series` objects in `series_group` based on pairwise tests.
    """

    def __init__(self, series_group):
        """Initialize the SeriesGroup object.

        Parameters
        ----------
        series_group : list or tuple or SeriesGroup
            The group of series to be contained.

        Returns
        -------
        None.
        """

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
        """Return the number of series in the container.

        Returns
        -------
        int
            The number of series in the container.
        """

        return len(self.series_group)

    def __getitem__(self, idx):
        """Return the series at the given index.

        Parameters
        ----------
        idx : int
            The index of the series to be returned.

        Returns
        -------
        Series
            The series at the given index.
        """

        return self.series_group[idx]

    def __iter__(self):
        """Return an iterator over the series in the container.

        Returns
        -------
        iterator
            An iterator over the series in the container.
        """

        return iter(self.series_group)

    def rank(self, threshold=0.05, mode="max", **kwargs):
        """Rank the series in the container based on their mean values.

        Parameters
        ----------
        threshold : float, optional
            The p-value threshold for statistical testing, by default 0.05.
        mode : str, optional
            The mode of ranking, either 'max' or 'min', by default 'max'.
        **kwargs : dict, optional
            Additional keyword arguments to be passed to the basic_test function.

        Returns
        -------
        list
            A list of rankings for the series in the container.
        """

        return Series.rank_series(self, threshold=threshold, mode=mode, **kwargs)


class GroupedSeries(object):
    """Represents a group of `SeriesGroup` objects.

    Parameters
    ----------
    grouped_series : list-like
        List of `SeriesGroup` objects or values to be converted to `SeriesGroup` objects.

    Attributes
    ----------
    grouped_series : list
        The list of `SeriesGroup` objects in this `GroupedSeries` instance.
    series : `Series`
        A `Series` object that combines all `Series` objects in all `SeriesGroup` objects in `grouped_series`.

    Methods
    -------
    __len__()
        Returns the length of the `grouped_series` attribute.
    __getitem__(idx)
        Returns the `SeriesGroup` object at index `idx` in `grouped_series`.
    __iter__()
        Returns an iterator over the `grouped_series` attribute.
    """

    def __init__(self, grouped_series):
        """Initialize the `GroupedSeries` object.

        Parameters
        ----------
        grouped_series : `SeriesGroup` object, list of `SeriesGroup` objects, tuple of `SeriesGroup` objects, or iterable of `SeriesGroup` objects
            The `SeriesGroup` objects to group together.
        """

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
        """Returns the number of `SeriesGroup` objects in the `grouped_series`.

        Returns
        -------
        int
            The number of `SeriesGroup` objects in the `grouped_series`.
        """

        return len(self.grouped_series)

    def __getitem__(self, idx):
        """Returns the `SeriesGroup` object at the given index in the `grouped_series`.

        Parameters
        ----------
        idx : int
            The index of the `SeriesGroup` object to retrieve.

        Returns
        -------
        `SeriesGroup` object
            The `SeriesGroup` object at the given index in the `grouped_series`.
        """

        return self.grouped_series[idx]

    def __iter__(self):
        """Returns an iterator over the `SeriesGroup` objects in the `grouped_series`.

        Returns
        -------
        iterator
            An iterator over the `SeriesGroup` objects in the `grouped_series`.
        """

        return iter(self.grouped_series)


class Figure(object):
    """A class to create a figure object.

    Parameters
    ----------
    xalim : tuple of two floats
        The minimum and maximum values for the x-axis.
    yalim : tuple of two floats
        The minimum and maximum values for the y-axis.
    awidthininches : float, optional
        The width of the axis in inches, by default 5.
    aheightininches : float, optional
        The height of the axis in inches, by default 5.
    dxininches : float, optional
        The size of the padding between the axis and the edge of the figure in the x-direction, by default 0.25.
    dyininches : float, optional
        The size of the padding between the axis and the edge of the figure in the y-direction, by default 0.25.
    lmwidthininches : float, optional
        The width of the left margin in inches, by default 1.
    rmwidthininches : float, optional
        The width of the right margin in inches, by default 0.
    bmheightininches : float, optional
        The height of the bottom margin in inches, by default 1.
    tmheightininches : float, optional
        The height of the top margin in inches, by default 0.
    top_extent : float, optional
        The amount by which the top of the axis extends beyond the maximum y value, by default 0.
    right_extent : float, optional
        The amount by which the right side of the axis extends beyond the maximum x value, by default 0.
    fs : int, optional
        The font size for the axis labels and tick labels, by default 20.
    lw : int, optional
        The linewidth for the axis spines, by default 2.
    ms : int, optional
        The size of the markers in the plot, by default 10.
    mask_inner_region : bool, optional
        Whether to mask the inner region of the axis, by default True.
    mask_outer_region : bool, optional
        Whether to mask the outer region of the axis, by default True.
    plot_xaxis : bool, optional
        Whether to plot the x-axis, by default True.
    plot_yaxis : bool, optional
        Whether to plot the y-axis, by default True.
    use_tex : bool, optional
        Whether to use TeX to render text, by default True.
    **kwargs : optional
        Additional keyword arguments to be passed to the figure object.

    Methods
    -------
    __call__()
        Returns the figure and axis objects.
    set_title(title)
        Sets the title for the axis.
    set_xlabel(xlabel)
        Sets the label for the x-axis.
    set_ylabel(ylabel)
        Sets the label for the y-axis.
    set_xticks(xticks)
        Sets the ticks for the x-axis.
    set_xticklabels(xticklabels)
        Sets the tick labels for the x-axis.
    set_yticks(yticks)
        Sets the ticks for the y-axis.
    set_yticklabels(yticklabels)
        Sets the tick labels for the y-axis.
    add_patch(patch)
        Adds a patch to the axis.
    text(x, y, s, fontdict=None, **kwargs)
        Adds text to the axis.
    legend(*args, fs=None, **kwargs)
        Adds a legend to the axis.
    show()
        Shows the figure.
    savefig(file_path, **kwargs)
        Saves the
    """

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
        """Initialize a new Matplotlib axis with customizable dimensions and styling.

        Parameters
        ----------
        xalim : tuple of float
            A tuple (xmin, xmax) defining the x-axis limits.
        yalim : tuple of float
            A tuple (ymin, ymax) defining the y-axis limits.
        awidthininches : float, optional
            The desired width of the plotting area in inches. Default is 5.
        aheightininches : float, optional
            The desired height of the plotting area in inches. Default is 5.
        dxininches : float, optional
            The desired width of the horizontal padding on each side of the plotting area in inches. Default is 0.25.
        dyininches : float, optional
            The desired height of the vertical padding on each side of the plotting area in inches. Default is 0.25.
        lmwidthininches : float, optional
            The desired width of the left margin in inches. Default is 1.
        rmwidthininches : float, optional
            The desired width of the right margin in inches. Default is 0.
        bmheightininches : float, optional
            The desired height of the bottom margin in inches. Default is 1.
        tmheightininches : float, optional
            The desired height of the top margin in inches. Default is 0.
        top_extent : float, optional
            The desired extra height of the plotting area above yalim in units of y-axis range. Default is 0.
        right_extent : float, optional
            The desired extra width of the plotting area to the right of xalim in units of x-axis range. Default is 0.
        fs : float, optional
            The desired font size for the axis labels and title. Default is 20.
        lw : float, optional
            The desired line width for the axis spines and ticks. Default is 2.
        ms : float, optional
            The desired marker size for scatter plots. Default is 10.
        mask_inner_region : bool, optional
            Whether to add a white patch to mask the inner padding region of the plot. Default is True.
        mask_outer_region : bool, optional
            Whether to add white patches to mask the outer padding region of the plot. Default is True.
        plot_xaxis : bool, optional
            Whether to plot the x-axis and its labels. Default is True.
        plot_yaxis : bool, optional
            Whether to plot the y-axis and its labels. Default is True.
        use_tex : bool, optional
            Whether to use LaTeX for rendering the text in the plot. Default is True.
        **kwargs : optional
            Additional keyword arguments are passed to the set_xlabel(), set_ylabel(), set_xticks(), set_yticks(), and set_title() methods of the plot axis. In particular, the keywords xlabel, ylabel, xticks, xticklabels, yticklabels.
        """

        if use_tex:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "serif",
                "font.family": "cm",
                "text.latex.preamble": r'\usepackage{amssymb,amsthm,amsmath}'
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
        self.fxmin, self.fxmax = self.xmin - self.awidth * self.lmwidthininches / self.awidthininches, self.xmax + self.awidth * self.rmwidthininches / self.awidthininches
        # calculate y dimensions
        self.yamin, self.yamax = self.yalim[0], self.yalim[1]
        self.aheight = self.yamax - self.yamin
        self.dy = self.aheight * self.dyininches / self.aheightininches
        self.ymin, self.ymax = self.yamin - self.dy, self.yamax + self.dy + self.aheight * self.top_extent
        self.height = self.ymax - self.ymin
        self.heightininches = self.aheightininches * self.height / self.aheight
        self.fheightininches = self.tmheightininches + self.heightininches + self.bmheightininches
        self.fymin, self.fymax = self.ymin - self.aheight * self.bmheightininches / self.aheightininches, self.ymax + self.aheight * self.tmheightininches / self.aheightininches
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

        else:
            self.ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labeltop=False)

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

        else:
            self.ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False, labelright=False)

        if "title" in kwargs and kwargs["title"] is not None:
            self.set_title(kwargs["title"])

    def __call__(self):
        """Returns the figure and axis objects.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axis object.
        """

        return self.fig, self.ax

    def set_title(self, title):
        """Set the title of the plot.

        Parameters
        ----------
        title : str
            The title of the plot.
        """

        # self.fig.suptitle(title, fontsize=self.fs, x=0.5, y=1 - (self.rmwidthininches / 5) / self.fheightininches, horizontalalignment='center', verticalalignment='top')
        self.ax.text((self.fxmin + self.fxmax) / 2, self.fymax - self.lhic, title, fontsize=self.fs, horizontalalignment='center', verticalalignment='top')

    def set_xlabel(self, xlabel):
        """Set the label for the x-axis.

        Parameters
        ----------
        xlabel : str
            The label for the x-axis.
        """

        # self.ax.set_xlabel(xlabel, fontsize=self.fs, horizontalalignment='center', verticalalignment='bottom')
        # self.ax.xaxis.set_label_coords((self.dxininches + self.awidthininches / 2) / self.widthininches, -self.bmheightininches / self.heightininches)
        self.ax.text((self.xamin + self.xamax) / 2, self.fymin + self.lhic, xlabel, fontsize=self.fs, horizontalalignment='center', verticalalignment='bottom')

    def set_ylabel(self, ylabel):
        """Set the label for the y-axis.

        Parameters
        ----------
        ylabel : str
            The label for the y-axis.
        """

        # self.ax.set_ylabel(ylabel, fontsize=self.fs, horizontalalignment='center', verticalalignment='top')
        # self.ax.yaxis.set_label_coords(-self.lmwidthininches / self.widthininches, (self.dyininches + self.aheightininches / 2) / self.heightininches)
        self.ax.text(self.fxmin + self.lwic, (self.yamin + self.yamax) / 2, ylabel, fontsize=self.fs, horizontalalignment='left', verticalalignment='center', rotation=90)

    def set_xticks(self, xticks):
        """Set the locations of the x-axis ticks.

        Parameters
        ----------
        xticks : array-like
            The locations of the x-axis ticks.
        """

        self.ax.set_xticks(xticks)

    def set_xticklabels(self, xticklabels):
        """Set the labels for the x-axis ticks.

        Parameters
        ----------
        xticklabels : array-like
            The labels for the x-axis ticks.
        """

        self.ax.set_xticklabels(xticklabels)

    def set_yticks(self, yticks):
        """Set the locations of the y-axis ticks.

        Parameters
        ----------
        yticks : array-like
            The locations of the y-axis ticks.
        """

        self.ax.set_yticks(yticks)

    def set_yticklabels(self, yticklabels):
        """Set the labels for the y-axis ticks.

        Parameters
        ----------
        yticklabels : array-like
            The labels for the y-axis ticks.
        """

        self.ax.set_yticklabels(yticklabels)

    def add_patch(self, patch):
        """Adds a `matplotlib.patches.Patch` to the plot.

        Parameters
        ----------
        patch : matplotlib.patches.Patch
            The patch to be added to the plot.
        """
            
        self.ax.add_patch(patch)

    def text(self, x, y, s, fontdict=None, **kwargs):
        """Adds text to the plot.

        Parameters
        ----------
        x : float
            The x-coordinate of the text.
        y : float
            The y-coordinate of the text.
        s : str
            The text to be displayed.
        fontdict : dict, optional
            A dictionary containing font properties. See `matplotlib.text.Text` documentation for details.
        **kwargs
            Additional keyword arguments to be passed to `matplotlib.text.Text`.

        Returns
        -------
        matplotlib.text.Text
            The `Text` instance that was added to the plot.
        """
        
        self.ax.text(x, y, s, fontdict=fontdict, **kwargs)

    def legend(self, *args, fs=None, **kwargs):
        """Place a legend on the plot.

        Parameters
        ----------
        *args
            Variable length argument list. See `matplotlib.axes.Axes.legend` documentation for details.
        fs : float, optional
            The font size of the legend. If not specified, uses the `fs` value specified in `__init__`.
        **kwargs
            Additional keyword arguments to be passed to `matplotlib.axes.Axes.legend`.

        Returns
        -------
        matplotlib.legend.Legend
            The `Legend` instance of the added legend.
        """

        if "fontsize" in kwargs and fs is not None:
            assert kwargs["fontsize"] == fs

        kwargs["fontsize"] = kwargs.get("fontsize", self.fs if fs is None else fs)
        self.ax.legend(*args, **kwargs)

    def show(self):
        """Displays the plot.
        """

        self.fig.show()

    def savefig(self, file_path, **kwargs):
        """Saves the plot to a file.

        Parameters
        ----------
        file_path : str
            The path to the file where the plot should be saved.
        **kwargs
            Additional keyword arguments to be passed to `matplotlib.figure.Figure.savefig`.
        """

        self.fig.savefig(file_path, **kwargs)

    @staticmethod
    def prepare(mode, x_and_y_data, xalim=None, yalim=None, colors=None, color_mode="series", **kwargs):
        """Prepare data and formatting parameters for plotting.

        Parameters
        ----------
        mode : str
            Mode for preparing data, one of "series", "series_group", or "grouped_series".
        x_and_y_data : list
            List of data for x and y axis. For "series" mode, a list of two Series objects, for "series_group" mode,
            a list of two SeriesGroup objects, and for "grouped_series" mode, a list of two GroupedSeries objects.
        xalim : tuple, optional
            Tuple of lower and upper x-axis limits. Default is None.
        yalim : tuple, optional
            Tuple of lower and upper y-axis limits. Default is None.
        colors : list or str, optional
            List of colors or color map string for plotting. Default is None.
        color_mode : str, optional
            Color mode for "series_group" and "grouped_series" modes, either "series" or "group". Default is "series".
        **kwargs : dict, optional
            Additional keyword arguments for formatting.

        Returns
        -------
        tuple
            Tuple of formatted data and formatting parameters for plotting.

        Raises
        ------
        ValueError
            If an unknown data prepare mode is specified.

        """
            
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
        """Convert a color string or RGB(A) tuple to a RGBA tuple.

        Parameters
        ----------
        color : str or tuple
            The color to convert. If a string, it should be a key from the `color_dict` dictionary. If a tuple, it should
            be a tuple of RGB or RGBA values.
        alpha : float or None, optional
            The alpha (transparency) value to use for the color. If None and the color is a tuple with 4 values, the
            alpha value from the color tuple is used. Default is None.

        Returns
        -------
        color_tuple : tuple
            A tuple of RGBA values for the specified color, with an alpha value of 1.0 if not provided.
        """
            
        color_tuple = color_dict[color] if isinstance(color, str) else color
        if len(color_tuple) == 3:
            return color_tuple + (1 if alpha is None else alpha,)

        if len(color_tuple) == 4:
            assert alpha is None or alpha == color_tuple[3]
            return color_tuple

    def plot(self, *args, **kwargs):
        """Make a plot on the underlying matplotlib axis object using the `plot` method.

        Parameters
        ----------
        *args : array-like
            Variable length argument list passed to the `plot` method of the underlying matplotlib axis object.
        **kwargs : dict
            Keyword arguments passed to the `plot` method of the underlying matplotlib axis object.
        """
        
        self.ax.plot(*args, **kwargs)

    def lineplot(self, series_x, series_y, color=(0, 0, 1, 1), alpha=1, marker=".", linestyle="-", linewidth=None, markersize=None, **kwargs):
        """Make a line plot of the given x and y series on the underlying matplotlib axis object.

        Parameters
        ----------
        series_x : array-like
            x-series data for the line plot.
        series_y : array-like
            y-series data for the line plot.
        color : str or tuple, optional
            Color of the line plot in the form of a string or a tuple of RGB values. Default is (0, 0, 1, 1) (blue).
        alpha : float, optional
            Alpha (transparency) value of the line plot. Default is 1.
        marker : str, optional
            Marker style for the data points in the line plot. Default is ".".
        linestyle : str, optional
            Style of the line in the line plot. Default is "-".
        linewidth : float, optional
            Width of the line in the line plot. Default is None (i.e., use default linewidth).
        markersize : float, optional
            Size of the markers used in the line plot. Default is None (i.e., use default markersize).
        **kwargs : dict
            Additional keyword arguments passed to the `plot` method of the underlying matplotlib axis object.
        """

        series_x, series_y = Series(series_x), Series(series_y)
        color = self.get_color(color, alpha)
        # self.plot(series_x, series_y, color=color, linewidth=self.lw if linewidth is None else linewidth, zorder=1.9999, linestyle=linestyle)
        if marker is None:
            self.plot(series_x, series_y, color=color, linewidth=self.lw if linewidth is None else linewidth, zorder=1.9999, linestyle=linestyle)

        else:
            self.plot(series_x, series_y, color=color, linewidth=self.lw if linewidth is None else linewidth, zorder=1.9999, linestyle=linestyle, marker=marker, markersize=self.ms if markersize is None else markersize)

    def lineplotwithstats(self, series_group_x, series_group_y, color=(0, 0, 1, 1), alpha=1, marker=".", linestyle="-", alpha_stats=0.5, linestyle_stats=None, plot_std=False, plot_ste=True, plot_iqr=False, grouped_per_line=True, **kwargs):
        """Plot a line plot with statistics on the axes.

        Parameters
        ----------
        series_group_x : pandas.Series or array-like
            Data for the x-axis, grouped by line.
        series_group_y : pandas.Series or array-like
            Data for the y-axis, grouped by line.
        color : str or tuple, optional
            Color of the line plot.
        alpha : float, optional
            Alpha value of the line plot.
        marker : str or None, optional
            Marker style of the line plot.
        linestyle : str, optional
            Line style of the line plot.
        alpha_stats : float, optional
            Alpha value of the statistics line plot.
        linestyle_stats : str, optional
            Line style of the statistics line plot.
        plot_std : bool, optional
            Whether to plot standard deviation statistics.
        plot_ste : bool, optional
            Whether to plot standard error statistics.
        plot_iqr : bool, optional
            Whether to plot interquartile range statistics.
        grouped_per_line : bool, optional
            Whether data is grouped per line or not.
        **kwargs : dict
            Additional plotting options.
        """
            
        series_group_x, series_group_y = SeriesGroup(series_group_x), SeriesGroup(series_group_y)
        series_x = [Series([series[i] for series in series_group_x]) for i in range(len(series_group_x[0]))] if grouped_per_line else series_group_x
        series_y = [Series([series[i] for series in series_group_y]) for i in range(len(series_group_y[0]))] if grouped_per_line else series_group_y
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
        """Creates a scatter plot of two Series objects, with optional additional visualizations such as mean and KDE plots.

        Parameters:
        -----------
        series_x : Series
            A Series object containing the x-axis values for the scatter plot.
        series_y : Series
            A Series object containing the y-axis values for the scatter plot.
        color : tuple or str, optional
            The color of the scatter plot. If a tuple, must be a RGB(A) tuple with values between 0 and 1. If a str, must be a key in the color_dict attribute. Default is (0, 0, 1, 1) (blue).
        alpha : float, optional
            The transparency of the scatter plot markers. Must be a value between 0 and 1. Default is 1 (fully opaque).
        marker : str, optional
            The marker type for the scatter plot markers. Must be a valid marker string. Default is "." (dot).
        markerfill : str, optional
            The fill style of the scatter plot markers. Must be "full", "left", "right", "bottom", "top", or "none". Default is "full".
        markersize : float, optional
            The size of the scatter plot markers. If None, uses the value of the ms attribute. Default is None.
        plot_scatter : bool, optional
            Whether to plot the scatter plot markers. Default is True.
        plot_unity : bool, optional
            Whether to plot a diagonal unity line in the scatter plot. Default is False.
        plot_mean : bool, optional
            Whether to plot the mean point of the scatter plot. Default is False.
        plot_kde : bool, optional
            Whether to plot a 2D KDE plot of the scatter plot. Uses seaborn's kdeplot. Default is False.
        nbins : int, optional
            If > 0, bins the x-axis data into nbins bins, and plots the mean y value of each bin against the mean x value of the bin. Default is 0.
        groupn : int, optional
            If > 0, groups the x-axis data into groupn groups, and plots the mean y value of each group against the mean x value of the group. Default is 0.
        linestyle : str, optional
            The linestyle for the lines connecting the binned or grouped means. Default is "-".
        ncontours : int, optional
            If plot_kde is True, the number of contour lines to plot in the KDE plot. Default is 10.
        fillcontours : bool, optional
            If plot_kde is True, whether to fill the contour lines in the KDE plot. Default is False.
        markeredgewidth : float, optional
            The width of the marker edges in the scatter plot. Default is 0 (no edge).
        print_correlation : bool, optional
            Whether to print the Pearson correlation coefficient of the x and y data on the plot. Default is False.
        **kwargs :
            Additional keyword arguments to pass to the underlying plotting methods.
        """

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
        """Create a Bland-Altman plot showing the difference between two series against their mean.

        Parameters
        ----------
        series_x : array-like
            The first series to compare.
        series_y : array-like
            The second series to compare.
        color : str or tuple, optional
            The color of the plot (default is '(0, 0, 1, 1)', which is blue).
        alpha : float, optional
            The opacity of the plot (default is 1).
        marker : str, optional
            The marker style (default is '.').
        plot_unity : bool, optional
            Whether to plot the unity line (default is True).
        nbins : int, optional
            The number of bins to use for binning the data (default is 0, which means no binning).
        alpha_stats : float or None, optional
            The opacity of the statistical lines (default is None, which means it uses the same value as 'alpha').
        **kwargs
            Additional keyword arguments to pass to the plotting functions.
        """

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

    def boxplot(self, series, pos, direction="vertical", width=0.8, fc=(0, 0, 1, 0.5), ec=None, project_stats=False, plot_violin=False, violin_color=None, print_mean=True, different_from=None, mean_formatting="{0:.3g}", confidences=False, alpha=None, **kwargs):
        """Generate a box plot.

        Parameters
        ----------
        series : array_like
            Input data.
        pos : float
            The x or y coordinate of the box plot. The interpretation of the value depends on the `direction` parameter.
        direction : {"vertical", "horizontal"}, optional
            The direction of the box plot. Default is "vertical".
        width : float, optional
            The width of the box plot. Default is 0.8.
        fc : array_like, optional
            The face color of the boxes. Default is (0, 0, 1, 0.5).
        ec : array_like or None, optional
            The edge color of the boxes. If None, it is set to the same color as `fc` but with alpha 1. Default is None.
        project_stats : bool, optional
            Whether to project the statistics onto the opposite axis. Default is False.
        plot_violin : bool, optional
            Whether to plot a violin plot instead of a box plot. Default is False.
        violin_color : array_like or None, optional
            The face color of the violin plot. If None, it is set to the same as `fc`. Default is None.
        print_mean : bool, optional
            Whether to print the mean value on the box plot. Default is True.
        different_from : float or None, optional
            A value to compare the input data to, to test if they are statistically different. If None, no significance testing is performed. Default is None.
        mean_formatting : str, optional
            The format string used to print the mean value. Default is "{0:.3g}".
        confidences : bool, optional
            Whether to plot confidence intervals instead of a box plot. Default is False.
        alpha : float or None, optional
            The alpha value for the box or violin plot. If None, the value is set automatically based on the number of box or violin plots. Default is None.
        **kwargs
            Additional arguments passed to the underlying plot function.
        """

        series = Series(series)
        width2 = width / 2
        fc = self.get_color(fc, alpha=alpha)
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
            if p_value_ < 0.1:
                sig_text += "$"
                sig_text += ">" if p_value > 0.9 else "<"
                sig_text += "*" if 0.01 < p_value_ < 0.1 else ("**" if 0.001 < p_value_ < 0.01 else "***")
                sig_text += "$"

        if plot_violin:
            violin_parts = self.ax.violinplot(series.series_, positions=[pos], vert=direction == "vertical", widths=width, showmeans=False, showmedians=False, showextrema=False, points=1000)
            for pc in violin_parts['bodies']:
                pc.set_color(self.get_color(fc if violin_color is None else violin_color))
                pc.set_linewidth(0)

            width = width / 2
            width2 = width2 / 2

        if direction == "vertical":
            if confidences:
                # self.text(pos, series.pm, "X", rotation=0, ha="center", va="center", color=ec, fontsize=self.fs)
                self.plot([pos], [series.pm], color=ec, linestyle="None", marker="x", markersize=self.ms)
                self.plot([pos, pos], [series.pm - series.std, series.pm + series.std], color=ec, linewidth=self.lw)
                self.plot([pos - width2, pos + width2], [series.pm - series.std, series.pm - series.std], color=ec, linewidth=self.lw)
                self.plot([pos - width2, pos + width2], [series.pm + series.std, series.pm + series.std], color=ec, linewidth=self.lw)

            else:
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
        """Draw a bar plot of a given series at the specified position.

        Parameters
        ----------
        series : array-like
            Input data for the bar plot.
        pos : float
            The position at which the bar plot will be drawn.
        offset : float, optional
            The offset from the left or bottom side of the plot.
        direction : {'vertical', 'horizontal'}, optional
            The direction of the bar plot.
        width : float, optional
            The width of the bars.
        fc : color or tuple, optional
            The face color of the bars.
        ec : color or tuple or None or False, optional
            The edge color of the bars. If None, it will be the same as the face color. If False, there will be no edge.
        print_mean : bool, optional
            Whether to print the mean value of the data on top of the bars.
        plot_error_bar : bool, optional
            Whether to plot the error bar of the data as a vertical line.
        """
        
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
    """A subclass of Figure for creating boxplots of grouped data.

    Parameters
    ----------
    grouped_series : list of lists of Series
        The data to be plotted in boxplots. Each list of Series represents a group of data to be plotted together.
    labels : list of str, optional
        The labels for each group of data.
    xalim : tuple of float, optional
        The limits of the x-axis.
    yalim : tuple of float, optional
        The limits of the y-axis.
    positions : list of lists of float, optional
        The positions of the boxes in the plot.
    inchesperposition : float, optional
        The size of the distance between positions.
    colors : list of lists of str, optional
        The colors of the boxes in the plot.
    alpha : float or list of lists of float, optional
        The transparency of the boxes in the plot.
    direction : {"vertical", "horizontal"}, optional
        The direction of the plot.
    l0_stats : bool, optional
        Whether to plot pairwise statistical significance comparisons within groups of data.
    l1_stats : bool, optional
        Whether to plot pairwise statistical significance comparisons between groups of data.
    p_value_threshold : float, optional
        The threshold for statistical significance.
    different_from : list of lists of bool, optional
        The groups of data that should be compared in pairwise comparisons.
    incremental_stats : bool, optional
        Whether to incrementally perform pairwise comparisons.

    Methods
    -------
    plot_boxplot(self, figure, grouped_series, positions, colors, alpha, direction, l0_stats, l1_stats, p_value_threshold, different_from, incremental_stats, **kwargs)
        A method for plotting boxplots.
    """
        
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
                            if (p_value_threshold is None and p < 0.1) or (p_value_threshold is not None and p < p_value_threshold):
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
                                    figure.text(*[position01, loc_][::1 if direction == "vertical" else -1], "${} {}$".format(">" if p_value > 0.9 else "<", "***" if p < 0.001 else ("**" if p < 0.01 else "*")),
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
                                if (p_value_threshold is None and p < 0.1) or (p_value_threshold is not None and p < p_value_threshold):
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
                                        figure.text(*[position01, loc_][::1 if direction == "vertical" else -1], "${} {}$".format(">" if p_value > 0.9 else "<", "***" if p < 0.001 else ("**" if p < 0.01 else "*")),
                                                  rotation=0 if direction == "vertical" else 270,
                                                  ha="center" if direction == "vertical" else "left",
                                                  va="bottom" if direction == "vertical" else "center",
                                                  fontsize=figure.fs / 2)


class Barplot(Figure):
    """A figure subclass that creates bar plots.

    Parameters
    ----------
    grouped_series : list of lists of Series or array-like of shape (n_groups, n_series)
        The data to plot. Each element in the outer list corresponds to a group of bars, and each element in the inner lists
        corresponds to a series of bars within that group. The Series should be numeric.
    labels : list of str, optional
        The labels to use for the x-axis ticks. If None, the labels will be auto-generated.
    xalim : tuple of 2 floats, optional
        The limits of the x-axis. If None, the limits will be auto-determined.
    yalim : tuple of 2 floats, optional
        The limits of the y-axis. If None, the limits will be auto-determined.
    positions : list of lists of floats, optional
        The x-positions of the bars. If None, the positions will be auto-generated.
    inchesperposition : float, optional
        The spacing between bar positions in inches. Only used if positions is None.
    colors : list of lists of str or tuple, optional
        The colors to use for each bar. If None, the colors will be auto-generated.
    alpha : float or list of lists of float, optional
        The transparency of each bar. If a single float is provided, it will be used for all bars. If a list of lists
        is provided, each element of the outer list should correspond to a group of bars, and each element of the inner
        lists should correspond to a series of bars within that group.
    direction : {'vertical', 'horizontal'}, optional
        The direction in which to plot the bars.
    grouped_offsets : list of lists of floats, optional
        The y-offsets of the bars for each group. If None, the bars will be plotted with zero offset.

    Attributes
    ----------
    fig : matplotlib.figure.Figure
        The Figure object.
    ax : matplotlib.axes.Axes
        The Axes object.

    Methods
    -------
    plot_barplot(figure, grouped_series, positions, colors, alpha=0.5, direction="vertical", grouped_offsets=None, **kwargs)
        Plot a bar plot on the given figure.
    """
        
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
    """A class for creating a line plot.

    Parameters
    ----------
    series_group_x : list of pandas.Series
        A list of pandas Series containing the x values for each line.
    series_group_y : list of pandas.Series
        A list of pandas Series containing the y values for each line.
    xalim : tuple, optional
        A tuple of the form (xmin, xmax) specifying the limits of the x-axis. If not specified, the limits will be
        automatically determined based on the data.
    yalim : tuple, optional
        A tuple of the form (ymin, ymax) specifying the limits of the y-axis. If not specified, the limits will be
        automatically determined based on the data.
    colors : list of str, optional
        A list of colors to use for the lines. If not specified, a default color cycle will be used.
    alpha : float or list of float, optional
        The alpha value(s) to use for the line(s). If a single value is given, it will be applied to all lines. If a
        list is given, it must be the same length as the number of lines.
    **kwargs : additional keyword arguments
        Additional keyword arguments to pass to the underlying plotting functions.

    Attributes
    ----------
    alpha : float or list of float
        The alpha value(s) used for the line(s).
    """
     
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
    """A class for creating line plots with statistical analysis.

    Parameters
    ----------
    grouped_series_x : list of lists of array-like
        The x-axis data for the line plot. Each element of the outer list represents a group of series, and each element of
        the inner list represents a series.
    grouped_series_y : list of lists of array-like
        The y-axis data for the line plot. Each element of the outer list represents a group of series, and each element of
        the inner list represents a series.
    xalim : tuple of 2 floats, optional
        The limits for the x-axis. The first float is the minimum limit and the second float is the maximum limit.
    yalim : tuple of 2 floats, optional
        The limits for the y-axis. The first float is the minimum limit and the second float is the maximum limit.
    colors : array-like or list of color codes, optional
        The color codes to use for the lines. If a single color code is given, it will be used for all lines. If a list of color
        codes is given, each element will be used for the corresponding line.
    alpha : float or array-like, optional
        The transparency of the lines. If a single float is given, it will be used for all lines. If an array-like object is
        given, each element will be used for the corresponding line.
    **kwargs
        Additional keyword arguments to be passed to the `Figure` constructor.

    Methods
    -------
    plot_lineplotwithstats(figure, grouped_series_x, grouped_series_y, colors, alpha, **kwargs)
        Plots the line plot with statistical analysis.
    """
     
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
    """A scatter plot figure.

    Parameters
    ----------
    series_group_x : list of array-like
        The x-values of the scatter plot. Each element of the list corresponds to a group of x-values.
    series_group_y : list of array-like
        The y-values of the scatter plot. Each element of the list corresponds to a group of y-values.
    xalim : tuple of 2 floats, optional
        The lower and upper limits of the x-axis.
    yalim : tuple of 2 floats, optional
        The lower and upper limits of the y-axis.
    colors : list of str or array-like, optional
        The colors of the scatter plot. If a list of strings is provided, each element corresponds to a group of points.
        If an array-like object is provided, it should have the same shape as `series_group_x` and `series_group_y`.
    alpha : float, optional
        The transparency of the scatter plot.
    **kwargs
        Additional keyword arguments to pass to `Figure`.

    Methods
    -------
    plot_scatterplot(figure, series_group_x, series_group_y, colors, alpha, **kwargs)
        Plots the scatter plot.
    """

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
    """A figure class for creating Bland-Altman plots.

    Parameters
    ----------
    series_group_x : list of Series or array-like
        Input data for the x-axis.
    series_group_y : list of Series or array-like
        Input data for the y-axis.
    xalim : tuple, optional
        Tuple specifying the lower and upper limits of the x-axis.
    yalim : tuple, optional
        Tuple specifying the lower and upper limits of the y-axis.
    colors : list or array-like, optional
        A list of colors to use for each series in the input data.
    alpha : float, optional
        The alpha value for the plotted points.
    **kwargs
        Additional keyword arguments to be passed to the `Figure` constructor.

    Attributes
    ----------
    alpha : float
        The alpha value for the plotted points.

    Methods
    -------
    plot_blandaltmanplot(figure, series_group_x, series_group_y, colors, alpha, **kwargs)
        Method to create the Bland-Altman plot.
    """

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
