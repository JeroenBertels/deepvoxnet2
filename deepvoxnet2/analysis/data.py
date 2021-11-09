import numpy as np
import pandas as pd
from collections import Iterable


class Series(object):
    def __init__(self, series):
        if not isinstance(series, Iterable):
            series = [series]

        self.series = np.array(series)
        self.series_, self.nnans, self.min, self.p25, self.p50, self.pm, self.p75, self.max, self.iqr, self.pmin, self.pmax, self.outliers = self.get_stats()
        self.median, self.mean = self.p50, self.pm
        self.std = np.std(self.series_)
        self.ste = self.std / np.sqrt(len(self.series_))

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

        if isinstance(series_group, tuple):
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


class GroupedSeries(object):
    def __init__(self, grouped_series):
        if not isinstance(grouped_series, Iterable):
            grouped_series = [grouped_series]

        if isinstance(grouped_series, tuple):
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


class Data(object):
    def __init__(self, df):
        assert len(df.index.names) == 3 and all([name == name_ for name, name_ in zip(df.index.names, ["dataset_id", "case_id", "record_id"])]),  "The index must be a MultiIndex with the three levels: dataset_id, case_id and record_id."
        assert df.shape[1] == 1, "The dataframe must have only a single column."
        assert isinstance(df, (Data, pd.DataFrame)), "A Data class needs to be constructed from a DataFrame or another Data class."
        self.df = df.sort_index() if isinstance(df, pd.DataFrame) else df.df.sort_index()
        self.index = self.df.index
        self.columns = self.df.columns
        self.shape = self.df.shape

    def get_empty_df(self, reduction_level=None, combine_fn_name=None, reduce_all_below=True):
        if reduction_level is None:
            indices = self.index

        else:
            indices = []
            if reduction_level == "record_id":
                for dataset_id, dataset_df in self.df.groupby(level="dataset_id", dropna=False):
                    for case_id, case_df in dataset_df.groupby(level="case_id", dropna=False):
                        for record_i, (record_id, record_df) in enumerate(case_df.groupby(level="record_id", dropna=False)):
                            indices.append((dataset_id, case_id, combine_fn_name))
                            break

            elif reduction_level == "case_id":
                for dataset_id, dataset_df in self.df.groupby(level="dataset_id", dropna=False):
                    for case_i, (case_id, case_df) in enumerate(dataset_df.groupby(level="case_id", dropna=False)):
                        if reduce_all_below:
                            indices.append((dataset_id, combine_fn_name, combine_fn_name))
                            break

                        elif case_i == 0:
                            indices_case_0 = []
                            for record_id, record_df in case_df.groupby(level="record_id", dropna=False):
                                indices_case_0.append((record_id,))
                                indices.append((dataset_id, combine_fn_name, record_id))

                        else:
                            indices_case_i = [(record_id,) for record_id, record_df in case_df.groupby(level="record_id", dropna=False)]
                            assert indices_case_0 == indices_case_i, "The cases don't have the same nested structure."

            elif reduction_level == "dataset_id":
                for dataset_i, (dataset_id, dataset_df) in enumerate(self.df.groupby(level="dataset_id", dropna=False)):
                    if reduce_all_below:
                        indices.append((combine_fn_name, combine_fn_name, combine_fn_name))
                        break

                    elif dataset_i == 0:
                        indices_dataset_0 = []
                        for case_id, case_df in dataset_df.groupby(level="case_id", dropna=False):
                            for record_id, record_df in case_df.groupby(level="record_id", dropna=False):
                                indices_dataset_0.append((case_id, record_id))
                                indices.append((combine_fn_name, case_id, record_id))

                    else:
                        indices_dataset_i = [(case_id, record_id) for case_id, case_df in dataset_df.groupby(level="case_id", dropna=False) for record_id, record_df in case_df.groupby(level="record_id", dropna=False)]
                        assert indices_dataset_0 == indices_dataset_i, "The datasets don't have the same nested structure."

            else:
                raise ValueError("The reduce_level specified does not exist.")

            indices = pd.MultiIndex.from_tuples(indices, names=["dataset_id", "case_id", "record_id"])

        return pd.DataFrame(index=indices, columns=self.columns)

    @staticmethod
    def iter_upper_level(df, level):
        if level == "dataset_id":
            return [((), df)]

        elif level == "case_id":
            return [((idx_,), df_) for idx_, df_ in df.groupby(level="dataset_id", dropna=False)]

        elif level == "record_id":
            return [((idx_, idx__), df__) for idx_, df_ in df.groupby(level="dataset_id", dropna=False) for idx__, df__ in df_.groupby(level="case_id", dropna=False)]

        else:
            raise ValueError("Unknown level value.")

    @staticmethod
    def iter_level(df, level):
        if level == "dataset_id":
            return [((idx_,), df_) for idx_, df_ in df.groupby(level="dataset_id", dropna=False)]

        elif level == "case_id":
            return [((idx_, idx__), df__) for idx_, df_ in df.groupby(level="dataset_id", dropna=False) for idx__, df__ in df_.groupby(level="case_id", dropna=False)]

        elif level == "record_id":
            return [((idx_, idx__, idx__), df___) for idx_, df_ in df.groupby(level="dataset_id", dropna=False) for idx__, df__ in df_.groupby(level="case_id", dropna=False) for idx___, df___ in df__.groupby(level="record_id", dropna=False)]

        else:
            raise ValueError("Unknown level value.")

    @staticmethod
    def iter_lower_level(df, level):
        if level == "dataset_id":
            return [((idx_, idx__), df__) for idx_, df_ in df.groupby(level="case_id", dropna=False) for idx__, df__ in df_.groupby(level="record_id", dropna=False)]

        elif level == "case_id":
            return [((idx_,), df_) for idx_, df_ in df.groupby(level="record_id", dropna=False)]

        elif level == "record_id":
            return [((), df)]

        else:
            raise ValueError("Unknown level value.")

    def combine(self, combine_fn, reduction_level="case_id", reduce_all_below=True, custom_combine_fn_name=False, exclude_nan=True, exclude_inf=True, **kwargs):
        combine_fn_name = custom_combine_fn_name if custom_combine_fn_name is not False else combine_fn.__name__
        combined_df = self.get_empty_df(reduction_level=reduction_level, combine_fn_name=combine_fn_name, reduce_all_below=reduce_all_below)
        for upper_index, upper_df in self.iter_upper_level(self.df, reduction_level):
            reduced_indices = []
            values = []
            if reduce_all_below:
                reduced_indices.append(upper_index + (combine_fn_name,) * (3 - len(upper_index)))
                values.append([df_.values[0, 0] for _, df_ in self.iter_level(upper_df, level="record_id")])

            else:
                for lower_index, lower_df in self.iter_lower_level(upper_df, reduction_level):
                    reduced_indices.append(upper_index + (combine_fn_name,) + lower_index)
                    values.append([df_.values[0, 0] for _, df_ in self.iter_level(upper_df, level="record_id")])

            for reduced_idx, values_ in zip(reduced_indices, values):
                if exclude_nan:
                    values_ = [value for value in values_ if (np.isscalar(value) and not np.isnan(value)) or (not np.isscalar(value) and (value.ndim > 0 or not np.isnan(value)))]

                if exclude_inf:
                    values_ = [value for value in values_ if (np.isscalar(value) and not np.isinf(value)) or (not np.isscalar(value) and (value.ndim > 0 or not np.isinf(value)))]

                combined_df.at[reduced_idx, combined_df.columns[0]] = combine_fn(values_, **kwargs) if len(values_) > 0 else np.nan

        return Data(combined_df)

    def combine_mean(self, axis=0, custom_combine_fn_name="mean", **kwargs):
        return self.combine(np.mean, axis=axis, custom_combine_fn_name=custom_combine_fn_name, **kwargs)

    def combine_sum(self, axis=0, custom_combine_fn_name="sum", **kwargs):
        return self.combine(np.sum, axis=axis, custom_combine_fn_name=custom_combine_fn_name, **kwargs)

    def combine_concat(self, axis=0, custom_combine_fn_name="concat", **kwargs):
        return self.combine(np.concatenate, axis=axis, custom_combine_fn_name=custom_combine_fn_name, **kwargs)

    def apply(self, apply_fn, *args, custom_apply_fn_name=None, **kwargs):
        columns = self.columns if custom_apply_fn_name is None else pd.MultiIndex.from_tuples([(custom_apply_fn_name,)], names=["apply_fn"])
        applied_df = pd.DataFrame(index=self.index, columns=columns)
        for ind in self.index:
            applied_df.at[ind, applied_df.columns[0]] = apply_fn(self.df.at[ind, self.df.columns[0]], *args, **kwargs)

        return Data(applied_df)

    def volume(self, voxel_volume=1, **kwargs):
        return self.apply(lambda value: np.sum(value) * voxel_volume, **kwargs)

    def mean(self, *args, **kwargs):
        return self.apply(np.mean, *args, **kwargs)

    def sum(self, *args, **kwargs):
        return self.apply(np.sum, *args, **kwargs)

    def reshape(self, *args, **kwargs):
        return self.apply(np.reshape, *args, **kwargs)

    def flatten(self, **kwargs):
        return self.reshape((-1,), **kwargs)

    def squeeze(self, *args, **kwargs):
        return self.apply(np.squeeze, *args, **kwargs)

    def round(self, decimals=0, **kwargs):
        return self.apply(np.round, decimals=decimals, **kwargs)

    def format(self, formatting="{}", **kwargs):
        return self.apply(lambda value: formatting.format(value), **kwargs)

    def expand_dims(self, *args, **kwargs):
        return self.apply(np.expand_dims, *args, **kwargs)

    def dropna(self, axis=0, how="any"):
        return Data(self.df.dropna(axis=axis, how=how))

    def reindex(self, *args, **kwargs):
        return Data(self.df.reindex(*args, **kwargs))

    def set_axis(self, *args, **kwargs):
        return Data(self.df.set_axis(*args, **kwargs))

    def rename(self, *args, **kwargs):
        return Data(self.df.rename(*args, **kwargs))

    def bootstrap(self, level="record_id", seed=0, n=None):
        dfs = [df for _, df in self.iter_level(self.df, level)]
        np.random.seed(seed)
        bootstrapped_dfs = []
        for i, j in enumerate(np.random.choice(range(len(dfs)), size=len(dfs) if n is None else n)):
            df = dfs[j]
            for level_name in ["dataset_id", "case_id", "record_id"]:
                df = df.rename(index=lambda ind: f"{ind}_{seed}", level=level_name)

            bootstrapped_dfs.append(df.rename(index=lambda ind: f"{ind}_{i}", level=level))

        return Data(pd.concat(bootstrapped_dfs))

    def get_stats(self, reduction_level="case_id", reduce_all_below=True, return_formatted=False, **kwargs):
        n = self.combine(lambda values: len([value for value in values if (np.isscalar(value) and not np.isnan(value) and not np.isinf(value)) or (not np.isscalar(value) and (value.ndim > 0 or (not np.isnan(value) and not np.isinf(value))))]), custom_combine_fn_name="n", reduction_level=reduction_level, reduce_all_below=reduce_all_below, exclude_nan=False, exclude_inf=False)
        nnan = self.combine(lambda values: len([value for value in values if (np.isscalar(value) and np.isnan(value)) or (not np.isscalar(value) and value.ndim == 0 and np.isnan(value))]), custom_combine_fn_name="nnan", reduction_level=reduction_level, reduce_all_below=reduce_all_below, exclude_nan=False, exclude_inf=False)
        ninf = self.combine(lambda values: len([value for value in values if (np.isscalar(value) and np.isinf(value)) or (not np.isscalar(value) and value.ndim == 0 and np.isinf(value))]), custom_combine_fn_name="ninf", reduction_level=reduction_level, reduce_all_below=reduce_all_below, exclude_nan=False, exclude_inf=False)
        pmin = self.combine(np.min, axis=0, custom_combine_fn_name="min", reduction_level=reduction_level, reduce_all_below=reduce_all_below, exclude_inf=False)
        p5 = self.combine(np.percentile, axis=0, q=5, custom_combine_fn_name="p5", reduction_level=reduction_level, reduce_all_below=reduce_all_below, exclude_inf=False)
        p25 = self.combine(np.percentile, axis=0, q=25, custom_combine_fn_name="p25", reduction_level=reduction_level, reduce_all_below=reduce_all_below, exclude_inf=False)
        p50 = self.combine(np.percentile, axis=0, q=50, custom_combine_fn_name="p50", reduction_level=reduction_level, reduce_all_below=reduce_all_below, exclude_inf=False)
        p75 = self.combine(np.percentile, axis=0, q=75, custom_combine_fn_name="p75", reduction_level=reduction_level, reduce_all_below=reduce_all_below, exclude_inf=False)
        p95 = self.combine(np.percentile, axis=0, q=95, custom_combine_fn_name="p95", reduction_level=reduction_level, reduce_all_below=reduce_all_below, exclude_inf=False)
        pmax = self.combine(np.max, axis=0, custom_combine_fn_name="max", reduction_level=reduction_level, reduce_all_below=reduce_all_below, exclude_inf=False)
        pmean = self.combine_mean(reduction_level=reduction_level, reduce_all_below=reduce_all_below)
        std = self.combine(np.std, axis=0, custom_combine_fn_name="std", reduction_level=reduction_level, reduce_all_below=reduce_all_below)
        ste = self.get_empty_df(combine_fn_name="ste", reduction_level=reduction_level, reduce_all_below=reduce_all_below)
        iqr = self.get_empty_df(combine_fn_name="iqr", reduction_level=reduction_level, reduce_all_below=reduce_all_below)
        nnaninf = self.get_empty_df(combine_fn_name="nnaninf", reduction_level=reduction_level, reduce_all_below=reduce_all_below)
        for i, _ in enumerate(std.index):
            ste.iat[i, 0] = std.df.iat[i, 0] / np.sqrt(n.df.iat[i, 0])
            iqr.iat[i, 0] = p75.df.iat[i, 0] - p25.df.iat[i, 0]
            nnaninf.iat[i, 0] = nnan.df.iat[i, 0] + ninf.df.iat[i, 0]

        data_stats = Data(pd.concat([n.df, nnan.df, ninf.df, nnaninf, pmin.df, p5.df, p25.df, p50.df, p75.df, p95.df, pmax.df, pmean.df, std.df, ste, iqr]))
        if return_formatted:
            data_stats = data_stats.print_stats(**kwargs)

        return data_stats

    def print_stats(self, printing_type=0, formatting="{}"):
        printing_indices = pd.MultiIndex.from_tuples([tuple(["print_stats" if idx[i] == "p50" else idx[i] for i in range(3)]) for idx in self.index if any([idx[i] == "p50" for i in range(3)])], names=self.index.names)
        printing_df = pd.DataFrame(index=printing_indices, columns=self.columns)
        for i, printing_idx in enumerate(printing_df.index):
            idx_fn = lambda x: tuple([x if idx_ == "print_stats" else idx_ for idx_ in printing_idx])
            column = printing_df.columns[0]
            p25 = self.df.at[idx_fn('p25'), column]
            p50 = self.df.at[idx_fn('p50'), column]
            p75 = self.df.at[idx_fn('p75'), column]
            pmean = self.df.at[idx_fn('mean'), column]
            nnaninf = self.df.at[idx_fn('nnaninf'), column]
            std = self.df.at[idx_fn('std'), column]
            n = self.df.at[idx_fn('n'), column]
            if n == 0:
                printing_df.at[printing_idx, column] = "/"

            elif printing_type == 0:
                s = f"{formatting} [{formatting}-{formatting}] {{}}"  # p50 [p25 - p75] [nnaninf]
                printing_df.at[printing_idx, column] = s.format(p50, p25, p75, "[{:.0f}]".format(float(nnaninf)) if float(nnaninf) > 0 else "")

            elif printing_type == 1:  # mean ± std [nnaninf]
                s = f"{formatting} ± {formatting} {{}}"
                printing_df.at[printing_idx, column] = s.format(pmean, std, "[{:.0f}]".format(float(nnaninf)) if float(nnaninf) > 0 else "")

            elif printing_type == 2:  # mean [nnaninf]
                s = f"{formatting} {{}}"
                printing_df.at[printing_idx, column] = s.format(pmean, "[{:.0f}]".format(float(nnaninf)) if float(nnaninf) > 0 else "")

            elif printing_type == 3:  # n
                s = f"{formatting}"
                printing_df.at[printing_idx, column] = s.format(n)

            else:
                raise ValueError("Unknown printing_type.")

        printing_data = Data(printing_df)
        # print(printing_data.df.transpose().to_latex(escape=True))
        return printing_data


if __name__ == "__main__":
    indices = pd.MultiIndex.from_tuples([("dataset_A", "case_0", "record_0"), ("dataset_A", "case_1", "record_0"), ("dataset_B", "case_0", "record_0"), ("dataset_B", "case_1", "record_0")], names=["dataset_id", "case_id", "record_id"])
    columns = pd.MultiIndex.from_tuples([("experiment_A", "metric_A")], names=["experiment_name", "metric_name"])
    df = pd.DataFrame([[np.array([[1, 2]])], [np.array([[3, 4]])], [np.array([[10, 11]])], [np.array([[10, 11]])]], index=indices, columns=columns)
    data = Data(df)

    print("The original dataframe: ")
    print(df, "\n")
    print("The empty dataframe: ")
    print(data.get_empty_df(), "\n")
    print("The empty dataframe mean-reduced at dataset level: ")
    print(data.get_empty_df(reduction_level="dataset_id", combine_fn_name="mean", reduce_all_below=False), "\n")
    print("The empty dataframe mean-reduced at dataset level and all below: ")
    print(data.get_empty_df(reduction_level="dataset_id", combine_fn_name="mean", reduce_all_below=True), "\n")
    print("The empty dataframe sum-reduced at case level: ")
    print(data.get_empty_df(reduction_level="case_id", combine_fn_name="sum", reduce_all_below=False), "\n")
    print("The empty dataframe sum-reduced at case level and all below: ")
    print(data.get_empty_df(reduction_level="case_id", combine_fn_name="sum", reduce_all_below=True), "\n")
    print("The empty dataframe concat-reduced at record level: ")
    print(data.get_empty_df(reduction_level="record_id", combine_fn_name="concat", reduce_all_below=False), "\n")
    print("The empty dataframe concat-reduced at record level and all below: ")
    print(data.get_empty_df(reduction_level="record_id", combine_fn_name="concat", reduce_all_below=True), "\n")

    print("MEAN dataframe reduced at dataset level: ")
    data = Data(df)
    print(data.combine_mean(reduction_level="dataset_id", reduce_all_below=False).df, "\n")
    print("MEAN dataframe reduced at dataset level and all below: ")
    data = Data(df)
    print(data.combine_mean(reduction_level="dataset_id", reduce_all_below=True).df, "\n")

    print("SUM dataframe reduced at dataset level: ")
    data = Data(df)
    print(data.combine_sum(reduction_level="dataset_id", reduce_all_below=False).df, "\n")
    print("SUM dataframe reduced at dataset level and all below: ")
    data = Data(df)
    print(data.combine_sum(reduction_level="dataset_id", reduce_all_below=True).df, "\n")

    print("CONCAT dataframe reduced at dataset level: ")
    data = Data(df)
    print(data.combine_concat(reduction_level="dataset_id", reduce_all_below=False).df, "\n")
    print("CONCAT dataframe reduced at dataset level and all below: ")
    data = Data(df)
    print(data.combine_concat(reduction_level="dataset_id", reduce_all_below=True).df, "\n")

    print("RESHAPE dataframe elements e.g. from (1, 2) to (2, 1): ")
    data = Data(df)
    print(data.reshape((2, 1)).df)

    print("FLATTEN dataframe elements: ")
    data = Data(df)
    print(data.flatten().df)

    print("MEAN dataframe elements along first axis: ")
    data = Data(df)
    print(data.mean(axis=0).df)

    print("SUM dataframe elements along second axis: ")
    data = Data(df)
    print(data.sum(axis=1).df)
