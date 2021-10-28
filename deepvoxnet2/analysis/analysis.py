import numpy as np
import pandas as pd
from deepvoxnet2.components.mirc import Mirc
from deepvoxnet2.keras.metrics import get_metric, get_metric_at_multiple_thresholds
from deepvoxnet2.factories.directory_structure import MircStructure
from deepvoxnet2.analysis.data import Data
from scripts.jberte3.KAROLINSKA2021.preprocessing.s2_datasets import mrclean, karolinska, crisp


class Analysis(object):
    def __init__(self, *data):
        assert all([isinstance(data_, Data) for data_ in data])
        if any([len(data_.index.names) != len(data[0].index.names) for data_ in data[1:]]):
            print("Watch out: not all indices have the same number of levels across all the data!")

        if any([len(data_.columns.names) != len(data[0].columns.names) for data_ in data[1:]]):
            print("Watch out: not all columns have the same number of levels across all the data!")

        all_indices = [ind for data_ in data for ind in data_.index]
        if any([set(data_.index) != set(data[0].index) for data_ in data]):
            print("Watch out: the given data have different indices!")

        all_columns = [col for data_ in data for col in data_.columns]
        if len(set(all_columns)) != len(all_columns):
            print("Watch out: the given data have overlapping columns!")

        all_indices = list(dict.fromkeys(all_indices))
        max_nb_index_levels = max([len(ind) for ind in all_indices])
        index_names = [None] * max_nb_index_levels
        for data_ in data:
            if len(data_.index.names) == max_nb_index_levels:
                index_names = data_.index.names
                break

        all_columns = list(dict.fromkeys(all_columns))
        max_nb_column_levels = max([len(col) for col in all_columns])
        column_names = [None] * max_nb_column_levels
        for data_ in data:
            if len(data_.columns.names) == max_nb_column_levels:
                column_names = data_.columns.names
                break

        self.df = pd.DataFrame(index=pd.MultiIndex.from_tuples(all_indices, names=index_names), columns=pd.MultiIndex.from_tuples(all_columns, names=column_names))
        self.index = self.df.index
        self.columns = self.df.columns
        for data_ in data:
            for ind in data_.index:
                for col in data_.columns:
                    self.df.loc[ind + (None,) * (max_nb_index_levels - len(ind)), col + (None,) * (max_nb_column_levels - len(col))] = data_.df.loc[ind, col]

    def __call__(self):
        return self.data()

    def data(self):
        return [Data(self.df.loc[:, [column]]) for column in self.columns]

    def get_empty_df(self):
        return pd.DataFrame(index=self.index, columns=self.columns)

    def apply(self, apply_fn, **kwargs):
        columns = pd.MultiIndex.from_tuples([(apply_fn.__name__,)], names=["apply_fn"])
        df = pd.DataFrame(index=self.index, columns=columns)
        for ind in self.df.dropna().index:
            df.loc[ind, :] = [apply_fn(*self.df.loc[ind, :].values, **kwargs).numpy()]

        return Data(df)

    def dropna(self):
        indices = self.df.dropna().index
        data = [Data(self.df.loc[indices, [column]]) for column in self.columns]
        return Analysis(*data)


if __name__ == "__main__":
    y_true_dfs = []
    for fold_i in range(0, 2):
        mirc = Mirc()
        mirc.add(mrclean(data='val', fold_i=fold_i))
        y_true_dfs.append(mirc.get_df("lesion_brain_nocsf"))

    y_pred_dfs = []
    for fold_i in range(1, 3):
        mirc = Mirc()
        mirc.add(mrclean(data='val', fold_i=fold_i))
        y_pred_dfs.append(MircStructure(
            base_dir="/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/tmp/datasets/Runs",
            run_name="mrclean_baseline",
            experiment_name="david_baseline",
            round_i=0,
            fold_i=fold_i,
            validation_mirc=mirc
        ).get_df("val", "full_val_recal"))

    y_true_df = pd.concat(y_true_dfs, axis=0)
    y_pred_df = pd.concat(y_pred_dfs, axis=0)
    y_true_data = Data(y_true_df)
    y_pred_data = Data(y_pred_df)

    analysis = Analysis(y_true_data, y_pred_data)
    dice = analysis.apply(get_metric("dice_coefficient", threshold=0.5)).dropna().squeeze()
    hd95 = analysis.apply(get_metric("hausdorff_distance", threshold=0.5)).dropna().squeeze()
    case_tpr = analysis.apply(get_metric_at_multiple_thresholds("true_positive_rate", thresholds=np.linspace(1e-7, 1 - 1e-7, 100), threshold_axis=0)).dropna()
    case_ppv = analysis.apply(get_metric_at_multiple_thresholds("positive_predictive_value", thresholds=np.linspace(1e-7, 1 - 1e-7, 100), threshold_axis=0)).dropna()
    case_auc = Analysis(case_tpr, case_ppv).apply(get_metric("riemann_sum")).squeeze()
    dataset_tpr = case_tpr.combine_mean()
    dataset_ppv = case_ppv.combine_mean()
    dataset_auc = Analysis(dataset_tpr, dataset_ppv).apply(get_metric("riemann_sum")).squeeze()
    ece_stats = analysis.apply(get_metric("ece", nbins=20, return_bin_stats=True)).dropna()
    case_ece = analysis.apply(get_metric("ece", nbins=20)).squeeze()

    y_true_data, y_pred_data = analysis.dropna()()  # remove nan rows and get Data objects
    analysis = Analysis(y_true_data.reshape((-1, 1, 1, 1, 1)).combine_concat(), y_pred_data.reshape((-1, 1, 1, 1, 1)).combine_concat())
    dataset_ece_stats = analysis.apply(get_metric("ece", nbins=20, return_bin_stats=True))
    dataset_ece = analysis.apply(get_metric("ece", nbins=20)).squeeze()
