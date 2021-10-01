import warnings
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
        if not all([sorted(data[0].df.index) == sorted(data_.df.index) for data_ in data[1:]]):
            warnings.warn("Watch out: not all indices are identical across all the data!")

        self.data_df = pd.concat([data_.df for data_ in data], axis=1)

    def __call__(self):
        df = self.data_df.dropna()
        return [Data(df.loc[:, [column]]) for column in df.columns]

    def apply(self, apply_fn, **kwargs):
        indices = self.data_df.dropna().index
        columns = pd.MultiIndex.from_tuples([(apply_fn.__name__,)], names=["apply_fn"])
        df = pd.DataFrame(index=indices, columns=columns)
        for index in indices:
            df.loc[index, :] = [apply_fn(*self.data_df.loc[index, :].values, **kwargs).numpy()]

        return Data(df)


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

    y_true_data, y_pred_data = analysis()  # remove nan rows
    analysis = Analysis(y_true_data.reshape((-1, 1, 1, 1, 1)).combine_concat(), y_pred_data.reshape((-1, 1, 1, 1, 1)).combine_concat())
    dataset_ece_stats = analysis.apply(get_metric("ece", nbins=20, return_bin_stats=True))
    dataset_ece = analysis.apply(get_metric("ece", nbins=20)).squeeze()
