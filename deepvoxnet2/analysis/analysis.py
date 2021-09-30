import numpy as np
import pandas as pd
from collections import Iterable
from deepvoxnet2.components.mirc import Mirc
from deepvoxnet2.keras.metrics import get_metric, get_metric_at_multiple_thresholds
from deepvoxnet2.factories.directory_structure import MircStructure
from deepvoxnet2.analysis.data import Data
from scripts.jberte3.KAROLINSKA2021.preprocessing.s2_datasets import mrclean, karolinska, crisp


class Analysis(object):
    def __init__(self, y_true_data, y_pred_data):
        assert isinstance(y_true_data, Data) and isinstance(y_pred_data, Data)
        assert sorted(y_true_data.df.index) == sorted(y_pred_data.df.index)
        self.y_true_data = y_true_data
        self.y_pred_data = y_pred_data

    def apply(self, apply_fn, columns=None, **kwargs):
        if columns is None:
            columns = pd.MultiIndex.from_tuples([(apply_fn.__name__,)], names=["metric_name"])
            
        result_df = self.y_true_data.get_empty_df(columns=columns)
        for index in result_df.index:
            y_true = self.y_true_data.df.loc[index, :].values[0]
            y_pred = self.y_pred_data.df.loc[index, :].values[0]
            result_df.loc[index, :] = [apply_fn(y_true, y_pred, **kwargs).numpy()]

        return Data(result_df)


if __name__ == "__main__":
    y_true_dfs = []
    y_pred_dfs = []
    for fold_i in range(1):
        mirc = Mirc()
        mirc.add(mrclean(data='val', fold_i=fold_i))
        y_true_dfs.append(mirc.get_df("lesion_brain_nocsf"))
        y_pred_dfs.append(MircStructure(
            base_dir="/usr/local/micapollo01/MIC/DATA/STAFF/jberte3/tmp/datasets/Runs",
            run_name="mrclean_baseline",
            experiment_name="david_baseline",
            round_i=0,
            fold_i=fold_i,
            validation_mirc=mirc
        ).get_df("val", "full_val_recal"))

    y_true_data = Data(pd.concat(y_true_dfs, axis=0))
    y_pred_data = Data(pd.concat(y_pred_dfs, axis=0))
    analysis = Analysis(y_true_data, y_pred_data)
    dice = analysis.apply(get_metric("dice_coefficient", threshold=0.5)).squeeze()
    hd95 = analysis.apply(get_metric("hausdorff_distance", threshold=0.5)).squeeze()
    tpr = analysis.apply(get_metric_at_multiple_thresholds("true_positive_rate", thresholds=np.linspace(1e-7, 1 - 1e-7, 100), threshold_axis=0)).combine_mean()
    ppv = analysis.apply(get_metric_at_multiple_thresholds("positive_predictive_value", thresholds=np.linspace(1e-7, 1 - 1e-7, 100), threshold_axis=0)).combine_mean()
    auc = Analysis(tpr, ppv).apply(get_metric("riemann_sum")).squeeze()
    ece = analysis.apply(get_metric("ece", nbins=100)).squeeze()
    dataset_ece = Analysis(y_true_data.flatten().combine_concat().reshape((-1, 1, 1, 1, 1)), y_pred_data.flatten().combine_concat().reshape((-1, 1, 1, 1, 1))).apply(get_metric("ece", nbins=100)).squeeze()
