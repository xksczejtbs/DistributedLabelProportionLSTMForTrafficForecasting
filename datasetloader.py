#from torch_geometric_temporal.dataset import PemsBayDatasetLoader
#from torch_geometric_temporal.dataset import METRLADatasetLoader
from PEMSBAYDatasetLoader import PemsBayDatasetLoader # overwrite original by custom loader
from METRLADatasetLoader import METRLADatasetLoader
from LuSTDatasetLoader import LuSTDatasetLoader
#import MoSTDatasetLoader
def get_loader_by_dataset_name(dataset_name_in,data_dir):
    datasetloader = None
    if dataset_name_in == "PemsBay":
        datasetloader = PemsBayDatasetLoader(raw_data_dir=data_dir+"/raw", data_statistics_path=data_dir)
    elif dataset_name_in == "METRLA":
        datasetloader = METRLADatasetLoader(raw_data_dir=data_dir+"/raw", data_statistics_path=data_dir)
    elif dataset_name_in == "LuST":
        datasetloader = LuSTDatasetLoader(raw_data_dir=data_dir+"/raw", data_statistics_path=data_dir)
    # elif dataset_name_in == "MoST":
    #    datasetloader = MoSTDatasetLoader()
    else:
        raise ValueError("dataset_name " + dataset_name_in + " not possible dataset")
    return datasetloader

from typing import Union, Tuple
from torch_geometric_temporal import StaticGraphTemporalSignal
from sklearn.model_selection import KFold
import torch
""" copied from torch_geometric_temporal.train_test_split and adapted to cross validation """
def temporal_signal_split(
    data_iterator, cvfold: int = 0, num_cvfolds: int = 5, cv_seed: int = 1234
) -> Tuple[StaticGraphTemporalSignal, StaticGraphTemporalSignal]:
    r"""Function to cross validate a data iterator according to a fixed cross validation fold id.

    Arg types:
        * **data_iterator** *(Signal Iterator)* - Node features.
        * **train_ratio** *(float)* - Graph edge indices.

    Return types:
        * **(train_iterator, test_iterator)** *(tuple of Signal Iterators)* - Train and test data iterators.
    """
    #from sklearn.model_selection import StratifiedShuffleSplit
    #kfold = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    # day and night same sample size < 6 a.m. and > 6 p.m.

    kfold = KFold(n_splits=num_cvfolds, random_state=cv_seed, shuffle=True)
    data = torch.empty((data_iterator.snapshot_count))
    train_ids, test_ids = list(kfold.split(data))[cvfold]
    #if needed: save test_ids for output here

    if type(data_iterator) == StaticGraphTemporalSignal:
        train_iterator = StaticGraphTemporalSignal(
            data_iterator.edge_index,
            data_iterator.edge_weight,
            [data_iterator.features[train_id] for train_id in train_ids],
            [data_iterator.targets[train_id] for train_id in train_ids],
            **{key: getattr(data_iterator, key)[train_ids] for key in data_iterator.additional_feature_keys}
        )

        test_iterator = StaticGraphTemporalSignal(
            data_iterator.edge_index,
            data_iterator.edge_weight,
            [data_iterator.features[test_id] for test_id in test_ids],
            [data_iterator.targets[test_id] for test_id in test_ids],
            **{key: getattr(data_iterator, key)[test_ids] for key in data_iterator.additional_feature_keys}
        )

    return train_iterator, test_iterator
if __name__ == "__main__":
    import torch
    edges = 5000
    features = 2
    timestamps = 12
    x = torch.randn(edges, features, timestamps)
    print("x",x.shape)
    cvfold = 0
    num_cvfolds = 5
    cv_seed = 1234
    kfold = KFold(n_splits=num_cvfolds, random_state=cv_seed, shuffle=True)
    train_ids, test_ids = list(kfold.split(x))[cvfold]
    print(test_ids)
    print(x[test_ids])
