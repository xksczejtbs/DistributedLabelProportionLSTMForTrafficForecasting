import os
import zipfile
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from six.moves import urllib
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from filelock import FileLock

class LuSTDatasetLoader(object):
    """A traffic forecasting dataset as described in
    L. Codeca, R. Frank, S. Faye and T. Engel,
    "Luxembourg SUMO Traffic (LuST) Scenario: Traffic Demand Evaluation"
    in IEEE Intelligent Transportation Systems Magazine, vol. 9, no. 2, pp. 52-63, Summer 2017.
    DOI: 10.1109/MITS.2017.2666585
    URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7906642&isnumber=7904753.

    This traffic dataset is collected by .... It is represented by a network of ... traffic sensors
    ...
    in 5 minute intervals.

    """

    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data"), data_statistics_path=os.path.join(os.getcwd(), "data")):
        super(LuSTDatasetLoader, self).__init__()
        self.raw_data_dir = raw_data_dir
        self.data_statistics_path = data_statistics_path
        self._read_web_data()

    def _download_url(self, url, save_path):  # pragma: no cover
        with urllib.request.urlopen(url) as dl_file:
            with open(save_path, "wb") as out_file:
                out_file.write(dl_file.read())

    def _read_web_data(self):
        url = "https://drive.google.com/file/d/1OjPkvptYb22eThm-0zOArBEFA_aLifqp/view?usp=sharing"

        # Check if zip file is in data folder from working directory, otherwise download
        if not os.path.isfile(
            os.path.join(self.raw_data_dir, "LuST.zip")
        ):  # pragma: no cover
            if not os.path.exists(self.raw_data_dir):
                os.makedirs(self.raw_data_dir)
            self._download_url(url, os.path.join(self.raw_data_dir, "LuST.zip"))

        if not os.path.isfile(
            os.path.join(self.raw_data_dir, "lust_adj_mat.npy")
        ) or not os.path.isfile(
            os.path.join(self.raw_data_dir, "lust_node_values.npy")
        ):  # pragma: no cover
            with zipfile.ZipFile(
                os.path.join(self.raw_data_dir, "LuST.zip"), "r"
            ) as zip_fh:
                zip_fh.extractall(self.raw_data_dir)

        A = np.load(os.path.join(self.raw_data_dir, "lust_adj_mat.npy"))
        X = np.load(os.path.join(self.raw_data_dir, "lust_node_values.npy"))#.transpose(
            #(1, 2, 0)
        #)
        X = X.astype(np.float32)

        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X, axis=(0, 2))
        X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2))
        X = X / stds.reshape(1, -1, 1)

        mins = np.min(X, axis=(0, 2))
        mins = mins.reshape(1, -1, 1)
        maxs = np.max(X, axis=(0, 2))
        maxs = maxs.reshape(1, -1, 1)
        if not os.path.isfile(self.data_statistics_path + '/Static_node_Features_mins_LuST.pt'):
            with FileLock(os.path.expanduser("~/.data.lock")):
                torch.save(torch.from_numpy(mins), self.data_statistics_path + '/Static_node_Features_mins_LuST.pt')
                torch.save(torch.from_numpy(maxs), self.data_statistics_path + '/Static_node_Features_maxs_LuST.pt')
                torch.save(torch.from_numpy(means), self.data_statistics_path + '/Static_node_Features_means_LuST.pt')
                torch.save(torch.from_numpy(stds), self.data_statistics_path + '/Static_node_Features_stds_LuST.pt')

        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 12, num_timesteps_out: int = 12):
        """Uses the node features of the graph and generates a feature/target
        relationship of the shape
        (num_nodes, num_node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        predicting the average traffic speed using num_timesteps_in to predict the
        traffic conditions in the next num_timesteps_out

        Args:
            num_timesteps_in (int): number of timesteps the sequence model sees
            num_timesteps_out (int): number of timesteps the sequence model has to predict
        """
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, :, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 12, num_timesteps_out: int = 12
    ) -> StaticGraphTemporalSignal:
        """Returns data iterator for LuSt dataset as an instance of the
        static graph temporal signal class.

        Return types:
            * **dataset** *(StaticGraphTemporalSignal)* - The LuST traffic
                forecasting dataset.
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )
        return dataset
