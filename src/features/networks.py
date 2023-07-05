import torch.nn as nn
import torch
from graph import Graph
import numpy as np
from common.model import *
from features.layers import STGConv, GCNodeEdgeModule, SENet, GCN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TGraphNet(nn.Module):
    """
    A temporal graph neural network (TGraphNet) model for time series prediction.

    Parameters:
        infeat_v (int): The number of input node features.
        infeat_e (int): The number of input edge features.
        nhid_v (List[int]): List of integers representing the number of hidden features in each GCN layer for nodes.
        nhid_e (List[int]): List of integers representing the number of hidden features in each GCN layer for edges.
        n_oute (int): The number of output edge features.
        n_outv (int): The number of output node features.
        gcn_window (List[int]): The length of the frames in the input time series.
        tcn_window (List[int]): The window sizes of the temporal convolutional layers.
        in_frames (int): The number of frames in the input time series.
        gconv_stages (List[int]): List of integers representing the number of stages in each GCN layer.
        use_residual_connections (bool): Use residual connections within the GCN layers.
        use_non_parametric (bool): Use non-parametric temporal aggregation.
        use_edge_conv (bool): Use edge convolutional layers.
    """
    def __init__(self,
                 infeat_v,
                 infeat_e,
                 nhid_v,
                 nhid_e,
                 n_oute,
                 n_outv,
                 gcn_window,
                 tcn_window,
                 aggregate,
                 in_frames,
                 gconv_stages,
                 dropout,
                 use_residual_connections,
                 use_non_parametric,
                 use_edge_conv,):
        super(TGraphNet, self).__init__()

        adj_v, adj_e, T = Graph(17, 16, gcn_window[0])()

        self.adj_v = nn.Parameter(adj_v).to(device)
        self.adj_e = nn.Parameter(adj_e).to(device)
        self.T = T.to(device)
        self.gcn_window = gcn_window
        self.tcn_window = tcn_window
        self.in_frames = in_frames
        self.infeat_v = infeat_v
        self.infeat_e = infeat_e
        self.use_edge_conv = use_edge_conv

        self.seq_len = in_frames // gcn_window[0]
        self.n_nodes = 17 * gcn_window[0]
        self.n_edges = 16 * gcn_window[0]

        self.pre = GCNodeEdgeModule(in_frames, infeat_v, infeat_e, nhid_v[0], nhid_e[0], dropout=0) if use_edge_conv else GCN(in_frames, infeat_v, nhid_v[0], dropout=0)
        self.layers = nn.ModuleList()

        n_stages = len(nhid_v)
        for i in range(n_stages):
            adj_v, adj_e, T = Graph(17, 16, self.gcn_window[i], norm=True)()

            self.layers.append(
                STGConv(
                    nhid_v=nhid_v[i],
                    nhid_e=nhid_e[i],
                    adj_e=adj_e.to(device),
                    adj_v=adj_v.to(device),
                    T=T.to(device),
                    n_in_frames=(self.in_frames // np.cumprod([1] + self.tcn_window)[i]),
                    gcn_window=self.gcn_window[i],
                    tcn_window=self.tcn_window[i],
                    dropout=dropout,
                    num_stages=gconv_stages[i],
                    residual=use_residual_connections,
                    use_non_parametric=use_non_parametric,
                    use_edge_conv=use_edge_conv,
                    aggregate=aggregate[i]),
                )

        self.post_node = nn. Sequential(
            SENet(dim=nhid_v[-1]),
            nn.Linear(nhid_v[-1], n_outv)
        )

        if use_edge_conv:
            self.senet_edge = SENet(dim=nhid_e[-1])
            self.post_edge = nn.Sequential(
                nn.Linear(n_outv * 17 + 16 * nhid_e[-1], nhid_e[-1]),
                nn.ReLU(),
                nn.Linear(nhid_e[-1],  n_oute * 16)
            )
        else:
            self.senet_edge = SENet(dim=nhid_v[-1])
            self.post_edge = nn.Sequential(
                nn.Linear(n_outv * 17 + 17 * nhid_v[-1], nhid_v[-1]),
                nn.ReLU(),
                nn.Linear(nhid_v[-1], n_oute * 16)
            )

    def forward(self, X, Z=None):
        if self.use_edge_conv:
            assert Z is not None, "If edge conv is used then edge features must be passed as input"

            X, Z = X.reshape(-1, self.seq_len, self.n_nodes, self.infeat_v), Z.reshape(-1, self.seq_len, self.n_edges, self.infeat_e)
            X, Z = self.pre(X, Z, self.adj_e, self.adj_v, self.T)

            for s in range(len(self.layers) - 1):
                X, Z = self.layers[s](X, Z)

            X, Z = self.layers[-1](X, Z)

            X = self.post_node(X)

            Z = self.senet_edge(Z)
            batch_size = Z.shape[0]
            feat = torch.cat((X.view(batch_size, -1), Z.reshape(batch_size, -1)), axis=1)
            Z = self.post_edge(feat).view(batch_size, 16, -1)

            return X.view(batch_size, 17, -1), Z
        else:
            X = X.reshape(-1, self.seq_len, self.n_nodes, self.infeat_v)
            X = self.pre(X, self.adj_v)

            for s in range(len(self.layers) - 1):
                X, _ = self.layers[s](X)

            features, _ = self.layers[-1](X)
            X = self.post_node(features)

            Z = self.senet_edge(features)
            batch_size = Z.shape[0]
            feat = torch.cat((X.view(batch_size, -1), Z.reshape(batch_size, -1)), axis=1)
            Z = self.post_edge(feat).view(batch_size, 16, -1)

            return X.view(batch_size, 17, -1), Z


if __name__ == "__main__":
    gcn = TGraphNet(infeat_v=2,
                    infeat_e=4,
                    nhid_v=[256, 256, 256, 256],
                    nhid_e=[256, 256, 256, 256],
                    n_oute=6,
                    n_outv=3,
                    gcn_window=[9, 9, 9, 9],
                    tcn_window=[3, 3, 3, 9],
                    in_frames=243,
                    gconv_stages=[1, 2, 2, 3],
                    dropout=0.25,
                    use_residual_connections=True,
                    use_non_parametric=False,
                    use_edge_conv=True,)

    for m in gcn.layers:
        print(m.string())

    print(gcn)
    print(count_parameters(gcn), get_model_size(gcn))
