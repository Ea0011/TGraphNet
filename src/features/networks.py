import torch.nn as nn
import torch
from graph import Graph
import numpy as np
from common.model import *
from features.layers import STGConv, GCNodeEdgeModule, SENet, GCN
from common.model import print_layers

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
                 num_groups,
                 dropout,
                 use_residual_connections,
                 use_non_parametric,
                 use_edge_conv,
                 learn_adj,):
        super(TGraphNet, self).__init__()

        g = Graph(17, 16, gcn_window[0], norm=True)()
        adj_v, adj_e, T = torch.stack((g['adj_v_root'], g['adj_v_close'], g['adj_v_further'], g['adj_v_sym'])), g['adj_e_wtemp'], g['ne_mapping']
        # adj_v, adj_e, T = torch.stack((g['adj_v'], g['adj_v_back'], g['adj_v_back'])), g['adj_e_wtemp'], g['ne_mapping']

        self.adj_v = nn.Parameter(adj_v.to(device), requires_grad=False)
        self.adj_e = nn.Parameter(adj_e.to(device), requires_grad=False)
        self.T = T.to(device)
        self.gcn_window = gcn_window
        self.tcn_window = tcn_window
        self.in_frames = in_frames
        self.infeat_v = infeat_v
        self.infeat_e = infeat_e
        self.use_edge_conv = use_edge_conv
        self.learn_adj = learn_adj

        self.seq_len = in_frames // gcn_window[0]
        self.n_nodes = 17 * gcn_window[0]
        self.n_edges = 16 * gcn_window[0]
        self.num_groups = num_groups

        self.pre = GCNodeEdgeModule(in_frames, infeat_v, infeat_e, nhid_v[0][0], nhid_e[0][0], dropout=0) if use_edge_conv else GCN(in_frames, infeat_v, nhid_v[0][0], num_groups=num_groups, dropout=dropout)
        self.layers = nn.ModuleList()

        n_stages = len(nhid_v)
        for i in range(n_stages):
            g = Graph(17, 16, gcn_window[i], norm=True)()
            adj_v, adj_e, T = torch.stack((g['adj_v_root'], g['adj_v_close'], g['adj_v_further'], g['adj_v_sym'])), g['adj_e_wtemp'], g['ne_mapping']
            # adj_v, adj_e, T = torch.stack((g['adj_v'], g['adj_v_back'], g['adj_v_back'])), g['adj_e_wtemp'], g['ne_mapping']

            if aggregate[i]:
                n_in_frames = (self.in_frames // np.cumprod([1] + self.tcn_window)[i])
            else:
                n_in_frames = self.in_frames

            self.layers.append(
                STGConv(
                    nin_v=nhid_v[i][0],
                    nin_e=nhid_e[i][0],
                    nhid_v=nhid_v[i][1],
                    nhid_e=nhid_e[i][1],
                    adj_e=adj_e.to(device),
                    adj_v=adj_v.to(device),
                    T=T.to(device),
                    n_in_frames=n_in_frames,
                    gcn_window=self.gcn_window[i],
                    tcn_window=self.tcn_window[i],
                    num_groups=self.num_groups,
                    dropout=dropout,
                    num_stages=gconv_stages[i],
                    residual=use_residual_connections,
                    use_non_parametric=use_non_parametric,
                    use_edge_conv=use_edge_conv,
                    aggregate=aggregate[i],
                    learn_adj=self.learn_adj),
                )

        self.post_node = nn.Sequential(
            SENet(dim=nhid_v[-1][-1]),
            nn.Linear(nhid_v[-1][-1], n_outv)
        )

        # self.post_node = nn.Linear(nhid_v[-1][-1] * 17, 51)
        # self.post_node = nn.Conv2d(nhid_v[-1][-1], 3, kernel_size=1)

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
                _, X, _ = self.layers[s](X)

            _, features, _ = self.layers[-1](X)
            B, F, J, D = features.shape

            # features = features.reshape(B, J * D)
            X = self.post_node(features)

            return X


class TGraphNetSeq(nn.Module):
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
                 num_groups,
                 dropout,
                 use_residual_connections,
                 use_non_parametric,
                 use_edge_conv,
                 learn_adj,):
        super().__init__()
        g = Graph(17, 16, gcn_window[0], norm=True)()
        adj_v, adj_e, T = torch.stack((g['adj_v_root'], g['adj_v_close'], g['adj_v_further'], g['adj_v_sym'])), g['adj_e_wtemp'], g['ne_mapping']
        # adj_v, adj_e, T = torch.stack((g['adj_v'], g['adj_v_back'], g['adj_v_back'])), g['adj_e_wtemp'], g['ne_mapping']

        self.adj_v = nn.Parameter(adj_v.to(device), requires_grad=False)
        self.adj_e = nn.Parameter(adj_e.to(device), requires_grad=False)
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
        self.num_groups = num_groups
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.merging_layers = nn.ModuleList()

        self.post_node = nn.Conv2d(nhid_v[0][1], 3, kernel_size=(1, 5), stride=1, padding="same")

        n_stages = len(nhid_v)
        for i in range(n_stages):
            g = Graph(17, 16, gcn_window[i], norm=True)()
            adj_v, adj_e, T = torch.stack((g['adj_v_root'], g['adj_v_close'], g['adj_v_further'], g['adj_v_sym'])), g['adj_e_wtemp'], g['ne_mapping']
            # adj_v, adj_e, T = torch.stack((g['adj_v'], g['adj_v_back'], g['adj_v_back'])), g['adj_e_wtemp'], g['ne_mapping']
            self.downsample_layers.append(
                STGConv(
                    nin_v=nhid_v[i][0],
                    nin_e=nhid_e[i][0],
                    nhid_v=nhid_v[i][1],
                    nhid_e=nhid_e[i][1],
                    adj_e=adj_e.to(device),
                    adj_v=adj_v.to(device),
                    T=T.to(device),
                    n_in_frames=(self.in_frames // np.cumprod([1] + self.tcn_window)[i]),
                    gcn_window=self.gcn_window[i],
                    tcn_window=self.tcn_window[i],
                    num_groups=self.num_groups,
                    dropout=dropout,
                    num_stages=gconv_stages[i],
                    residual=use_residual_connections,
                    use_non_parametric=use_non_parametric,
                    use_edge_conv=use_edge_conv,
                    aggregate=aggregate[i],
                    learn_adj=learn_adj),
                )

        for i in range(n_stages - 1):
            g = Graph(17, 16, gcn_window[i], norm=True)()
            adj_v, adj_e, T = torch.stack((g['adj_v_root'], g['adj_v_close'], g['adj_v_further'], g['adj_v_sym'])), g['adj_e_wtemp'], g['ne_mapping']
            # adj_v, adj_e, T = torch.stack((g['adj_v'], g['adj_v_back'], g['adj_v_back'])), g['adj_e_wtemp'], g['ne_mapping']
            self.upsample_layers.append(
                nn.ModuleList((
                    STGConv(
                        nin_v=nhid_v[-(i+1)][1],
                        nin_e=nhid_e[-(i+1)][1],
                        nhid_v=nhid_v[-(i+1)][0],
                        nhid_e=nhid_e[-(i+1)][0],
                        adj_e=adj_e.to(device),
                        adj_v=adj_v.to(device),
                        T=T.to(device),
                        n_in_frames=(self.in_frames // np.cumprod([1] + self.tcn_window)[-(i + 2)]),
                        gcn_window=self.gcn_window[i],
                        tcn_window=self.tcn_window[i],
                        num_groups=self.num_groups,
                        dropout=dropout,
                        num_stages=1,
                        residual=use_residual_connections,
                        use_non_parametric=use_non_parametric,
                        use_edge_conv=use_edge_conv,
                        aggregate=False,
                        learn_adj=learn_adj
                    ),
                    nn.Upsample(scale_factor=(1, self.tcn_window[-(i + 1)]), mode='bilinear', align_corners=True),
                ))
            )

        for i in range(n_stages - 1):
            g = Graph(17, 16, gcn_window[i], norm=True)()
            adj_v, adj_e, T = torch.stack((g['adj_v_root'], g['adj_v_close'], g['adj_v_further'], g['adj_v_sym'])), g['adj_e_wtemp'], g['ne_mapping']
            # adj_v, adj_e, T = torch.stack((g['adj_v'], g['adj_v_back'], g['adj_v_back'])), g['adj_e_wtemp'], g['ne_mapping']
            self.merging_layers.append(
                nn.ModuleList((
                    STGConv(
                        nin_v=nhid_v[-(i+2)][1],
                        nin_e=nhid_e[-(i+2)][1],
                        nhid_v=nhid_v[0][1],
                        nhid_e=nhid_e[0][1],
                        adj_e=adj_e.to(device),
                        adj_v=adj_v.to(device),
                        T=T.to(device),
                        n_in_frames=(self.in_frames // np.cumprod([1] + self.tcn_window)[-(i + 2)]),
                        gcn_window=self.gcn_window[i],
                        tcn_window=self.tcn_window[i],
                        num_groups=self.num_groups,
                        dropout=dropout,
                        num_stages=1,
                        residual=use_residual_connections,
                        use_non_parametric=use_non_parametric,
                        use_edge_conv=use_edge_conv,
                        aggregate=False,
                        learn_adj=learn_adj
                    ),
                    nn.Upsample(scale_factor=(1, np.cumprod([1] + self.tcn_window)[-(i + 2)]), mode='bilinear', align_corners=True),
                ))
            )

        self.post_merge = STGConv(
            nin_v=nhid_v[0][1],
            nin_e=nhid_e[0][1],
            nhid_v=nhid_v[0][1],
            nhid_e=nhid_e[0][1],
            adj_e=adj_e.to(device),
            adj_v=adj_v.to(device),
            T=T.to(device),
            n_in_frames=(self.in_frames),
            gcn_window=self.gcn_window[0],
            tcn_window=self.tcn_window[0],
            num_groups=self.num_groups,
            dropout=dropout,
            num_stages=1,
            residual=use_residual_connections,
            use_non_parametric=use_non_parametric,
            use_edge_conv=use_edge_conv,
            aggregate=False,
            learn_adj=learn_adj
        )
        self.merge_norm = nn.BatchNorm2d(nhid_v[0][1])

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
            downsample_out = [None] * (len(self.downsample_layers) - 1)  # create buffer to hold features
            merging_out = [None] * (len(self.merging_layers))

            X = X.reshape(-1, self.seq_len, self.n_nodes, self.infeat_v)

            for s in range(len(self.downsample_layers) - 1):
                downsample_out[s], X, _ = self.downsample_layers[s](X)

            _, X, _ = self.downsample_layers[-1](X)

            for s in range(len(self.upsample_layers)):
                skip = downsample_out[-(s + 1)]
                X, _, _ = self.upsample_layers[s][0](X)

                # Save for merging stage
                merging_out[s], _, _ = self.merging_layers[s][0](X)
                merging_out[s] = self.merging_layers[s][1](merging_out[s].transpose(1, -1)).transpose(1, -1)

                # Upsample and add the skip connection
                X = self.upsample_layers[s][1](X.transpose(1, -1)).transpose(1, -1)
                X = X + skip

            for f in merging_out:
                X = X + f

            X = self.merge_norm(X.transpose(1, -1)).transpose(1, -1)
            X, _, _ = self.post_merge(X)
            X = self.post_node(X.transpose(1, -1)).transpose(1, -1)

            return X


if __name__ == "__main__":
    gcn = TGraphNetSeq(infeat_v=2,
                    infeat_e=4,
                    nhid_v=[[256, 256], [256, 256], [256, 256], [256, 256]],
                    nhid_e=[[256, 256], [256, 256], [256, 256], [256, 256]],
                    n_oute=6,
                    n_outv=3,
                    gcn_window=[3, 3, 3, 3],
                    tcn_window=[3, 3, 3, 3],
                    in_frames=81,
                    num_groups=3,
                    aggregate=[True] * 4,
                    gconv_stages=[1, 2, 2, 3],
                    dropout=0.1,
                    use_residual_connections=True,
                    use_non_parametric=False,
                    use_edge_conv=False,
                    learn_adj=False,)

    for m in gcn.downsample_layers:
        print(m.string())

    print(gcn)
    print_layers(gcn)
    print(count_parameters(gcn), get_model_size(gcn))
