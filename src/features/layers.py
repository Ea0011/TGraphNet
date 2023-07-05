import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, dropout=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SeqAffinityModulation(nn.Module):
    def __init__(self, in_features, num_joints, num_groups=1, mlp_ratio=1):
        super().__init__()
        self.num_groups = num_groups
        self.num_joints = num_joints
        self.mlp = Mlp(in_features=in_features, hidden_features=in_features*mlp_ratio, out_features=num_joints*num_joints*num_groups)


    def forward(self, input_features, adj):
        x = torch.mean(input_features, dim=(1, 2)) if len(input_features.shape) > 3 else input_features # average over sequence dimension
        modulation = self.mlp(x)
        modulation = rearrange(modulation, 'b (j1 j2 g) -> b g j1 j2', g=self.num_groups, j1=self.num_joints, j2=self.num_joints)
        affinity = modulation + adj

        return affinity


class FrameAffinityModulation(nn.Module):
    def __init__(self, in_features, num_joints, num_groups=1, mlp_ratio=1):
        super().__init__()
        self.num_groups = num_groups
        self.num_joints = num_joints
        self.mlp = Mlp(in_features=in_features, hidden_features=in_features*mlp_ratio, out_features=num_joints*num_joints*num_groups)

    def forward(self, input_features, adj):
        x = torch.mean(input_features, dim=2) if len(input_features.shape) > 3 else input_features # average over sequence dimension
        modulation = self.mlp(x)
        modulation = rearrange(modulation, 'b t (j1 j2 g) -> b t g j1 j2', g=self.num_groups, j1=self.num_joints, j2=self.num_joints)
        affinity = modulation + adj

        return affinity


class AffinityGCN(nn.Module):
    def __init__(self, in_frames, in_features, out_features, num_groups=3, dropout=0):
        super(AffinityGCN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.in_frames = in_frames
        self.dropout = dropout
        self.num_groups = num_groups

        self.weight = nn.Parameter(torch.FloatTensor(in_features, self.num_groups * out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.norm = nn.BatchNorm2d(out_features)

        self.affinity_modulation = FrameAffinityModulation(in_features, 17, num_groups, mlp_ratio=6)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.zeros_(self.bias)

    def forward(self, input_features, adj):
        B, T, V, _ = input_features.shape
        support = input_features @ self.weight
        support = support.reshape(B, T, V, self.num_groups, self.out_features)
        support = support.transpose(-3, -2)

        affinitiy = self.affinity_modulation(input_features, adj)
        message_passing = (affinitiy @ support)
        output = torch.sum(message_passing, dim=-3)

        if self.bias is not None:
            output = output + self.bias

        output = self.norm(output.transpose(1, -1)).transpose(1, -1)
        output = F.relu(output)
        output = F.dropout(output, self.dropout, training=self.training)

        return output


class GCN(nn.Module):
    def __init__(self, in_frames, in_features, out_features, num_groups=3, dropout=0):
        super(GCN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.in_frames = in_frames
        self.dropout = dropout
        self.num_groups = num_groups

        self.weight = nn.Parameter(torch.FloatTensor(in_features, self.num_groups * out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.norm = nn.BatchNorm2d(out_features)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        nn.init.zeros_(self.bias)

    def forward(self, input_features, adj):
        B, T, V, _ = input_features.shape
        support = input_features @ self.weight
        support = support.reshape(B, T, V, self.num_groups, self.out_features)
        support = support.transpose(-3, -2)
        output = torch.sum(adj @ support, dim=-3)

        if self.bias is not None:
            output = output + self.bias

        output = self.norm(output.transpose(1, -1)).transpose(1, -1)
        output = F.relu(output)
        output = F.dropout(output, self.dropout, training=self.training)

        return output


class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, in_frames, kernel_size=(1, 7), stride=(1, 1), use_non_parametric=False, dropout=.0):
        super(TCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        # self.norm = nn.LayerNorm([out_channels, 17, in_frames // kernel_size[-1]])
        self.norm = nn.BatchNorm2d(out_channels)

        if use_non_parametric:
            self.conv = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
            self.sigma = nn.Identity()
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=True, kernel_size=kernel_size, stride=stride)
            self.sigma = nn.ReLU()

            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        out = F.dropout(self.sigma(self.conv(self.norm(x))), self.dropout, self.training)
        return out


class NodeResidualBlock(nn.Module):
    """
    Residual block with two node-edge modules
    """
    def __init__(self, in_frames, nfeat_v, nhid_v, num_groups, dropout):
        super(NodeResidualBlock, self).__init__()
        self.gc1 = GCN(in_frames, nfeat_v, nhid_v, num_groups, dropout)
        self.gc2 = GCN(in_frames, nhid_v, nhid_v, num_groups, dropout)

        self.residual_flag = "same"

        if (nfeat_v != nhid_v):
            self.residual_flag = "diff"
            self.residual_gc = GCN(in_frames, nfeat_v, nhid_v, num_groups, dropout)

    def forward(self, X, adj_v):
        if self.residual_flag == "same":
            residual_X = X
        else:
            residual_X = self.residual_gc(X, adj_v)

        X = self.gc1(X, adj_v)
        X = self.gc2(X, adj_v)

        X = X + residual_X

        return X


class NodeTCNResidualBlock(nn.Module):
    """
    Residual block with two node-edge modules
    """
    def __init__(self, in_frames, nfeat_v, nhid_v, num_groups, dropout):
        super(NodeTCNResidualBlock, self).__init__()
        self.in_frames = in_frames
        self.nhid_v = nhid_v
        self.gcn = GCN(in_frames, nfeat_v, nhid_v, num_groups, dropout)
        self.tcn = nn.Sequential(
            nn.Conv2d(nhid_v, nhid_v, (1, 5), stride=1, padding="same"),  # hardcode at the moment
            nn.BatchNorm2d(nhid_v),
            nn.Dropout(dropout),
        )

        self.residual_flag = "same"

        if (nfeat_v != nhid_v):
            self.residual_flag = "diff"
            self.residual_gc = GCN(in_frames, nfeat_v, nhid_v, num_groups, dropout)

    def forward(self, X, adj_v):
        if self.residual_flag == "same":
            residual_X = X
        else:
            residual_X = self.residual_gc(X, adj_v)

        B, T, J, _ = X.shape
        X = self.gcn(X, adj_v).reshape(-1, self.in_frames, 17, self.nhid_v).transpose(1, -1)
        X = self.tcn(X).transpose(1, -1).reshape(B, T, J, self.nhid_v)

        X = X + residual_X

        return X


class NodeEdgeTCN(nn.Module):
    def __init__(self, in_channels, out_channels, in_frames, kernel_size=(1, 7), stride=(1, 1), use_non_parametric=True, dropout=.0):
        super(NodeEdgeTCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.norm2 = nn.BatchNorm2d(out_channels)

        if use_non_parametric:
            self.conv_node = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
            self.conv_edge = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
            self.sigma = nn.Identity()
        else:
            self.conv_node = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, bias=True, stride=stride)
            self.conv_edge = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, bias=True, stride=stride)
            self.sigma = nn.ReLU()
            self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.conv_node.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.conv_edge.weight, nonlinearity='relu')

        if self.conv_node.bias is not None:
            nn.init.constant_(self.conv_node.bias, 0)
            nn.init.constant_(self.conv_edge.bias, 0)

    def forward(self, x, z):
        out_node = F.dropout(self.sigma(self.norm1(self.conv_node(x))), self.dropout, self.training)
        out_edge = F.dropout(self.sigma(self.norm2(self.conv_edge(z))), self.dropout, self.training)
        return out_node, out_edge


class NodeEdgeConv(nn.Module):
    def __init__(self,
                 in_features_v,
                 out_features_v,
                 in_features_e,
                 out_features_e,
                 bias=True,
                 node_layer=True,
                 num_node_groups=1,
                 num_edge_groups=1,):
        super(NodeEdgeConv, self).__init__()
        self.in_features_e = in_features_e
        self.out_features_e = out_features_e
        self.in_features_v = in_features_v
        self.out_features_v = out_features_v
        self.num_node_groups = num_node_groups
        self.num_edge_groups = num_edge_groups

        if node_layer:
            self.node_layer = True
            self.weight = nn.Parameter(torch.FloatTensor(in_features_v, out_features_v))
            self.weight_sec = nn.Parameter(torch.FloatTensor(in_features_e, in_features_v))
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features_v))
            else:
                self.register_parameter('bias', None)
        else:
            self.node_layer = False
            self.weight = nn.Parameter(torch.FloatTensor(in_features_e, out_features_e))
            self.weight_sec = nn.Parameter(torch.FloatTensor(in_features_v, in_features_e))
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features_e))
            else:
                self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')
        nn.init.kaiming_uniform_(self.weight_sec, nonlinearity='relu')

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        if self.node_layer:
            '''
            \sigma( ̃ A_v ( H_v + W_s (T H_e) ) W_v)
            '''
            # E, V = adj_e.shape[0], adj_v.shape[0]
            # msg_to_nodes = T @ H_e  # B x V x F
            # msg_to_nodes_t = msg_to_nodes.view(-1, H_e.shape[1], V, H_e.shape[-1]) @ self.weight_sec
            node_feat = H_v # + msg_to_nodes_t

            # node transformation
            temp = (node_feat @ self.weight)

            output = adj_v @ temp

            if self.bias is not None:
                ret = output + self.bias
            return ret, H_e
        else:
            '''
            \sigma(T.t() Φ(H_v P_v) T * ̃ A_e H_e W_e)
            '''
            # E, V = adj_e.shape[0], adj_v.shape[0]
            # msg_to_edges = T.t() @ H_v  # B x E x F
            # msg_to_edges_t = msg_to_edges.view(-1, H_e.shape[1], E, H_v.shape[-1]) @ self.weight_sec  # b x V x F'
            edge_feat = H_e # + msg_to_edges_t

            temp = (edge_feat @ self.weight)
            output = adj_e @ temp

            if self.bias is not None:
                ret = output + self.bias
            return H_v, ret


class GCNodeEdgeModule(nn.Module):
    def __init__(self,
                 in_frames,
                 nfeat_v,
                 nfeat_e,
                 nhid_v,
                 nhid_e,
                 dropout,
                 use_norm=True,
                 use_activ=True,):
        super(GCNodeEdgeModule, self).__init__()

        # node
        self.gc_node = NodeEdgeConv(nfeat_v, nhid_v, nfeat_e, nfeat_e, node_layer=True)

        # edge
        self.gc_edge = NodeEdgeConv(nhid_v, nhid_v, nfeat_e, nhid_e, node_layer=False)

        self.use_norm = use_norm
        self.use_activ = use_activ
        self.in_frames = in_frames
        self.nhid_v = nhid_v
        self.nhid_e = nhid_e

        if self.use_norm:
            self.norm1 = nn.BatchNorm2d(nhid_v)
            self.norm2 = nn.BatchNorm2d(nhid_e)

        self.dropout = dropout

    def forward(self, X, Z, adj_e, adj_v, T):
        X, Z = self.gc_node(X, Z, adj_e, adj_v, T)

        if self.use_norm:
            X = self.norm1(X.transpose(1, -1)).transpose(1, -1)
        if self.use_activ:
            X, Z = F.relu(X), F.relu(Z)

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        X, Z = self.gc_edge(X, Z, adj_e, adj_v, T)

        if self.use_norm:
            Z = self.norm2(Z.transpose(1, -1)).transpose(1, -1)
        if self.use_activ:
            X, Z = F.relu(X), F.relu(Z)

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        return X, Z


class GCResidualBlock(nn.Module):
    """
    Residual block with two node-edge modules
    """
    def __init__(self, in_frames, nfeat_v, nfeat_e, nhid_v, nhid_e, dropout):
        super(GCResidualBlock, self).__init__()

        self.gc1 = GCNodeEdgeModule(in_frames, nfeat_v, nfeat_e, nhid_v, nhid_e, dropout)
        self.gc2 = GCNodeEdgeModule(in_frames, nhid_v, nhid_e, nhid_v, nhid_e, dropout)
        self.residual_flag = "same"

        if (nfeat_v != nhid_v or nfeat_e != nhid_e):
            self.residual_flag = "diff"
            self.residual_gc = GCNodeEdgeModule(nfeat_v, nfeat_e, nhid_v, nhid_e, dropout)

    def forward(self, X, Z, adj_e, adj_v, T):
        '''
        X: node features
        Z: edge features
        adj_e: edge adjacency matrix
        adj_v: node adjacency matrix
        T: node to edge mapping
        '''

        if self.residual_flag == "same":
            residual_X = X
            residual_Z = Z
        else:
            residual_X, residual_Z = self.residual_gc(X, Z, adj_e, adj_v, T)

        X, Z = self.gc1(X, Z, adj_e, adj_v, T)

        X, Z = self.gc2(X, Z, adj_e, adj_v, T)

        X = X + residual_X
        Z = Z + residual_Z

        return X, Z


class STGConv(nn.Module):
    def __init__(self,
                 nin_v,
                 nin_e,
                 nhid_v,
                 nhid_e,
                 adj_e,
                 adj_v,
                 T,
                 n_in_frames,
                 gcn_window=9,
                 tcn_window=3,
                 num_groups=1,
                 dropout=.0,
                 num_stages=1,
                 residual=False,
                 use_non_parametric=False,
                 use_edge_conv=True,
                 aggregate=True,
                 learn_adj=True):
        super(STGConv, self).__init__()

        # self.adj_v = nn.Parameter(adj_v.clone().detach(), requires_grad=learn_adj)
        # self.adj_e = nn.Parameter(adj_e.clone().detach(), requires_grad=learn_adj)
        self.adj_v = adj_v
        self.adj_e = adj_e
        self.T = T

        self.frame_length = gcn_window
        self.n_in_frames = n_in_frames
        self.tcn_window = tcn_window
        self.use_edge_conv = use_edge_conv
        self.aggregate = aggregate
        self.nhid_v = nhid_v
        self.nhid_e = nhid_e
        self.nin_v = nin_v
        self.nin_e = nin_e
        self.num_groups = num_groups

        self.seq_len = self.n_in_frames // self.frame_length
        self.n_nodes_in_seq = self.frame_length * 17
        self.n_edges_in_seq = self.frame_length * 16

        self.graph_stages = nn.ModuleList()
        if residual:
            self.graph_stages.append(
                GCResidualBlock(n_in_frames, nin_v, nin_e, nhid_v, nhid_e, dropout) if use_edge_conv else NodeTCNResidualBlock(n_in_frames, nin_v, nhid_v, self.num_groups, dropout)
            )
        else:
            self.graph_stages.append(
                GCNodeEdgeModule(n_in_frames, nin_v, nin_e, nhid_v, nhid_e, dropout) if use_edge_conv else GCN(n_in_frames, nin_v, nhid_v, self.num_groups, dropout)
            )
        for s in range(num_stages - 1):
            if residual:
                self.graph_stages.append(
                    GCResidualBlock(n_in_frames, nhid_v, nhid_e, nhid_v, nhid_e, dropout) if use_edge_conv else NodeTCNResidualBlock(n_in_frames, nhid_v, nhid_v, self.num_groups, dropout)
                )
            else:
                self.graph_stages.append(
                    GCNodeEdgeModule(n_in_frames, nhid_v, nhid_e, nhid_v, nhid_e, dropout) if use_edge_conv else GCN(n_in_frames, nhid_v, nhid_v, self.num_groups, dropout)
                )

        # Add a non parametric option here
        if use_edge_conv and aggregate:
            self.tcn = NodeEdgeTCN(nhid_v,
                                   nhid_v,
                                   in_frames=n_in_frames,
                                   kernel_size=(1, self.tcn_window),
                                   stride=(1, self.tcn_window),
                                   use_non_parametric=use_non_parametric,
                                   dropout=dropout)
        elif aggregate:
            self.tcn = TCN(nhid_v,
                           nhid_v,
                           in_frames=n_in_frames,
                           kernel_size=(1, self.tcn_window),
                           stride=(1, self.tcn_window),
                           use_non_parametric=use_non_parametric,
                           dropout=dropout)

    def forward(self, X, Z=None):
        if self.use_edge_conv:
            for s in range(len(self.graph_stages)):
                X, Z = X.reshape(-1, self.seq_len, self.n_nodes_in_seq, self.nhid_v), Z.reshape(-1, self.seq_len, self.n_edges_in_seq, self.nhid_v)
                X, Z = self.graph_stages[s](X, Z, self.adj_e, self.adj_v, self.T)

            if self.aggregate:
                X, Z = X.reshape(-1, self.n_in_frames, 17, self.nhid_v).transpose(1, -1), Z.reshape(-1, self.n_in_frames, 16, self.nhid_e).transpose(1, -1)
                X, Z = self.tcn(X, Z)

                X = X.transpose(1, -1)
                Z = Z.transpose(1, -1)
            return X, Z
        else:
            X = X.reshape(-1, self.seq_len, self.n_nodes_in_seq, self.nin_v)
            X = self.graph_stages[0](X, self.adj_v)
            X = X.reshape(-1, self.n_in_frames, 17, self.nhid_v)

            for s in range(len(self.graph_stages) - 1):
                X = X.reshape(-1, self.seq_len, self.n_nodes_in_seq, self.nhid_v)
                X = self.graph_stages[s + 1](X, self.adj_v)
                X = X.reshape(-1, self.n_in_frames, 17, self.nhid_v)

            if self.aggregate:
                X_down = self.tcn(X.transpose(1, -1)).transpose(1, -1)
                return X, X_down, Z
            return X, X, Z

    def string(self):
        return f"In Frames: {self.n_in_frames} -> Out Frames: {self.n_in_frames // self.tcn_window}, Processed Window: {self.tcn_window}"


class SENet(nn.Module):
    """
    Squeeze and Excitation block (SE block)
    """

    def __init__(self, dim, ratio=8):
        super(SENet, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.squeeze = nn.Linear(dim, dim//ratio)
        self.excite = nn.Linear(dim//ratio, dim)

    def forward(self, x):
        if len(x.shape) > 2:
            sq = torch.mean(x, dim=(2), keepdim=True)
        else:
            sq = x
        sq = self.relu(self.squeeze(sq))
        ex = self.sigmoid(self.excite(sq))
        return x * ex
