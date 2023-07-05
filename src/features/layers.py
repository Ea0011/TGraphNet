import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class GCN(nn.Module):
    def __init__(self, in_features, out_features, dropout=0):
        super(GCN, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

        # Store the adjacency matrix as an attribute of the model

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, input_features, adj):
        support = input_features @ self.weight
        output = adj @ support

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class TCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 7), stride=(1, 1), use_non_parametric=False):
        super(TCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if use_non_parametric:
            self.conv = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
            self.sigma = nn.Identity()
        else:
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=True, kernel_size=kernel_size, stride=stride)
            self.sigma = nn.ReLU()

            self.reset_parameters()

    def reset_parameters(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.sigma(self.conv(x))
        return out


class NodeResidualBlock(nn.Module):
    """
    Residual block with two node-edge modules
    """
    def __init__(self, nfeat_v, nhid_v, dropout):
        super(NodeResidualBlock, self).__init__()
        self.gc1 = GCN(nfeat_v, nhid_v,)
        self.gc2 = GCN(nfeat_v, nhid_v,)

    def forward(self, X, adj_v):
        residual_X = X

        X = self.gc1(X, adj_v)
        X = self.gc2(X, adj_v)

        X = X + residual_X

        return X


class NodeEdgeTCN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 7), stride=(1, 1), use_non_parametric=True):
        super(NodeEdgeTCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

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
        self.conv_node.weight.data.normal_(0, 0.01)
        self.conv_edge.weight.data.normal_(0, 0.01)

    def forward(self, x, z):
        out_node = self.sigma(self.conv_node(x))
        out_edge = self.sigma(self.conv_edge(z))
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
            self.weight = nn.Parameter(torch.FloatTensor(in_features_v, self.num_node_groups * out_features_v))
            self.weight_sec = nn.Parameter(torch.FloatTensor(in_features_e, in_features_v))
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features_v))
            else:
                self.register_parameter('bias', None)
        else:
            self.node_layer = False
            self.weight = nn.Parameter(torch.FloatTensor(in_features_e, self.num_edge_groups * out_features_e))
            self.weight_sec = nn.Parameter(torch.FloatTensor(in_features_v, in_features_e))
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features_e))
            else:
                self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        stdv = 1. / math.sqrt(self.weight_sec.size(1))
        self.weight_sec.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, H_v, H_e, adj_e, adj_v, T):
        batch_size = H_v.shape[0]
        num_joints = adj_v.shape[-1]
        num_edges = adj_e.shape[-1]

        if self.node_layer:
            '''
            \sigma( ̃ A_v ( H_v + W_s (T H_e) ) W_v)
            '''
            msg_to_nodes = T @ H_e # B x V x F
            msg_to_nodes_t = msg_to_nodes.view(batch_size, -1, msg_to_nodes.shape[-1]) @ self.weight_sec
            node_feat = H_v + msg_to_nodes_t

            # node transformation
            temp = (node_feat @ self.weight)
            temp = temp.reshape(batch_size, num_joints, self.num_node_groups, self.out_features_v)
            temp = temp.transpose(-3, -2)
            output = torch.sum(adj_v @ temp, dim=-3)

            if self.bias is not None:
                ret = output + self.bias
            return ret, H_e
        else:
            '''
            \sigma(T.t() Φ(H_v P_v) T * ̃ A_e H_e W_e)
            '''

            msg_to_edges = T.t() @ H_v  # B x E x F
            msg_to_edges_t = msg_to_edges @ self.weight_sec # b x V x F'
            edge_feat = H_e + msg_to_edges_t

            temp = (edge_feat @ self.weight)
            temp = temp.reshape(batch_size, num_edges, self.num_edge_groups, self.out_features_e)
            temp = temp.transpose(-3, -2)
            output = torch.sum(adj_e @ temp, dim=-3)

            if self.bias is not None:
                ret = output + self.bias
            return H_v, ret


class GCNodeEdgeModule(nn.Module):
    def __init__(self, nfeat_v, nfeat_e, nhid_v, nhid_e, dropout, use_bn=True, use_activ=True):
        super(GCNodeEdgeModule, self).__init__()

        # node
        self.gc_node = NodeEdgeConv(nfeat_v, nhid_v, nfeat_e, nfeat_e, node_layer=True)

        # edge
        self.gc_edge = NodeEdgeConv(nhid_v, nhid_v, nfeat_e, nhid_e, node_layer=False)

        self.use_bn = use_bn
        self.use_activ = use_activ

        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(nhid_v)
            self.bn2 = nn.BatchNorm1d(nhid_e)

        self.dropout = dropout

    def forward(self, X, Z, adj_e, adj_v, T):
        X, Z = self.gc_node(X, Z, adj_e, adj_v, T)

        if self.use_bn:
            X = self.bn1(X.transpose(1, 2)).transpose(1, 2)
        if self.use_activ:
            X, Z = F.relu(X), F.relu(Z)

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        X, Z = self.gc_edge(X, Z, adj_e, adj_v, T)

        if self.use_bn:
            Z = self.bn2(Z.transpose(1, 2)).transpose(1, 2)
        if self.use_activ:
            X, Z = F.relu(X), F.relu(Z)

        X = F.dropout(X, self.dropout, training=self.training)
        Z = F.dropout(Z, self.dropout, training=self.training)

        return X, Z


class GCResidualBlock(nn.Module):
    """
    Residual block with two node-edge modules
    """
    def __init__(self, nfeat_v, nfeat_e, nhid_v, nhid_e, dropout):
        super(GCResidualBlock, self).__init__()

        self.gc1 = GCNodeEdgeModule(nfeat_v, nfeat_e, nhid_v, nhid_e, dropout)
        self.gc2 = GCNodeEdgeModule(nhid_v, nhid_e, nhid_v, nhid_e, dropout)
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
                 nhid_v,
                 nhid_e,
                 adj_e,
                 adj_v,
                 T,
                 n_in_frames,
                 gcn_window=9,
                 tcn_window=3,
                 dropout=.0,
                 num_stages=1,
                 residual=False,
                 use_non_parametric=False,
                 use_edge_conv=True):
        super(STGConv, self).__init__()

        self.adj_v = nn.Parameter(adj_v)
        self.adj_e = nn.Parameter(adj_e)
        self.T = T

        self.frame_length = gcn_window
        self.n_in_frames = n_in_frames
        self.tcn_window = tcn_window
        self.use_edge_conv = use_edge_conv

        self.graph_stages = nn.ModuleList()
        for s in range(num_stages):
            if residual:
                self.graph_stages.append(
                    GCResidualBlock(nhid_v, nhid_e, nhid_v, nhid_e, dropout) if use_edge_conv else NodeResidualBlock(nhid_v, nhid_v, dropout)
                )
            else:
                self.graph_stages.append(
                    GCNodeEdgeModule(nhid_v, nhid_e, nhid_v, nhid_e, dropout) if use_edge_conv else GCN(nhid_v, nhid_v, dropout)
                )

        # Add a non parametric option here
        if use_edge_conv:
            self.tcn = NodeEdgeTCN(nhid_v,
                                   nhid_v,
                                   kernel_size=(1, self.tcn_window),
                                   stride=(1, self.tcn_window),
                                   use_non_parametric=use_non_parametric,)
        else:
            self.tcn = TCN(nhid_v,
                           nhid_v,
                           kernel_size=(1, self.tcn_window),
                           stride=(1, self.tcn_window),
                           use_non_parametric=use_non_parametric,)

    def forward(self, X, Z=None):
        if self.use_edge_conv:
            for s in range(len(self.graph_stages)):
                X, Z = self.graph_stages[s](X, Z, self.adj_e, self.adj_v, self.T)

            hdim_v = X.shape[-1]
            hdim_e = Z.shape[-1]

            X, Z = X.reshape(-1, hdim_v, 17, self.n_in_frames), Z.reshape(-1, hdim_e, 16, self.n_in_frames)
            X, Z = self.tcn(X, Z)

            n_frames = X.shape[-1]
            return X.view(-1, n_frames, 17, hdim_v), Z.view(-1, n_frames, 16, hdim_e)
        else:
            for s in range(len(self.graph_stages)):
                X = self.graph_stages[s](X, self.adj_v,)

            hdim_v = X.shape[-1]

            X = X.reshape(-1, hdim_v, 17, self.n_in_frames)
            X = self.tcn(X)

            n_frames = X.shape[-1]
            return X.view(-1, n_frames, 17, hdim_v), Z

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
            sq = torch.mean(x, dim=1, keepdim=True)
        else:
            sq = x
        sq = self.relu(self.squeeze(sq))
        ex = self.sigmoid(self.excite(sq))
        return x * ex
