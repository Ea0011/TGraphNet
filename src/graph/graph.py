from common.h36m_skeleton import *
from vizualization import plot_adjacency_matrix, plot_node_to_edge_map
import matplotlib.pyplot as plt
import numpy as np


class Graph:
    def __init__(self,
                 num_joints,
                 num_edges,
                 num_frames,
                 connections="neighbour",
                 norm=True,) -> None:

        if connections not in ["all", "middle", "neighbour"]:
            raise f"{connections} is not an available option, all and middle are the options for this param"

        self.num_joints = num_joints
        self.num_edges = num_edges
        self.num_frames = num_frames
        self.connections = connections
        self.norm = norm
        self.dist_center = self.get_distance_to_center()

    def get_distance_to_center(self):
        """
        :return: get the distance of each node to center
        """

        dist_center = np.zeros(self.num_joints)
        # center = hip
        dist_center[0:7] = [0, 1, 2, 3, 1, 2, 3] # legs
        dist_center[7:11] = [1, 2, 3, 4] # body
        dist_center[11:17] = [3, 4, 5, 3, 4, 5] #arms

        return dist_center

    def get_spatial_adjacency(self):
        """
        returns the (F X 17, F X 17) human3.6m style adjacency matrix where only spatial links are established
        """
        self_link = [(i, i) for i in range(self.num_joints)]

        neighbour_link = [(0,1), (1,2), (2,3), # Rleg
                            (0,4), (4,5), (5,6), # Lleg
                            (0,7), (7,8), (8,9), (9,10), # body
                            (8,11), (11,12), (12,13), # Larm
                            (8,14), (14,15), (15,16) # Rarm
                        ]

        edges = self_link + neighbour_link

        # initializing the matrix
        spatial_adjecency = torch.zeros(self.num_joints, self.num_joints)
        a_root = torch.zeros((self.num_joints, self.num_joints))
        a_close = torch.zeros((self.num_joints, self.num_joints))
        a_further = torch.zeros((self.num_joints, self.num_joints))
        for e in edges:
            source, sink = e
            spatial_adjecency[source, sink] = spatial_adjecency[sink, source] = 1
        for i in range(self.num_joints):
            for j in range(self.num_joints):
                if (i, j) not in edges and (j, i) not in edges:
                    continue
                if self.dist_center[j] == self.dist_center[i]:
                    a_root[j, i] = 1
                elif self.dist_center[j] > self.dist_center[i]:
                    a_close[j, i] = 1
                else:
                    a_further[j, i] = 1

                if self.dist_center[j] == self.dist_center[i]:
                    a_root[j, i] = 1
                elif self.dist_center[j] > self.dist_center[i]:
                    a_close[j, i] = 1
                else:
                    a_further[j, i] = 1

        A = torch.zeros(self.num_frames * self.num_joints, self.num_frames * self.num_joints)
        A_root = torch.zeros(self.num_frames * self.num_joints, self.num_frames * self.num_joints)
        A_close = torch.zeros(self.num_frames * self.num_joints, self.num_frames * self.num_joints)
        A_further = torch.zeros(self.num_frames * self.num_joints, self.num_frames * self.num_joints)

        # init spatial connections
        for i in range(self.num_frames):
            frame = (i * self.num_joints)
            A[frame:frame+self.num_joints, frame:frame+self.num_joints] = spatial_adjecency.clone()
            A_root[frame:frame+self.num_joints, frame:frame+self.num_joints] = a_root.clone()
            A_close[frame:frame+self.num_joints, frame:frame+self.num_joints] = a_close.clone()
            A_further[frame:frame+self.num_joints, frame:frame+self.num_joints] = a_further.clone()

        return A, A_root, A_close, A_further

    def get_temporal_adjacency(self, num_joints):
        """
        Generate a temporal adjacency matrix for a graph with a given number of frames and nodes.
        The adjacency matrix is used to represent connections between nodes in different time frames,
        and can be constructed in different ways depending on the value of the `connections` parameter.

        Parameters
        ----------
        self.num_frames : int, optional
            The number of time frames being considered.
        self.num_joints : int, optional
            The number of nodes in each time frame.
        connections : str, optional
            The strategy used to establish temporal connections between the nodes.
            Possible values are "all", "middle", and "neighbour",
            which correspond to different strategies for connecting the nodes across time.

        Returns
        -------
        A : torch.Tensor
                The temporal adjacency matrix for the graph.

        """
        if self.connections not in ["all", "middle", "neighbour"]:
            raise f"{self.connections} is not an available option, all and middle are the options for this param"

        A_back = torch.zeros(self.num_frames * num_joints, self.num_frames * num_joints)
        A_frw = torch.zeros(self.num_frames * num_joints, self.num_frames * num_joints)

        if self.connections == "all":
            for i in range(self.num_frames):
                for j in range(num_joints):
                    frame = (i * num_joints)
                    neighbour_frame = (j * num_joints)
                    A_frw[frame:frame+self.num_joints, neighbour_frame:neighbour_frame+num_joints].fill_diagonal_(1)
                    A_back[neighbour_frame:neighbour_frame+self.num_joints, frame:frame+num_joints].fill_diagonal_(1)

        if self.connections == "middle":
            middle_frame = (self.num_frames // 2) * num_joints
            for i in range(self.num_frames):
                neighbour_frame = (i * num_joints)
                A_frw[middle_frame:middle_frame+num_joints, neighbour_frame:neighbour_frame+num_joints].fill_diagonal_(1)
                A_back[neighbour_frame:neighbour_frame+num_joints, middle_frame:middle_frame+num_joints].fill_diagonal_(1)

        if self.connections == "neighbour":
            for i in range(self.num_frames):
                frame = (i * num_joints)
                next_frame = ((i + 1) * num_joints)

                if next_frame >= A_frw.shape[0]:
                    break

                A_frw[frame:frame+num_joints, next_frame:next_frame+num_joints].fill_diagonal_(1)  # time forward
                A_back[next_frame:next_frame+num_joints, frame:frame+num_joints].fill_diagonal_(1)  # time back

        A_frw.fill_diagonal_(0)
        A_back.fill_diagonal_(0)

        return A_back.fill_diagonal_(0), A_frw.fill_diagonal_(0) # remove self links

    def get_temporal_edge_adjacency(self):
        edge_connections_back, edge_connections_back_frw = self.get_temporal_adjacency(num_joints=self.num_edges)

        A_back = torch.zeros(self.num_frames * self.num_edges, self.num_frames * self.num_edges)
        A_frw = torch.zeros(self.num_frames * self.num_edges, self.num_frames * self.num_edges)

        A_back[0:(self.num_frames * self.num_edges), 0:(self.num_frames * self.num_edges)] = edge_connections_back
        A_frw[0:(self.num_frames * self.num_edges), 0:(self.num_frames * self.num_edges)] = edge_connections_back_frw

        return A_back, A_frw

    def get_edge_adjacency(self):
        """
        Generate an adjacency matrix for a graph with edges representing connections between different
        parts of the human body over multiple time frames.

        Parameters
        ----------
        self.num_frames : int, optional
                The number of time frames being considered.

        Returns
        -------
        A_ext : torch.Tensor
                The adjacency matrix for the graph.

        """
        adj_e = torch.zeros((self.num_edges, self.num_edges))
        adj_e_parent = torch.zeros((self.num_edges, self.num_edges))
        adj_e_child = torch.zeros((self.num_edges, self.num_edges))
        adj_e_crosshead = torch.zeros((self.num_edges, self.num_edges))
        adj_e_root = torch.zeros((self.num_edges, self.num_edges))

        edge_names_to_id = {edge: i for i, edge in enumerate(edge_names)}

        # parent adjacency
        for edge, parent in edge_parent.items():
            for p in parent:
                adj_e[edge_names_to_id[edge]][edge_names_to_id[p]] = 1
                adj_e_parent[edge_names_to_id[edge]][edge_names_to_id[p]] = 1

        # child adjacency
        for edge, children in edge_children.items():
            for child in children:
                adj_e[edge_names_to_id[edge]][edge_names_to_id[child]] = 1
                adj_e_child[edge_names_to_id[edge]][edge_names_to_id[child]] = 1

        # crossheads
        for edge, neighbors in edge_crosshead.items():
            for n in neighbors:
                adj_e[edge_names_to_id[edge]][edge_names_to_id[n]] = 1
                adj_e_crosshead[edge_names_to_id[edge]][edge_names_to_id[child]] = 1

        # self loops
        for i in range(self.num_edges):
            adj_e[i][i] = 1
            adj_e_root[i][i] = 1

        A = torch.zeros(self.num_frames * self.num_edges, self.num_frames * self.num_edges)
        A_root = torch.zeros(self.num_frames * self.num_edges, self.num_frames * self.num_edges)
        A_parent = torch.zeros(self.num_frames * self.num_edges, self.num_frames * self.num_edges)
        A_child = torch.zeros(self.num_frames * self.num_edges, self.num_frames * self.num_edges)
        A_crosshead = torch.zeros(self.num_frames * self.num_edges, self.num_frames * self.num_edges)

        # init spatial connections
        for i in range(self.num_frames):
            frame = (i * self.num_edges)
            A[frame:frame+self.num_edges, frame:frame+self.num_edges] = adj_e.clone()
            A_root[frame:frame+self.num_edges, frame:frame+self.num_edges] = adj_e_root.clone()
            A_parent[frame:frame+self.num_edges, frame:frame+self.num_edges] = adj_e_parent.clone()
            A_child[frame:frame+self.num_edges, frame:frame+self.num_edges] = adj_e_child.clone()
            A_crosshead[frame:frame+self.num_edges, frame:frame+self.num_edges] = adj_e_crosshead.clone()

        return A, A_root, A_parent, A_child, A_crosshead

    def get_node_to_edge_mapping(self):
        """
        Generate a mapping between nodes and edges in a graph with edges representing
        connections between different parts of the human body over multiple time frames.

        Parameters
        ----------
        self.num_frames : int, optional
                The number of time frames being considered.

        Returns
        -------
        T : torch.Tensor
                The node-to-edge mapping matrix for the graph.

        """
        joint_names_to_id =  {joint: i for i, joint in enumerate(joint_names)} 
        edge_names_to_id = {edge: i for i, edge in enumerate(edge_names)}

        T = torch.zeros((self.num_frames * self.num_joints, (self.num_frames * self.num_edges)))
        T_local = torch.zeros(self.num_joints, self.num_edges)

        for joint, edges in joint_to_edge_mapping.items():
            joint_id = joint_names_to_id[joint]
            for edge in edges:
                T_local[joint_id][edge_names_to_id[edge]] = 1

        # Spatial node to edge mapping
        for i in range(self.num_frames):
            joint_frame = (i * self.num_joints)
            edge_frame = (i * self.num_edges)
            T[joint_frame:joint_frame+self.num_joints, edge_frame:edge_frame+self.num_edges] = T_local.clone()

        return T[:, :(self.num_frames * self.num_edges)]

    def normalize_undigraph(self, A):
        Dl = torch.sum(A, 0)
        num_node = A.shape[0]
        Dn = torch.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-0.5)

        DAD = Dn @ (A @ Dn)
        return DAD

    def normalize_digraph(self, A):
        Dl = torch.sum(A, 0)
        num_node = A.shape[0]
        Dn = torch.zeros((num_node, num_node))
        for i in range(num_node):
            if Dl[i] > 0:
                Dn[i, i] = Dl[i]**(-1)
        AD = A @ Dn
        return AD

    def __call__(self):
        temporal_adj_v_back, temporal_adj_v_frw = self.get_temporal_adjacency(self.num_joints)
        temporal_adj_v = temporal_adj_v_frw + temporal_adj_v_back
        temporal_adj_e_back, temporal_adj_e_frw = self.get_temporal_edge_adjacency()
        temporal_adj_e = temporal_adj_e_back + temporal_adj_e_frw
        AV_full, AV_root, AV_close, AV_further = self.get_spatial_adjacency()
        AE_full, AE_root, AE_parent, AE_child, AE_crosshead = self.get_edge_adjacency()

        if self.norm:
            normed_adj_v = self.normalize_undigraph((AV_full + temporal_adj_v).clamp(0, 1))
            AV_close[AV_close == 1] = normed_adj_v[AV_close == 1]
            AV_further[AV_further == 1] = normed_adj_v[AV_further == 1]
            AV_root[AV_root == 1] = normed_adj_v[AV_root == 1]
            AV_full[AV_full == 1] = normed_adj_v[AV_full == 1]
            temporal_adj_v[temporal_adj_v == 1] = normed_adj_v[temporal_adj_v == 1]
            temporal_adj_v_frw[temporal_adj_v_frw == 1] = normed_adj_v[temporal_adj_v_frw == 1]
            temporal_adj_v_back[temporal_adj_v_back == 1] = normed_adj_v[temporal_adj_v_back == 1]
            AE_full = self.normalize_digraph(AE_full)
            AE_root = self.normalize_digraph(AE_root)
            AE_parent = self.normalize_digraph(AE_parent)
            AE_child = self.normalize_digraph(AE_child)
            AE_crosshead = self.normalize_digraph(AE_crosshead)
            # AV_full = self.normalize_digraph(AV_full)
            # AV_root = self.normalize_digraph(AV_root)
            # AV_close = self.normalize_digraph(AV_close)
            # AV_further = self.normalize_digraph(AV_further)
            # temporal_adj_v = self.normalize_digraph(temporal_adj_v)
            # temporal_adj_e = self.normalize_digraph(temporal_adj_e)
            # temporal_adj_v_back = self.normalize_digraph(temporal_adj_v_back)
            # temporal_adj_v_frw = self.normalize_digraph(temporal_adj_v_frw)

        T = self.get_node_to_edge_mapping().clamp(0, 1)

        return {
            "adj_v": AV_full,
            "adj_v_root": AV_root,
            "adj_v_temp": temporal_adj_v,
            "adj_v_close": AV_close,
            "adj_v_further": AV_further,
            "adj_e": AE_full,
            "adj_e_root": AE_root,
            "adj_e_temp": temporal_adj_e,
            "adj_e_close": AE_parent,
            "adj_e_further": AE_child,
            "adj_e_crosshead": AE_crosshead,
            "ne_mapping": T,
            "adj_v_wtemp": (AV_full + temporal_adj_v).clamp(0, 1),
            "adj_e_wtemp": (AE_full + temporal_adj_e).clamp(0, 1),
            "adj_v_frw": temporal_adj_v_frw,
            "adj_v_back": temporal_adj_v_back,
        }


if __name__ == "__main__":
    adjs = Graph(num_joints=17, num_edges=16, num_frames=1, norm=True)()
    grp = torch.stack((adjs['adj_v_further'], adjs['adj_v']))
    plot_adjacency_matrix(grp[0], annotate_frames=False, annotate_values=True, node_names=get_node_names(1))
    # plot_node_to_edge_map(ajds["adj_v_close"], num_frames=3, annotate_frames=False, node_names=get_node_names(3), edge_names=get_edge_names(3))
