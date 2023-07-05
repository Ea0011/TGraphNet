from common.h36m_skeleton import *
from vizualization import plot_adjacency_matrix, plot_node_to_edge_map
import matplotlib.pyplot as plt


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
        for e in edges:
            source, sink = e
            spatial_adjecency[source, sink] = spatial_adjecency[sink, source] = 1

        A = torch.zeros(self.num_frames * self.num_joints, self.num_frames * self.num_joints)

        # init spatial connections
        for i in range(self.num_frames):
            frame = (i * self.num_joints)
            A[frame:frame+self.num_joints, frame:frame+self.num_joints] = spatial_adjecency.clone()

        return A

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

        A = torch.zeros(self.num_frames * num_joints, self.num_frames * num_joints)

        if self.connections == "all":
            for i in range(self.num_frames):
                for j in range(num_joints):
                    frame = (i * num_joints)
                    neighbour_frame = (j * num_joints)
                    print(frame, neighbour_frame)
                    A[frame:frame+self.num_joints, neighbour_frame:neighbour_frame+num_joints].fill_diagonal_(1)
                    A[neighbour_frame:neighbour_frame+self.num_joints, frame:frame+num_joints].fill_diagonal_(1)

        if self.connections == "middle":
            middle_frame = (self.num_frames // 2) * num_joints
            for i in range(self.num_frames):
                neighbour_frame = (i * num_joints)
                A[middle_frame:middle_frame+num_joints, neighbour_frame:neighbour_frame+num_joints].fill_diagonal_(1)
                A[neighbour_frame:neighbour_frame+num_joints, middle_frame:middle_frame+num_joints].fill_diagonal_(1)

        if self.connections == "neighbour":
            for i in range(self.num_frames):
                frame = (i * num_joints)
                next_frame = ((i + 1) * num_joints)

                if next_frame >= A.shape[0]:
                    break

                A[frame:frame+num_joints, next_frame:next_frame+num_joints].fill_diagonal_(1)
                A[next_frame:next_frame+num_joints, frame:frame+num_joints].fill_diagonal_(1)

        return A.fill_diagonal_(0) # remove self links

    def get_temporal_edge_adjacency(self):
        edge_connections = self.get_temporal_adjacency(num_joints=self.num_edges)

        A = torch.zeros(self.num_frames * self.num_edges, self.num_frames * self.num_edges)

        A[0:(self.num_frames * self.num_edges), 0:(self.num_frames * self.num_edges)] = edge_connections

        return A[0:(self.num_frames * self.num_edges), 0:(self.num_frames * self.num_edges)].fill_diagonal_(0)

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
        edge_names_to_id = {edge: i for i, edge in enumerate(edge_names)}

        # parent adjacency
        for edge, parent in edge_parent.items():
            for p in parent:
                adj_e[edge_names_to_id[edge]][edge_names_to_id[p]] = 1

        # child adjacency
        for edge, children in edge_children.items():
            for child in children:
                adj_e[edge_names_to_id[edge]][edge_names_to_id[child]] = 1

        # crossheads
        for edge, neighbors in edge_crosshead.items():
            for n in neighbors:
                adj_e[edge_names_to_id[edge]][edge_names_to_id[n]] = 1

        # self loops
        for i in range(self.num_edges):
            adj_e[i][i] = 1

        A = torch.zeros(self.num_frames * self.num_edges, self.num_frames * self.num_edges)

        # init spatial connections
        for i in range(self.num_frames):
            frame = (i * self.num_edges)
            A[frame:frame+self.num_edges, frame:frame+self.num_edges] = adj_e.clone()

        return A

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

        DAD = (Dn @ A) @ Dn
        return DAD

    def __call__(self):
        adj_e = (self.get_edge_adjacency() + self.get_temporal_edge_adjacency()).clamp(0, 1)
        adj_v = (self.get_spatial_adjacency() + self.get_temporal_adjacency(num_joints=self.num_joints)).clamp(0, 1)

        if self.norm:
            adj_e, adj_v = self.normalize_undigraph(adj_e), self.normalize_undigraph(adj_v)

        T = self.get_node_to_edge_mapping().clamp(0, 1)

        return adj_v, adj_e, T


if __name__ == "__main__":
    g = Graph(num_joints=17, num_edges=16, num_frames=3)()
    # plot_adjacency_matrix(g[1], annotate_frames=False, node_names=get_edge_names(3))
    plot_node_to_edge_map(g[-1], num_frames=3, annotate_frames=False, node_names=get_node_names(3), edge_names=get_edge_names(3))

