import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_adjacency_matrix(A, num_frames=3, annotate_frames=True, annotate_values=False, annotate_neighbour=False, node_names=[], num_nodes=17):
    plt.figure(figsize=(24, 24))
    ax = sns.heatmap(
        A,
        xticklabels=node_names,
        yticklabels=node_names,
        cmap="Blues",
        linecolor='gray',
        cbar=None,
        linewidth=.5,
        annot=annotate_values,
        square=True)

    if annotate_neighbour:
        for i in range(num_frames):
            frame = (i * num_nodes)
            next_frame = ((i + 1) * num_nodes)

            if next_frame < num_frames * num_nodes:
                ax.add_patch(Rectangle((frame, next_frame), num_nodes, num_nodes, edgecolor='darkgreen', lw=4, clip_on=True, fill=False,))
                ax.add_patch(Rectangle((next_frame, frame), num_nodes, num_nodes, edgecolor='deeppink', lw=4, clip_on=True, fill=False,))

    if annotate_frames:
        for i in range(num_frames):
            frame = (i * num_nodes)
            ax.add_patch(Rectangle((frame, frame), num_nodes, num_nodes, edgecolor='darkorange', lw=4, clip_on=False, fill=False))

    plt.show()


def plot_node_to_edge_map(T, num_frames=3, annotate_frames=True, node_names=[], edge_names=[], num_nodes=17, num_edges=16):
    plt.figure(figsize=(24, 24))
    ax = sns.heatmap(
        T,
        xticklabels=edge_names,
        yticklabels=node_names,
        cmap="Blues",
        linecolor='gray',
        cbar=None,
        linewidth=.5,
        square=True)
    
    if annotate_frames:
        for i in range(num_frames):
            node_frame = (i * num_nodes)
            edge_frame = (i * num_edges)
            ax.add_patch(Rectangle((edge_frame, node_frame), num_edges, num_nodes, edgecolor='darkorange', lw=4, clip_on=False, fill=False))

    plt.show()
