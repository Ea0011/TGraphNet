import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pytransform3d import rotations as pr
from pytransform3d.plot_utils import make_3d_axis
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.transforms import Affine2D
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from common.h36m_skeleton import *
from matplotlib.animation import FuncAnimation


def plot_adjacency_matrix(A, num_frames=3, annotate_frames=True, annotate_values=False, annotate_neighbour=False, node_names=[], num_nodes=17):
    plt.figure(figsize=(24, 24))
    ax = sns.heatmap(
        A,
        xticklabels=node_names,
        yticklabels=node_names,
        cmap="coolwarm",
        linecolor='gray',
        cbar=None,
        linewidth=.5,
        annot=annotate_values,
        robust=True,
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


def plot_adjacency_matrix_cluster(A, num_frames=3, annotate_frames=True, annotate_values=False, annotate_neighbour=False, node_names=[], num_nodes=17):
    sns.clustermap(
        A,
        xticklabels=node_names,
        yticklabels=node_names,
        cmap="coolwarm",
        linecolor='gray',
        # cbar=None,
        linewidth=.5,
        annot=annotate_values,
        robust=True,
        square=True,
        figsize=(24, 24),
        method="complete")

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


def text3d(ax, xyz, s, zdir="z", size=None, angle=0, usetex=False, **kwargs):
    """
    Plots the string *s* on the axes *ax*, with position *xyz*, size *size*,
    and rotation angle *angle*. *zdir* gives the axis which is to be treated as
    the third dimension. *usetex* is a boolean indicating whether the string
    should be run through a LaTeX subprocess or not.  Any additional keyword
    arguments are forwarded to `.transform_path`.

    Note: zdir affects the interpretation of xyz.
    """
    x, y, z = xyz
    if zdir == "y":
        xy1, z1 = (x, z), y
    elif zdir == "x":
        xy1, z1 = (y, z), x
    else:
        xy1, z1 = (x, y), z

    text_path = TextPath((0, 0), s, size=size, usetex=usetex)
    trans = Affine2D().rotate(angle).translate(xy1[0], xy1[1])

    p1 = PathPatch(trans.transform_path(text_path), **kwargs)
    ax.add_patch(p1)
    art3d.pathpatch_2d_to_3d(p1, z=z1, zdir=zdir)


def plot_pose_and_orientation(pos3d, rot, edges, colors=['red'], pose_scale=60, rot_ax_scale=2, rot_ax_width=3, x_offset=10, y_offset=4, ax=None):
    pos3d /= pose_scale
    pos3d[:, 0] += x_offset # translate X
    pos3d[:, 1] += y_offset # translate Y
    pos3d[:, 2] += 0 # translate Z
    for jid in range(len(pos3d)):        
        pos = pos3d[jid]
        ax.scatter([pos[0]], [pos[1]], [pos[2]], color="crimson", s=5)
        if rot is not None:
            r = rot[jid]
            pr.plot_basis(ax=ax, R=r, p=pos, s=rot_ax_scale, lw=rot_ax_width, strict_check=False)

    for e in edges:
        j1 = pos3d[e[0]]
        j2 = pos3d[e[1]]
        if joint_id_to_names[int(e[0])].startswith("L"):
            ax.plot([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]], linestyle='-', color=colors[1], linewidth=5)
        else:
            ax.plot([j1[0], j2[0]], [j1[1], j2[1]], [j1[2], j2[2]], linestyle='-', color=colors[0], linewidth=5)


def plot_poses_only(gt_pos, pred_pos, action="", mpjpe=0, pose_scale=60, x_offset=10, y_offset=4, ax=None):
    ax.text(0, 14, 0, f"Action: {action}", zdir="x", color="blue", fontweight="normal", horizontalalignment="center", fontsize=22)
    ax.text(0, 12, 0, f"MPJPE: {mpjpe:.1f}mm", zdir="x", color="black", fontweight="normal", horizontalalignment="center", fontsize=20)

    plot_pose_and_orientation(-gt_pos, None, edges=edge_index, colors=['limegreen', 'orange'], pose_scale=pose_scale, x_offset=-x_offset, y_offset=y_offset, ax=ax)
    plot_pose_and_orientation(-pred_pos, None, edges=edge_index,colors=['deepskyblue', 'hotpink'], pose_scale=pose_scale, x_offset=x_offset, y_offset=y_offset, ax=ax)


def plot_pred_and_gt_poses(gt_pos, pred_pos, gt_rot, pred_rot, action="", mpjpe=0, mpjae=0,pose_scale=60, rot_ax_scale=2, rot_ax_width=3, x_offset=10, y_offset=4, ax=None):
    ax.text(0, 14, 0, f"Action: {action}", zdir="x", color="blue", fontweight="normal", horizontalalignment="center", fontsize=22)
    ax.text(0, 12, 0, f"MPJPE: {mpjpe:.1f}mm", zdir="x", color="black", fontweight="normal", horizontalalignment="center", fontsize=20)
    ax.text(0, 10, 0, f"MPJAE: {mpjae:.2f}rad", zdir="x", color="black", fontweight="normal", horizontalalignment="center", fontsize=20)

    plot_pose_and_orientation(-gt_pos, gt_rot, edges=edge_index, colors=['limegreen', 'orange'], pose_scale=pose_scale, rot_ax_scale=rot_ax_scale, rot_ax_width=rot_ax_width, x_offset=-x_offset, y_offset=y_offset, ax=ax)
    plot_pose_and_orientation(-pred_pos, pred_rot, edges=edge_index,colors=['deepskyblue', 'hotpink'], pose_scale=pose_scale, rot_ax_scale=rot_ax_scale, rot_ax_width=rot_ax_width, x_offset=x_offset, y_offset=y_offset, ax=ax)


def plot_pose_animation(pred_pos3d, gt_pos3d, mpjpe_err=None, action="", num_frames=81, save_path=None):
    assert save_path is not None

    fig = plt.figure(figsize=(12, 12))
    ax = make_3d_axis(20, pos=int('11{}'.format(1)), n_ticks=5,)

    def animate(i):
        ax.clear()
        pred_pos = pred_pos3d[i]
        gt_pos = gt_pos3d[i]
        pos_err = mpjpe_err[i] if mpjpe_err is not None else 0
        ax.view_init(-80, 90) # view them from front
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.set_ylim(-30, 30)
        ax.set_xlim(-30, 30)
        ax.set_zlim(-30, 30)

        plot_poses_only(gt_pos, pred_pos, f"{action} Frame: {i}", pos_err, ax=ax, x_offset=-10, y_offset=-4)

    ani = FuncAnimation(fig, animate, frames=num_frames, repeat=False, interval=10) #interval: Delay between frames in milliseconds

    return ani