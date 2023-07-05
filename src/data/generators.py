# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np
import torch
from torch.nn import functional as F
from einops import rearrange, repeat


def eval_data_prepare(receptive_field, inputs_2d, out_3d,):
    assert inputs_2d.shape[:-1] == out_3d.shape[:-1], "2d and 3d inputs shape must be same! " + str(inputs_2d.shape) + str(out_3d.shape)
    inputs_2d_p = torch.squeeze(inputs_2d)
    out_3d_p = torch.squeeze(out_3d)

    if inputs_2d_p.shape[0] / receptive_field > inputs_2d_p.shape[0] // receptive_field: 
        out_num = inputs_2d_p.shape[0] // receptive_field+1
    elif inputs_2d_p.shape[0] / receptive_field == inputs_2d_p.shape[0] // receptive_field:
        out_num = inputs_2d_p.shape[0] // receptive_field

    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    eval_out_3d = torch.empty(out_num, receptive_field, out_3d_p.shape[1], out_3d_p.shape[2])

    for i in range(out_num-1):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
        eval_out_3d[i,:,:,:] = out_3d_p[i*receptive_field:i*receptive_field+receptive_field,:,:]
    if inputs_2d_p.shape[0] < receptive_field:
        pad_right = receptive_field-inputs_2d_p.shape[0]
        inputs_2d_p = rearrange(inputs_2d_p, 'b f c -> f c b')
        inputs_2d_p = F.pad(inputs_2d_p, (0,pad_right), mode='replicate')
        inputs_2d_p = rearrange(inputs_2d_p, 'f c b -> b f c')
    if out_3d_p.shape[0] < receptive_field:
        pad_right = receptive_field-out_3d_p.shape[0]
        out_3d_p = rearrange(out_3d_p, 'b f c -> f c b')
        out_3d_p = F.pad(out_3d_p, (0,pad_right), mode='replicate')
        out_3d_p = rearrange(out_3d_p, 'f c b -> b f c')
    eval_input_2d[-1,:,:,:] = inputs_2d_p[-receptive_field:,:,:]
    eval_out_3d[-1,:,:,:] = out_3d_p[-receptive_field:,:,:]

    return eval_out_3d, eval_input_2d


class ChunkedGenerator_Seq:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.

    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of input frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_2d, rot_6d, edge_feat,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, future_frame_pred=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_2d[i].shape[0] == poses_3d[i].shape[0]
            seq_len = poses_2d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if future_frame_pred:
                pairs = [p for p in pairs if p[2] + pad <= seq_len]  # exclude seqs with last frame padding

            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        # Initialize buffers
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        if rot_6d is not None:
            self.batch_6d = np.empty((batch_size, chunk_length, rot_6d[0].shape[-2], rot_6d[0].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))
        self.batch_edge = np.empty((batch_size, chunk_length, edge_feat[0].shape[-2], edge_feat[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        self.rot_6d = rot_6d
        self.edge_feat = edge_feat

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    # start_2d = start_3d - self.pad - self.causal_shift
                    start_2d = start_3d
                    # end_2d = end_3d + self.pad - self.causal_shift
                    end_2d = end_3d

                    # 2D poses
                    seq_2d = self.poses_2d[seq_i]
                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_2d.shape[0])
                    pad_left_2d = low_2d - start_2d
                    pad_right_2d = end_2d - high_2d
                    if pad_left_2d != 0 or pad_right_2d != 0:
                        self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
                    else:
                        self.batch_2d[i] = seq_2d[low_2d:high_2d]

                    if flip:
                        # Flip 2D keypoints
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]

                    # 3D poses
                    if self.poses_3d is not None:
                        seq_3d = self.poses_3d[seq_i]
                        low_3d = max(start_3d, 0)
                        high_3d = min(end_3d, seq_3d.shape[0])
                        pad_left_3d = low_3d - start_3d
                        pad_right_3d = end_3d - high_3d
                        if pad_left_3d != 0 or pad_right_3d != 0:
                            self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_3d[i] = seq_3d[low_3d:high_3d]

                        if flip:
                            # Flip 3D joints
                            self.batch_3d[i, :, :, 0] *= -1
                            self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                    self.batch_3d[i, :, self.joints_right + self.joints_left]

                    # 6D rotations
                    if self.rot_6d is not None:
                        seq_6d = self.rot_6d[seq_i]
                        low_6d = max(start_3d, 0)
                        high_6d = min(end_3d, seq_6d.shape[0])
                        pad_left_6d = low_6d - start_3d
                        pad_right_6d = end_3d - high_6d
                        if pad_left_6d != 0 or pad_right_6d != 0:
                            self.batch_6d[i] = np.pad(seq_6d[low_6d:high_6d], ((pad_left_6d, pad_right_6d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_6d[i] = seq_6d[low_6d:high_6d]

                        # TODO: maybe use a 180 degree rotation here if we use flipping of joints
                        # if flip:
                        #     # Flip 3D joints
                        #     self.batch_3d[i, :, :, 0] *= -1
                        #     self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                        #             self.batch_3d[i, :, self.joints_right + self.joints_left]

                    # edge feature rotations
                    if self.edge_feat is not None:
                        seq_edge = self.edge_feat[seq_i]
                        low_edge = max(start_3d, 0)
                        high_edge = min(end_3d, seq_edge.shape[0])
                        pad_left_edge = low_edge - start_3d
                        pad_right_edge = end_3d - high_edge
                        if pad_left_edge != 0 or pad_right_edge != 0:
                            self.batch_edge[i] = np.pad(seq_edge[low_edge:high_edge], ((pad_left_edge, pad_right_edge), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_edge[i] = seq_edge[low_edge:high_edge]

                        # TODO: maybe use a 180 degree rotation here if we use flipping of joints
                        # if flip:

                    # Cameras
                    if self.cameras is not None:
                        self.batch_cam[i] = self.cameras[seq_i]
                        if flip:
                            # Flip horizontal distortion coefficients
                            self.batch_cam[i, 2] *= -1
                            self.batch_cam[i, 7] *= -1

                if self.endless:
                    self.state = (b_i + 1, pairs)
                if self.poses_3d is None and self.cameras is None:
                    yield None, None, None, self.batch_2d[:len(chunks)], self.batch_edge[:len(chunks)]
                elif self.poses_3d is not None and self.cameras is None:
                    yield None, self.batch_3d[:len(chunks)], self.batch_6d[:len(chunks)], self.batch_2d[:len(chunks)], self.batch_edge[:len(chunks)]
                elif self.poses_3d is None:
                    yield self.batch_cam[:len(chunks)], None, None, self.batch_2d[:len(chunks)], self.batch_edge[:len(chunks)]
                else:
                    yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_6d[:len(chunks)], self.batch_2d[:len(chunks)], self.batch_edge[:len(chunks)]

            if self.endless:
                self.state = None
            else:
                enabled = False


class UnchunkedGenerator_Seq:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.

    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.

    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """

    def __init__(self, cameras, poses_3d, poses_2d, rot_6d, edge_feat, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
        self.edge_feat = edge_feat
        self.rot_6d = [] if rot_6d is None else rot_6d

    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count

    def augment_enabled(self):
        return self.augment

    def set_augment(self, augment):
        self.augment = augment

    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d, seq_6d, seq_edge in zip_longest(self.cameras, self.poses_3d, self.poses_2d, self.rot_6d, self.edge_feat):
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
            batch_2d = None if seq_2d is None else np.expand_dims(seq_2d, axis=0)
            batch_6d = None if seq_6d is None else np.expand_dims(seq_6d, axis=0)
            batch_edge = None if seq_6d is None else np.expand_dims(seq_edge, axis=0)
            # batch_2d = np.expand_dims(np.pad(seq_2d,
            #                 ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
            #                 'edge'), axis=0)
            if self.augment:
                # Append flipped version
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1

                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam, batch_3d, batch_6d, batch_2d, batch_edge


class UnchunkedGenerator_Seq2Seq:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.

    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.

    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """

    def __init__(self, cameras, poses_3d, poses_2d, rot_6d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = False
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right

        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
        self.rot_6d = [] if rot_6d is None else rot_6d

    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count

    def augment_enabled(self):
        return self.augment

    def set_augment(self, augment):
        self.augment = augment

    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d, seq_6d in zip_longest(self.cameras, self.poses_3d, self.poses_2d, self.rot_6d):
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(np.pad(seq_3d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            batch_6d = None if seq_6d is None else np.expand_dims(np.pad(seq_6d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            if self.augment:
                # Append flipped version
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1

                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam, batch_3d, batch_2d, batch_6d


class ChunkedGenerator_Frame:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    def __init__(self, batch_size, cameras, poses_3d, rot_6d, edge_feat, poses_2d,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, future_frame_pred=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)
    
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]
            seq_len = poses_2d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = (np.arange(n_chunks+1)*chunk_length - offset)
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)

            if future_frame_pred:
                pairs = [p for p in pairs if p[2] + pad <= seq_len]  # exclude seqs with last frame padding
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        # Initialize buffers
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        if rot_6d is not None:
            self.batch_6d = np.empty((batch_size, chunk_length, rot_6d[0].shape[-2], rot_6d[0].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length + 2*pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))
        self.batch_edge = np.empty((batch_size, chunk_length + 2*pad, edge_feat[0].shape[-2], edge_feat[0].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None
        
        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        self.rot_6d = rot_6d
        self.edge_feat = edge_feat
        
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state
    
    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    start_2d = start_3d - self.pad - self.causal_shift
                    end_2d = end_3d + self.pad - self.causal_shift

                    # 2D poses
                    seq_2d = self.poses_2d[seq_i]
                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_2d.shape[0])
                    pad_left_2d = low_2d - start_2d
                    pad_right_2d = end_2d - high_2d
                    if pad_left_2d != 0 or pad_right_2d != 0:
                        self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
                    else:
                        self.batch_2d[i] = seq_2d[low_2d:high_2d]

                    if flip:
                        # Flip 2D keypoints
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]
                    
                    # edge feature rotations
                    if self.edge_feat is not None:
                        seq_edge = self.edge_feat[seq_i]
                        low_edge = max(start_2d, 0)
                        high_edge = min(end_2d, seq_edge.shape[0])
                        pad_left_edge = low_edge - start_2d
                        pad_right_edge = end_2d - high_edge
                        if pad_left_edge != 0 or pad_right_edge != 0:
                            self.batch_edge[i] = np.pad(seq_edge[low_edge:high_edge], ((pad_left_edge, pad_right_edge), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_edge[i] = seq_edge[low_edge:high_edge]

                    # 3D poses
                    if self.poses_3d is not None:
                        seq_3d = self.poses_3d[seq_i]
                        low_3d = max(start_3d, 0)
                        high_3d = min(end_3d, seq_3d.shape[0])
                        pad_left_3d = low_3d - start_3d
                        pad_right_3d = end_3d - high_3d
                        if pad_left_3d != 0 or pad_right_3d != 0:
                            self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d], ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_3d[i] = seq_3d[low_3d:high_3d]

                        if flip:
                            # Flip 3D joints
                            self.batch_3d[i, :, :, 0] *= -1
                            self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                    self.batch_3d[i, :, self.joints_right + self.joints_left]
                    
                    # 6D rotations
                    if self.rot_6d is not None:
                        seq_6d = self.rot_6d[seq_i]
                        low_6d = max(start_3d, 0)
                        high_6d = min(end_3d, seq_6d.shape[0])
                        pad_left_6d = low_6d - start_3d
                        pad_right_6d = end_3d - high_6d
                        if pad_left_6d != 0 or pad_right_6d != 0:
                            self.batch_6d[i] = np.pad(seq_6d[low_6d:high_6d], ((pad_left_6d, pad_right_6d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_6d[i] = seq_6d[low_6d:high_6d]

                    # Cameras
                    if self.cameras is not None:
                        self.batch_cam[i] = self.cameras[seq_i]
                        if flip:
                            # Flip horizontal distortion coefficients
                            self.batch_cam[i, 2] *= -1
                            self.batch_cam[i, 7] *= -1

                if self.endless:
                    self.state = (b_i + 1, pairs)
                if self.poses_3d is None and self.cameras is None:
                    yield None, None, None, self.batch_2d[:len(chunks)].copy(), self.batch_edge[:len(chunks)].copy()
                elif self.poses_3d is not None and self.cameras is None:
                    yield None, self.batch_3d[:len(chunks)].copy(), self.batch_6d[:len(chunks)].copy(), self.batch_2d[:len(chunks)].copy(), self.batch_edge[:len(chunks)].copy()
                elif self.poses_3d is None:
                    yield self.batch_cam[:len(chunks)].copy(), None, None, self.batch_2d[:len(chunks)].copy(), self.batch_edge[:len(chunks)].copy()
                else:
                    yield self.batch_cam[:len(chunks)].copy(), self.batch_3d[:len(chunks)].copy(), self.batch_6d[:len(chunks)].copy(), self.batch_2d[:len(chunks)].copy(), self.batch_edge[:len(chunks)].copy()

            if self.endless:
                self.state = None
            else:
                enabled = False

class UnchunkedGenerator_Frame:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, cameras, poses_3d, poses_2d, rot_6d, edge_feat, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.rot_6d = [] if rot_6d is None else rot_6d
        self.edge_feat = edge_feat
        self.poses_2d = poses_2d
        
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count
    
    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d, seq_6d, seq_edge in zip_longest(self.cameras, self.poses_3d, self.poses_2d, self.rot_6d, self.edge_feat):
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(seq_3d, axis=0)
            batch_6d = None if seq_6d is None else np.expand_dims(seq_6d, axis=0)
            batch_edge = np.expand_dims(np.pad(seq_edge,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            if self.augment:
                # Append flipped version
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1
                
                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam, batch_3d.copy(), batch_6d.copy(), batch_2d.copy(), batch_edge.copy()


class ChunkedGenerator_Seq2Seq:
    def __init__(self, batch_size, cameras, poses_3d, poses_2d,
                 chunk_length=1, pad=0, causal_shift=0,
                 shuffle=False, random_seed=1234,
                 augment=False, reverse_aug= False,kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False, out_all = False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)

        pairs = []
        self.saved_index = {}
        start_index = 0

        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_2d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks + 1) * chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            idx = np.repeat(i, len(bounds - 1))
            pairs += list(zip(idx, bounds[:-1], bounds[1:], augment_vector))
            if reverse_aug:
                pairs += list(zip(idx, bounds[:-1], bounds[1:], augment_vector,))
            if augment:
                if reverse_aug:
                    pairs += list(zip(idx, bounds[:-1], bounds[1:], ~augment_vector))
                else:
                    pairs += list(zip(idx, bounds[:-1], bounds[1:], ~augment_vector))

            end_index = start_index + poses_3d[i].shape[0]
            self.saved_index[i] = [start_index,end_index]
            start_index = start_index + poses_3d[i].shape[0]


        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[i].shape[-1]))

        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length + 2 * pad, poses_3d[i].shape[-2], poses_3d[i].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length + 2 * pad, poses_2d[i].shape[-2], poses_2d[i].shape[-1]))

        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None

        self.cameras = cameras
        if cameras is not None:
            self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        self.out_all = out_all

    def num_frames(self):
        return self.num_batches * self.batch_size

    def random_state(self):
        return self.random

    def set_random_state(self, random):
        self.random = random

    def augment_enabled(self):
        return self.augment

    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    seq_2d = self.poses_2d[seq_i]
                    start_2d = start_3d - self.pad - self.causal_shift
                    end_2d = end_3d + self.pad - self.causal_shift
                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_2d.shape[0])
                    pad_left_2d = low_2d - start_2d
                    pad_right_2d = end_2d - high_2d
                    if pad_left_2d != 0 or pad_right_2d != 0:
                        self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
                    else:
                        self.batch_2d[i] = seq_2d[low_2d:high_2d]

                    if flip:
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :,
                                                                            (self.kps_right + self.kps_left)]

                    if self.poses_3d is not None:
                        seq_3d = self.poses_3d[seq_i]
                        if self.out_all:
                            low_3d = low_2d
                            high_3d = high_2d
                            pad_left_3d = pad_left_2d
                            pad_right_3d = pad_right_2d
                        else:
                            low_3d = max(start_3d, 0)
                            high_3d = min(end_3d, seq_3d.shape[0])
                            pad_left_3d = low_3d - start_3d
                            pad_right_3d = end_3d - high_3d
                        if pad_left_3d != 0 or pad_right_3d != 0:
                            self.batch_3d[i] = np.pad(seq_3d[low_3d:high_3d],
                                                    ((pad_left_3d, pad_right_3d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_3d[i] = seq_3d[low_3d:high_3d]

                        if flip:
                            self.batch_3d[i, :, :, 0] *= -1
                            self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                self.batch_3d[i, :, self.joints_right + self.joints_left]

                    if self.cameras is not None:
                        self.batch_cam[i] = self.cameras[seq_i]
                        if flip:
                            self.batch_cam[i, 2] *= -1
                            self.batch_cam[i, 7] *= -1

                if self.endless:
                    self.state = (b_i + 1, pairs)
                if self.poses_3d is None and self.cameras is None:
                    yield None, None, None, self.batch_2d[:len(chunks)].copy(), None
                elif self.poses_3d is not None and self.cameras is None:
                    yield None, self.batch_3d[:len(chunks)].copy(), None, self.batch_2d[:len(chunks)].copy(), None
                elif self.poses_3d is None:
                    yield self.batch_cam[:len(chunks)].copy(), None, None, self.batch_2d[:len(chunks)].copy(), None
                else:
                    yield self.batch_cam[:len(chunks)].copy(), self.batch_3d[:len(chunks)].copy(), None, self.batch_2d[:len(chunks)].copy(), None

            if self.endless:
                self.state = None
            else:
                enabled = False
if __name__ == "__main__":
    seq_2d = torch.randn((500, 17, 2))
    seq_3d = torch.randn((500, 17, 3))
    seq_6d = torch.randn((500, 17, 6))
    seq_edges = torch.randn((500, 17, 4))

    for i in range(seq_2d.shape[0]):
        seq_2d[i] = torch.ones(17, 2) * (i + 1)
        seq_3d[i] = torch.ones(17, 3) * (i + 1)
    gen = ChunkedGenerator_Seq(batch_size=128, cameras=None,
                               poses_2d=[seq_2d,], poses_3d=[seq_3d,], rot_6d=[seq_6d,],
                               edge_feat=[seq_edges,], chunk_length=81, pad=0, shuffle=False,)

    print(f"N Frames: {gen.num_frames()}, N Batches {gen.num_batches}")
    for cam, batch_3d, batch_6d, batch_2d, batch_edge in gen.next_epoch():
        for i in range(batch_6d.shape[0]):
            print("SEQS", batch_6d[i, :, 0, 0].reshape(-1)[0], batch_6d[i, :, 0, 0].reshape(-1)[-1])
            print(batch_2d.shape, batch_3d.shape, batch_6d.shape, batch_edge.shape)

    # unchunked_gen = UnchunkedGenerator_Seq(cameras=None, poses_2d=[seq_2d], poses_3d=[seq_3d], rot_6d=[seq_6d], edge_feat=[seq_edges])
    # print(unchunked_gen.num_frames())
    # for cam, batch_3d, batch_2d, batch_6d, batch_edge in unchunked_gen.next_epoch():
    #     print(batch_2d.shape, batch_3d.shape, batch_6d.shape, batch_edge.shape)
    #     data = eval_data_prepare(82, torch.from_numpy(batch_2d), torch.from_numpy(batch_edge), torch.from_numpy(batch_3d), torch.from_numpy(batch_6d))
    #     # print("Unchunked SEQS", data[0][-2, :, 0, 0])
    #     for i in range(data[0].shape[0]):
    #         print("Unchunked SEQS", data[0][i, :, 0, 0].reshape(-1))
