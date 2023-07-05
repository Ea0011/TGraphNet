import numpy as np
import os
import glob
import scipy.io as sio
# import cv2
from angles import *
from common.h36m_skeleton import *
from common.camera_params import h36m_cameras_intrinsic_params, normalize_screen_coordinates, project_to_2d_linear, project_to_2d, normalize_screen_coordinates_torch
from data.generators import ChunkedGenerator_Seq, ChunkedGenerator_Frame, ChunkedGenerator_Seq2Seq


class Human36M:
    joint_id_to_names = dict(zip(range(len(joint_names)), joint_names))
    joint_name_to_id = {joint: i for i, joint in enumerate(joint_names)}
    edge_id_to_names = dict(zip(range(len(edge_names)), edge_names))
    edge_name_to_id = {edge: i for i, edge in enumerate(edge_names)}

    def get_bonelength(self, pos):
        '''
        pos: F x J x dim
        '''
        F, J, _ = pos.shape
        bone_length = np.zeros((F, len(edge_names), 1))
        for idx, edge in enumerate(edge_index):
            joint1 = pos[:, edge[0]]
            joint2 = pos[:, edge[1]]
            bone_length[:, idx] = np.linalg.norm((joint1-joint2), axis=1).reshape(-1, 1)
        return bone_length

    def get_bone_orientation(self, pos):
        '''
        pos: F x J x dim
        '''
        F, J, _ = pos.shape
        bone_orientation = np.zeros((F, len(edge_names), 4))

        # for each edge
        for idx, edge in enumerate(edge_index):
            # find edge vector
            e = pos[:, edge[1]] - pos[:, edge[0]]

            # find parent edge vector
            parent_name = edge_parents[edge_names[idx]]
            if parent_name == 'root':
                # TO DO: put comments
                edge_parent='Hip'
                root_v = np.copy(pos[:, 0] )
                root_v[:, 1] = 0
                e_p = pos[:, 0] - root_v
            else:
                edge_parent = edge_index[self.edge_name_to_id[parent_name]]
                e_p = pos[:, edge_parent[1]] - pos[:, edge_parent[0]]

            bone_orientation[:, idx] = rotmat(angle_between1(e, e_p))

        return bone_orientation

    def prepare_camera_params(self):
        pass

    def load_source_data(self, bpath, dim, dim_to_use, disp=False):
        """
        Loads 3d/ 2d ground truth from disk, and
        puts it in an easy-to-acess dictionary
        Args
        bpath: String. Path where to load the data from
        subjects: List of integers. Subjects whose data will be loaded
        actions: List of strings. The actions to load
        dim: Integer={2,3}. Load 2 or 3-dimensional data
        dim_to_use: out of 32 joints, select 17 joints
        Returns:
        data: Dictionary with keys k=(subject, action, seqname)
          values v=(nx(32*2) matrix of 2d ground truth)
          There will be 2 entries per subject/action if loading 3d data
          There will be 8 entries per subject/action if loading 2d data
        """
        if not dim in [2,3]:
            raise(ValueError, 'dim must be 2 or 3')

        data = {}
        total_recs=0
        sub_actions = []

        for subj in self.subjects:
            if dim == 3:
                dpath = os.path.join( bpath, 'S{0}'.format(subj), 'MyPoseFeatures/D3_Positions_mono/*.cdf.mat')
            elif dim == 2:
                dpath =  os.path.join( bpath, 'S{0}'.format(subj), 'MyPoseFeatures/D2_Positions/*.cdf.mat')

            fnames = glob.glob( dpath )
            num_recs=0

            for fname in fnames:
                seqname = os.path.basename( fname )

                action = seqname.split(".")[0]
                action = action.replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog')

                cam = seqname.split(".")[1]
                cam_id = h36m_cameras_intrinsic_params[cam]['id']

                if subj == 11 and action == "Directions":
                    continue # corrupt video

                sub_actions.append(action)
                poses = sio.loadmat(fname)['data'][0][0]
                poses = poses.reshape(-1, 32, dim)[:, dim_to_use]
                data[ (subj, action, cam_id) ] = poses
                num_recs+=poses.shape[0]


            if disp:
                print("subject: ", subj, " num_files: ", len(fnames), " num_recs: ", num_recs)
            total_recs+=num_recs

        sub_actions = np.asarray(sub_actions)
        sub_actions = np.unique(sub_actions)
        if disp:
            print("load_source_data / records loaded: ", total_recs, " x ", poses.shape[1], " x ", poses.shape[2], "\n")

        return data, sub_actions

    def load_cpn_detection(self, cpn_file="data_2d_h36m_cpn_ft_h36m_dbb.npz", disp=False):
        keypoints = np.load("/media/HDD3/datasets/Human3.6M/" + cpn_file, allow_pickle=True)
        keypoints = keypoints['positions_2d'].item() # a nested dict with Subject->Action

        # Check for >= instead of == because some videos in H3.6M contain extra frames

        for subj in self.subjects:
            subject = "S"+str(subj)

            for action in keypoints[subject].keys():
                # assert action in self.actions, 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)

                for cam_idx in range(len(keypoints[subject][action])):

                    mocap_length = self.data_2d[(subj,action,cam_idx)].shape[0]
                    assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

                    if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                        # Shorten sequence
                        keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        data = {}
        total_recs=0
        sub_actions = []

        for subj in self.subjects:
            subject = "S" + str(subj)
            subj_keypoints = keypoints[subject]
            num_recs=0
            num_files = 0

            for action in subj_keypoints.keys():

                action = action.replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog')

                if subject == 'S11' and action == "Directions":
                    continue # corrupt video

                for cam_id, kps in enumerate(subj_keypoints[action]):
                    sub_actions.append(action)
                    poses = kps # -1, 17, 2
                    poses = np.array(poses)
                    data[ (subj, action, cam_id) ] = poses
                    num_recs+= poses.shape[0]
                    num_files+=1

            if disp:
                print("subject: ", subj, " num_files: ", num_files, "num_recs: ", num_recs)
            total_recs+=num_recs

        sub_actions = np.asarray(sub_actions)
        sub_actions = np.unique(sub_actions)
        if disp:
            print("load_cpn_detection / records loaded: ", total_recs, " x ", poses.shape[1], " x ", poses.shape[2], "\n")

        return data, sub_actions

    def load_source_angle(self, bpath):
        data_angles = {}

        for subj in self.subjects:

            dpath = os.path.join( bpath, 'S{0}'.format(subj), 'MyPoseFeatures/D3_Angles_mono/*.cdf.mat')
            fnames = glob.glob( dpath )

            for fname in fnames:
                seqname = os.path.basename( fname )

                action = seqname.split(".")[0]
                action = action.replace('TakingPhoto', 'Photo').replace('WalkingDog', 'WalkDog')
                cam = seqname.split(".")[1]
                cam_id = h36m_cameras_intrinsic_params[cam]['id']

                if subj == 11 and action == "Directions":
                    continue # corrupt video

                angles = sio.loadmat(fname)['data'][0][0]  # 25 rotation + 1 for trajectory
                angles = angles.reshape(-1, 26, 3)

                data_angles[(subj, action, cam_id)] = angles.astype('float32')

        return data_angles

    def preprocess_rotations(self, data):
        '''
        https://github.com/kamisoel/kinematic_pose_estimation/tree/fd0fa7ce87b8b690e86572b2689604763c283d73
        r = np.array([
        'hips', 'rightUpLeg', 'rightLowLeg','RightFoot', <'RightToeBase'>,
        'LeftUpLeg', 'LeftLowLeg', 'LeftFoot', <'LeftToeBase'>,
        'Spine', 'Spine1', 'Neck', 'Head',
        'LeftShoulder', 'LeftUpArm', 'LeftFOreArm', <'LeftHand', 'LeftHandThumb', 'L_Wrist'>,
        'RightShoulder', 'RightUpArm','RightForeArm', <'RightHand', 'RightHandThumb', 'R_Wrist'>])
        '''
        rot_6d = {}
        traj = {}
        self.dim_to_use_angle=[0,1,2,3,5,6,7,9,10,11,12,13,14,15,19,20,21]

        for key in data.keys():
            subject, actions, cam_id = key

            r = data[key][:, 1:, :]
            t = data[key][:, 0, :]

            # keep angles in [-180, 180 ]
            r = (r + 180) % 360 - 180

            r_mat = eul2mat_np(r)[:, self.dim_to_use_angle]
            F, J, _, _ = r_mat.shape
            r_6d = rotmat_to_rot6d_np(r_mat.reshape(-1, 3, 3)).reshape(F, J, 6)
            rot_6d[key] = r_6d
            traj[key] = t / 1000. # convert to meter

        return rot_6d, traj

    def normalize_2d(self, data):
        """
        Normalize so that [0, Width] is mapped to [-1, 1], while preserving the aspect ratio
        """
        data_out = {}

        for key in data.keys():
            # extract cam id from filename
            _, _, cam_id = key

            for k, v in h36m_cameras_intrinsic_params.items():
                if v['id'] == cam_id:
                    camera_id = k

            joint_data = data[ key ]

            # get cam resolution
            height = h36m_cameras_intrinsic_params[camera_id]['res_h']
            width = h36m_cameras_intrinsic_params[camera_id]['res_w']

            # Normalize
            data_out[key] = joint_data/width*2 -[1, height/width]

        return data_out

    def postprocess_3d(self, poses_set):
        """
        Center 3d points around root and extract 17 out of 32 joints
        Args
        poses_set: Dictionary with keys k=(subject, action, seqname)
        and value v=(nx(32*3) matrix of 3d ground truth)
        Returns
        poses_set: dictionary with 3d data centred around root (center hip) joint
        root_positions: dictionary with the original 3d position of each pose
        """
        poses_set_bkp = poses_set.copy()

        for k in poses_set_bkp.keys():
            # Subtract the root from the 1st position onwards
            # 0th position tracks the root position for future use
            poses = poses_set[k]
            root_joint = poses[:, :1]
            poses[:, 1:] -= root_joint  # keep global position intact
            poses_set[k] = poses

        return poses_set

    def __len__(self):
        return len(self.pos2d)

    def __init__(self, data_dir, train=True, ds_category="gt", actions="all", camera="all", train_subjects=[1, 5, 6, 7, 8], test_subjects=[9, 11]):
        """
        set joint names /dim_to_use
        set training and test subjects
        set camera resolutions
        load 2d (normalized) to input and 3d skeleton (unnormalized, root centered) to output
        """
        if True:

            # Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
            self.H36M_NAMES = ['']*32
            self.H36M_NAMES[0]  = 'Hip'
            self.H36M_NAMES[1]  = 'RHip'
            self.H36M_NAMES[2]  = 'RKnee'
            self.H36M_NAMES[3]  = 'RFoot'
            self.H36M_NAMES[6]  = 'LHip'
            self.H36M_NAMES[7]  = 'LKnee'
            self.H36M_NAMES[8]  = 'LFoot'
            self.H36M_NAMES[12] = 'Spine'
            self.H36M_NAMES[13] = 'Thorax'
            self.H36M_NAMES[14] = 'Neck/Nose'
            self.H36M_NAMES[15] = 'Head'
            self.H36M_NAMES[17] = 'LShoulder'
            self.H36M_NAMES[18] = 'LElbow'
            self.H36M_NAMES[19] = 'LWrist'
            self.H36M_NAMES[25] = 'RShoulder'
            self.H36M_NAMES[26] = 'RElbow'
            self.H36M_NAMES[27] = 'RWrist'

            self.map_camid_to_camnum = {}
            for camnum, value in h36m_cameras_intrinsic_params.items():
                self.map_camid_to_camnum[value['id']] = camnum

            self.dim_to_use = np.where(np.array([x != '' for x in self.H36M_NAMES]))[0]
            self.data_dir = data_dir

            if train:
                self.subjects = train_subjects
            else:
                self.subjects =  test_subjects

        self.image_names, self.pos2d, self.pos2d_centered, self.pos3d, self.pos3d_centered, self.cam, \
            self.gt_angles_euler, self.gt_angles_6d, self.global_ori = [], [], [], [], [], [], [], [], []
        self.edge_features = []
        self.gt_angles_mat = []

        # Load data
        # load 2d keypoints (camera frame)
        if ds_category == "gt":
            self.data_2d, self.actions = self.load_source_data(self.data_dir, 2, self.dim_to_use)
        elif ds_category == "cpn":
            self.data_2d, self.actions = self.load_source_data(self.data_dir, 2, self.dim_to_use)
            self.data_2d, self.actions = self.load_cpn_detection()

        # load 3d ground truth camera frame positions
        data_3d, _ = self.load_source_data(self.data_dir, 3, self.dim_to_use)

        # Load angles
        _data_angles = self.load_source_angle(self.data_dir)

        ## Post-processing
        # normalize 2d skeleton -1 to +1
        self.data_2d = self.normalize_2d(self.data_2d) #temporarily not normalize

        # root center 3d skeleton
        data_3d_centered = self.postprocess_3d(data_3d)

        # post process angles
        data_rot6d, globalpos = self.preprocess_rotations(_data_angles)

        for key in self.data_2d.keys():
            subject, act, cam_id = key
            camnum = self.map_camid_to_camnum[cam_id]

            cam_params = h36m_cameras_intrinsic_params[camnum]
            for k, v in cam_params.items():
                if k not in ['id', 'res_w', 'res_h']:
                    cam_params[k] = np.array(v, dtype=np.float32)
            
            # Normalize camera frame
            # cam_params['center'] = normalize_screen_coordinates(cam_params['center'], w=cam_params['res_w'], h=cam_params['res_h']).astype('float32')
            # cam_params['focal_length'] = cam_params['focal_length']/cam_params['res_w']*2

            cam_intrinsics = np.concatenate((
                cam_params['focal_length'],
                cam_params['center'],
                cam_params['radial_distortion'],
                cam_params['tangential_distortion']
            ))
            if actions != "all":
                if not (actions in act):
                    continue
                
                # WOW!
                if (actions == "Sitting" and act == "SittingDown"):
                    continue

            if camera != "all":
                if camnum != camera:
                    continue

            pose_2d = self.data_2d[key]
            pose_3d_centered = data_3d_centered[key]
            globalori = data_rot6d[key][:, :1]
            pose_angle_6d = data_rot6d[key][:, 1:]  # exclude global orientation i.e. index 0
            edge_features = self.get_bone_orientation(pose_2d)

            image_dir = os.path.join(self.data_dir, 'S{0}/Images/{1}.{2}/'.format(subject, act, camnum))

            frame_name =  "%s.%s" %(act, camnum)
            image_fpath = os.path.join(image_dir, frame_name+".jpg")
            self.image_names.append(image_fpath)

            self.pos2d.append(pose_2d)
            self.pos3d_centered.append(pose_3d_centered)
            self.gt_angles_6d.append(pose_angle_6d)
            self.edge_features.append(edge_features)
            self.global_ori.append(globalori)
            self.cam.append(cam_intrinsics)

            # Can add stride here


        print("Subjects: ", self.subjects, ds_category)
        print("2d: ", len(self.pos2d), "3d: ", len(self.pos3d_centered), \
             "angles_6d: ", len(self.gt_angles_6d), "edge_feat:", len(self.edge_features), \
             "global_ori: ", len(self.global_ori), "camera params", len(self.cam))
        print(sum([len(s) for s in self.pos2d]))



if __name__ == "__main__":
    import time
    from time import strftime, gmtime

    start = time.time()
    train_dataset = Human36M(data_dir="/media/HDD3/datasets/Human3.6M/pose_zip", train=True, ds_category="cpn",)
    pos2d, pos3d, angles_6d, edge_features, cameras = train_dataset.pos2d, train_dataset.pos3d_centered, train_dataset.gt_angles_6d, train_dataset.edge_features, train_dataset.cam

    # gen = ChunkedGenerator_Frame(256, cameras=cameras, poses_2d=pos2d, poses_3d=pos3d, rot_6d=angles_6d,
    #                                              edge_feat=edge_features, chunk_length=81, pad=0, shuffle=False,)

    kps_left, kps_right = [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16]

    gen = ChunkedGenerator_Seq2Seq(128, cameras, pos3d, pos2d, chunk_length=11, pad=35, out_all=True, shuffle=False,
                                                    augment=False, reverse_aug=False,
                                                    kps_left=kps_left, kps_right=kps_right,
                                                    joints_left=kps_left,
                                                    joints_right=kps_right)
    print(f"N Frames: {gen.num_frames()}, N Batches {gen.num_batches}")
    for cam, batch_3d, batch_6d, batch_2d, batch_edge in gen.next_epoch():
        cam = torch.from_numpy(cam.astype('float32')).cuda()
        batch_3d = torch.FloatTensor(batch_3d).cuda()
        traj = batch_3d[:, :, :1].clone()
        batch_3d[:, :, 0] = 0
        prj = project_to_2d((batch_3d + traj) / 1000, cam)
        prj_norm = normalize_screen_coordinates_torch(prj, 1000, 1000)
        print(torch.mean(torch.norm(prj_norm - torch.from_numpy(batch_2d).cuda(), dim=3)))

    print("Elapsed time: ", strftime("%H:%M:%S", gmtime(time.time() - start)))