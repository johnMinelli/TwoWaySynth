from __future__ import division
import json
import os.path

import imageio
import numpy as np
import scipy.misc
import torch
from PIL import Image
from path import Path
from scipy.spatial.transform import Rotation as ROT

# if opt.category == 'kitti':
#     intrinsics = np.array([718.9, 0., 128,
#                            0., 718.9, 128,
#                            0., 0., 1.]).reshape((3, 3))
#     self.depth_bias = 0
#     self.depth_scale = 1
#     self.depth_scale_vis = 250. / self.depth_scale
#     self.depth_bias_vis = 0.
#     self.intrinsics = torch.Tensor(intrinsics.astype(np.float32)).cuda().unsqueeze(0)
# 
# elif opt.category in ['shapenet']:
#     intrinsics = np.array([480, 0, 128,
#                            0, 480, 128,
#                            0, 0, 1]).reshape((3, 3))


class ShapeNetRawLoader(object):
    def __init__(self,
                 data_path,
                 split='train',
                 img_height=171,
                 img_width=416,
                 depth=None,
                 get_pose=False):
        self.data_path = Path(data_path)
        self.split = split
        self.img_height = img_height
        self.img_width = img_width
        self.shape_list = ['03001627', '03211117',]
        self.difficulty_folder = "easy"  # "hard"
        self.depth = depth
        self.get_pose = get_pose
        self.collect_train_folders()

    def collect_train_folders(self):
        self.scenes = []
        for shape in self.shape_list:
            shape_scenes = (self.data_path/"image"/shape).dirs()
            for dir in shape_scenes:
                self.scenes.append(dir)

    def collect_from_scene(self, scene):
        train_scenes = []
        scene_id = scene.name
        scene = scene/self.difficulty_folder

        img_files = sorted((scene).files('*.png'))
        scene_data = {'dir': scene, 'frame_id': [], 'pose': [], 'rel_path': scene_id}

        # create scene data dicts, and subsample scene every two frames
        for i, img in enumerate(img_files):
            scene_data['frame_id'].append('{:02d}'.format(i))
        # read metadata
        parse_line_metadata = lambda s : np.array(json.loads("[" + s.replace("\n", "")[:-2] + "]"))
        with open(scene/'rendering_metadata.txt') as file:
            s = []
            for pos in parse_line_metadata(file.read()):
                # r = self.pose_vec2mat(translation=[0,0,0], rototranslation=[0,0,0], rotation=[pos2[0],pos2[1],0], scale=pos2[3])
                r = ROT.from_euler('xyz', [-pos[1], pos[0], pos[2]], degrees=True).as_matrix()
                scale = pos[3]
                s.append((r, scale))
            scene_data['pose'] = s
        scene_data['intrinsics'] = self.load_intrinsics(img_files[0])

        train_scenes.append(scene_data)
        return train_scenes

    def load_intrinsics(self, frame_path):  # HERE this could be wrong
        focal_len = 35
        sensor_size = 32
        intrinsics = np.array([[focal_len * self.img_width / sensor_size, 0, self.img_width / 2.],
                         [0, focal_len * self.img_height / sensor_size, self.img_height / 2.],
                         [0, 0, 1]])

        img = imageio.imread(frame_path)
        zoom_y = self.img_height/img.shape[0]
        zoom_x = self.img_width/img.shape[1]

        intrinsics[0] *= zoom_x
        intrinsics[1] *= zoom_y
        return intrinsics

    def get_scene_imgs(self, scene_data):
        def construct_sample(scene_data, i, frame_id):
            sample = {"img": self.load_image(scene_data, i)[0], "id": frame_id}

            if self.get_pose:
                sample['pose'] = scene_data['pose'][i]
            if self.depth is not None:
                sample['sparse_depth'] = self.load_depth_image(scene_data, i)
            return sample

        for (i, frame_id) in enumerate(scene_data['frame_id']):
            yield construct_sample(scene_data, i, frame_id)

    def load_image(self, scene_data, n_frame):
        img_file = scene_data['dir']/scene_data['frame_id'][n_frame]+'.png'

        if not img_file.isfile():
            return None
        img = imageio.imread(img_file)
        zoom_y = self.img_height/img.shape[0]
        zoom_x = self.img_width/img.shape[1]
        img = Image.fromarray(img).resize(size=(self.img_width, self.img_height))
        return img, zoom_x, zoom_y

    def load_depth_image(self, scene_data, n_frame):
        depth_file = Path(scene_data['dir'].replace("image", "depth"))/scene_data['frame_id'][n_frame]+".png"
        if not depth_file.isfile():
            return None
        img = imageio.imread(depth_file)
        img = Image.fromarray(img).resize(size=(self.img_width, self.img_height))
        return img

    def pose_vec2mat(self, cam_pos_vec):
        """
        Convert 6DoF parameters to transformation matrix.

        Args:s
            vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
        Returns:
            A transformation matrix -- [3, 4]
        """
        translation = np.expand_dims(cam_pos_vec[:3], -1)  # [3, 1]

        x, y, z = np.deg2rad(cam_pos_vec[3]), np.deg2rad(cam_pos_vec[4]), np.deg2rad(cam_pos_vec[5])

        cosz = np.cos(z)
        sinz = np.sin(z)

        zeros = z * 0
        ones = zeros + 1
        zmat = np.stack([cosz, -sinz, zeros, sinz, cosz, zeros, zeros, zeros, ones], axis=0).reshape(3, 3)

        cosy = np.cos(y)
        siny = np.sin(y)

        ymat = np.stack([cosy, zeros, siny, zeros, ones, zeros, -siny, zeros, cosy], axis=0).reshape(3, 3)

        cosx = np.cos(x)
        sinx = np.sin(x)

        xmat = np.stack([ones, zeros, zeros, zeros, cosx, -sinx, zeros, sinx, cosx], axis=0).reshape(3, 3)

        rot_mat = xmat @ ymat @ zmat  # [B, 3, 3]

        transform_mat = np.concatenate([rot_mat, translation], axis=1)  # [ 3, 4]
        return transform_mat

    def pose_from_position(self, cam_pos):
        yaw, roll, pitch = cam_pos  #z_a, x, y

        def get_blender_proj(camera):
            # deg2rad = lambda angle: (angle / 180.) * np.pi
            sa = np.sin(np.deg2rad(-camera[0]))
            ca = np.cos(np.deg2rad(-camera[0]))
            se = np.sin(np.deg2rad(-camera[1]))
            ce = np.cos(np.deg2rad(-camera[1]))
            sz = np.sin(np.deg2rad(-camera[2]))
            R_world2obj = np.eye(3)
            R_world2obj[0, 0] = ca * ce
            R_world2obj[0, 1] = sa * ce
            R_world2obj[0, 2] = -se
            R_world2obj[1, 0] = -sa
            R_world2obj[1, 1] = ca
            R_world2obj[2, 0] = ca * se
            R_world2obj[2, 1] = sa * se
            R_world2obj[2, 2] = ce
            R_obj2cam = np.array((
            (1.910685676922942e-15, 4.371138828673793e-08, 1.0), (1.0, -4.371138828673793e-08, -0.0),
            (4.371138828673793e-08, 1.0, -4.371138828673793e-08))).T
            R_world2cam = np.dot(R_obj2cam, R_world2obj)
            cam_location = np.zeros((3, 1))
            cam_location[0, 0] = camera[2] * 1.75
            T_world2cam = -1 * np.dot(R_obj2cam, cam_location)
            R_camfix = np.array(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
            R_world2cam = np.dot(R_camfix, R_world2cam)
            T_world2cam = np.dot(R_camfix, T_world2cam)
            RT = np.concatenate([R_world2cam, T_world2cam], axis=1)
            return RT

        return get_blender_proj((-yaw, -roll, 0))  # FIXME is this the correct order?

    def pose_from_oxts_packet(self, metadata, scale):  # TODO another try

        lat, lon, alt, roll, pitch, yaw = metadata
        """Helper method to compute a SE(3) pose matrix from an OXTS packet.
        Taken from https://github.com/utiasSTARS/pykitti
        """

        er = 1.75  # earth radius (approx.) in meters
        # Use a Mercator projection to get the translation vector
        ty = lat * np.pi * er / 180.

        tx = scale * lon * np.pi * er / 180.
        # ty = scale * er * \
        #     np.log(np.tan((90. + lat) * np.pi / 360.))
        tz = alt
        t = np.array([tx, ty, tz]).reshape(-1, 1)

        # Use the Euler angles to get the rotation matrix
        Rx = rotx(roll)
        Ry = roty(pitch)
        Rz = rotz(yaw)
        R = Rz.dot(Ry.dot(Rx))
        return transform_from_rot_trans(R, t)

def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def rotz(t):
    """Rotation about the z_a-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def transform_from_rot_trans(R, t):
    """Transforation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))