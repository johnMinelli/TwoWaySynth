import random
import numpy as np
import torch.utils.data as data
import os
from PIL import Image
from path import Path
from scipy.spatial.transform import Rotation as ROT

from datasets.transform_list import RandomCropNumpy, EnhancedCompose, RandomColor, RandomHorizontalFlip, ArrayToTensorNumpy, Normalize


class ShapeNetDataset(data.Dataset):
    def __init__(self, args, train=True):
        self.train = train
        self.args = args

        self.use_depth = args.depth is not None
        self.depth_scale = 1.75
        self.data_root = Path(args.data_path)

        self.transform = Transformer(args)
        if train is True:
            self.datafile = args.train_file
        else:
            self.datafile = args.test_file

        # Read split file of ids
        self.samples = []
        self.scenes_render = {}
        with open(self.datafile, 'r') as f:
            self.ids = [s.strip() for s in f.readlines() if s]

        # Read intrinsics and poses
        scenes_intrinsics = {}; scenes_poses = {}
        for scene in self.data_root.dirs():
            scenes_intrinsics[str(scene.name)] = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            scene_poses = {}
            for pose in np.genfromtxt(scene/'poses.txt'):
                scene_poses[int(pose[0])] = (pose[1:10].astype(np.float64).reshape((3, 3)), pose[10])
            scenes_poses[str(scene.name)] = scene_poses
            # additional
            # with open(scene/'rendering_metadata.txt') as file:
            #     self.scenes_render[str(scene.name)] = json.loads("["+file.read().replace("\n", "")[:-2]+"]")

        for scene_id in self.ids:
            if not os.path.exists(self.data_root/scene_id): continue

            scene_path = self.data_root/scene_id

            intrinsics = scenes_intrinsics[scene_id]
            poses = scenes_poses[scene_id]
            # Make sample: the target is a list in case a future implementation wants to use multiple views
            for i in range(len(poses)):
                view_shift = i+np.random.randint(1, self.args.max_seq_distance+1)
                sample = {'intrinsics': intrinsics, 'pose_ref': poses[i], 'pose_targets': [poses[(view_shift+j) % len(poses)] for j in range(self.args.n_targets)],
                          'ref': Path(scene_path/'%.2d' % i), 'targets': [Path(scene_path/'%.2d' % ((view_shift+j) % len(poses))) for j in range(self.args.n_targets)]}
                self.samples.append(sample)

        random.shuffle(self.samples)
        self.dataset_size = len(self.samples)
        self.args.rot_range = 4

    def __getitem__(self, index):
        sample_data = self.samples[index]
        DA =None; DB = None

        # create the pair
        id_source = sample_data["ref"]
        id_target = sample_data["targets"][0]

        # load images [-1,1]
        A = self.load_image(id_source+".png") / 255. * 2 - 1
        B = self.load_image(id_target+".png") / 255. * 2 - 1
        # load depth [0,1] where 1 is the furthest
        if self.use_depth:
            DA = self.depth_scale/(((self.load_depth_image((self.data_root/id_source)+"_sparse_depth.png") / 255.) * self.depth_scale)+1.e-17)
            DB = self.depth_scale/(((self.load_depth_image((self.data_root/id_target)+"_sparse_depth.png") / 255.) * self.depth_scale)+1.e-17)

        # intrinsics
        I = sample_data["intrinsics"].astype(np.float32)

        # given the poses compute rotation mat between images
        # PA = sample_data["pose_ref"]
        # PB = sample_data["pose_targets"]
        # 
        # poses = np.stack([PA]+PB)
        # first_pose = poses[0]
        # poses[:, :, -1] -= first_pose[:, -1]
        # compensated_poses = np.linalg.inv(first_pose[:, :3]) @ poses
        # RT1 = np.array([np.vstack([pose, np.column_stack([np.zeros((1, 3)), 1])]).astype(np.float32) for pose in compensated_poses])[1]

        PA, PA_scale = sample_data["pose_ref"]
        PB, PB_scale = sample_data["pose_targets"][0]

        S = (1 - (PB_scale - PA_scale))
        T = np.array([0, 0, 2]).reshape((3, 1))
        R = np.linalg.inv(np.linalg.inv(PA) @ PB) @ (np.eye(3) * (1 - (PA_scale - PB_scale)))

        T = -R.dot(T)+T
        RT = np.block([[R, T], [np.zeros((1, 3)), 1]]).astype(np.float32)

        A, B = self.transform([A, B], self.train)  # as tensors; change CHW format; crop; normalize
        return {'A': A, 'B': B, 'RT': RT, 'S': S, 'DA': DA, 'DB': DB, 'I': I}

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'ShapeNetDataLoader'

    def load_image(self, filename):
        """
        Load image, turn into narray and normalize in [0,1] range
        :param filename: filepath to the image
        :return: normalized HWC narray
        """
        image_path = os.path.join(self.data_root, filename)
        im = Image.open(image_path)
        image = np.asarray(im.convert('RGB'))
        im.close()
        return image

    def load_depth_image(self, filename):
        """
        Load image, turn into narray and normalize in [0,1] range
        :param filename: filepath to the image
        :return: normalized single 1HW narray
        """
        image_path = os.path.join(self.data_root, filename)
        im = Image.open(image_path)
        image = np.expand_dims(np.asarray(im, dtype=np.float32)[:,:,0],0)
        # norm_img = (img - img.min()) / (img.max() - img.min())
        # image = np.clip(image, 0, self.args.max_depth)
        im.close()

        return image

class Transformer(object):
    def __init__(self, args):
        self.train_transform = EnhancedCompose([
            # RandomCropNumpy((args.height,args.width)),
            # RandomHorizontalFlip(),
            # [RandomColor(multiplier_range=(0.9, 1.1)), None, None],
            ArrayToTensorNumpy(),
            # [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
        ])
        self.test_transform = EnhancedCompose([
            ArrayToTensorNumpy(),
            # [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
        ])
    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)





def pose_vec2mat(translation, rototranslation, rotation, scale=1):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [3, 4]
    """
    # translation
    transvec = np.expand_dims(translation, -1)  # [3, 1]

    # rototranslation
    azimuth, elevation, radius = rototranslation
    sRadians = np.deg2rad(azimuth)
    tRadians = np.deg2rad(elevation)
    x = radius * scale * np.cos(sRadians) * np.sin(tRadians)
    y = radius * scale * np.sin(sRadians) * np.sin(tRadians)
    z = radius * scale * np.cos(tRadians)
    rototvec = np.array([x, y, z]).reshape((3, 1))  # [3, 1]

    sumtransvec = transvec+rototvec  # [3, 1]

    # rotation
    x, y, z = scale*np.deg2rad(rotation[0]), scale*np.deg2rad(rotation[1]), scale*np.deg2rad(rotation[2])

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
    sumtransvec= -rot_mat.dot(transvec) + transvec
    transform_mat = np.concatenate([rot_mat, sumtransvec], axis=1)  # [ 3, 4]
    return transform_mat