import random

import cv2
from scipy.spatial.transform import Rotation as ROT
from datasets.transform_list import RandomCropNumpy, EnhancedCompose, RandomColor, RandomHorizontalFlip, ArrayToTensor, Normalize
from torchvision import transforms
import os
import torch.utils.data as data
from PIL import Image
import numpy as np
from path import Path

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

class KITTIDataset(data.Dataset):
    def __init__(self, args, train, valid, eval):
        self.train = train
        self.valid = valid
        self.eval = eval
        self.args = args

        self.use_depth = args.gt_depth is not None
        self.data_root = Path(args.data_path)

        self.transform = Transformer(args)
        if train is True:
            self.datafile = args.train_file
        else:
            self.datafile = args.valid_file

        # Read split file of ids
        self.samples = []
        self.intrinsics = np.genfromtxt(self.data_root.dirs()[0] / 'cam.txt').astype(np.float32).reshape((3,3))
        self.intrinsics[2, 0] = 256 / 2  # being a crop, not a resize, only cx and cy should be updated
        self.intrinsics[2, 1] = 256 / 2

        with open(self.datafile, 'r') as f:
            self.fileset = f.readlines()

        # Read intrinsics and poses
        scenes_poses = {}
        for scene in self.data_root.dirs():
            scene_poses = {}
            for pose in np.genfromtxt(scene/'poses.txt'):
                scene_poses[int(pose[0])] = pose[1:].astype(np.float64).reshape((3, 4))
            scenes_poses[str(scene.name)] = scene_poses

        for sample_file in self.fileset:
            if not os.path.exists(self.data_root/sample_file): continue
            scene = str(Path(sample_file).dirname().name)
            poses = scenes_poses[scene]

            id_ref = int(Path(sample_file).name.stripext())
            last = int((self.data_root/sample_file).parent.files('*.png')[-1].name.stripext())
            if id_ref+3 > last: continue

            id_tgt = np.random.randint(id_ref+3, min(id_ref+self.args.max_kitti_distance+1, last))  # random target
            tgt_file = sample_file.replace(str(id_ref), str(id_tgt))
            if not os.path.exists(self.data_root/tgt_file):
                print('dropped')  # TODO
                continue

            # Make samples for sampled image
            sample = {'pose_ref': poses[id_ref], 'pose_targets': poses[id_tgt],
                      'ref': self.data_root/sample_file, 'targets': self.data_root/tgt_file}
            self.samples.append(sample)

        random.shuffle(self.samples)  # to handle data storage ordering, however seed is fixed
        self.dataset_size = len(self.samples)


    def __getitem__(self, index):
        sample_data = self.samples[index]
        DA = None; DB = None

        # create the pair
        id_source = sample_data["ref"]
        id_target = sample_data["targets"]

        # load images [-1,1]
        A = self.load_image(id_source) / 255. * 2 - 1
        B = self.load_image(id_target) / 255. * 2 - 1
        # load depth [0,max_depth], -1 is invalid
        if self.use_depth:
            DA = self.load_depth_image((self.data_root/id_source).stripext()+"_depth.png") if \
                    os.path.exists((self.data_root/id_source).stripext()+"_depth.png") else np.zeros_like(A)
            DB = self.load_depth_image((self.data_root/id_target).stripext()+"_depth.png") if \
                    os.path.exists((self.data_root/id_target).stripext()+"_depth.png") else np.zeros_like(B)

            # garg/eigen crop
            maskA = np.logical_and(DA >= MIN_DEPTH, DA <= MAX_DEPTH)
            maskB = np.logical_and(DB >= MIN_DEPTH, DB <= MAX_DEPTH)
            crop_mask = np.zeros(maskA.shape)
            crop_mask[153:371, 44:1197] = 1
            DA = DA * np.logical_and(maskA, crop_mask)
            DB = DB * np.logical_and(maskB, crop_mask)
            DA[DA==0] = -1
            DB[DB==0] = -1

        # cropping dimensions to match input 256x256
        h = A.height
        w = A.width
        bound_left = (w - 256) // 2
        bound_right = bound_left + 256
        bound_top = h - 256
        bound_bottom = bound_top + 256

        A = A[bound_top:bound_bottom, bound_left:bound_right]
        B = B[bound_top:bound_bottom, bound_left:bound_right]
        if self.use_depth:
            DA = DA[bound_top:bound_bottom, bound_left:bound_right]
            DB = DB[bound_top:bound_bottom, bound_left:bound_right]

        # intrinsics
        I = self.intrinsics

        PA = sample_data["pose_ref"]
        PB = sample_data["pose_target"]

        R = PA[:, :3].T @ PB[:, :3]
        T = PA[:, :3].T.dot(PB[:, 3:] - PA[:, 3:])

        RT = np.block([[R, T], [np.zeros((1, 3)), 1]]).astype(np.float32)

        A, B, DA, DB = self.transform([A, B, DA, DB], self.train)  # as tensors; change CHW format; crop; normalize
        return {'A': A, 'B': B, 'RT': RT, 'DA': DA, 'DB': DB, 'I': I}

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'KITTIDataLoader'

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
        Load image, turn into narray
        :param filename: filepath to the image
        :return: HW1 narray representing a depth map
        """
        image_path = os.path.join(self.data_root, filename)
        im = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        im = im/np.max(im) * MAX_DEPTH
        im[im == np.inf] = -1
        return im


class Transformer(object):
    def __init__(self, args):
        self.train_transform = EnhancedCompose([
            # RandomCropNumpy((args.height,args.width)),
            # RandomHorizontalFlip(),
            # [RandomColor(multiplier_range=(0.9, 1.1)), None, None],
            ArrayToTensor(),
            # [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
        ])
        self.test_transform = EnhancedCompose([
            ArrayToTensor(),
            # [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
        ])
    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)
