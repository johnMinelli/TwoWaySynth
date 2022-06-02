import random
import numpy as np
import torch.utils.data as data
import os
from PIL import Image
from path import Path

from datasets.transform_list import RandomCropNumpy, EnhancedCompose, RandomColor, RandomHorizontalFlip, ArrayToTensorNumpy, Normalize


class ShapeNetDataset(data.Dataset):
    def __init__(self, args, train, valid, eval):
        self.train = train
        self.valid = valid
        self.eval = eval
        self.args = args

        self.use_depth = args.depth is not None
        self.depth_scale = args.max_depth
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

        if self.eval:
            self.pairs = self.ids
            scene_id = None
            for pair_ids in self.pairs:
                id_s, id_t = pair_ids.split(" ")
                if scene_id != id_s.split("/")[0]:
                    scene_id = id_s.split("/")[0]
                    scene_path = self.data_root / scene_id
                    intrinsics = scenes_intrinsics[scene_id]
                    poses = scenes_poses[scene_id]
                sample = {'intrinsics': intrinsics, 'pose_ref': poses[int(Path(id_s).name)], 'pose_targets': [poses[int(Path(id_t).name)]],
                          'ref': Path(self.data_root / id_s), 'targets': [Path(self.data_root / id_t)]}
                self.samples.append(sample)
        else:
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

        random.shuffle(self.samples)  # to handle data storage ordering, however seed is fixed
        self.dataset_size = len(self.samples)

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
