import random
import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
from path import Path

from datasets.transform_list import RandomCropNumpy, EnhancedCompose, RandomColor, RandomHorizontalFlip, ArrayToTensor, \
    Normalize, Resize

EL_RANGE = 4  # different positions: dataset dependant
AZ_RANGE = 18  # different positions: dataset dependant

class ShapeNetDataset(data.Dataset):
    def __init__(self, args, train, valid, eval):
        self.train = train
        self.valid = valid
        self.eval = eval
        self.args = args

        self.use_depth = args.gt_depth is not None
        self.data_root = Path(args.data_path)
        self.positions = AZ_RANGE*EL_RANGE

        self.transform = Transformer(args)
        if train is True:
            self.datafile = args.train_file
        else:
            self.datafile = args.valid_file

        # Read split file of ids
        self.samples = []
        self.intrinsics = np.genfromtxt(self.data_root / '..' / '..' / 'camera_settings' / 'cam_K' / 'cam_K.txt').astype(np.float32).reshape((3,3))
        self.intrinsics[:-1, :] = self.intrinsics[:-1, :] / 2  # halved since the images are resized from 512 to 256

        with open(self.datafile, 'r') as f:
            self.ids = [s.strip() for s in f.readlines() if s]

        # Read intrinsics and poses
        poses = []
        for i in range(self.positions):
            poses.append(np.genfromtxt(self.data_root / '..' / '..' / 'camera_settings' / 'cam_RT' / 'cam_RT_{:03d}.txt'.format(i+1)).astype(np.float32).reshape((3,4)))

        if self.eval:
            self.pairs = self.ids
            scene_id = None
            for pair_ids in self.pairs:
                id_s, id_t = pair_ids.split(" ")
                scene_id,b,c = id_s.split("_")
                id_s = (int(b)/20*EL_RANGE) + (int(c)/10) + 1
                _,b,c = id_t.split("_")
                id_t = (int(b)/20*EL_RANGE) + (int(c)/10) + 1
                scene_path = self.data_root / scene_id
                sample = {'pose_ref': poses[id_s], 'pose_target': poses[id_t],
                          'ref': scene_path/'color_{:03d}.png'.format(id_s), 'target': scene_path/'color_{:03d}.png'.format(id_t)}
                self.samples.append(sample)
        else:
            for scene_id in self.ids:
                if not os.path.exists(self.data_root/scene_id): continue
                scene_path = self.data_root/scene_id

                # Make samples for sampled object
                for i in range(self.positions):
                    view_shift = (((((i//EL_RANGE) +  # in range [0, AZ_RANGE]
                                 np.random.randint(1, self.args.max_az_distance+1))*EL_RANGE) % self.positions) +  # random az distance
                                 np.random.randint(0, EL_RANGE) if self.args.rand_el_distance else i % 4)  # optional random el distance
                    sample = {'pose_ref': poses[i], 'pose_target': poses[view_shift],
                              'ref': scene_path/'color_{:03d}.png'.format(i+1), 'target': scene_path/'color_{:03d}.png'.format(view_shift+1)}
                    self.samples.append(sample)

        random.shuffle(self.samples)  # to handle data storage ordering, however seed is fixed
        self.dataset_size = len(self.samples)

    def __getitem__(self, index):
        sample_data = self.samples[index]
        DA = None; DB = None

        # create the pair
        id_source = sample_data["ref"]
        id_target = sample_data["target"]

        # load images [-1,1]
        A = self.load_image(id_source) / 255. * 2 - 1
        B = self.load_image(id_target) / 255. * 2 - 1
        # load depth [0,max_depth], -1 is invalid
        if self.use_depth:
            DA = self.load_depth_image(id_source.dirname()/id_source.name.replace("color", "depth").replace("png", "exr"))
            DB = self.load_depth_image(id_target.dirname()/id_target.name.replace("color", "depth").replace("png", "exr"))

        # intrinsics
        I = self.intrinsics

        PA = sample_data["pose_ref"]
        PB = sample_data["pose_target"]

        T = np.array([0, 0, 2]).reshape((3, 1))
        R = PA[:,:3].T @ PB[:,:3]
        # R = np.linalg.inv(np.linalg.inv(PB[:,:3]) @ PA[:,:3])  # equivalent

        T = -R.dot(T)+T
        RT = np.block([[R, T], [np.zeros((1, 3)), 1]]).astype(np.float32)

        A, B, DA, DB = self.transform([A, B, DA, DB], self.train)  # as tensors; change CHW format; crop; normalize
        return {'A': A, 'B': B, 'RT': RT, 'DA': DA, 'DB': DB, 'I': I}

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
        Load image, turn into narray
        :param filename: filepath to the image
        :return: HW1 narray representing a depth map
        """
        image_path = os.path.join(self.data_root, filename)
        im = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)[:,:,:1]
        im[im == np.inf] = -1
        return im

class Transformer(object):
    def __init__(self, args):
        self.train_transform = EnhancedCompose([
            Resize((256,256)),
            # RandomCropNumpy((args.height,args.width)),
            # RandomHorizontalFlip(),
            # [RandomColor(multiplier_range=(0.9, 1.1)), None, None],
            ArrayToTensor()
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.test_transform = EnhancedCompose([
            Resize((256, 256)),
            ArrayToTensor()
        ])
    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)
