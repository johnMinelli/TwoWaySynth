import random
from scipy.spatial.transform import Rotation as ROT
from datasets.transform_list import RandomCropNumpy, EnhancedCompose, RandomColor, RandomHorizontalFlip, ArrayToTensorNumpy, Normalize
from torchvision import transforms
import os
import csv
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
from PIL import ImageFile
from path import Path
ImageFile.LOAD_TRUNCATED_IMAGES = True

def _is_pil_image(img):
    return isinstance(img, Image.Image)

class KITTIDataset(data.Dataset):
    def __init__(self, args, train=True):
        self.train = train
        self.args = args
        self.use_dense_depth = args.depth == "dense"
        self.use_sparse_depth = args.depth == "sparse"
        self.data_root = Path(args.data_path)

        self.transform = Transformer(args)
        if train is True:
            self.datafile = args.train_file
            self.angle_range = (-1, 1)
            self.depth_scale = 256.0
        else:
            self.datafile = args.test_file
            self.depth_scale = 256.0

        # Read split file of ids
        self.samples = []
        with open(self.datafile, 'r') as f:
            self.fileset = f.readlines()

        # Read intrinsics and poses
        scenes_intrinsics = {}; scenes_poses = {}
        for scene in self.data_root.dirs()[:-1]:  # FIXME togli il -1
            scenes_intrinsics[str(scene.name)] = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            scene_poses = {}
            for pose in np.genfromtxt(scene/'poses.txt'):
                scene_poses[int(pose[0])] = pose[1:].astype(np.float64).reshape((3, 4))
            scenes_poses[str(scene.name)] = scene_poses

        for sample_files in [s.strip().split(' ') for s in self.fileset if s]:
            if not os.path.exists(self.data_root/sample_files[0]) or not os.path.exists(self.data_root/sample_files[1]): continue
            scene = str(Path(sample_files[0]).dirname().name)
            id_ref = int(Path(sample_files[0]).name.stripext())
            ids_tgt = [int(Path(sample_files[1]).name.stripext())]

            intrinsics = scenes_intrinsics[scene]
            poses = scenes_poses[scene]
            # Make sample
            sample = {'intrinsics': intrinsics, 'pose_ref': poses[id_ref], 'pose_targets': [poses[id_tgt] for id_tgt in ids_tgt],
                      'ref': self.data_root/sample_files[0], 'targets': [self.data_root/sample_files[1]]}
            self.samples.append(sample)

        random.shuffle(self.samples)
        self.dataset_size = int(len(self.samples))


    def __getitem__(self, index):
        sample_data = self.samples[index]
        DA = None; DB = None

        # create the pair
        id_source = sample_data["ref"]  # FIXME questo sarebbe invertito in the original dataloader 
        id_target = sample_data["targets"][0]

        # load images
        A = self.load_image(id_source)
        B = self.load_image(id_target)

        if self.use_sparse_depth:
            DA = self.load_image((self.data_root/id_source).stripext()+"_sparse_depth.png", rgb=False)
            DB = self.load_image((self.data_root/id_target).stripext()+"_sparse_depth.png", rgb=False)
        elif self.use_dense_depth:
            DA = self.load_image((self.data_root/id_source).stripext()+"_dense_depth.png", rgb=False)
            DB = self.load_image((self.data_root/id_target).stripext()+"_dense_depth.png", rgb=False)

        # cropping dimensions that can be divided by 16
        h = A.height
        w = A.width
        bound_left = (w - 1216) // 2
        bound_right = bound_left + 1216
        bound_top = h - 352
        bound_bottom = bound_top + 352

        # crop and normalize 0 to 1 ==>  rgb range:(0,1),  depth range: (0, max_depth)
        A = A.crop((bound_left, bound_top, bound_right, bound_bottom))
        A = np.asarray(A, dtype=np.float32) / 255.0
        B = B.crop((bound_left, bound_top, bound_right, bound_bottom))
        B = np.asarray(B, dtype=np.float32) / 255.0

        if _is_pil_image(DA):
            DA = DA.crop((bound_left, bound_top, bound_right, bound_bottom))
            DA = (np.asarray(DA, dtype=np.float32)) / self.depth_scale
            DA = np.expand_dims(DA, axis=2)
            DA = np.clip(DA, 0, self.args.max_depth)
        if _is_pil_image(DB):
            DB = DB.crop((bound_left, bound_top, bound_right, bound_bottom))
            DB = (np.asarray(DB, dtype=np.float32)) / self.depth_scale
            DB = np.expand_dims(DB, axis=2)
            DB = np.clip(DB, 0, self.args.max_depth)

        # A,B,DA,DB = self.transform([A,B,DA,DB], self.train)  #TODO

        # pose between images
        RA = sample_data["pose_ref"]
        RB = sample_data["pose_targets"]

        poses = np.stack([RA]+RB)
        first_pose = poses[0]
        poses[:, :, -1] -= first_pose[:, -1]
        compensated_poses = np.linalg.inv(first_pose[:, :3]) @ poses

        full_mat_poses = [np.vstack([pose, np.column_stack([np.zeros((1, 3)), 1])]).astype(np.float32) for pose in compensated_poses]

        # make tensors in CHW data format
        A = torch.from_numpy(A.astype(np.float32)).permute((2, 0, 1))
        B = torch.from_numpy(B.astype(np.float32)).permute((2, 0, 1))
        full_mat_poses = torch.tensor(full_mat_poses)
        
        # if not self.train:
        #     angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
        #     rgb = Image.open(self.data_path+"/"+id_source)
        #     gt = Image.open(gt_file)
        #     rgb = rgb.rotate(angle, resample=Image.BILINEAR)
        #     gt = gt.rotate(angle, resample=Image.NEAREST)
        #     if self.use_dense_depth:
        #         gt_dense = Image.open(gt_dense_file)
        #         gt_dense = gt_dense.rotate(angle, resample=Image.NEAREST)

        # Poses computations
        # # ---- shapenet files [skip]
        # T = np.array([0, 0, 2]).reshape((3, 1))
        #
        # RA = ROT.from_euler('xyz', [-elev_a, rot_a, 0], degrees=True).as_matrix()
        # RB = ROT.from_euler('xyz', [-elev_b, rot_b, 0], degrees=True).as_matrix()
        # R = RA.T @ RB
        #
        # T = -R.dot(T) + T
        # mat = np.block([[R, T], [np.zeros((1, 3)), 1]])
        # mat_A = np.block([[RA.T, T], [np.zeros((1, 3)), 1]])
        # mat_B = np.block([[RB.T, T], [np.zeros((1, 3)), 1]])
        #
        # # ---- kitti odo poses.txt 6values [skip]
        # poseA = self.poses[int(id_source.name.stripext())]
        # poseB = self.poses[int(id_target.name.stripext())]
        # TA = poseA[3:].reshape(3, 1)
        # RA = ROT.from_euler('xyz', poseA[0:3]).as_matrix()
        # TB = poseB[3:].reshape(3, 1)
        # RB = ROT.from_euler('xyz', poseB[0:3]).as_matrix()
        # T = RA.T.dot(TB - TA) / 50.  # ???
        #
        # mat = np.block([[RA.T @ RB, T], [np.zeros((1, 3)), 1]])
        #
        # # ---- pose_Evaluation_utils.py [unsup]
        # poses = np.stack(pose_list[i] for i in snippet_indices)
        # first_pose = poses[0]
        # poses[:, :, -1] -= first_pose[:, -1]
        # compensated_poses = np.linalg.inv(first_pose[:, :3]) @ poses
        # # ----

        return {'A': A, 'B': B, 'RT': full_mat_poses, 'DA': DA, 'DB': DB}

    def __len__(self):
        return self.dataset_size

    def name(self):
        return 'KITTIDataLoader'

    def load_image(self, filename, rgb=True):
        image_path = os.path.join(self.data_root, filename)
        im = Image.open(image_path)
        return im.convert('RGB') if rgb else im


class Transformer(object):
    def __init__(self, args):
        self.train_transform = EnhancedCompose([
            RandomCropNumpy((args.height,args.width)),
            RandomHorizontalFlip(),
            [RandomColor(multiplier_range=(0.9, 1.1)), None, None],
            ArrayToTensorNumpy(),
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
        ])
        self.test_transform = EnhancedCompose([
            ArrayToTensorNumpy(),
            [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
        ])
    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)
