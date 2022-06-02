from __future__ import division
import json

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from path import Path
from scipy.spatial.transform import Rotation as ROT


class ShapeNetRawLoader(object):
    def __init__(self,
                 data_path,
                 split='train',
                 img_height=256,
                 img_width=256,
                 depth=None,
                 get_pose=False):
        self.data_path = Path(data_path)
        self.split = split
        self.img_height = img_height
        self.img_width = img_width
        self.shape_list = ['03001627', '02958343']
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
                r = ROT.from_euler('xyz', [-pos[1], pos[0], pos[2]], degrees=True).as_matrix()
                scale = pos[3]
                s.append((r, scale))
            scene_data['pose'] = s
        scene_data['intrinsics'] = self.load_intrinsics(img_files[0])

        train_scenes.append(scene_data)
        return train_scenes

    def load_intrinsics(self, frame_path):
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
