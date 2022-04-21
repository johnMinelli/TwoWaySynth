import argparse
from shutil import copyfile

import imageio
import numpy as np
from pebble import ProcessPool
import sys
from tqdm import tqdm
from path import Path

from options.preprocess_options import PreprocessOptions


def dump_example(args, scene):
    scene_list = data_loader.collect_from_scene(scene)
    for scene_data in scene_list:
        dump_dir = args.dump_root/scene_data['rel_path']
        dump_dir.makedirs_p()
        intrinsics = scene_data['intrinsics']

        dump_cam_file = dump_dir/'cam.txt'

        np.savetxt(dump_cam_file, intrinsics)
        poses_file = dump_dir/'poses.txt'
        poses = []

        for sample in data_loader.get_scene_imgs(scene_data):
            img, frame_nb = sample["img"], sample["id"]
            dump_img_file = dump_dir/'{}.png'.format(frame_nb)
            imageio.imsave(dump_img_file, img)
            if "pose" in sample.keys():
                p=[frame_nb]  # id
                p.extend(['%.6e' % s for s in sample["pose"][0].reshape(-1, 9)[0]])  # pose matrix
                p.append(sample["pose"][1])  # scale
                poses.append(p)
            if "dense_depth" in sample.keys():
                dump_depth_file = dump_dir/'{}.png'.format(frame_nb+"_dense_depth")
                imageio.imsave(dump_depth_file, sample["dense_depth"].astype(np.uint16))
            if "sparse_depth" in sample.keys() and  sample["sparse_depth"] is not None:
                dump_depth_file = dump_dir / '{}.png'.format(frame_nb+"_sparse_depth")
                imageio.imsave(dump_depth_file, sample["sparse_depth"])

        if len(poses) != 0:
            np.savetxt(poses_file, poses, fmt='%s')

        if len(dump_dir.files('*.png')) < 3:
            dump_dir.rmtree()


def main():
    args = PreprocessOptions().parse()

    args.dump_root = Path(args.dump_root)
    args.dump_root.mkdir_p()

    global data_loader

    if args.dataset_format == 'kitti':
        from kitti_raw_loader import KittiRawLoader
        data_loader = KittiRawLoader(args.data_path,
                                     static_frames_file=args.static_frames,
                                     img_height=args.height,
                                     img_width=args.width,
                                     depth=args.depth,
                                     get_pose=args.with_pose,
                                     depth_size_ratio=args.depth_size_ratio)

    if args.dataset_format == 'shapenet':
        from shapenet_raw_loader import ShapeNetRawLoader
        data_loader = ShapeNetRawLoader(args.data_path,
                                        img_height=args.height,
                                        img_width=args.width,
                                        depth=args.depth,
                                        get_pose=args.with_pose)

    n_scenes = len(data_loader.scenes)
    print('Found {} potential scenes'.format(n_scenes))
    print('Retrieving frames')
    if args.num_threads == 1:
        for i, scene in enumerate(tqdm(data_loader.scenes)):
            # if i<150: continue
            dump_example(args, scene)
    else:
        with ProcessPool(max_workers=args.num_threads) as pool:
            tasks = pool.map(dump_example, [args]*n_scenes, data_loader.scenes)
            try:
                for _ in tqdm(tasks.result(), total=n_scenes):
                    pass
            except KeyboardInterrupt as e:
                tasks.cancel()
                raise e

    # print('Generating train val lists')  # TODO remove
    # np.random.seed(8964)
    # # to avoid data snooping, we will make two cameras of the same scene to fall in the same set, train or val
    # subdirs = args.dump_root.dirs()
    # canonic_prefixes = set([subdir.basename()[:-2] for subdir in subdirs])
    # with open(args.dump_root / 'train.txt', 'w') as tf:
    #     with open(args.dump_root / 'val.txt', 'w') as vf:
    #         for pr in tqdm(canonic_prefixes):
    #             corresponding_dirs = args.dump_root.dirs('{}*'.format(pr))
    #             if np.random.random() < 0.1:
    #                 for s in corresponding_dirs:
    #                     vf.write('{}\n'.format(s.name))
    #             else:
    #                 for s in corresponding_dirs:
    #                     tf.write('{}\n'.format(s.name))
    #                     if args.with_depth and args.no_train_gt:
    #                         for gt_file in s.files('*.npy'):
    #                             gt_file.remove_p()


if __name__ == '__main__':
    main()
