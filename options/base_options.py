import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # basic info
        self.parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
        self.parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--workers', default=2, type=int, help='# threads for loading data')

        # visualizer initialization settings
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--save_path', type=str, default='./save', help='images and checkpoints are saved here')
        self.parser.add_argument('--save_images', action='store_true', help='if specified, result images will be saved in save_path')

        # data loading related
        self.parser.add_argument('--data_path', type=str, required=True, help='path to dataset')
        self.parser.add_argument("--dataset", type=str, default='kitti', choices=["kitti", "shapenet"])
        self.parser.add_argument("--dataset_format", type=str, default='kitti', choices=["kitti", "shapenet"])

        # train and eval: models hyperparameters
        # experiment related
        self.parser.add_argument('--nz_geo', type=int, default=200, help='number of latent points')
        self.parser.add_argument('--padding_mode', type=str, choices=['zeros', 'border'], default='border',
                            help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                                 ' zeros will null gradients outside target image.'
                                 ' border will only null gradients of the coordinate outside (x or y)')
        self.parser.add_argument('--upsample_mode', type=str, choices=['nearest', 'bilinear'], default='bilinear', help='to specify')
        self.parser.add_argument('--norm_layer', type=str, choices=['batch', 'none'], default='batch', help='to specify')
        self.parser.add_argument('--nl_layer', type=str, choices=['relu', 'lrelu', 'elu'], default='lrelu', help='to specify')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(".", self.opt.name)
        try:
            os.makedirs(expr_dir)
        except: pass
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
