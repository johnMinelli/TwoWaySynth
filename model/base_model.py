import glob

import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import torch

from model.network_utils.depth_decoder import DepthDecoder
from model.network_utils.util import get_scheduler, print_network, init_weights
from model.network_utils.losses import depth_loss, get_depth_smoothness, photometric_reconstruction_loss, \
    VGGPerceptualLoss
from model.network_utils.metrics import compute_depth_metrics
from model.network_utils.nvs_decoder import NvsDecoder
from model.network_utils.projection_layer import inverse_warp, transform_code
from model.network_utils.metrics import ssim
from model.network_utils.projection_utils import get_RT, scale_K
from model.network_utils.resnet_encoder import ResnetEncoder
from model.network_utils.util import tensor2im
from collections import OrderedDict
import numpy as np
import itertools
from path import Path
import os


class BaseModel():
    def name(self):
        return 'TwoWaySynthModel'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.category = opt.dataset
        self.start_epoch = 0

        self.train_mode = opt.isTrain

        # Setup training devices
        if opt.gpu_ids[0] < 0 or not torch.cuda.is_available():
            print("Training on CPU")
            self.device = torch.device("cpu")
        else:
            print("Training on GPU")
            if len(opt.gpu_ids) > 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_ids)[1:-1]
            self.device = torch.device("cuda")

        # If your system support parallelism, wrap modules with nn.DataParallel( <module> ).to(self.device)
        self.enc = ResnetEncoder(18, nz=opt.nz_geo * 3, pretrained=True).to(self.device)
        self.dec = NvsDecoder(self.enc.num_ch_enc, nz=opt.nz_geo * 3).to(self.device)
        self.depthdec = DepthDecoder(self.enc.num_ch_enc, nz=opt.nz_geo * 3, norm_layer_type=opt.norm_layer, nl_layer_type=opt.nl_layer, upsample_mode=opt.upsample_mode).to(self.device)
        self.vgg = VGGPerceptualLoss().to(self.device)

        self.net_dict = {'enc': self.enc,
                         'dec': self.dec,
                         'depthdec': self.depthdec}
        param_list = []
        for name, model in self.net_dict.items():
            if not self.isTrain or opt.continue_train is not None:
                if self.isTrain:
                    self.start_epoch = self._load_network(model, load_dir=self.backup_dir,epoch_label=opt.continue_train,network_label=name)
                else:
                    self.start_epoch = self._load_network(model, load_dir=opt.models_path, epoch_label=opt.model_epoch, network_label=name)
            else:
                init_weights(model, init_type=opt.init_type)

            if self.isTrain:
                model.train()
                param_list.append(model.parameters())
            else:
                model.eval()
            print_network(model)

        if self.isTrain:
            # train parameters
            self.backup_dir = os.path.join(opt.save_path, opt.name)
            self.nvs_mode = not opt.fast_train
            # initialize optimizers
            self.schedulers, self.optimizers = [], []
            self.optimizer_G = torch.optim.Adam(itertools.chain(*param_list), lr=opt.lr, betas=(opt.momentum, opt.beta), weight_decay=opt.weight_decay)

            self.optimizers.append(self.optimizer_G)

            for optimizer in self.optimizers:
                if self.start_epoch > 0: optimizer.param_groups[0].update({"initial_lr": opt.lr})
                self.schedulers.append(get_scheduler(optimizer, opt, self.start_epoch-1))

        if opt.dataset == 'kitti':
            self.depth_bias , self.depth_scale = 0, 1
            # intrinsics = np.array(  # <-- dynamically computed in the deta_loader
            #     [718.9, 0., 128, \
            #      0., 718.9, 128, \
            #      0., 0., 1.]).reshape((3, 3))
            # self.intrinsics = torch.tensor(intrinsics).float().to(self.device).unsqueeze(0)
        elif opt.dataset == 'shapenet':  # map into [0,3]
            self.depth_bias, self.depth_scale = 1., 2
            # intrinsics = np.array(  # <-- dynamically computed in the deta_loader
            #     [280, 0, 128, \
            #      0, 280, 128, \
            #      0, 0, 1]).reshape((3, 3))
            # self.intrinsics = torch.tensor(intrinsics).float().to(self.device).unsqueeze(0)

    def set_input(self, input):
        self.real_A = Variable(input['A'].to(self.device))
        # self.real_depth_A = Variable(input['DA'].to(self.device))
        self.real_B = Variable(input['B'].to(self.device))
        self.real_depth_B = Variable(input['DB'].to(self.device))
        self.real_RT = Variable(input['RT'].to(self.device))
        self.intrinsics = input['I'][:,:,:3].to(self.device)
        # self.scale_factor = input['S'].numpy()
        self.batch_size = self.real_A.size(0)

    def forward(self):
        self.z_a, self.z_features_a = self.encode(self.real_A)  # [b,nz*3] [high to low features]
        self.depth_a_skip, _ = self.depthdecode(self.z_a, self.z_features_a[:-1])  # for loss
        self.z_b, self.z_features_b = self.encode(self.real_B)  # for loss
        self.depth_b_unskip, _ = self.depthdecode(self.z_b)  # for loss
        self.depth_b_skip, self.depth_scales_b = self.depthdecode(self.z_b, self.z_features_b[:-1])  # for loss

        self.z_a2b = self.transform(self.z_a, self.real_RT)  # [b,nz*3]
        # get depth from latent no_skip
        self.depth_a2b_unskip, self.depth_scales_a2b_unskip = self.depthdecode(self.z_a2b)  # [low to high features]

        # warp 'z_features_a' with 'depth_scales_a2b_unskip' for depth skip connections
        self.z_features_a2b_u = self.warp_features(self.z_features_a, self.depth_scales_a2b_unskip[::-1], intrinsics_ratios=[0.5, 0.25, 0.125, 0.0625, 0.03125])
        self.depth_a2b_skip, self.depth_scales_a2b_skip = self.depthdecode(self.z_a2b, self.z_features_a2b_u[:-1])

        # again warp 'z_features_a' with better 'depth_scales_a2b_skip' to obtain features for nvs skip connections
        self.z_features_a2b_s = self.warp_features(self.z_features_a, self.depth_scales_a2b_skip[::-1], intrinsics_ratios=[0.5, 0.25, 0.125, 0.0625, 0.03125])
        self.fake_B3, self.fake_B2, self.fake_B1, self.fake_B = self.decode(self.z_a2b, self.z_features_a2b_s[:-1] if not self.train_mode or self.nvs_mode else self.z_features_b[:-1])
        # self.fake_direct_B3, self.fake_direct_B2, self.fake_direct_B1, self.fake_direct_B = self.decode(self.z_b, self.z_features_b)  # for loss

    def warp_features(self, z_features, depth_scales, intrinsics_ratios):
        return [inverse_warp(z_features[i], depth_scales[i], self.real_RT, scale_K(self.intrinsics, intrinsics_ratios[i]), self.opt.padding_mode)[0] for i in range(len(z_features))]

    def encode(self, image_tensor):
        return self.enc(image_tensor)

    def transform(self, z, RT):
        return transform_code(z, RT.inverse(), object_centric=self.opt.dataset in ['shapenet'])

    def decode(self, z, z_features):
        output = self.dec(z, z_features)
        return [torch.tanh(out) for out in output]

    def depthdecode(self, z, z_features=None):
        outputs = self.depthdec(z, z_features)
        if self.opt.dataset in ['kitti']:
            return [1 / (10 * torch.sigmoid(output_scale) + 0.01) for output_scale in outputs]  # predict disparity instead of depth for natural scenes
        elif self.opt.dataset in ['shapenet']:
            outputs = [torch.tanh(output_scale) * self.depth_scale + self.depth_bias for output_scale in outputs]  # here do inversion if needed
            return outputs[-1], [F.interpolate(output, scale_factor=0.5, mode=self.opt.upsample_mode) for output in outputs]  # TODO io qui toglierei una scale e il downsample

    def backward_G(self):

        # https://www.sciencedirect.com/science/article/pii/S0923596508001197
        # Multiscale reconstruction loss for NVS output
        self.loss_reco = F.l1_loss(self.fake_B,self.real_B) \
                + 0.5*F.l1_loss(F.interpolate(self.fake_B1, scale_factor=2, mode=self.opt.upsample_mode), self.real_B) \
                + 0.2*F.l1_loss(F.interpolate(self.fake_B2, scale_factor=4, mode=self.opt.upsample_mode), self.real_B) \
                + 0.1*F.l1_loss(F.interpolate(self.fake_B3, scale_factor=8, mode=self.opt.upsample_mode), self.real_B)

        # VGG perceptual loss
        self.vgg_loss = self.vgg(self.fake_B, self.real_B)

        # Image with depth quality
        self.loss_depth_smooth = get_depth_smoothness(self.depth_b_skip, self.real_B) \
                                  + get_depth_smoothness(self.depth_a_skip, self.real_A)  # FIXME maybe is better unskip here?

        # Multiscale transformation/warping loss for depth quality: to supervise the warped scales
        self.loss_warp, warped, diff = photometric_reconstruction_loss(self.real_B, self.real_A, self.intrinsics, self.depth_scales_b + [self.depth_b_skip], self.real_RT)
        # FIXME as it is optimize depth decoder for ground truth input then force similarity between warped and gt depth. Maybe a direct optimization would be harder but better?

        # Loss to improve unskipped depth quality: to improve warping of skipped scales
        depth_b_skip = self.depth_b_skip.detach().clone()
        self.loss_unskip = F.l1_loss(self.depth_a2b_unskip * (torch.median(depth_b_skip) / torch.median(self.depth_a2b_unskip)), depth_b_skip)
        self.loss_skip = F.l1_loss(self.depth_a2b_skip * (torch.median(depth_b_skip) / torch.median(self.depth_a2b_skip)), depth_b_skip)

        # Encoder quality and depth_decoder quality with gt # to enable if you want to use the gt_depth
        # self.loss_depth = depth_loss(self.real_depth_B, self.real_flow_B, self.depth_a2b, self.fake_flow_B)
        # self.loss_depth = F.l1_loss(self.real_depth_B, self.depth_b_skip)

        self.loss_G = (self.loss_reco+self.loss_warp) * self.opt.lambda_recon  # 10.0
        self.loss_G += (self.loss_unskip+self.loss_skip) * self.opt.lambda_skip  # 1.0
        self.loss_G += self.vgg_loss * self.opt.lambda_vgg  # 1.0
        self.loss_G += self.loss_depth_smooth * self.opt.lambda_smooth  # 10.0
        # self.loss_G += self.loss_depth * self.opt.lambda_depth  # 10.0

        self.loss_G.backward()

    def optimize_parameters(self):
        """
        Optimize the parameters computing and backpropagating the losses over the gradients accumulated.
        If the model is in not in train mode the call is ineffective.
        """
        if self.train_mode:
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()

    def get_current_errors(self):
        """
        Returns the computed losses over the current inputs.
        If the model is in not in train mode the call returns a null value.
        """
        return OrderedDict({
                            'loss_reco': self.loss_reco.item(),
                            'loss_vgg': self.vgg_loss.item(),
                            'loss_skip': self.loss_skip.item(),
                            'loss_warp': self.loss_warp.item(),
                            'loss_smooth': self.loss_depth_smooth.item(),
                            # 'loss_depth': self.opt.lambda_depth*self.loss_depth.item(),
                            }) if self.train_mode else {}

    def get_current_metrics(self):
        """
        Returns the metrics for the current inputs.
        """

        return OrderedDict({
                            'L1': F.l1_loss(self.fake_B, self.real_B).item(),
                            'SSIM': ssim(self.fake_B * 0.5 + 0.5, self.real_B * 0.5 + 0.5).item(),
                            'depth_L1_real': F.l1_loss(self.depth_a2b_skip[torch.logical_and(self.real_depth_B > 0, self.real_depth_B < self.opt.max_depth)], self.real_depth_B[torch.logical_and(self.real_depth_B > 0, self.real_depth_B < self.opt.max_depth)]).item(),
                            'depth_L1_direct': F.l1_loss(self.depth_a2b_skip, self.depth_b_skip).item(),
                            **compute_depth_metrics(self.real_depth_B, self.depth_b_skip, max_depth=self.opt.max_depth)})

    def get_current_visuals(self):
        """
        Returns the images computed from current inputs.
        """
        return OrderedDict({'real_A': tensor2im(self.real_A.data[0]),
                            'real_B': tensor2im(self.real_B.data[0]),
                            'real_depth_B': tensor2im(self.real_depth_B.data[0]),
                            'fake_B': tensor2im(self.fake_B.data[0]),
                            # 'fake_direct_B': tensor2im(self.fake_direct_B.data[0]),
                            'depth_B_unskip_warped': tensor2im(self.depth_a2b_unskip.data[0]),
                            'depth_B_skip_warped': tensor2im(self.depth_a2b_skip.data[0]),
                            'depth_B_unskip': tensor2im(self.depth_b_unskip.data[0]),
                            'depth_B_skip': tensor2im(self.depth_b_skip.data[0]),
                            })

    def get_current_anim(self):
        self.switch_mode('eval')

        self.anim_dict = {'vis':[]}
        self.real_A = self.real_A[:1]
        self.real_depth_A = self.real_depth_A[:1]
        self.real_B = self.real_B[:1]
        self.real_depth_B = self.real_depth_B[:1]
        self.intrinsics = self.intrinsics[:1]

        NV = 60
        for i in range(NV):
            pose = np.array([0, -(i-NV/2)*np.pi/180, 0, 0, 0, 0]) if self.opt.dataset in ['shapenet'] \
                else np.array([0, 0, 0,0, 0, i / 1000])
            self.real_RT = get_RT(pose, dataset_type=self.opt.dataset).to(self.device)
            self.forward()
            self.anim_dict['vis'].append(tensor2im(self.fake_B.data[0]))

        self.switch_mode('train')
        return self.anim_dict

    def switch_mode(self, mode):
        assert(mode in ['train', 'eval'])
        self.train_mode = mode == "train"
        for name, model in self.net_dict.items():
            if mode == 'eval': model.eval()
            if mode == 'train': model.train()

    def print(self):
        for _, model in self.net_dict.items():
            print_network(model)

    def save(self, epoch, save_dir=None):
        for name, model in self.net_dict.items():
            self._save_network(model, name, epoch, save_dir)

    # helper saving function that can be used by subclasses
    def _save_network(self, network, network_label, epoch_label, save_dir=None):
        save_filename = '{:04}_net_{}.pth'.format(epoch_label, network_label)
        if save_dir is None: save_dir = self.backup_dir
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def _load_network(self, network, network_label, epoch_label, load_dir=None):
        if load_dir is None: load_dir = self.backup_dir
        if epoch_label == -1:
            load_filename = '*_net_%s.pth' % (network_label)
            load_path = Path(sorted(glob.glob(os.path.join(load_dir, load_filename)))[-1])
        else:
            load_filename = '{:04}_net_{}.pth'.format(epoch_label, network_label)
            load_path = Path(os.path.join(load_dir, load_filename))
        network.load_state_dict(torch.load(load_path))
        return int(load_path.name.split('_')[0])

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
           scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
