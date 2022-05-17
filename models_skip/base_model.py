import glob

import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torchvision
from torch.optim import lr_scheduler

from models_skip.network_utils.depth_decoder import DepthDecoder
from models_skip.network_utils.losses import depth_loss, get_depth_smoothness, photometric_reconstruction_loss
from models_skip.network_utils.metrics import compute_depth_metrics
from models_skip.network_utils.nvs_decoder import NvsDecoder
from models_skip.network_utils.projection_layer import inverse_warp
from models_skip.network_utils import networks
from models_skip.network_utils.metrics import ssim
from models_skip.network_utils.resnet_encoder import ResnetEncoder
from util_skip.util import tensor2im
from collections import OrderedDict
from skimage.transform import warp
import numpy as np
import itertools
from path import Path
import os


class BaseModel():
    def name(self):
        return 'BaseSkipModel'

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.backup_dir = os.path.join(opt.save_path, opt.name)
        self.category = opt.dataset
        self.start_epoch = 0

        self.nvs_mode = False
        self.train_mode = True

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
                self.start_epoch = self.load_network(model, load_dir=self.backup_dir, epoch_label=opt.continue_train, network_label=name)
            else:
                networks.init_weights(model, init_type=opt.init_type)

            if self.isTrain:
                model.train()
                param_list.append(model.parameters())
            else:
                model.eval()
            networks.print_network(model)

        if self.isTrain:
            # initialize optimizers
            self.schedulers, self.optimizers = [], []
            self.optimizer_G = torch.optim.Adam(itertools.chain(*param_list), lr=opt.lr, betas=(opt.momentum, opt.beta), weight_decay=opt.weight_decay)

            self.optimizers.append(self.optimizer_G)

            for optimizer in self.optimizers:
                if self.start_epoch>0: optimizer.param_groups[0].update({"initial_lr":opt.lr})
                self.schedulers.append(self.get_scheduler(optimizer, opt, self.start_epoch-1))

        if opt.dataset == 'kitti':
            self.depth_bias , self.depth_scale = 0, 1
            # intrinsics = np.array(  # <-- dynamically computed in the loader
            #     [718.9, 0., 128, \
            #      0., 718.9, 128, \
            #      0., 0., 1.]).reshape((3, 3))
            # self.intrinsics = torch.tensor(intrinsics).float().to(self.device).unsqueeze(0)
        elif opt.dataset == 'shapenet':  # map into [0,3]
            self.depth_bias, self.depth_scale = 1., 2
            # intrinsics = np.array(  # <-- dynamically computed in the loader
            #     [280, 0, 128, \
            #      0, 280, 128, \
            #      0, 0, 1]).reshape((3, 3))
            # self.intrinsics = torch.tensor(intrinsics).float().to(self.device).unsqueeze(0)


    def set_input(self, input):
        self.real_A = Variable(input['A'].to(self.device))
        self.real_depth_A = Variable(input['DA'].to(self.device))
        self.real_B = Variable(input['B'].to(self.device))
        self.real_depth_B = Variable(input['DB'].to(self.device))
        self.real_RT = Variable(input['RT'].squeeze().to(self.device))
        self.intrinsics = input['I'][:,:,:3].to(self.device)
        # self.scale_factor = input['S'].numpy()
        self.batch_size = self.real_A.size(0)

    def forward(self):
        self.z_a, self.z_features_a = self.encode(self.real_A)  # [b,nz*3] [high to low features]
        self.z_b, self.z_features_b = self.encode(self.real_B)  # for loss
        self.depthunskipped_b, _ = self.depthdecode(self.z_b)  # for loss
        self.depthskipped_b, self.depth_scales_b = self.depthdecode(self.z_features_b)  # for loss

        self.z_a2b = self.transform(self.z_a, self.real_RT)  # [b,nz*3]
        # get depth from latent no_skip
        self.depthunskipped_a2b, self.depth_scales_a2b = self.depthdecode(self.z_a2b)  # [low to high features]
        self.depthskipped_a, _ = self.depthdecode(self.z_features_a)  # for loss

        # warp 'z_features_a' with 'depth_scales_a2b' for depth skip connections
        self.z_features_warped = self.warp_features(self.z_features_a, self.depth_scales_a2b[::-1], intrinsics_ratios=[0.5, 0.25, 0.125, 0.0625, 0.03125])
        self.depthskipped_b_warped, self.depth_scales_b_skip_warped = self.depthdecode(self.z_features_warped)

        # again warp 'z_features_a' with better 'depth_scales_a2b' for nvs skip connections
        self.z_features_skip_warped = self.warp_features(self.z_features_a, self.depth_scales_b_skip_warped[::-1], intrinsics_ratios=[0.5, 0.25, 0.125, 0.0625, 0.03125])
        self.fake_B3, self.fake_B2, self.fake_B1, self.fake_B = self.decode(self.z_a2b, self.z_features_skip_warped[:-1] if self.nvs_mode or not self.train_mode else self.z_features_b)

    def warp_features(self, z_features, depth_scales, intrinsics_ratios):
        return [inverse_warp(z_features[i], depth_scales[i], self.real_RT, self.scale_K(self.intrinsics, intrinsics_ratios[i]), self.opt.padding_mode)[0] for i in range(len(z_features))]

    def encode(self, image_tensor):
        return self.enc(image_tensor)

    def transform(self, z, RT):
        return networks.transform_code(z, self.opt.nz_geo, RT.inverse(), object_centric=self.opt.dataset in ['shapenet'])

    def decode(self, z, z_features):
        output = self.dec(z, z_features)
        return [torch.tanh(out) for out in output]

    def depthdecode(self, z):
        outputs = self.depthdec(z)
        if self.opt.dataset in ['kitti']:
            return [1 / (10 * torch.sigmoid(output_scale) + 0.01) for output_scale in outputs]  # predict disparity instead of depth for natural scenes
        elif self.opt.dataset in ['shapenet']:
            outputs = [torch.tanh(output_scale) * self.depth_scale + self.depth_bias for output_scale in outputs]  # here do inversion if needed
            return outputs[-1], [F.interpolate(output, scale_factor=0.5, mode=self.opt.upsample_mode) for output in outputs]

    def backward_G(self):

        # https://www.sciencedirect.com/science/article/pii/S0923596508001197
        # Multiscale reconstruction loss for NVS output
        self.loss_reco = (F.l1_loss(self.fake_B,self.real_B) \
                + 0.5*F.l1_loss(F.interpolate(self.fake_B1, scale_factor=2, mode=self.opt.upsample_mode), self.real_B) \
                + 0.2*F.l1_loss(F.interpolate(self.fake_B2, scale_factor=4, mode=self.opt.upsample_mode), self.real_B) \
                + 0.1*F.l1_loss(F.interpolate(self.fake_B3, scale_factor=8, mode=self.opt.upsample_mode), self.real_B))

        # VGG perceptual loss
        self.vgg_loss = self.vgg(self.fake_B, self.real_B)

        # Image with depth quality
        self.loss_depth_smooth = get_depth_smoothness(self.depthskipped_b, self.real_B) \
                                  + get_depth_smoothness(self.depthskipped_a, self.real_A)
        # Multiscale transformation/warping loss for depth quality
        #     scaled_A = self.scale_image(self.real_A, self.scale_factor, self.scale_factor)
        #     self.loss_warp = F.l1_loss(self.warp_fake_B[scaled_A>0].requires_grad_(), self.real_B[scaled_A>0])
        self.loss_warp, warped, diff = photometric_reconstruction_loss(self.real_B, self.real_A, self.intrinsics, self.depth_scales_b+[self.depthskipped_b], self.real_RT)  # supervise the warped scales

        # Loss to improve unskipped depth quality to improve warping of skipped scales
        depthskipped_b = self.depthskipped_b.detach().clone()
        self.loss_unskip = F.l1_loss(self.depthunskipped_a2b * (torch.median(depthskipped_b) / torch.median(self.depthunskipped_a2b)), depthskipped_b)
        self.loss_skip = F.l1_loss(self.depthskipped_b_warped * (torch.median(depthskipped_b) / torch.median(self.depthskipped_b_warped)), depthskipped_b)  # sure about this normalization?

        # Encoder quality and depth_decoder quality with gt # TODO to enable if we want to use the gt_depth
        self.loss_depth = 0  # depth_loss(self.real_depth_B, self.real_flow_B, self.depth_a2b, self.fake_flow_B)
        # self.loss_depth = F.l1_loss(self.real_depth_B, self.depthskipped_b)

        self.loss_G = (self.loss_reco+self.loss_warp) * self.opt.lambda_recon  # 10.0
        self.loss_G += (self.loss_unskip+self.loss_skip) * self.opt.lambda_skip  # 1.0
        self.loss_G += self.vgg_loss * self.opt.lambda_vgg  # 1.0
        self.loss_G += self.loss_depth_smooth * self.opt.lambda_smooth  # 10.0
        self.loss_G += self.loss_depth * self.opt.lambda_depth  # 10.0

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

    def get_current_visuals(self):
        """
        Returns the images computed from current inputs.
        """
        return OrderedDict({'real_A': tensor2im(self.real_A.data[0]),
                            'real_B': tensor2im(self.real_B.data[0]),
                            'real_depth_B': tensor2im(self.real_depth_B.data[0]),
                            'fake_B': tensor2im(self.fake_B.data[0]),
                            'depth_B_unskip_warped': tensor2im(self.depthunskipped_a2b.data[0]),
                            'depth_B_skip_warped': tensor2im(self.depthskipped_b_warped.data[0]),
                            'depth_B_unskip': tensor2im(self.depthunskipped_b.data[0]),
                            'depth_B_skip': tensor2im(self.depthskipped_b.data[0]),
                            })

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
                            'loss_depth': 0 # self.opt.lambda_depth*self.loss_depth.item(),
                            }) if self.train_mode else {}

    def get_current_metrics(self):
        """
        Returns the metrics for the current inputs.
        """

        return OrderedDict({
                            'L1': F.l1_loss(self.fake_B, self.real_B).item(),
                            'SSIM': ssim(self.fake_B * 0.5 + 0.5, self.real_B * 0.5 + 0.5).item(),
                            'depth_L1_real': F.l1_loss(self.depthskipped_b_warped[torch.logical_and(self.real_depth_B>0, self.real_depth_B<self.opt.max_depth)], self.real_depth_B[torch.logical_and(self.real_depth_B>0, self.real_depth_B<self.opt.max_depth)]).item(),
                            'depth_L1_direct': F.l1_loss(self.depthskipped_b_warped, self.depthskipped_b).item(),
                            **compute_depth_metrics(self.real_depth_B, self.depthskipped_b, max_depth=self.opt.max_depth)})

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
            self.real_RT = self.get_RT(pose)
            self.forward()
            self.anim_dict['vis'].append(tensor2im(self.fake_B.data[0]))

        self.switch_mode('train')
        return self.anim_dict

    def get_RT(self, vec):  # TODO move utils
        from scipy.spatial.transform import Rotation as ROT
        if self.opt.dataset in ['shapenet']:
            T = np.array([0, 0, 2]).reshape((3, 1))
            R = ROT.from_euler('xyz', vec[:3]).as_matrix()
            T = -R.dot(T) + T
        else:
            R = ROT.from_euler('xyz', vec[0:3]).as_matrix()
            T = vec[3:].reshape((3, 1))
        mat = np.block([[R, T], [np.zeros((1, 3)), 1]])

        return torch.Tensor(mat).float().to(self.device).unsqueeze(0)

    def scale_K(self, intrinsics, scale):  # TODO move utils
        #scale fx, fy, cx, cy according to the scale

        K = intrinsics.clone()
        K[:, 0, 0] *= scale
        K[:, 1, 1] *= scale
        K[:, 0, 2] *= scale
        K[:, 1, 2] *= scale
        return K

    def scale_image(self, batch_img, scale_x, scale_y):  # TODO move utils
        scale_mat = torch.eye(3).unsqueeze(0).repeat(self.batch_size, 1, 1).numpy()
        scale_mat[:, 0, 0] = scale_x
        scale_mat[:, 0, 2] = (batch_img.size()[3] - (batch_img.size()[3] * scale_x)) / 2
        scale_mat[:, 1, 1] = scale_y
        scale_mat[:, 1, 2] = (batch_img.size()[2] - (batch_img.size()[2] * scale_y)) / 2
        scaled_image = torch.from_numpy(np.array([warp(tensor2im(img), np.linalg.inv(sc)) for img, sc in
                                              zip(batch_img, scale_mat)]).transpose((0, 3, 1, 2)))
        return scaled_image

    def switch_mode(self, mode):
        assert(mode in ['train', 'eval'])
        self.train_mode = mode == "train"
        for name, model in self.net_dict.items():
            if mode == 'eval': model.eval()
            if mode == 'train': model.train()

    def save(self, epoch, save_dir=None):
        for name, model in self.net_dict.items():
            self.save_network(model, name, epoch, save_dir)

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, save_dir=None):  # TODO make private
        save_filename = '{:04}_net_{}.pth'.format(epoch_label, network_label)
        if save_dir is None: save_dir = self.backup_dir
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, load_dir=None):  # TODO make private
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

    def get_scheduler(self, optimizer, opt, last_epoch):  # TODO clean all these methods
        if opt.lr_policy == 'lambda':
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 - opt.lr_niter_frozen) / float(opt.lr_niter_decay + 1)
                return lr_l
            scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
        elif opt.lr_policy == 'step':
            scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_every, gamma=0.1, last_epoch=last_epoch)
        elif opt.lr_policy == 'plateau':
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5, last_epoch=last_epoch)
        else:
            return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
        return scheduler


class VGGPerceptualLoss(torch.nn.Module):
    # VGG loss, Cite from https://gist.github.com/alper111/8233cdb0414b4cb5853f2f730ab95a49

    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss


# import bpy
# import bpy_extras
# from mathutils import Matrix
# from mathutils import Vector
# # ---------------------------------------------------------------
# # 3x4 P matrix from Blender camera
# # ---------------------------------------------------------------
# # Build intrinsic camera parameters from Blender camera data
# # See notes on this in 
# # blender.stackexchange.com/questions/15102/what-is-blenders-camera-projection-matrix-model
# def get_calibration_matrix_K_from_blender(camd):
#     f_in_mm = camd.lens
#     scene = bpy.context.scene
#     resolution_x_in_px = scene.render.resolution_x
#     resolution_y_in_px = scene.render.resolution_y
#     scale = scene.render.resolution_percentage / 100
#     sensor_width_in_mm = camd.sensor_width
#     sensor_height_in_mm = camd.sensor_height
#     pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
#     if (camd.sensor_fit == 'VERTICAL'):
#         # the sensor height is fixed (sensor fit is horizontal), 
#         # the sensor width is effectively changed with the pixel aspect ratio
#         s_u = resolution_x_in_px * scale / sensor_width_in_mm / pixel_aspect_ratio
#         s_v = resolution_y_in_px * scale / sensor_height_in_mm
#     else:  # 'HORIZONTAL' and 'AUTO'
#         # the sensor width is fixed (sensor fit is horizontal), 
#         # the sensor height is effectively changed with the pixel aspect ratio
#         pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
#         s_u = resolution_x_in_px * scale / sensor_width_in_mm
#         s_v = resolution_y_in_px * scale * pixel_aspect_ratio / sensor_height_in_mm
#     # Parameters of intrinsic calibration matrix K
#     alpha_u = f_in_mm * s_u
#     alpha_v = f_in_mm * s_v
#     u_0 = resolution_x_in_px * scale / 2
#     v_0 = resolution_y_in_px * scale / 2
#     skew = 0  # only use rectangular pixels
#     K = Matrix(((alpha_u, skew, u_0), (0, alpha_v, v_0), (0, 0, 1)))
#     return K
# # Returns camera rotation and translation matrices from Blender.
# # There are 3 coordinate systems involved:
# #    1. The World coordinates: "world"
# #       - right-handed
# #    2. The Blender camera coordinates: "bcam"
# #       - x is horizontal
# #       - y is up
# #       - right-handed: negative z_a look-at direction
# #    3. The desired computer vision camera coordinates: "cv"
# #       - x is horizontal
# #       - y is down (to align to the actual pixel coordinates 
# #         used in digital images)
# #       - right-handed: positive z_a look-at direction
# def get_3x4_RT_matrix_from_blender(cam):
#     # bcam stands for blender camera
#     R_bcam2cv = Matrix(((1, 0, 0), (0, -1, 0), (0, 0, -1)))
#     # Transpose since the rotation is object rotation, 
#     # and we want coordinate rotation
#     # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
#     # T_world2bcam = -1*R_world2bcam * location
#     # Use matrix_world instead to account for all constraints
#     location, rotation = cam.matrix_world.decompose()[0:2]
#     R_world2bcam = rotation.to_matrix().transposed()
#     # Convert camera location to translation vector used in coordinate changes
#     # T_world2bcam = -1*R_world2bcam*cam.location
#     # Use location from matrix_world to account for constraints:     
#     T_world2bcam = -1 * R_world2bcam * location
#     R_world2cv = R_bcam2cv * R_world2bcam
#     T_world2cv = R_bcam2cv * T_world2bcam
#     print(T_world2cv, R_bcam2cv, T_world2bcam)
#     RT = Matrix((R_world2cv[0][:] + (T_world2cv[0],), R_world2cv[1][:] + (T_world2cv[1],), R_world2cv[2][:] + (T_world2cv[2],)))
#     print(RT)
#     return RT
# def get_3x4_P_matrix_from_blender(cam):
#     K = get_calibration_matrix_K_from_blender(cam.data)
#     RT = get_3x4_RT_matrix_from_blender(cam)
#     return K * RT, K, RT
# def project_by_object_utils(cam, point):
#     scene = bpy.context.scene
#     co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point)
#     render_scale = scene.render.resolution_percentage / 100
#     render_size = (int(scene.render.resolution_x * render_scale), int(scene.render.resolution_y * render_scale),)
#     return Vector((co_2d.x * render_size[0], render_size[1] - co_2d.y * render_size[1]))
# def start():
#     cam = bpy.data.objects['Camera.001']
#     P, K, RT = get_3x4_P_matrix_from_blender(cam)
#     print("K")
#     print(K)
#     print("RT")
#     print(RT)
#     print("P")
#     print(P)
#     print("==== Tests ====")
#     e1 = Vector((1, 0, 0, 1))
#     e2 = Vector((0, 1, 0, 1))
#     e3 = Vector((0, 0, 1, 1))
#     O = Vector((0, 0, 0, 1))
#     p1 = P @ e1
#     p1 /= p1[2]
#     print("Projected e1")
#     print(p1)
#     print("proj by object_utils")
#     print(project_by_object_utils(cam, Vector(e1[0:3])))
#     p2 = P @ e2
#     p2 /= p2[2]
#     print("Projected e2")
#     print(p2)
#     print("proj by object_utils")
#     print(project_by_object_utils(cam, Vector(e2[0:3])))
#     p3 = P @ e3
#     p3 /= p3[2]
#     print("Projected e3")
#     print(p3)
#     print("proj by object_utils")
#     print(project_by_object_utils(cam, Vector(e3[0:3])))
#     pO = P @ O
#     pO /= pO[2]
#     print("Projected world origin")
#     print(pO)
#     print("proj by object_utils")
#     print(project_by_object_utils(cam, Vector(O[0:3])))
#     # Bonus code: save the 3x4 P matrix into a plain text file
#     # Don't forget to import numpy for this
#     nP = numpy.matrix(P)
#     numpy.savetxt("/tmp/P3x4.txt", nP)  # to select precision, use e.g. fmt='%.2f'