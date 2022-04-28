import glob

from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch
import torchvision

from models_skip.network_utils.depth_decoder import DepthDecoder
from models_skip.network_utils.losses import depth_loss, get_depth_smoothness, photometric_reconstruction_loss
from models_skip.network_utils.metrics import compute_depth_metrics
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
        self.backup_dir = opt.save_path
        self.category = opt.dataset
        self.start_epoch = 0

        self.count = 0
        self.train_mode = True

        # Setup training devices
        if opt.gpu_ids[0] < 0 or not torch.cuda.is_available():
            print("Training on CPU")
            self.device = torch.device("cpu")
        else:
            print("Training on GPU")
            if len(opt.gpu_ids) > 1:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu_ids)[1:-1]
            # self.device = torch.device('cuda:%d' % opt.gpu_ids[0])
            self.device = torch.device("cuda")

        # self.enc = ResnetEncoder(18, 600, pretrained=True).to(self.device)
        # self.dec = networks.Decoder(output_nc=3, nz=opt.nz_geo * 3).to(self.device)
        # self.depthdec = DepthDecoder(self.enc.num_ch_enc, 600, [0, 1, 2, 3]).to(self.device)
        # self.vgg = VGGPerceptualLoss().to(self.device)

        self.enc = nn.DataParallel(networks.Encoder(input_nc=3, nz=opt.nz_geo * 3)).to(self.device)
        self.dec = nn.DataParallel(networks.Decoder(output_nc=3, nz=opt.nz_geo * 3)).to(self.device)
        self.depthdec = nn.DataParallel(networks.depthDecoder(output_nc=1, nz=opt.nz_geo * 3)).to(self.device)
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
            # define loss functions
            self.old_lr = opt.lr
            # initialize optimizers
            self.schedulers,self.optimizers = [],[]
            self.optimizer_G = torch.optim.Adam(itertools.chain(*param_list), lr=opt.lr, betas=(opt.momentum, opt.beta), weight_decay=opt.weight_decay)

            self.optimizers.append(self.optimizer_G)

            #for optimizer in self.optimizers:
            #    self.schedulers.append(networks.get_scheduler(optimizer, opt))

        if opt.dataset == 'kitti':
            intrinsics = np.array(
                [718.9, 0., 128, \
                 0., 718.9, 128, \
                 0., 0., 1.]).reshape((3, 3))
            self.depth_bias = 0
            self.depth_scale = 1
            self.depth_scale_vis = 250. / self.depth_scale
            self.depth_bias_vis = 0.
            self.intrinsics = torch.Tensor(intrinsics.astype(np.float32)).cuda().unsqueeze(0)

        elif opt.dataset == 'shapenet':
            # intrinsics = np.array([280, 0, 128,
            #                        0, 280, 128,
            #                        0, 0, 1]).reshape((3, 3))
            self.depth_bias, self.depth_scale = 2, 1.
            self.depth_scale_vis = 255. / self.depth_scale
            self.depth_bias_vis = self.depth_bias - self.depth_scale
            # self.intrinsics = torch.Tensor(intrinsics).float().to(self.device).unsqueeze(0)


    def set_input(self, input):
        self.real_A = Variable(input['A'].to(self.device))
        self.real_depth_A = Variable(input['DA'].to(self.device))
        self.real_B = Variable(input['B'].to(self.device))
        self.real_depth_B = Variable(input['DB'].to(self.device)) # TODO check if *self.depth_scale+self.depth_bias should be moved here
        self.real_RT = Variable(input['RT'].squeeze().to(self.device))
        self.intrinsics = input['I'][:,:,:3].to(self.device)
        self.scale_factor = input['S'].numpy()
        self.batch_size = self.real_A.size(0)

    def forward(self):
        self.z_a, self.conv0, self.conv2, self.conv3, self.conv4 = self.encode(self.real_A)

        self.z_a2b = self.transform(self.z_a, self.real_RT)
        # get depth from latent
        self.depth_a = self.depthdecode(self.z_a)  # for loss
        self.depth_a2b = self.depthdecode(self.z_a2b)  # TODO add skip and use DA
        # warp with real and fake depth
        self.warp_real_B, _, _ = inverse_warp(self.real_A, self.real_depth_B*self.depth_scale+self.depth_bias, self.real_RT, self.intrinsics)
        self.warp_fake_B, _, _ = inverse_warp(self.real_A, self.depth_a2b, self.real_RT, self.intrinsics)

        # _, self.conv0_w, self.conv2_w, self.conv3_w, self.conv4_w = self.encode(self.warp_fake_B)  does it make sense?
        self.fake_A = self.decode(self.z_a, self.conv0, self.conv2, self.conv3, self.conv4)[0]  # for visualization


        #warp features for skip connections
        self.conv0_tf, _, _ = inverse_warp(self.conv0, self.depth_a2b,
                                           self.real_RT, self.intrinsics)

        self.conv2_tf, _, _ = inverse_warp(self.conv2, torch.nn.functional.upsample(self.depth_a2b, scale_factor=0.25),
                                           self.real_RT, self.scale_K(self.intrinsics, 0.25))

        self.conv3_tf, _, _ = inverse_warp(self.conv3, torch.nn.functional.upsample(self.depth_a2b, scale_factor=0.125),
                                           self.real_RT, self.scale_K(self.intrinsics, 0.125))

        self.conv4_tf, _, _ = inverse_warp(self.conv4, torch.nn.functional.upsample(self.depth_a2b, scale_factor=0.0625),
                                           self.real_RT, self.scale_K(self.intrinsics, 0.0625))

        # decode new view with latent transformed and warped features
        self.fake_B, self.fake_B3, self.fake_B2, self.fake_B1 = self.decode(self.z_a2b,
                                                                            self.conv0_tf,
                                                                            self.conv2_tf,
                                                                            self.conv3_tf,
                                                                            self.conv4_tf)

    def forward_res(self):
        self.z_features = self.encode(self.real_A)

        self.z_features_a2b = self.transform_resnet(self.z_features, self.real_RT)
        # get depth from latent
        self.depth_scales_a2b = self.depthdecode(self.z_features_a2b)

        # warp depth scales
        # use 'self.depth_scales_a2b' to warp 'self.z_features' into 'self.z_features_tf'

        self.fake_B, self.fake_B3, self.fake_B2, self.fake_B1 = self.decode(self.z_a2b, self.z_features_tf)

    def encode(self, image_tensor):
        return self.enc(image_tensor)

    def transform(self,z,RT):
        return networks.transform_code(z, self.opt.nz_geo, RT.inverse(), object_centric=self.opt.dataset in ['shapenet'])

    def transform_resnet(self, z_features, RT):
        return [self.transform(z, RT) for z in z_features]

    def decode(self,z_a, conv0, conv2, conv3, conv4):
        output = self.dec(z_a, conv0, conv2, conv3, conv4)
        return [torch.tanh(out) for out in output]

    def depthdecode(self,z):
        output = self.depthdec(z)
        if self.opt.dataset in ['kitti']:
            return 1 / (10 * torch.sigmoid(output) + 0.01)  # predict disparity instead of depth for natural scenes
        elif self.opt.dataset in ['shapenet']:
            return torch.tanh(output) * self.depth_scale + self.depth_bias

    def backward_G(self):

        # https://www.sciencedirect.com/science/article/pii/S0923596508001197
        # multiscale reconstruction loss
        self.loss_reco = F.l1_loss(self.fake_B,self.real_B) \
                 + 0.5*F.l1_loss(torch.nn.functional.upsample(self.fake_B3, scale_factor=2),self.real_B) \
                 + 0.2*F.l1_loss(torch.nn.functional.upsample(self.fake_B2, scale_factor=4),self.real_B) \
                 + 0.1*F.l1_loss(torch.nn.functional.upsample(self.fake_B1, scale_factor=8), self.real_B)
        # image with depth quality
        self.loss_depth_smooth = get_depth_smoothness(self.depth_a2b, self.real_B) \
                                 + get_depth_smoothness(self.depth_a, self.real_A)

        # transformation/warping quality  # TODO equivalent to multiscale version 'photometric_reconstruction_loss'
        # scaled_A = self.scale_image(self.real_A, self.scale_factor, self.scale_factor)
        # self.loss_warp = F.l1_loss(self.warp_fake_B[scaled_A>0].requires_grad_(), self.real_B[scaled_A>0])
        self.loss_warp, warped, diff = photometric_reconstruction_loss(self.real_B, self.real_A, self.intrinsics, self.depth_a2b, self.real_RT)
        # self.loss_warp = F.l1_loss(self.warp_fake_B, self.real_B)
        # encoder quality and depth_decoder quality  # TODO to enable if we want to use the gt_depth
        self.loss_depth = 0  # depth_loss(self.real_depth_B, self.real_flow_B, self.depth_a2b, self.fake_flow_B)
        # F.l1_loss(self.depth_a2b, self.depth_b)
        # VGG perceptual loss
        self.vgg_loss = self.vgg(self.fake_B, self.real_B)

        self.loss_G = (self.loss_reco+self.loss_warp) * self.opt.lambda_recon  # 10.0
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
            self.count += 1

    def get_current_visuals(self):
        """
        Returns the images computed from current inputs.
        """
        return OrderedDict({'real_A': tensor2im(self.real_A.data[0]),
                            'real_B': tensor2im(self.real_B.data[0]),
                            'warp_B_fake': tensor2im(self.warp_fake_B.data[0]),
                            'depth_B': tensor2im(self.depth_a2b.data[0]),
                            'warp_B_real': tensor2im(self.warp_real_B.data[0]),
                            'real_depth_B': tensor2im(self.real_depth_B.data[0]),
                            'fake_A': tensor2im(self.fake_A.data[0]),
                            'fake_B': tensor2im(self.fake_B.data[0]),
                            })

    def get_current_errors(self):
        """
        Returns the computed losses over the current inputs.
        If the model is in not in train mode the call returns a null value.
        """
        return OrderedDict({
                            'loss_reco': self.opt.lambda_recon*self.loss_reco.item(),
                            'loss_vgg': self.opt.lambda_vgg*self.vgg_loss.item(),
                            'loss_warp': self.opt.lambda_recon*self.loss_warp.item(),
                            'loss_smooth': self.opt.lambda_smooth*self.loss_depth_smooth.item(),
                            'loss_depth': 0,  # self.opt.lambda_depth*self.loss_depth.item(),
                            }) if self.train_mode else None

    def get_current_metrics(self):
        """
        Returns the metrics for the current inputs.
        """

        return OrderedDict({
                            'SSIM': ssim((self.fake_B + 1) / 2, (self.real_B + 1) / 2).item(),
                            'L1': F.l1_loss(self.depth_a2b, self.real_depth_B).item()
                            },**compute_depth_metrics(self.real_depth_B, self.depth_a2b))

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

    def get_RT(self, vec):
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

    def scale_K(self, intrinsics, scale):
        #scale fx, fy, cx, cy according to the scale

        K = self.intrinsics.clone()
        K[:, 0, 0] *= scale
        K[:, 1, 1] *= scale
        K[:, 0, 2] *= scale
        K[:, 1, 2] *= scale
        return K

    def scale_image(self, img, scale_x, scale_y):
        scale_mat = torch.eye(3).unsqueeze(0).repeat(self.batch_size, 1, 1).numpy()
        scale_mat[:, 0, 0] = scale_x
        scale_mat[:, 0, 2] = (self.real_A.size()[3] - (self.real_A.size()[3] * scale_x)) / 2
        scale_mat[:, 1, 1] = scale_y
        scale_mat[:, 1, 2] = (self.real_A.size()[2] - (self.real_A.size()[2] * scale_y)) / 2
        scaled_image = torch.from_numpy(np.array([warp(tensor2im(im), np.linalg.inv(sc)) for im, sc in
                                              zip(self.real_A, scale_mat)]).transpose((0, 3, 1, 2)))
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
    def save_network(self, network, network_label, epoch_label, save_dir=None):
        save_filename = '{:04}_net_{}.pth'.format(epoch_label, network_label)
        if save_dir is None: save_dir = self.backup_dir
        save_path = os.path.join(save_dir, save_filename)
        torch.save(network.state_dict(), save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, load_dir=None):
        if load_dir is None: load_dir = self.backup_dir
        if epoch_label == -1:
            load_filename = '*_net_%s.pth' % (network_label)
            load_path = Path(sorted(glob.glob(os.path.join(load_dir, load_filename)))[-1])
        else:
            load_filename = '%s_net_%s.pth' % (epoch_label, network_label)
            load_path = os.path.join(load_dir, load_filename)
        network.load_state_dict(torch.load(load_path))
        return int(load_path.name.split('_')[0])

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        #for scheduler in self.schedulers:
        #    scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


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
            loss += torch.nn.functional.l1_loss(x, y)
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