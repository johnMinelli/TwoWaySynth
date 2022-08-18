from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # input settings
        self.parser.add_argument("--train_file", type=str, required=True, help='file with the pairs of the train split')
        self.parser.add_argument("--valid_file", type=str, required=True, help='file with the pairs of the test split')
        self.parser.add_argument("--gt_depth", action='store_true', help="Use depth ground truth to compute metrics")
        self.parser.add_argument('--max_az_distance', type=int, default=2, help='[ShapeNet] pairs of images will be created with a random image at a maximum azimuth distanca of max_az_distance.'
                                                                                  'The higher the value, the more rotation will occur between the images. Min 1.')
        self.parser.add_argument('--rand_el_distance', action='store_true', help='[ShapeNet] pairs of images will be created with a images at a random elevation distance.')
        self.parser.add_argument('--max_kitti_distance', type=int, default=5, help='[KITTI] pairs of images will be created with random images at a maximum range distance of max_kitti_distance. Min 3.')
        # output settings
        self.parser.add_argument('--display_freq', type=int, default=1, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=50, help='frequency of showing training results on console')
        self.parser.add_argument('--tensorboard', action='store_true', help='log stats on tensorboard local dashboard')
        self.parser.add_argument('--wandb', action='store_true', help='log stats on wandb dashboard')
        self.parser.add_argument('--sweep_id', type=str, help='sweep id for wandb hyperparameters search')

        self.parser.add_argument('--validate', action='store_true', help='evaluate the model on validation set during training')
        self.parser.add_argument('--continue_train', type=int, default=None, help='continue training: if set to -1 load the latest model from save_path')
        self.parser.add_argument("--n_targets", type=int, default=1, help="number of target images for each reference")
        self.parser.add_argument('--epochs', type=int, default=50, help='number of total epochs to run')
        self.parser.add_argument('--shuffle_batches', action='store_true', help='if true, takes images randomly to make batches otherwise takes them in order')
        self.parser.add_argument('--lr_niter_frozen', type=int, default=30, help='[lr_policy=lambda] # of iter at starting learning rate')
        self.parser.add_argument('--lr_niter_decay', type=int, default=30, help='[lr_policy=lambda] # of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate for adam')
        self.parser.add_argument('--momentum', default=0.5, type=float, help='momentum for sgd, alpha parameter for adam')
        self.parser.add_argument('--beta', default=0.999, type=float, help='beta parameters for adam')
        self.parser.add_argument('--weight_decay', default=0, type=float, help='weight decay for adam')
        self.parser.add_argument('--lr_policy', type=str, default='lambda', help='learning rate policy: lambda|step|plateau')
        self.parser.add_argument('--lr_decay_every', type=int, default=50, help='[lr_policy=step] multiply by a gamma by lr_decay_every iterations')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--lambda_recon', type=float, default=131.0, help='scale for reconstruction loss')
        self.parser.add_argument('--lambda_consistency', type=float, default=78.0, help='scale for skip features loss')
        self.parser.add_argument('--lambda_smooth', type=float, default=159.0, help='scale for edge smoothness loss')
        self.parser.add_argument('--lambda_vgg', type=float, default=56.0, help='scale for vgg perceptual loss')
        self.parser.add_argument('--fast_train', action='store_true', help='initally train NVSDecoder with GT features untill valid_depth_L1_direct stays under a threshold')

        self.isTrain = True