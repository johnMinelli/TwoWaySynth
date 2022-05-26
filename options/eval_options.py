from .base_options import BaseOptions


class EvalOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--models_path', type=str, required=True, default='./models_ckp/shapenet', help='path where models are stored')
        self.parser.add_argument('--model_epoch', type=int, required=True, default='-1', help='which epoch of the model to load from save_path. If set to -1 load the latest model')
        self.parser.add_argument("--test_file", type=str, required=True, help='file with the pairs of the test split')
        self.parser.add_argument("--depth", type=str, default=None, choices=["", "sparse", "dense"], help="If available (e.g. with KITTI), will use depth ground truth in evaluation")
        self.isTrain = False
