from .base_options import BaseOptions


class PreprocessOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--static_frames", default=None, help="list of imgs to discard for being static, if not set will discard them based on speed (careful, on KITTI some frames have incorrect speed)")
        self.parser.add_argument("--depth", type=str, default=None, choices=["", "sparse", "dense"], help="If available (e.g. with KITTI), will store depth ground truth along with images, for validation")
        self.parser.add_argument("--with_pose", action='store_true', help="If available (e.g. with KITTI), will store pose ground truth along with images, for validation")
        self.parser.add_argument("--no_train_gt", action='store_true', help="If selected, will delete ground truth depth to save space")
        self.parser.add_argument("--dump_root", type=str, default='dump', help="Where to dump the data")
        self.parser.add_argument("--height", type=int, default=128, help="image height")
        self.parser.add_argument("--width", type=int, default=416, help="image width")
        self.parser.add_argument("--depth_size_ratio", type=int, default=1, help="will divide depth size by that ratio")
        self.parser.add_argument("--num_threads", type=int, default=4, help="number of threads to use")
        self.isTrain = False