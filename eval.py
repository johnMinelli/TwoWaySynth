import torch
import torch.optim
import torch.utils.data
from tqdm import tqdm

from datasets.dataset_loader import CreateDataset
from logger.logger import AverageMeter
from logger.visualizer import Visualizer
from model.base_model import BaseModel
from model.network_utils.util import fix_random
from options.eval_options import EvalOptions


def main():
    args = EvalOptions().parse()
    fix_random(args.seed)

    # Data
    if args.dataset == 'kitti':
        args.max_depth = 80.0
    elif args.dataset == 'shapenet':
        args.max_depth = 4

    eval_ds = CreateDataset(args, eval=True)
    print(f'{len(eval_ds)} samples found for evaluation')

    # Loaders
    data_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True)

    visualizer = Visualizer(args)
    metrics = AverageMeter(precision=4)

    model = BaseModel(args)
    model.switch_mode('eval')
    for i, data in enumerate(tqdm(data_loader)):
        current_batch_size = data["A"].size(0)

        model.set_input(data)
        model.forward()

        metrics.update(list(model.get_current_metrics().items()), current_batch_size)

        # to save images
        visualizer.display_current_results(model.get_current_visuals(), i, False)
        visualizer.reset()
        # if i==5000: break
    avg_metrics = metrics.avg
    print(' * Avg Metrics : ' + ', '.join(["{}: {:.3f}".format(n, v) for n, v in zip(metrics.names, avg_metrics)]))


if __name__ == '__main__':
    main()
