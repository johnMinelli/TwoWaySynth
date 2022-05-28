import torch
import torch.optim
import torch.utils.data

from datasets.dataset_loader import CreateDataset
from models_skip.base_model import BaseModel
from options.eval_options import EvalOptions
from util_skip.logger import Logger, AverageMeter
from util_skip.util import fix_random
from util_skip.visualizer import Visualizer


def main():
    args = EvalOptions().parse()
    fix_random(args.seed)

    # Data
    if args.dataset == 'kitti':
        args.max_depth = 80.0
    elif args.dataset == 'shapenet':
        args.max_depth = 1.75

    eval_ds = CreateDataset(args, eval=True)
    print(f'{len(eval_ds)} samples found for evaluation')

    # Loaders
    data_loader = torch.utils.data.DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    visualizer = Visualizer(args)
    metrics = AverageMeter(precision=4)

    model = BaseModel(args)

    model.switch_mode('eval')
    for i, data in enumerate(data_loader):
        current_batch_size = data["A"].size(0)
        if current_batch_size == 1: break

        model.set_input(data)
        model.forward()

        metrics.update(list(model.get_current_metrics().items()), current_batch_size)
        visualizer.display_current_results(model.get_current_visuals(), i, True)
        visualizer.reset()
    avg_metrics = metrics.avg
    print(' * Avg Metrics : ' + ', '.join(["{}: {:.3f}".format(n, v) for n, v in zip(metrics.names, avg_metrics)]))

if __name__ == '__main__':
    main()
