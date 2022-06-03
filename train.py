import os
from copy import deepcopy

import pandas as pd
import torch
import torch.optim
import torch.utils.data

from datasets.dataset_loader import CreateDataset
from model.base_model import BaseModel
from options.train_options import TrainOptions
from util_skip.logger import Logger
from util_skip.util import fix_random
from util_skip.visualizer import Visualizer
from tensorboardX import SummaryWriter
import wandb

best_result = -1

def main():
    global best_result, parser_config
    args = deepcopy(parser_config)
    fix_random(args.seed)

    # Data
    if args.dataset == 'kitti':
        args.max_depth = 80.0
    elif args.dataset == 'shapenet':
        args.max_depth = 1.75

    train_ds = CreateDataset(args, train=True)
    val_ds = CreateDataset(args, valid=True)

    print(f'{len(train_ds)} samples found in train split')
    print(f'{len(val_ds)} samples found in valid split')

    # Loaders
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=args.shuffle_batches,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    model = BaseModel(args)

    visualizer = Visualizer(args)

    if args.tensorboard:
        tb_writer = SummaryWriter()
    else: tb_writer = None
    if args.wandb:
        wandb.init(project="TwoWaySinth", entity="johnminelli")
        if args.sweep_id is not None: args.__dict__.update(wandb.config)
        wandb.config = args
        wandb.log({"params": wandb.Table(data=pd.DataFrame({k: [v] for k, v in vars(args).items()}))})
    train_logger = Logger(mode="train", n_epochs=args.epochs, data_size=len(train_loader), terminal_print_freq=args.print_freq, display_freq=args.display_freq, tensorboard=tb_writer, visualizer=visualizer, wand=args.wandb)
    valid_logger = Logger(mode="valid", n_epochs=args.epochs, data_size=len(val_loader), terminal_print_freq=-1, display_freq=args.display_freq, tensorboard=tb_writer, visualizer=visualizer, wand=args.wandb)

    for epoch in range(model.start_epoch, args.epochs):
        if not model.nvs_mode and (model.start_epoch > 0 or (epoch > 0 and ((args.validate and valid_metrics[3] < 0.055) or (not args.validate and train_metrics[3] < 0.055)))):
            model.nvs_mode = True

        # train for one epoch
        train_time, train_loss, train_metrics = run_model(train_loader, model, train_logger.epoch_start(epoch))

        # validate
        if args.validate:
            model.switch_mode('eval')

            with torch.no_grad():
                valid_time, valid_loss, valid_metrics = run_model(val_loader, model, valid_logger.epoch_start(epoch))

            model.switch_mode('train')

            # choose the most relevant as measure of performance to determine the best model: NVS_L1 is used
            # /* note that some measures are to maximize (e.g. a1,a2,a3) */
            ref_metric = valid_metrics[0]
            if best_result < 0:
                best_result = ref_metric
            elif ref_metric < best_result:
                best_result = ref_metric
                model.save(epoch, os.path.join(args.save_path, args.name))

        train_logger.epoch_stop()
        model.update_learning_rate()
    train_logger.progress_bar.finish()
    # final save
    model.save(args.epochs, os.path.join(args.save_path, args.name))


def run_model(dataloader, model, logger):
    """
    Train or eval for an epoch the model.
    :param dataloader: data to iterate
    :param model: model to use
    :param logger: logger object to record times and outputs
    :return: the average of gathered (times, losses, metrics)
    """
    for i, data in enumerate(dataloader):
        current_batch_size = data["A"].size(0)
        if current_batch_size == 1: break

        model.set_input(data)
        model.forward()

        logger.display_results(i, model.get_current_visuals())

        model.optimize_parameters()

        logger.epoch_step(i, current_batch_size=current_batch_size, errors=model.get_current_errors(), metrics=model.get_current_metrics())

    # logger.anim(model.get_current_anim())
    avg_time, avg_loss, avg_metrics = logger.epoch_stop()
    return avg_time, avg_loss, avg_metrics


if __name__ == '__main__':
    global parser_config
    parser_config = TrainOptions().parse()
    fix_random(parser_config.seed)
    if parser_config.sweep_id is not None:
        wandb.agent(parser_config.sweep_id, main)
    else:
        main()

    # To get a SWEEP ID:
    #   sweep_configuration = {
    #       "name": "my-awesome-sweep", "metric": {"name": "accuracy", "goal": "maximize"}, "method": "grid",
    #       "parameters": {"a": {"values": [1, 2, 3, 4]}}
    #   }
    #   print(wandb.sweep(sweep_configuration))
    #
    # Or from CommandLine:
    #   wandb sweep config.yaml
    #
    # Or from web interface
