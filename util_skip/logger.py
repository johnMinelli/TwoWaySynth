import random

import numpy as np
import progressbar
import sys
import time

from util_skip.util import tensor2im


class Logger(object):
    def __init__(self, mode, n_epochs, data_size, terminal_print_freq=1, display_freq=1, tensorboard=None, wand=None, visualizer=None):
        self.n_epochs = n_epochs
        self.data_size = data_size
        self.print_freq = terminal_print_freq
        self.display_freq = display_freq
        self.tensorboard = tensorboard
        self.wandb = wand
        self.visualizer = visualizer

        s = 10
        e = 1   # epoch bar position
        tr = 3  # train bar position
        ts = 6  # valid bar position
        h = 100

        self.progress_bar = None
        self.epoch_bar = None
        self.epoch = None
        self.t = None

        if mode == "valid":
            self.prefix = "Valid"
            self.writer = Writer(self.t, (0, h - s + ts))
            self.bar_writer = Writer(self.t, (0, h - s + ts + 1))
        elif mode == "train":
            self.prefix = "Train"
            self.writer = Writer(self.t, (0, h - s + tr))
            self.bar_writer = Writer(self.t, (0, h - s + tr + 1))
            self.progress_bar = progressbar.ProgressBar(maxval=n_epochs, fd=Writer(self.t, (0, h - s + e)))
            [print('') for i in range(2)]
            self.progress_bar.start()

    def set_tensorboard(self, writer):
        self.tensorboard = writer

    def set_wandb(self, writer):
        self.wandb = writer

    def set_visualizer(self, display, display_freq):
        self.visualizer = display
        self.display_freq = display_freq

    def anim(self, vis):
        self.visualizer.display_current_anim(vis, self.epoch)

    def log(self, text):
        self.writer.write(text)

    def epoch_start(self, epoch):
        self.epoch = epoch
        self.total_steps = epoch * self.data_size
        self.step_time = AverageMeter()
        self.losses = AverageMeter(precision=4)
        self.metrics = AverageMeter(precision=4)
        self.epoch_start_time = time.time()
        self.chronometer = time.time()
        self.epoch_bar = progressbar.ProgressBar(maxval=self.data_size, fd=self.bar_writer)
        self.epoch_bar.start()
        return self

    def epoch_step(self, step, current_batch_size, errors, metrics):
        if self.epoch is not None:
            # losses error
            if errors is not None:
                for name, error in errors.items():
                    self.tensorboard.add_scalar(self.prefix+"/"+name, error, self.total_steps + step)
                self.losses.update(list(errors.items()), current_batch_size)
            # metrics
            for name, error in metrics.items():
                self.tensorboard.add_scalar(self.prefix+"/"+name, error, self.total_steps + step)
            self.metrics.update(list(metrics.items()), current_batch_size)
            # time
            self.step_time.update(time.time() - self.chronometer, current_batch_size)
            # console prints
            self.epoch_bar.update(step + 1)
            if step % self.print_freq == 0:
                self.writer.write('{}[{}/{}]: Time {},  Loss  {}'.format(self.prefix, self.epoch, self.n_epochs, self.step_time, self.losses))

            self.chronometer = time.time()

    def epoch_stop(self):
        if self.epoch is not None:
            self.epoch_bar.finish()
            self.writer.write('End of epoch %d / %d \t Time Taken: %d sec' % (self.epoch, self.n_epochs, time.time() - self.epoch_start_time))

            avg_time = self.step_time.avg[0]
            avg_losses = self.losses.avg[0] if len(self.losses.avg) > 0 else 0
            avg_metrics = self.metrics.avg[0]
            self.log(' * Avg Loss : {:.3f} - Avg Metrics : {:.3f} - Avg Time : {:.3f}'.format(avg_losses, avg_metrics, avg_time))

            self.progress_bar.update(self.epoch + 1)
            if self.epoch + 1 == self.n_epochs:
                self.progress_bar.finish()

            self.step_time.avg = None
            self.losses.avg = None
            self.metrics.avg = None
            self.epoch = None
            return avg_time, avg_losses, avg_metrics

    def display_results(self, step, images):
        if step % self.display_freq == 0:
            self.visualizer.display_current_results(images, self.total_steps+step, False)
            for name, image in images.items():
                self.tensorboard.add_image('{} {}'.format(self.prefix, name), image, self.total_steps+step, dataformats='HWC')

class Writer(object):
    """Create an object with a write method that writes to a
    specific place on the screen, defined at instantiation.

    This is the glue between blessings and progressbar.
    """

    def __init__(self, t, location):
        """
        Input: location - tuple of ints (x, y), the position
                        of the bar in the terminal
        """
        self.location = location
        self.t = t

    def write(self, string):
        print(string)

    def flush(self):
        return


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, n_meters=0, precision=3):
        self.n_meters = n_meters
        self.precision = precision
        self.reset(self.n_meters)

    def reset(self, i):
        self.val = [0]*i
        self.avg = [0]*i
        self.sum = [0]*i
        self.names = [""]*i
        self.count = 0

    def update(self, val, n=1):
        # check input and correct init
        if not isinstance(val, list):
            val = [val]
        if self.n_meters == 0:
            self.n_meters = len(val)
            self.reset(self.n_meters)
        assert(len(val) == self.n_meters)

        # update
        self.count += n
        for i,v in enumerate(val):
            if isinstance(v, tuple):
                self.names[i] = v[0]
                v = v[1]
            self.val[i] = v
            self.sum[i] += v * n
            self.avg[i] = self.sum[i] / self.count

    def __repr__(self):
        val = ', '.join(['{} {:.{}f} ({:.{}f})'.format(n, v, self.precision, a, self.precision)
                         for n,v,a in zip(self.names, self.val, self.avg)])
        return val


