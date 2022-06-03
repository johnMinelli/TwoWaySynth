import progressbar
import time
import wandb
last_step = 0

class Logger(object):
    global last_step

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
            self.log_only_at_end = True
            self.writer = Writer(self.t, (0, h - s + ts))
            self.bar_writer = Writer(self.t, (0, h - s + ts + 1))
        elif mode == "train":
            self.prefix = "Train"
            self.log_only_at_end = False
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
        self.visualizer.display_current_anim(vis, self.epoch, True)

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
            # losses error & metrics
            self.losses.update(list(errors.items()), current_batch_size)
            self.metrics.update(list(metrics.items()), current_batch_size)
            if not self.log_only_at_end:
                current_steps = self.total_steps + step
                self._log_stats_to_dashboards(current_steps, errors)
                self._log_stats_to_dashboards(current_steps, metrics)
                global last_step
                last_step = current_steps
            # time
            self.step_time.update(time.time() - self.chronometer, current_batch_size)
            # console prints
            self.epoch_bar.update(step + 1)
            if self.print_freq > 0 and step % self.print_freq == 0:
                self.log('{}[{}/{}]: Time {}'.format(self.prefix, self.epoch, self.n_epochs, self.step_time))
                if str(self.losses) != '': self.log('\tLoss  {}'.format(self.losses))

            self.chronometer = time.time()

    def epoch_stop(self):
        if self.epoch is not None:
            self.epoch_bar.finish()
            self.log('End of epoch %d / %d \t Time Taken: %d sec' % (self.epoch, self.n_epochs, time.time() - self.epoch_start_time))

            avg_time = self.step_time.avg[0]
            avg_losses = self.losses.avg
            avg_metrics = self.metrics.avg

            self.log(' * Avg Metrics : '+', '.join(["{}: {:.3f}".format(n,v) for n,v in zip(self.metrics.names, avg_metrics)])+' - Avg Time : {:.3f}'.format(avg_time)+'\n\n')
            if self.progress_bar is not None:
                self.progress_bar.update(self.epoch + 1)
                if self.epoch + 1 == self.n_epochs:
                    self.progress_bar.finish()
            if self.log_only_at_end:
                global last_step
                self._log_stats_to_dashboards(last_step, dict(zip(self.losses.names, avg_losses)))
                self._log_stats_to_dashboards(last_step, dict(zip(self.metrics.names, avg_metrics)))
            if self.visualizer is not None:
                self.visualizer.reset()

            self.step_time.avg = None
            self.losses.avg = None
            self.metrics.avg = None
            self.epoch = None
            return avg_time, avg_losses, avg_metrics

    def display_results(self, step, images, save=False):
        if step % self.display_freq == 0:
            current_steps = self.total_steps + step
            self.visualizer.display_current_results(images, current_steps, save)
            if self.tensorboard is not None:
                for name, image in images.items():
                    self.tensorboard.add_image('{} {}'.format(self.prefix, name), image, current_steps, dataformats='HWC')
            # if self.wandb:
            #     wandb.log({'{}_{}'.format(self.prefix, name): wandb.Image(image, caption=name) for name, image in images.items()}, step)

    def _log_stats_to_dashboards(self, step, stats):
        for name, value in stats.items():
            namet = self.prefix + "/" + name
            namew = self.prefix + "/" + self.prefix.lower() + "_" + name
            if self.tensorboard is not None:
                self.tensorboard.add_scalar(namet, value, step)
            if self.wandb:
                wandb.log({namew: value}, step)

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


