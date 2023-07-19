
try:
    # from tensorboardX import SummaryWriter     # pytorch version lower than 1.2.0
    from torch.utils.tensorboard import SummaryWriter
except ImportError as error:
    print('tensorboard not installed, visualizing wont be available')
    SummaryWriter = None


class Writer:
    def __init__(self, opt):
        self.opt = opt

        if opt["vis"] and SummaryWriter is not None:
            self.display = SummaryWriter(log_dir=opt["log"])  # example: './exp/tooth/log'
        else:
            self.display = None

    def plot_train_loss(self, loss, epoch, i, n):
        iters = i + (epoch - 1) * n
        if self.display and self.opt["show_train_loss"]:
            self.display.add_scalar('data/train_loss', loss, iters)

    def plot_val_loss(self, loss, epoch):
        if self.display and self.opt["show_val_loss"]:
            self.display.add_scalar('data/val_loss', loss, epoch)

    def plot_acc(self, acc, epoch):
        if self.display:
            self.display.add_scalar('data/test_acc', acc, epoch)

    def plot_model_wts(self, net, epoch):
        if self.display and self.opt["show_model_weights"]:
            for name, param in net.named_parameters():
                self.display.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    def close(self):
        if self.display is not None:
            self.display.close()
