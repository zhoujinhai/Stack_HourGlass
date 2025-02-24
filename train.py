import os
import shutil
from pprint import pprint
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from models.hour_glass import StackHourglass
from datas.my_dataset import ToothDataSet, get_peak_points
from config.config import get_config
from visualizer.writer import Writer

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True


def calculate_mask(heatmaps_targets, device="cpu"):
    """

    :param heatmaps_targets: Variable (N,15,96,96)
    :return: Variable (N,15,96,96)
    """
    N, C, H, W = heatmaps_targets.size()
    # print("N, C, H, W: ", N, C, H, W)
    N_idx = []
    C_idx = []
    for n in range(N):
        for c in range(C):
            # print(heatmaps_targets[n, c, :, :].max().data.item())
            max_v = heatmaps_targets[n, c, :, :].max().data.item()
            if max_v != 0.0:
                N_idx.append(n)
                C_idx.append(c)
    mask = Variable(torch.zeros(heatmaps_targets.size()))
    mask[N_idx, C_idx, :, :] = 1.
    mask = mask.float().to(device)
    return mask, [N_idx, C_idx]


def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    """
    from pytorch/examples
    """
    basename = os.path.dirname(filename)
    if not os.path.exists(basename):
        os.makedirs(basename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(basename, 'model_best.pt'))


def save(model, epoch, exp, is_best=False):
    resume = os.path.join('./exp', exp)
    resume_file = os.path.join(resume, 'checkpoint.pt')

    save_checkpoint({
            'state_dict': model.state_dict(),
            # 'optimizer': config['train']['optimizer'].state_dict(),
            'epoch': epoch,
        }, is_best, filename=resume_file)
    print('=> save checkpoint')


def get_mse(pred_points, gts, indices_valid=None):
    """

    :param pred_points: numpy (N,3,2)
    :param gts: numpy (N,3,2)
    :return:
    """
    pred_points = pred_points[indices_valid[0], indices_valid[1], :]
    gts = gts[indices_valid[0], indices_valid[1], :]
    pred_points = Variable(torch.from_numpy(pred_points).float(), requires_grad=False)
    gts = Variable(torch.from_numpy(gts).float(), requires_grad=False)
    criterion = torch.nn.MSELoss()
    loss = criterion(pred_points, gts)
    return loss


def train(config):
    # writer
    writer = Writer(config)

    # model
    model = StackHourglass(config["n_stack"], config["in_dim"], config["n_kp"], config["n_hg_layer"])
    model.float().to(config["device"])
    # loss
    criterion = torch.nn.MSELoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    lambda1 = lambda e: 0.99 ** (e / 10)  # adjust learning rate 
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
    # data
    dataset = ToothDataSet(config["img_dir"], config["label_file"], config["in_dim"], config["out_dim"], config["n_kp"])
    n_train = int(config["train_ratio"] * len(dataset))
    n_val = len(dataset) - n_train
    print("data number: {}, train: {}, val: {}".format(len(dataset), n_train, n_val))
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, config["batch_size"], True)
    val_loader = DataLoader(val_dataset, config["batch_size"], False)
    # print(config["pre_trained_model"], os.path.isfile(config["pre_trained_model"]))
    if config["pre_trained_model"] != "" and os.path.isfile(config["pre_trained_model"]):
        checkpoint = torch.load(config["pre_trained_model"])
        config["start_epoch"] = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        print("Load {} successed!, start epoch{}".format(config["pre_trained_model"], checkpoint['epoch']))

    print("=========Training===========")
    min_loss = 9999999999.0
    for epoch in range(config["start_epoch"], config["epoch_num"]):
        print("Lr:{}， scheduler lr: {}".format(optimizer.state_dict()['param_groups'][0]['lr'], scheduler.get_lr()))
        # train
        model.train()
        for i, (inputs, target_kps, target_heatmaps) in enumerate(train_loader):
            # print(i, inputs.shape, target_kps.shape, target_heatmaps.shape)
            inputs = Variable(inputs).to(config["device"])
            target_heatmaps = Variable(target_heatmaps).to(config["device"])
            mask, indices_valid = calculate_mask(target_heatmaps, config["device"])
            optimizer.zero_grad()
            outputs = model(inputs)
            # print("outputs: *********** ", outputs.shape)
            last_out = outputs[:, -1] * mask
            target_heatmaps = target_heatmaps * mask
            loss = criterion(last_out, target_heatmaps)  # just compare last output
            loss.backward()
            optimizer.step()
            writer.plot_train_loss(loss, epoch, i, len(train_loader))
            print('[ Train Epoch {:005d} -> {:005d} / {} ] loss : {:15} '.format(
                epoch, i, len(train_loader), loss.item()))

        scheduler.step()  # update learning rate

        # val
        model.eval()
        is_best = False
        with torch.no_grad():
            val_loss = 0.0

            for i, (inputs, target_kps, target_heatmaps) in enumerate(val_loader):
                inputs = Variable(inputs).to(config["device"])
                target_heatmaps = Variable(target_heatmaps).to(config["device"])
                mask, indices_valid = calculate_mask(target_heatmaps, config["device"])
                outputs = model(inputs)
                # print("outputs: *********** ", outputs.shape)
                all_peak_points = get_peak_points(outputs[:, -1].cpu().data.numpy())
                val_loss += get_mse(all_peak_points, target_kps.numpy(), indices_valid)

            val_loss /= len(val_loader)
            print('******* val  loss : {:15} '.format(val_loss))
            if ((epoch + 1) % config["save_freq"] == 0 or epoch == config["epoch_num"] - 1) and val_loss < min_loss:
                is_best = True
                min_loss = val_loss

            writer.plot_val_loss(val_loss, epoch)

        # save
        if (epoch + 1) % config["save_freq"] == 0 or epoch == config["epoch_num"] - 1:
            save(model, epoch + 1, config["exp"], is_best=is_best)
            writer.plot_model_wts(model, epoch)

    writer.close()


if __name__ == "__main__":
    torch.manual_seed(0)
    print("============Config==================")
    # config
    config = get_config()
    pprint(config)
    print("==============================")

    # train
    train(config)
