import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, random_split
from models.hour_glass import StackHourglass
from datas.my_dataset import ToothDataSet, get_peak_points
import os
import shutil

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
        shutil.copyfile(filename, 'model_best.pt')


def save(model, epoch, exp, is_best=False):
    resume = os.path.join('exp', exp)
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


if __name__ == "__main__":
    print("------------")
    # config
    # device epoch batchsize  continue
    device = "cpu"
    epoch_num = 50
    save_freq = 10
    start_epoch = 0
    batch_size = 4
    lr = 0.001
    train_img_dir = r"E:\data\SplitTooth\AddFDIClassAndKeyPoint\keyPoint\black"
    txt_file = r"E:\data\SplitTooth\AddFDIClassAndKeyPoint\keyPoint\kp_label.txt"
    train_ratio = 0.9
    pre_trained_model = "./exp/tooth/checkpoint50.pt"

    torch.manual_seed(0)
    # model
    model = StackHourglass(4, 256, 3, 4)
    model.float().to(device)

    # loss
    criterion = torch.nn.MSELoss()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lambda1 = lambda e: 0.99 ** (e % 10)  # adjust learning rate for every 10 epochs
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])
    # data
    dataset = ToothDataSet(train_img_dir, txt_file, 256)
    n_train = int(train_ratio * len(dataset))
    n_val = len(dataset) - n_train
    print("data number: {}, train: {}, val: {}".format(len(dataset), n_train, n_val))
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size, True)
    val_loader = DataLoader(val_dataset, batch_size, False)

    if pre_trained_model != "":
        checkpoint = torch.load(pre_trained_model)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

    for epoch in range(start_epoch, epoch_num):
        print("Lr:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        # train
        model.train()
        for i, (inputs, target_kps, target_heatmaps) in enumerate(train_loader):
            # print(i, inputs.shape, target_kps.shape, target_heatmaps.shape)
            inputs = Variable(inputs).to(device)
            target_heatmaps = Variable(target_heatmaps).to(device)
            mask, indices_valid = calculate_mask(target_heatmaps, device)
            # print(mask.shape, indices_valid)
            optimizer.zero_grad()
            outputs = model(inputs) * mask
            # print("outputs: *********** ", outputs.shape)
            target_heatmaps = target_heatmaps * mask
            loss = criterion(outputs[:, -1], target_heatmaps)   # just compare last output
            loss.backward()
            optimizer.step()

            print('[ Train Epoch {:005d} -> {:005d} / {} ] loss : {:15} '.format(
                epoch, i, len(train_loader), loss.item()))

        scheduler.step()

        # val
        model.eval()
        with torch.no_grad():
            val_loss = 0.0

            for i, (inputs, target_kps, target_heatmaps) in enumerate(val_loader):
                inputs = Variable(inputs).to(device)
                target_heatmaps = Variable(target_heatmaps).to(device)
                mask, indices_valid = calculate_mask(target_heatmaps, device)
                outputs = model(inputs)
                # print("outputs: *********** ", outputs.shape)
                all_peak_points = get_peak_points(outputs[:, -1].cpu().data.numpy())
                val_loss += get_mse(all_peak_points, target_kps.numpy(), indices_valid)

            val_loss /= len(val_loader)
            print('******* val  loss : {:15} '.format(val_loss))

        if (epoch+1) % save_freq == 0 or epoch == epoch_num - 1:
            save(model, epoch + 1, "tooth", is_best=False)
