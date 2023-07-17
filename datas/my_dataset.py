import numpy as np
import os
import time
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class GenerateHeatmap(object):
    def __init__(self, output_res, num_parts, sigma=3):
        self.output_res = output_res
        self.num_parts = num_parts

        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape=(self.num_parts, self.output_res, self.output_res), dtype=np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(keypoints):
            if pt[0] > 0:
                x, y = int(pt[0] + 0.5), int(pt[1] + 0.5)
                if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                    continue
                ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                c, d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                a, b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                cc, dd = max(0, ul[0]), min(br[0], self.output_res)
                aa, bb = max(0, ul[1]), min(br[1], self.output_res)
                hms[idx, aa:bb, cc:dd] = np.maximum(hms[idx, aa:bb, cc:dd], self.g[a:b, c:d])
        return hms


def get_peak_points(heatmaps):
    """
    Args:
        heatmaps: (N, C, H, W)

    Returns:
        N, C, 2
    """
    N, C, H, W = heatmaps.shape
    all_peaks = []
    for i in range(N):
        peaks = []
        for j in range(C):
            y, x = np.unravel_index(heatmaps[i, j].argmax(), heatmaps[i, j].shape)
            peaks.append([x, y])
        all_peaks.append(peaks)
    return np.array(all_peaks)


class HeatmapParser:
    def __init__(self):
        from torch import nn
        self.pool = nn.MaxPool2d(3, 1, 1)

    def nms(self, det):
        maxm = self.pool(det)
        maxm = torch.eq(maxm, det).float()
        det = det * maxm
        return det

    def calc(self, det):
        with torch.no_grad():
            det = torch.autograd.Variable(torch.Tensor(det))
            # This is a better format for future version pytorch

        det = self.nms(det)  # N, C, H, W
        h = det.size()[2]
        w = det.size()[3]
        det = det.view(det.size()[0], det.size()[1], -1)
        val_k, ind = det.topk(1, dim=2)
        x = ind % w
        y = (ind / w).long()
        ind_k = torch.stack((x, y), dim=3)
        ans = {'loc_k': ind_k, 'val_k': val_k}
        ans = {key: ans[key].cpu().data.numpy() for key in ans}

        loc = ans['loc_k'][0, :, 0, :]
        val = ans['val_k'][0, :, :]
        ans = np.hstack((loc, val))
        ans = np.expand_dims(ans, axis=0)
        ret = []
        ret.append(ans)
        return ret

    def adjust(self, ans, det):
        for batch_id, people in enumerate(ans):
            for people_id, i in enumerate(people):
                for joint_id, joint in enumerate(i):
                    if joint[2] > 0:
                        y, x = joint[0:2]
                        xx, yy = int(x), int(y)
                        tmp = det[0][joint_id]
                        if tmp[xx, min(yy+1, tmp.shape[1]-1)] > tmp[xx, max(yy-1, 0)]:
                            y += 0.25
                        else:
                            y -= 0.25

                        if tmp[min(xx+1, tmp.shape[0]-1), yy] > tmp[max(0, xx-1), yy]:
                            x += 0.25
                        else:
                            x -= 0.25
                        ans[0][0, joint_id, 0:2] = (y+0.5, x+0.5)
        return ans

    def parse(self, det, adjust=True):
        ans = self.calc(det)
        if adjust:
            ans = self.adjust(ans, det)
        return ans


def denormalize(x_hat):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # x_hat = (x - mean) / std
    # x = x_hat * std + mean
    # x:[C,H,W]
    # mean: [3] -> [3, 1, 1]

    mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
    std = torch.tensor(std).unsqueeze(1).unsqueeze(1)
    x = x_hat * std + mean
    return x


def pil_to_torch_tensor(pil_img, norm=True):
    tensor_img = torch.from_numpy(np.array(pil_img))  # H, W, C
    if norm:
        tensor_img = tensor_img.permute(2, 0, 1).float() / 255.0
    else:
        tensor_img = tensor_img.permute(2, 0, 1).float()
    return tensor_img


def torch_tensor_to_pil(tensor_img, norm=True):
    if norm:
        tensor_img = denormalize(tensor_img)
    tensor_img = tensor_img.squeeze(0).permute(1, 2, 0)
    pil_img = tensor_img.numpy()
    if norm:
        pil_img = Image.fromarray((pil_img * 255).astype(np.uint8))
    else:
        pil_img = Image.fromarray(pil_img.astype(np.uint8))
    return pil_img


def cv_to_torch(cv_img):
    cv_img = np.transpose(cv_img, (2, 0, 1))  # convert to C, H, W
    tensor_img = torch.from_numpy(cv_img).float()
    return tensor_img


def torch_tensor_to_cv(tensor_img, norm=True):
    if norm:
        tensor_img = denormalize(tensor_img)
    tensor_img = tensor_img.squeeze(0).permute(1, 2, 0)
    cv_img = tensor_img.numpy()
    if norm:
        cv_img = (cv_img * 255).astype(np.uint8)
    else:
        cv_img = cv_img.astype(np.uint8)
    return cv_img


def pil_to_cv(pil_img):
    cv_img = cv2.cvtColor(np.asarray(pil_img), cv2.COLOR_RGB2BGR)
    return cv_img


def cv_to_pil(cv_img):
    pil_img = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))
    return pil_img


def get_transform_matrix(center, angle, ori_w, ori_h, scale=1.0):
    # mat = cv2.getRotationMatrix2D(center, angle, scale)
    # print("mat cv: ", mat)
    radian = angle * np.pi / 180.0
    alpha = scale * np.cos(radian)
    beta = scale * np.sin(radian)
    t_x = (1 - alpha) * center[0] - beta * center[1]
    t_y = beta * center[0] + (1 - alpha) * center[1]
    mat = np.array([[alpha, beta, t_x], [-beta, alpha, t_y]])
    # print("mat: ", mat)

    new_w = abs(mat[0][0]) * ori_w + abs(mat[0][1]) * ori_h
    new_h = abs(mat[0][1]) * ori_w + abs(mat[0][0]) * ori_h

    mat[0][2] += (new_w - ori_w) / 2.0  # 中心点偏移
    mat[1][2] += (new_h - ori_h) / 2.0

    return mat


def img_rotate_center(img, angle, scale=1.0):
    ori_h, ori_w = img.shape[:2]
    center = (ori_w/2.0, ori_h / 2.0)
    m = get_transform_matrix(center, angle, ori_w, ori_h, scale)
    new_w = abs(m[0][0]) * ori_w + abs(m[0][1]) * ori_h
    new_h = abs(m[0][1]) * ori_w + abs(m[0][0]) * ori_h

    # print("offset: ", (new_w - ori_w) / 2.0, (new_h - ori_h) / 2.0)
    new_img = cv2.warpAffine(img, m, (int(new_w + 0.5), int(new_h + 0.5)))
    return new_img


def pt_rotate_center(pts, angle, ori_w, ori_h, scale=1.0):
    center = (ori_w / 2.0, ori_h / 2.0)
    m = get_transform_matrix(center, angle, ori_w, ori_h, scale)

    # # 思路一： 中心点偏移
    # new_w = abs(m[0][0]) * ori_w + abs(m[0][1]) * ori_h
    # new_h = abs(m[0][1]) * ori_w + abs(m[0][0]) * ori_h
    # m[0][2] += (new_w - ori_w) / 2.0
    # m[1][2] += (new_h - ori_h) / 2.0
    # # print("offset: ", (new_w - ori_w) / 2.0, (new_h - ori_h) / 2.0)
    #
    # # # # 思路二： 参考角点偏移
    # # corner_pts = np.array([[0, 0, 1], [ori_w, 0, 1], [0, ori_h, 1], [ori_w, ori_h, 1]])
    # # new_corner_pts = np.dot(m, corner_pts.T).T
    # # w_offset, h_offset = np.min(new_corner_pts, axis=0)  # 将最小坐标移动至坐标原点
    # # print("offset w, h: ", w_offset, h_offset)
    # # m[0][2] -= w_offset
    # # m[1][2] -= h_offset

    pts = np.insert(pts, 2, 1, axis=1)
    new_pts = np.dot(m, pts.T).T

    return new_pts


def crop_img(cv_img):
    gray_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    thresh, binary_img = cv2.threshold(gray_img, 10, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    x, y, w, h = cv2.boundingRect(contours[0])
    cropped = cv_img[y:y+h, x:x+w]

    return cropped, x, y


def crop_pts(pts, x, y):
    return pts - np.array([x, y])


class ToothDataSet(Dataset):
    def __init__(self, img_dir, label_txt_file, img_size=256, out_size=64, n_kps=3):
        self.img_dir = img_dir
        self.label_txt_file = label_txt_file            # img_name, x1, y1, x2, y2...
        self.img_size = img_size
        self.out_size = out_size
        self.ratio = out_size / img_size
        self.n_kps = n_kps
        self.gen_heat_map = GenerateHeatmap(self.out_size, self.n_kps)

        self.img_names = []
        self.keypoints = []
        self.visibles = []
        self.load_data()

    def load_data(self):
        print('loading data...')
        tic = time.time()

        labels = np.loadtxt(self.label_txt_file,
                            dtype={'names': ('img_name', 'left_x', 'left_y', 'mid_x', 'mid_y', 'right_x', 'right_y'),
                                   'formats': ('|S200', float, float, float, float, float, float)},
                            delimiter=';', skiprows=0)

        img_names = []
        kps = []
        visibles = []
        for i in range(len(labels)):
            img_names.append(str(labels['img_name'][i].decode("utf8")))
            kps.append([[labels['left_x'][i], labels['left_y'][i]], [labels['mid_x'][i], labels['mid_y'][i]], [labels['right_x'][i], labels['right_y'][i]]])
            visibles.append([1, 1, 1])

        self.img_names = img_names
        self.keypoints = kps
        self.visibles = visibles

        print('Done (t={:0.2f}s)'.format(time.time() - tic))

    def __getitembak__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        img = Image.open(img_name).convert("RGB")
        w, h = img.size
        kps = []
        w_ratio = self.img_size / w
        h_ratio = self.img_size / h
        for i in range(len(self.keypoints[idx])):
            if self.visibles[idx][i] == 0:
                # set keypoints to 0 when were not visible initially (so heatmap all 0s)
                kps.append([0.0, 0.0])
            else:
                ori_kp = self.keypoints[idx][i]
                x = ori_kp[0] * w_ratio * self.ratio
                y = ori_kp[1] * h_ratio * self.ratio
                kps.append([x, y])
        img = img.resize((self.img_size, self.img_size))
        heatmap = self.gen_heat_map(kps)
        kps = np.array(kps)
        # can add data augment
        return img, torch.from_numpy(kps), torch.from_numpy(heatmap)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.img_names[idx])
        img = Image.open(img_name).convert("RGB")
        ori_kps = self.keypoints[idx]
        if np.random.randint(2) == 0:
            img = pil_to_cv(img)
            angle = np.random.random() * 360                    # [0, 360]
            scale = np.random.random() * (1.25 - 0.75) + 0.75   # [0.75, 1.25]
            new_img = img_rotate_center(img, angle, scale)
            ori_h, ori_w = img.shape[:2]
            cropped_img, left_x, left_y = crop_img(new_img)
            new_pts = pt_rotate_center(ori_kps, angle, ori_w, ori_h, scale)
            cropped_pts = crop_pts(new_pts, left_x, left_y)

            img = cv_to_pil(cropped_img)
            ori_kps = cropped_pts
        w, h = img.size
        kps = []
        w_ratio = self.img_size / w
        h_ratio = self.img_size / h
        for i in range(len(self.keypoints[idx])):
            if self.visibles[idx][i] == 0:
                # set keypoints to 0 when were not visible initially (so heatmap all 0s)
                kps.append([0.0, 0.0])
            else:
                ori_kp = ori_kps[i]
                x = ori_kp[0] * w_ratio * self.ratio
                y = ori_kp[1] * h_ratio * self.ratio
                kps.append([x, y])

        tf = transforms.Compose([
            transforms.Resize((int(self.img_size), int(self.img_size))),
            # # can add data augment
            # transforms.RandomRotation(15),
            # transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = tf(img)
        heatmap = self.gen_heat_map(kps)

        return img, np.array(kps), heatmap

    def __len__(self):
        return len(self.img_names)


if __name__ == "__main__":
    # angle = 60
    # scale = 0.75
    # img = pil_to_cv(Image.open(r"E:\data\SplitTooth\AddFDIClassAndKeyPoint\keyPoint\black\1 (1)_top_black_flip.png").convert("RGB"))
    # print(img.shape)
    # pts = [[636.1309523809524, 89.28571428571429], [330.4761904761905, 500.0], [80.47619047619048, 53.57142857142858]]
    # for i in range(len(pts)):
    #     cv2.circle(img, (int(pts[i][0]), int(pts[i][1])), 10, (0, 0, 255), -1)
    #
    # cv2.imshow("img", img)
    #
    # img = pil_to_cv(Image.open(r"E:\data\SplitTooth\AddFDIClassAndKeyPoint\keyPoint\black\1 (1)_top_black_flip.png").convert("RGB"))
    # new_img = img_rotate_center(img, angle, scale)
    # ori_h, ori_w = img.shape[:2]
    # cropped_img, left_x, left_y = crop_img(new_img)
    # new_pts = pt_rotate_center(pts, angle, ori_w, ori_h, scale)
    # cropped_pts = crop_pts(new_pts, left_x, left_y)
    # for i in range(len(pts)):
    #     cv2.circle(cropped_img, (int(cropped_pts[i][0]), int(cropped_pts[i][1])), 10, (0, 255, 0), -1)
    #
    # cv2.imshow("crop_img", cropped_img)
    # cv2.waitKey(0)

    train_img_dir = r"E:\data\SplitTooth\AddFDIClassAndKeyPoint\keyPoint\black"
    txt_file = r"E:\data\SplitTooth\AddFDIClassAndKeyPoint\keyPoint\kp_label.txt"
    im_resize, keypoints, heatmaps = ToothDataSet(train_img_dir, txt_file, 256)[0]
    print("keypoints: ", keypoints, keypoints.shape)
    print("im shape: ", im_resize.shape)
    print("heatmaps shape: ", heatmaps.shape)
    print(get_peak_points(heatmaps[np.newaxis, :]))

    parser = HeatmapParser()
    print(parser.parse(heatmaps[np.newaxis, :]))
    import matplotlib.pyplot as plt
    print(im_resize.shape)
    pil_img = torch_tensor_to_pil(im_resize, True)
    # pil_img.rotate(10)
    im = pil_img.resize((64, 64))
    print(im.size)
    plt.imshow(im)
    plt.imshow(heatmaps[0], alpha=0.5)
    plt.imshow(heatmaps[1], alpha=0.5)
    plt.imshow(heatmaps[2], alpha=0.5)
    plt.show()

