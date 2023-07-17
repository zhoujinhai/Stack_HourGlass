from torchvision import transforms
import torch
import os
from PIL import Image
import glob
import numpy as np
import cv2
from datas.my_dataset import get_peak_points, HeatmapParser
from models.hour_glass import StackHourglass


if __name__ == "__main__":
    # model
    device = "cpu"
    model = StackHourglass(4, 256, 3, 4)
    model.float().to(device)

    weights = "./exp/tooth/checkpoint.pt"
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    # test data
    img_dir = r"E:\data\SplitTooth\AddFDIClassAndKeyPoint\keyPoint\black"
    img_files = glob.glob(os.path.join(img_dir, "*.png"))

    # data process
    input_size = 256
    tf = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # test
    parser = HeatmapParser()
    with torch.no_grad():
        for img_path in img_files:
            print("Test: ", img_path)
            img = Image.open(img_path).convert("RGB")
            w, h = img.size
            w_ratio = w / input_size
            h_ratio = h / input_size
            img = tf(img)
            img = img.unsqueeze(0)
            output = model(img.to(device))
            all_peak_points = get_peak_points(output[:, -1].cpu().data.numpy())
            # all_peak_points = parser.parse(output[:, -1].cpu().data.numpy())
            # print(all_peak_points)
            kps = all_peak_points * 4 * np.array([w_ratio, h_ratio])
            kps = kps[0]
            # print(kps.shape, kps)
            img = cv2.imread(img_path)
            for i in range(kps.shape[0]):
                x, y = kps[i]
                print("w: {}, h: {}, x: {}, y: {}".format(w, h, x, y))
                cv2.circle(img, (int(x), int(y)), 5, [(0, 0, 255), (0, 255, 0), (255, 0, 0)][i], -1)
            cv2.imshow("img", img)
            cv2.waitKey(0)

