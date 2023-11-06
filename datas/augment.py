# ref: https://albumentations.ai/docs/getting_started/installation/

import albumentations as A
import cv2
from PIL import Image
import numpy as np
import random


# 分类
# 大致流程： 读取图片==>通过定义的增强管道==>输出增强后的图片
# 其他增强如下雨、雾气、雪花等参考：https://albumentations.ai/docs/examples/example_weather_transforms/
tf_class = A.Compose([
    # A.RandomCrop(width=512, height=512),           # 随机裁剪
    A.HorizontalFlip(),                            # 水平翻转
    A.RandomBrightnessContrast(p=0.2),             # 亮度和对比度随机增强
])

# 目标框
# 支持voc(x_min, y_min, x_max, y_max)
# coco(x_min, y_min, w, h)
# albumentations(x_min/W, y_min/H, x_max/W, y_max/H)
# yolo格式目标框(x_center/W, y_center/H, w/W, h/H)

# 大致流程： 读取图片和框信息==>通过定义的增强管道==>输出增强后的图片和框
tf_box = A.Compose([
    # A.RandomCrop(width=512, height=512),           # 随机裁剪
    A.RandomSizedBBoxSafeCrop(width=448, height=336, erosion_rate=0.2),
    A.HorizontalFlip(),                            # 水平翻转
    A.RandomBrightnessContrast(p=0.2),             # 亮度和对比度随机增强
], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))  # pascal_voc, albumentations, coco or yolo

# 分割掩膜
# 大致流程： 读取图片和掩膜信息==>通过定义的增强管道==>输出增强后的图片和掩膜
tf_seg = A.Compose([
    A.RandomCrop(width=512, height=512),           # 随机裁剪
    # A.HorizontalFlip(),  # 水平翻转
    A.RandomBrightnessContrast(p=0.2),  # 亮度和对比度随机增强
], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

# 关键点
# 大致流程： 读取图片和关键点信息==>通过定义的增强管道==>输出增强后的图片和关键点
# support xy/xya/xys/xyas/xysa (x,y) is coord,a is angle,s is scale.
tf_kp = A.Compose([
    A.RandomCrop(width=512, height=512),           # 随机裁剪
    A.HorizontalFlip(),  # 水平翻转
    A.RandomBrightnessContrast(p=0.2),  # 亮度和对比度随机增强
], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']))


# ref: yolov5: https://github.com/ultralytics/yolov5/blob/master/utils/augmentations.py
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py
# mosaic
# 大致流程： 先预定一个2*2的图片, 划分由中心点确定, center在(0.5*size, 1.5*size)之间
#          按照左上、右上、右下、左下进行裁剪, 最后对标签进行相关处理
def mosaic(out_h, out_w, imgs, bboxes=None, classes=None, masks=None, kps=None):
    """
    Arrange the images in a 2x2 grid. Images can have different shape. And Deal the labels.
    Args:
        out_h:(int) The height of output.
        out_w:(int) The width of output.
        imgs:(List[np.ndarray]) The input images, it have 4 single image.
        bboxes:(List[List[np.ndarray]]) The boxes infos for every image.
        classes:(List[List[str]]) The class name for every boxes.
        masks:(List[List[np.ndarray]]) The mask info for every image.
        kps:(List[List[np.ndarray]]) The keypoints for every image.

    Returns:
        the output is image and label after mosaic operator.
    """
    if len(imgs) != 4:
        raise ValueError(f"Length of image_batch should be 4. Got {len(imgs)}")

    mosaic_border = [out_h // 2, out_w // 2]
    print(mosaic_border)
    yc, xc = (int(random.uniform(0.8 * bd, 1.2 * bd)) for bd in mosaic_border)  # mosaic center [0.4, 0.6]
    print("xc: ", xc, "yc: ", yc)
    indices = [0, 1, 2, 3]
    random.shuffle(indices)
    print(indices)
    if len(imgs[0].shape) == 2:
        out_shape = [out_h, out_w]
    else:
        out_shape = [out_h, out_w, imgs[0].shape[2]]
    img4 = np.full(out_shape, 0, dtype=np.uint8)  # base image with 4 tiles
    print(img4.shape)

    boxes4, masks4, kps4, cls4 = [], [], [], []
    if masks is not None:
        masks4 = np.full((out_h, out_w), 0, dtype=np.uint8)

    for i, index in enumerate(indices):
        # place img in img4
        img = imgs[index]
        h, w = img.shape[:2]
        x1a, y1a, x2a, y2a, x1b, y1b, x2b, y2b, x_offset, y_offset = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        if i == 0:  # top left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            x_offset = 0
            y_offset = 0
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, out_w), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            x_offset = xc
            y_offset = 0
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(out_h, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            x_offset = 0
            y_offset = yc
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, out_w), min(out_h, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            x_offset = xc
            y_offset = yc

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]

        # label
        if bboxes is not None and classes is not None:
            print("=====", bboxes)
            boxes = bboxes[index]
            class_names = classes[index]
            tf = A.Compose([
                A.Crop(x1b, y1b, x2b, y2b)
            ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

            img_crop = tf(image=img, bboxes=boxes, class_labels=class_names)
            crop_boxes = img_crop["bboxes"]
            crop_class = img_crop["class_labels"]
            if len(crop_class) > 0:
                boxes4.extend(np.add(crop_boxes, [x_offset, y_offset, 0, 0]))
                cls4.extend(crop_class)

        # mask
        if masks is not None and classes is not None:
            mask = masks[index]
            masks4[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]
            class_names = classes[index]
            tf = A.Compose([
                A.Crop(x1b, y1b, x2b, y2b)
            ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))

            w = mask.shape[1]
            h = mask.shape[0]
            single_masks = np.zeros((np.max(mask), h, w), dtype=np.uint8)
            for n in range(np.max(mask)):
                single_masks[n, :, :][mask == n + 1] = 1
            temp_bboxes = [cv2.boundingRect(cv2.findNonZero(temp_mask)) for temp_mask in single_masks]
            img_crop = tf(image=img, mask=mask, bboxes=temp_bboxes, class_labels=class_names)
            crop_boxes = img_crop["bboxes"]
            crop_class = img_crop["class_labels"]
            if len(crop_class) > 0:
                boxes4.extend(np.add(crop_boxes, [x_offset, y_offset, 0, 0]))
                cls4.extend(crop_class)

        # kps
        if kps is not None and classes is not None:
            tf = A.Compose([
                A.Crop(x1b, y1b, x2b, y2b)
            ], keypoint_params=A.KeypointParams(format='xy', label_fields=['class_labels']))
            keypoint = kps[index]
            kp_names = classes[index]
            img_crop = tf(image=img, keypoints=keypoint, class_labels=kp_names)
            crop_kps = img_crop["keypoints"]
            crop_labels = img_crop["class_labels"]
            if len(crop_labels) > 0:
                kps4.extend(np.add(img_crop["keypoints"], [x_offset, y_offset]))
                cls4.extend(img_crop["class_labels"])

    return img4, cls4, boxes4, masks4, kps4


def show_aug_img_cv(cv_img, b_convert=True):
    if b_convert:
        img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
        cv2.imshow("img", img)
    else:
        cv2.imshow("img", cv_img)
    cv2.waitKey(0)


def show_aug_img_pil(image):
    from matplotlib import pyplot as plt
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)
    plt.show()


def show_box(img, bbox, class_name, color=(255, 0, 0), thickness=2):
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    ((text_w, text_h), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_h)), (x_min + text_w, y_min), color, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_h)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
    )
    return img


def show_img_box(image, bboxes, class_names):
    from matplotlib import pyplot as plt
    img = image.copy()
    for bbox, class_name in zip(bboxes, class_names):
        img = show_box(img, bbox, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.imshow(img)
    plt.show()


def show_kp(img, kp, class_name):
    x, y = int(kp[0]), int(kp[1])
    r = 5
    cv2.circle(img, (x, y), r, (255, 0, 0), -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x + r + 2, y),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=(255, 255, 255),
        lineType=cv2.LINE_AA,
    )
    return img


def show_img_kps(image, kps, class_names):
    from matplotlib import pyplot as plt
    img = image.copy()
    for kp, class_name in zip(kps, class_names):
        img = show_kp(img, kp, class_name)
    plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    # #==========class==========
    # img_path = "D:/data/aug/classify.png"
    # img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    # pil_img = Image.open(img_path)
    # pil_img = np.array(pil_img)
    #
    # cls_img_tf = tf_class(image=pil_img)["image"]
    # show_aug_img_pil(cls_img_tf)

    # #==========detect==========
    # detect_img_path = "D:/data/aug/detect.jpg"   # img ref: https://cocodataset.org/#explore?id=386298
    # detect_img = cv2.imread(detect_img_path)
    # detect_img = cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB)
    # bboxes = [[20.07, 185.64, 169.81, 208.42], [473.05, 96.53, 165.95, 244.47]]
    # class_names = ["cat", "dog"]
    # # 这里可以把class_names分别放在bboxes后面如：[[20.07, 185.64, 169.81, 208.42, "cat"], [473.05, 96.53, 165.95, 244.47, "dog"]]
    # # 相应的tf_box把bbox_params中的label_fields进行删除
    # detect_img_tf = tf_box(image=detect_img, bboxes=bboxes, class_labels=class_names)
    # detect_aug_img = detect_img_tf["image"]
    # detect_aug_bboxes = detect_img_tf["bboxes"]
    # detect_aug_class_labels = detect_img_tf["class_labels"]
    # print(detect_aug_bboxes, detect_aug_class_labels)
    # show_img_box(detect_aug_img, detect_aug_bboxes, detect_aug_class_labels)

    # # ==========segmentation==========
    # # 借助json_to_dataset.py可以把labelme标注的数据转换为mask图(背景为0, 第一个实例为1， 第二个实例为2 依次递增.....)
    # # 或者参考https://www.cnblogs.com/xiaxuexiaoab/p/14743827.html  单通道彩色图要用pillow读取 opencv读取会自动转换为三通道彩图
    # seg_img_path = "D:/data/aug/seg/img.png"
    # mask_path = "D:/data/aug/seg/label.png"
    # seg_img = cv2.imread(seg_img_path)
    # seg_img = cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB)
    # mask = np.array(Image.open(mask_path))
    #
    # mask_number = np.max(mask)
    # label_names = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    # assert mask_number == len(label_names)
    # w = mask.shape[1]
    # h = mask.shape[0]
    # masks = np.zeros((mask_number, h, w), dtype=np.uint8)
    # for n in range(mask_number):
    #     masks[n, :, :][mask == n + 1] = 1
    # print(masks.shape)
    # # cv2.boundingRect return (x_min, y_min, w, h)
    # bboxes = [cv2.boundingRect(cv2.findNonZero(temp_mask)) for temp_mask in masks]
    # print(bboxes)
    #
    # seg_tf = tf_seg(image=seg_img, mask=mask, bboxes=bboxes, class_labels=label_names)
    # seg_aug_mask = seg_tf["mask"]
    # seg_aug_img = seg_tf["image"]
    # seg_aug_labels = seg_tf["class_labels"]
    # show_aug_img_pil(seg_aug_mask)
    # print(seg_aug_labels)
    #
    # # 将mask拆成单个mask
    # aug_masks = np.zeros((mask_number, seg_aug_mask.shape[0], seg_aug_mask.shape[1]), dtype=np.uint8)
    # for n in range(mask_number):
    #     aug_masks[n, :, :][seg_aug_mask == n + 1] = 1
    #     if np.max(aug_masks[n, :, :]) > 0:
    #         show_aug_img_pil(aug_masks[n, :, :])
    #         print(label_names[n])

    # # ==========keypoint==========
    # kp_img_path = r"D:\data\aug\classify.png"
    # kps = [(643.47, 87.50), (329.78, 505.36), (63.71, 48.81), (357.76, 264.29)]
    # kp_names = ["L", "M", "R", "C"]
    #
    # kp_img = cv2.imread(kp_img_path)
    # kp_img = cv2.cvtColor(kp_img, cv2.COLOR_RGB2BGR)
    # show_img_kps(kp_img, kps, kp_names)
    # print(tf_kp)
    # kp_tf = tf_kp(image=kp_img, keypoints=kps, class_labels=kp_names)
    # kp_aug_img = kp_tf["image"]
    # kp_aug_kps = kp_tf["keypoints"]
    # kp_aug_labels = kp_tf["class_labels"]
    # print(kp_aug_labels)
    # show_img_kps(kp_aug_img, kp_aug_kps, kp_aug_labels)

    # ==========mosaic==========
    # ref: https://github.com/mljack/albumentations_mosaic/commit/886e3e8b0d37889481fff30acdff358edda14a65
    # ref: yolov5/utils/dataloaders.py ==> load_mosaic()
    # 思路: 先预定一个2*2网格图片, 中心点预留在（0.5*size, 1.5*size之间）, 然后计算各个图片的大小及维度并进行裁剪, 对标签进行处理

    test_img_path = r"D:\data\aug\classify.png"
    test_img_path2 = "D:/data/aug/detect.jpg"
    img1 = cv2.imread(test_img_path)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread(test_img_path2)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # # box
    # boxes = [[20.07, 185.64, 169.81, 208.42], [473.05, 96.53, 165.95, 244.47]]
    # class_names = ["cat", "dog"]
    # batch_imgs = [img2, img2, img2, img2]
    # batch_class_names = [class_names] * 4
    # batch_boxes = [boxes] * 4
    # aug_img, aug_cls, aug_boxes, aug_masks, aug_kps = \
    #     mosaic(600, 700, batch_imgs, bboxes=batch_boxes, classes=batch_class_names)
    # show_img_box(aug_img, aug_boxes, aug_cls)

    # masks
    mask_path = "D:/data/aug/seg/label.png"
    mask = np.array(Image.open(mask_path))
    label_names = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "R1", "R2", "R3", "R4", "R5", "R6", "R7"]

    batch_imgs = [img1, img1, img1, img1]
    batch_masks = [mask, mask, mask, mask]
    batch_class_names = [label_names] * 4
    aug_img, aug_cls, aug_boxes, aug_masks, aug_kps = \
        mosaic(600, 700, batch_imgs, masks=batch_masks, classes=batch_class_names)
    show_img_box(aug_img, aug_boxes, aug_cls)
    show_aug_img_pil(aug_masks)
    print("aug_cls: ", aug_cls)

    # # kps
    # kps = [(643.47, 87.50), (329.78, 505.36), (63.71, 48.81), (357.76, 264.29)]
    # kp_names = ["L", "M", "R", "C"]
    # batch_imgs = [img1, img1, img1, img1]
    # batch_class_names = [kp_names] * 4
    # batch_kps = [kps] * 4
    # print(batch_kps)
    # aug_img, aug_cls, aug_boxes, aug_masks, aug_kps = \
    #     mosaic(600, 700, batch_imgs, kps=batch_kps, classes=batch_class_names)
    # show_img_kps(aug_img, aug_kps, aug_cls)
