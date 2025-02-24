import json
import glob
import os
import numpy as np


JSON_Dir = r"E:\data\SplitTooth\AddFDIClassAndKeyPoint\keyPoint\black"
SAVE_LABEL_PATH = r"D:/kp_label.txt"


if __name__ == "__main__":
    import cv2
    img = cv2.imread(os.path.join(JSON_Dir, "12707-LW-W1010 UBP_top_black_flip.png"))
    cv2.circle(img, (int(136.207), int(544.688)), 5, (0, 0, 255), -1)
    cv2.circle(img, (int(168.527), int(85.312)), 5, (0, 255, 0), -1)
    cv2.circle(img, (int(524.051), int(356.562)), 5, (255, 0, 0), -1)
    cv2.imshow("img", img)
    cv2.waitKey(0)

  #   json_files = glob.glob(os.path.join(JSON_Dir, "*.json"))
  #
  #   datas = []
  #   for json_file in json_files:
  #       with open(json_file, "r") as f:
  #           json_data = json.load(f)
  #           shapes = json_data["shapes"]
  #           img_name = json_data["imagePath"]
  #           print(img_name)
  #           data = [[0, 0], [0, 0], [0, 0]]
  #           for idx, shape in enumerate(shapes):
  #               if shape["label"] == "L":
  #                   data[0] = np.mean(np.array(shape["points"]), axis=0)
  #               elif shape["label"] == "M":
  #                   data[1] = np.mean(np.array(shape["points"]), axis=0)
  #               elif shape["label"] == "R":
  #                   data[2] = np.mean(np.array(shape["points"]), axis=0)
  #
  #           # datas.append([img_name, str(data[0][0]), str(data[0][1]), str(data[1][0]), str(data[1][1]), str(data[2][0]), str(data[2][1])])
  #           datas.append([img_name, data[0][0], data[0][1], data[1][0], data[1][1], data[2][0], data[2][1]])
  #
  #   print(len(datas))
  #
  #   # # save label to txt
  #   # np.savetxt(SAVE_LABEL_PATH, datas, fmt="%s", delimiter=";")
  #
  #   import cv2
  #   data_0 = datas[0]
  #   print(data_0)
  #   img = cv2.imread(os.path.join(JSON_Dir, data_0[0]))
  #
  #   img_resize = cv2.resize(img, (256, 256))
  #   left_pt = (int(data_0[1]), int(data_0[2]))
  #   mid_pt = (int(data_0[3]), int(data_0[4]))
  #   right_pt = (int(data_0[5]), int(data_0[6]))
  #   print(left_pt, mid_pt, right_pt)
  #   cv2.circle(img, left_pt, 10, (0, 0, 255), -1)
  #   cv2.circle(img, mid_pt, 10, (255, 0, 0), -1)
  #   cv2.circle(img, right_pt, 10, (0, 255, 0), -1)
  #
  #   cv2.imshow("img", img)
  #
  #   left_pt_resize = (int(left_pt[0] / img.shape[1] * 256), int(left_pt[1] / img.shape[0] * 256))
  #   mid_pt_resize = (int(mid_pt[0] / img.shape[1] * 256), int(mid_pt[1] / img.shape[0] * 256))
  #   right_pt_resize = (int(right_pt[0] / img.shape[1] * 256), int(right_pt[1] / img.shape[0] * 256))
  #   print(left_pt_resize, mid_pt_resize, right_pt_resize)
  #   cv2.circle(img_resize, left_pt_resize, 5, (0, 0, 255), -1)
  #   cv2.circle(img_resize, mid_pt_resize, 5, (255, 0, 0), -1)
  #   cv2.circle(img_resize, right_pt_resize, 5, (0, 255, 0), -1)
  #   cv2.imshow("img_resize", img_resize)
  #
  #
  #   """
  #   [275.383 534.176   0.012]
  # [422.617 151.012   0.022]
  # [ 95.43  187.074   0.011]]
  #   """
  #   img2 = cv2.imread(os.path.join(JSON_Dir, data_0[0]))
  #   cv2.circle(img2, (275, 534), 10, (0, 0, 255), -1)
  #   cv2.circle(img2, (423, 151), 10, (255, 0, 0), -1)
  #   cv2.circle(img2, (95, 187), 10, (0, 255, 0), -1)
  #
  #   cv2.imshow("img2", img2)
  #   cv2.waitKey(0)
