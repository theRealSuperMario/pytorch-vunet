from torch.utils.data import Dataset
import numpy as np
import cv2
import pickle
import os
import pandas as pd

import cv2
from src import stickman
import edflow.datasets.utils as edu
from supermariopy import imageutils

cv2.setNumThreads(0)


class VUnetDeepfashionDataset(Dataset):
    def __init__(
        self, data_dir, index_p, split="train", img_shape=(256, 256), box_factor=2
    ):
        self.data_dir = data_dir
        self.index_p = index_p

        # filter out valid indices based on valid_joints heuristic
        data = pickle.load(open(self.index_p, "rb"))
        df = pd.DataFrame(data["train"])
        valid = [
            stickman.valid_joints(data["joints"][i]) for i in range(len(data["joints"]))
        ]

        # joint_order : list of strings indicating keypoint order
        self.joint_order = data["joint_order"]

        # boolean list indicating if
        # element is from train or validations set
        self.is_train = data["train"].copy()

        # list of strings for filenames
        self.imgs = data["imgs"].copy()
        # np array indicating keypoints
        self.joints = data["joints"].copy()

        # needed for keypoint image
        self.img_shape = img_shape
        self.box_factor = box_factor

        # filter out elements based on valid
        valid_train = np.logical_and(np.array(valid), np.array(self.is_train))
        valid_test = np.logical_and(
            np.array(valid), np.logical_not(np.array(self.is_train))
        )

        # self.is_train = [self.is_train[i] for i, v in enumerate(valid_train) if v]
        # self.imgs = [self.imgs[i] for i, v in enumerate(valid) if v]
        # self.joints = [self.imgs[i] for i, v in enumerate(valid) if v]
        if split == "train":
            self.imgs = list(np.array(self.imgs)[valid_train])
            self.joints = list(np.array(self.joints)[valid_train])
        else:
            self.imgs = list(np.array(self.imgs)[valid_test])
            self.joints = list(np.array(self.joints)[valid_test])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        keypoints = self.joints[i]
        image_path = os.path.join(self.data_dir, self.imgs[i])
        image = cv2.imread(image_path, -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = imageutils.convert_range(image, [0, 255], [-1, 1])

        image_stickman = make_joint_img(
            self.img_shape,
            self.joint_order,
            keypoints * np.array(self.img_shape).reshape((-1, 2)),
        )
        image_stickman = imageutils.convert_range(image_stickman, [0, 255], [-1, 1])

        part_img, part_stickman = stickman.VUNetStickman.normalize(
            image,
            keypoints * np.array(self.img_shape).reshape((-1, 2)),
            image_stickman,
            self.joint_order,
            self.box_factor,
        )
        # part_img = imageutils.convert_range(part_img, [0, 255], [0, 1])
        # part_stickman = imageutils.convert_range(part_stickman, [0, 255], [0, 1])

        # build example for output
        example = {
            "image_path": image_path,
            "image": image,
            "image_stickman": image_stickman,
            "part_image": part_img,
            "part_stickman": part_stickman,
            "keypoints": keypoints,
            "joint_order" : self.joint_order
        }
        return example


class DeepfashionTrain(edu.DatasetMixin):
    def __init__(self, config):
        super(DeepfashionTrain, self).__init__()
        self.config = config
        dataset_params = config["dataset_params"]

        self.dset = VUnetDeepfashionDataset(**dataset_params, split="train")

    def __len__(self):
        return len(self.dset)

    def get_example(self, i):
        return self.dset[i]


class DeepfashionVal(DeepfashionTrain):
    def __init__(self, config):
        self.config = config
        dataset_params = config["dataset_params"]

        self.dset = VUnetDeepfashionDataset(**dataset_params, split="valid")


def make_joint_img(img_shape, jo, joints):
    # three channels: left, right, center
    scale_factor = img_shape[1] / 128
    thickness = int(3 * scale_factor)
    imgs = list()
    for i in range(3):
        imgs.append(np.zeros(img_shape[:2], dtype="uint8"))

    body = ["lhip", "lshoulder", "rshoulder", "rhip"]
    body_pts = np.array([[joints[jo.index(part), :] for part in body]])
    if np.min(body_pts) >= 0:
        body_pts = np.int_(body_pts)
        cv2.fillPoly(imgs[2], body_pts, 255)

    right_lines = [
        ("rankle", "rknee"),
        ("rknee", "rhip"),
        ("rhip", "rshoulder"),
        ("rshoulder", "relbow"),
        ("relbow", "rwrist"),
    ]
    for line in right_lines:
        l = [jo.index(line[0]), jo.index(line[1])]
        if np.min(joints[l]) >= 0:
            a = tuple(np.int_(joints[l[0]]))
            b = tuple(np.int_(joints[l[1]]))
            cv2.line(imgs[0], a, b, color=255, thickness=thickness)

    left_lines = [
        ("lankle", "lknee"),
        ("lknee", "lhip"),
        ("lhip", "lshoulder"),
        ("lshoulder", "lelbow"),
        ("lelbow", "lwrist"),
    ]
    for line in left_lines:
        l = [jo.index(line[0]), jo.index(line[1])]
        if np.min(joints[l]) >= 0:
            a = tuple(np.int_(joints[l[0]]))
            b = tuple(np.int_(joints[l[1]]))
            cv2.line(imgs[1], a, b, color=255, thickness=thickness)

    rs = joints[jo.index("rshoulder")]
    ls = joints[jo.index("lshoulder")]
    cn = joints[jo.index("cnose")]
    neck = 0.5 * (rs + ls)
    a = tuple(np.int_(neck))
    b = tuple(np.int_(cn))
    if np.min(a) >= 0 and np.min(b) >= 0:
        cv2.line(imgs[0], a, b, color=127, thickness=thickness)
        cv2.line(imgs[1], a, b, color=127, thickness=thickness)

    cn = tuple(np.int_(cn))
    leye = tuple(np.int_(joints[jo.index("leye")]))
    reye = tuple(np.int_(joints[jo.index("reye")]))
    if np.min(reye) >= 0 and np.min(leye) >= 0 and np.min(cn) >= 0:
        cv2.line(imgs[0], cn, reye, color=255, thickness=thickness)
        cv2.line(imgs[1], cn, leye, color=255, thickness=thickness)

    img = np.stack(imgs, axis=-1)
    if img_shape[-1] == 1:
        img = np.mean(img, axis=-1)[:, :, None]
    return img
