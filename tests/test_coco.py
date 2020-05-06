import pytest
import sys

sys.path.insert(0, "data")
sys.path.insert(0, ".")
from src.datasets import coco


class Test_VUnetCocoDataset:
    def test_getitem_train(self):
        dset = coco.VUnetCocoDataset(
            "datasets/coco", index_p="datasets/coco/index.p", split="train"
        )
        len_ = len(dset)
        assert len(dset) == 18345

        example = dset[0]

        for k in ["image", "image_stickman", "part_image", "part_stickman"]:
            assert example[k].max() <= 1.0
        

    def test_getitem_test(self):
        dset = coco.VUnetCocoDataset(
            "datasets/coco", index_p="datasets/coco/index.p", split="test"
        )
        len_ = len(dset)
        assert len(dset) == 690

        example = dset[0]

        for k in ["image", "image_stickman", "part_image", "part_stickman"]:
            assert example[k].max() <= 1.0
