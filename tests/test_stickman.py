import pytest
from src import stickman
from supermariopy import imageutils
from supermariopy import plotting
from matplotlib import pyplot as plt
import numpy as np
import math
from supermariopy import imageutils


class Test_example_joint_models:
    @pytest.mark.mpl_image_compare
    def test_joint_model(self):
        # kps = stickman.EXAMPLE_JOINT_MODELS["JointModel_15"]
        # joint_img = stickman.make_joint_img((128, 128), kps, stickman.JointModel_15)
        kps = stickman.VUNetStickman.get_example_valid_keypoints_deepfashion()
        joint_img = stickman.VUNetStickman.make_joint_img(
            (128, 128),
            stickman.VUNET_JOINT_ORDER_DEEPFASHION,
            stickman.VUNetStickman.get_example_valid_keypoints_deepfashion() * 128,
        )

        plt.imshow(joint_img)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_get_bounding_box(self):
        kps = stickman.VUNetStickman.get_example_valid_keypoints_deepfashion()
        box = stickman.get_bounding_boxes(kps, (128, 128), 32)
        joint_img = stickman.VUNetStickman.make_joint_img(
            (128, 128),
            stickman.VUNET_JOINT_ORDER_DEEPFASHION,
            stickman.VUNetStickman.get_example_valid_keypoints_deepfashion() * 128,
        )

        box_image = plotting.overlay_boxes_without_labels(joint_img, box)
        plt.imshow(box_image)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_invalid_stickman(self):
        kps = stickman.VUNetStickman.get_example_invalid_keypoints()
        box = stickman.get_bounding_boxes(kps, (128, 128), 32)
        joint_img = stickman.VUNetStickman.make_joint_img(
            (128, 128),
            stickman.VUNET_JOINT_ORDER_DEEPFASHION,
            stickman.VUNetStickman.get_example_invalid_keypoints() * 128,
        )

        box_image = plotting.overlay_boxes_without_labels(joint_img, box)
        plt.imshow(box_image)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_filter_parts(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        kps = stickman.VUNetStickman.get_example_valid_keypoints_deepfashion()
        box = stickman.get_bounding_boxes(kps, (128, 128), 32)
        joint_img = stickman.VUNetStickman.make_joint_img(
            (128, 128),
            stickman.VUNET_JOINT_ORDER_DEEPFASHION,
            stickman.VUNetStickman.get_example_valid_keypoints_deepfashion() * 128,
        )

        box_image = plotting.overlay_boxes_without_labels(joint_img, box)
        ax[0].imshow(box_image)

        kps = stickman.VUNetStickman.get_example_valid_keypoints_deepfashion()
        jo = ["lankle", "rankle"]
        new_kps = stickman.filter_keypoints(
            kps, stickman.VUNET_JOINT_ORDER_DEEPFASHION, jo
        )
        box = stickman.get_bounding_boxes(new_kps, (128, 128), 32)
        # joint_img = stickman.VUNetStickman.make_joint_img((128, 128), jo, new_kps * 128)

        box_image = plotting.overlay_boxes_without_labels(
            joint_img, box, colors=[[0, 0, 255],] * len(new_kps)
        )
        ax[1].imshow(box_image)
        return plt.gcf()

    @pytest.mark.mpl_image_compare
    def test_crop(self):
        kps = stickman.VUNetStickman.get_example_valid_keypoints_deepfashion()
        joint_img = stickman.VUNetStickman.make_joint_img(
            (128, 128), stickman.VUNET_JOINT_ORDER_DEEPFASHION, kps * 128,
        )
        joint_order = stickman.VUNET_JOINT_ORDER_DEEPFASHION

        crops = [
            stickman.VUNetStickman.normalize(
                joint_img, kps * 128, joint_img, joint_order, 1
            )
        ]
        crops = [c[0] for c in crops]
        crops = np.stack(crops, axis=0)
        crops = np.split(crops, 8, axis=-1)
        n_crops = len(crops)
        cols = math.ceil(math.sqrt(n_crops))
        crops = np.concatenate(crops, axis=0)
        crops = imageutils.batch_to_canvas(crops, cols)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(crops)
        plt.savefig("test_crop.png")
        return fig

    @pytest.mark.mpl_image_compare
    def test_crop_affine(self):
        kps = stickman.VUNetStickman.get_example_valid_keypoints_deepfashion()
        joint_img = stickman.VUNetStickman.make_joint_img(
            (128, 128), stickman.VUNET_JOINT_ORDER_DEEPFASHION, kps * 128,
        )
        joint_order = stickman.VUNET_JOINT_ORDER_DEEPFASHION

        crops = [
            stickman.VUNetStickman.normalize_affine(
                joint_img, kps * 128, joint_img, joint_order, 1
            )
        ]
        crops = [c[0] for c in crops]
        crops = np.stack(crops, axis=0)
        crops = np.split(crops, 8, axis=-1)
        n_crops = len(crops)
        cols = math.ceil(math.sqrt(n_crops))
        crops = np.concatenate(crops, axis=0)
        crops = imageutils.batch_to_canvas(crops, cols)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(crops)
        plt.savefig("test_crop_affine.png")
        return fig

    @pytest.mark.mpl_image_compare
    def test_crop_scale(self):
        kps = stickman.VUNetStickman.get_example_valid_keypoints_deepfashion()
        joint_img = stickman.VUNetStickman.make_joint_img(
            (128, 128), stickman.VUNET_JOINT_ORDER_DEEPFASHION, kps * 128,
        )
        joint_order = stickman.VUNET_JOINT_ORDER_DEEPFASHION

        crops = [
            stickman.VUNetStickman.normalize_scale(
                joint_img, kps * 128, joint_img, joint_order, 1
            )
        ]
        crops = [c[0] for c in crops]
        crops = np.stack(crops, axis=0)
        crops = np.split(crops, 8, axis=-1)
        n_crops = len(crops)
        cols = math.ceil(math.sqrt(n_crops))
        crops = np.concatenate(crops, axis=0)
        crops = imageutils.batch_to_canvas(crops, cols)
        fig, ax = plt.subplots(1, 1)
        ax.imshow(crops)
        plt.savefig("test_crop_scaling.png")
        return fig

    @pytest.mark.mpl_image_compare
    def test_visualize_body_part(self):
        kps = stickman.VUNetStickman.get_example_valid_keypoints_deepfashion()
        joint_img = stickman.VUNetStickman.make_joint_img(
            (128, 128), stickman.VUNET_JOINT_ORDER_DEEPFASHION, kps * 128,
        )
        joint_order = stickman.VUNET_JOINT_ORDER_DEEPFASHION
        bparts = stickman.VUNetStickman.VUNET_CROP_PARTS
        colors = plt.cm.hsv(np.linspace(0, 1, len(bparts)), bytes=True)[
            :, :3
        ]  # no alpha
        visualized_image = stickman.VUNetStickman.visualize_body_parts(
            joint_img, kps * 128, joint_order, bparts, colors
        )
        plt.imshow(visualized_image)
        plt.savefig("test_visualize_bparts.png")
        return fig
