import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from edflow import TemplateIterator, get_logger
from src.vunet_org import VunetOrg
from supermariopy.ptutils import losses as ptlosses
from supermariopy.ptutils import nn as ptnn
from src import losses
from supermariopy import imageutils
from typing import *
from dotmap import DotMap


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = DotMap(config)
        spatial_size = config["spatial_size"]
        vunet_params = config["vunet_params"]

        self.spatial_size = spatial_size
        self.vunet_params = vunet_params
        self.lr = config["lr"]

        self.vunet = VunetOrg(spatial_size=spatial_size, **vunet_params)

    def forward(self, stickman, appearance_cropped):
        imgs, qs, ps, activations = self.vunet(appearance_cropped, stickman)
        return imgs, qs, ps, activations

    def transfer(self, stickman, appearance_cropped):
        imgs = self.vunet.transfer(appearance_cropped, stickman)
        return imgs


class Iterator(TemplateIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.config = self.model.config
        self.model = self.model.to(self.device)
        self.vgg_loss = ptlosses.PerceptualVGG(
            feature_weights=self.config.perceptual_loss_params.feature_weights,
            use_gram=self.config.perceptual_loss_params.use_gram,
            gram_weights=self.config.perceptual_loss_params.gram_weights,
        )
        self.vgg_loss = self.vgg_loss.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model.lr)
        self.criterion = torch.nn.MSELoss()

    @property
    def callbacks(self):
        # return {"eval_op": {"acc_callback": acc_callback}}
        pass

    def save(self, checkpoint_path):
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        torch.save(state, checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])

    def step_op(self, model, **kwargs):
        # get inputs
        image, stickman, part_image, part_stickman = (
            kwargs["image"],
            kwargs["image_stickman"],
            kwargs["part_image"],
            kwargs["part_stickman"],
        )
        image = to_torch(image, permute=True)
        stickman = to_torch(stickman, permute=True)
        part_image = to_torch(part_image, permute=True)
        part_stickman = to_torch(part_stickman, permute=True)

        def train_op():
            # compute losses and run optimization
            model.train()
            reconstruction, qs, ps, activations = model(stickman, part_image)
            # reconstruction_loss = self.criterion(image, reconstruction)
            reconstruction_losses = self.vgg_loss.loss(
                convert_range(image, [-1, 1], [0, 1]),
                convert_range(reconstruction, [-1, 1], [0, 1]),
            )
            reconstruction_loss = torch.sum(torch.stack(reconstruction_losses, dim=-1))

            loss_kl = losses.aggregate_kl_loss(ps, qs)
            mean_loss = (
                self.config.r_lambda * reconstruction_loss
                + self.config.kl_lambda * loss_kl
            )

            self.optimizer.zero_grad()
            mean_loss.backward()
            self.optimizer.step()

        def log_op():
            # calculate logs every now and then
            with torch.no_grad():
                fixed_examples = self.get_fixed_examples(
                    ["image", "image_stickman", "part_image"]
                )
                model.eval()
                logs = {"images": {}, "scalars": {}}  # images and scalars
                reconstruction, qs, ps, activations = model(stickman, part_image)
                fixed_reconstruction, _, _, _ = model(
                    fixed_examples.image_stickman, fixed_examples.part_image
                )
                fixed_transfer = model.transfer(
                    fixed_examples.image_stickman,
                    ptnn.flip(fixed_examples.part_image, 0),
                )
                reconstruction_losses = self.vgg_loss.loss(
                    convert_range(image, [-1, 1], [0, 1]),
                    convert_range(reconstruction, [-1, 1], [0, 1]),
                )
                reconstruction_loss = self.config.r_lambda * torch.sum(
                    torch.stack(reconstruction_losses, dim=-1)
                )

                loss_kl = losses.aggregate_kl_loss(ps, qs)
                mean_loss = reconstruction_loss + self.config.kl_lambda * loss_kl
                transfer = model.transfer(stickman, ptnn.flip(part_image, 0))

                part_image_log = imageutils.batch_to_canvas(
                    to_numpy(split_stack_reshape(part_image, 3), permute=True), 8,
                )
                part_image_log = np.transpose(
                    part_image_log[np.newaxis, ...], (0, 3, 1, 2)
                )
                part_stickman_log = imageutils.batch_to_canvas(
                    to_numpy(split_stack_reshape(part_stickman, 3), permute=True), 8
                )
                part_stickman_log = np.transpose(
                    part_stickman_log[np.newaxis, ...], (0, 3, 1, 2)
                )

                logs["scalars"]["loss_reconstruction"] = reconstruction_loss
                logs["scalars"]["loss_kl"] = loss_kl
                logs["scalars"]["loss_kl_weighted "] = loss_kl * self.config.kl_lambda
                logs["scalars"]["lambda_kl"] = self.config.kl_lambda
                logs["scalars"]["loss_total"] = mean_loss
                logs["scalars"]["lr"] = self.model.lr
                logs["images"]["reconstruction"] = reconstruction
                logs["images"]["fixed_reconstruction"] = fixed_reconstruction
                logs["images"]["image"] = image
                logs["images"]["fixed_image"] = fixed_examples.image
                logs["images"]["transfer"] = transfer
                logs["images"]["fixed_transfer"] = fixed_transfer
                logs["images"]["part_image"] = part_image_log
                logs["images"]["part_stickman"] = part_stickman_log
                logs["images"]["stickman"] = stickman
                logs["images"]["fixed_stickman"] = fixed_examples.image_stickman
                import functools

                func = functools.partial(to_numpy, permute=True)
                logs["images"] = recursive_apply(logs["images"], func)
                logs["scalars"] = recursive_apply(logs["scalars"], to_numpy)
            return logs

        def eval_op():
            model.eval()
            # eval_logs = {"outputs": {}, "labels": {}}  # eval_logs
            eval_logs = None
            return eval_logs

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}

    def get_fixed_examples(self, names):
        """collect fixed examples from dataset with names"""
        if not hasattr(self, "fixed_examples"):
            fixed_random_indices = np.random.RandomState(1).choice(
                len(self.dataset), self.config["batch_size"]
            )
            fixed_examples = {}
            for n in names:
                fixed_examples_n = [self.dataset[i][n] for i in fixed_random_indices]
                fixed_examples_n = np.stack(fixed_examples_n)
                fixed_examples[n] = to_torch(fixed_examples_n, True)
            self.fixed_examples = DotMap(fixed_examples)
        return self.fixed_examples


# TODO: perceptual loss


def convert_range(
    array: torch.Tensor, input_range: Iterable[int], target_range: Iterable[int]
) -> torch.Tensor:
    """convert range of array from input range to target range

    Parameters
    ----------
    array: torch.Tensor
        array in any shape
    input_range: Iterable[int]
        range of array values in format [min, max]
    output_range: Iterable[int]
        range of rescaled array values in format [min, max]

    Returns
    -------
    torch.Tensor
        rescaled array
        
    Examples
    --------
        t = imageutils.convert_range(np.array([-1, 1]), [-1, 1], [0, 1])
        assert np.allclose(t, np.array([0, 1]))
        t = imageutils.convert_range(np.array([0, 1]), [0, 1], [-1, 1])
        assert np.allclose(t, np.array([-1, 1]))
    """
    if input_range[1] <= input_range[0]:
        raise ValueError
    if target_range[1] <= target_range[0]:
        raise ValueError

    a = input_range[0]
    b = input_range[1]
    c = target_range[0]
    d = target_range[1]
    return (array - a) / (b - a) * (d - c) + c


def to_numpy(x, permute=False):
    """automatically detach and move to cpu if necessary."""
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.detach().cpu().numpy()
        else:
            x = x.detach().numpy()
    if isinstance(x, np.ndarray):
        if permute:
            x = np.transpose(x, (0, 2, 3, 1))  # NCHW --> NHWC
    return x


def to_torch(x, permute=False):
    """automatically convert numpy array to torch and permute channels from NHWC to NCHW"""
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        x = x.to(device)

    if permute:
        x = x.permute((0, 3, 1, 2))  # NHWC --> NCHW
    if x.dtype is torch.float64:
        x = x.type(torch.float32)
    return x


def recursive_apply(d: dict, func: callable):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = recursive_apply(v, func)
        else:
            d[k] = func(v)
    return d


def split_stack(x, split_sizes, split_dim, stack_dim):
    """Split x along dimension split_dim and stack again at dimension stack_dim"""
    t = torch.stack(torch.split(x, split_sizes, dim=split_dim), dim=stack_dim)
    return t


def split_stack_reshape(x, split_sizes=3):
    t = split_stack(x, split_sizes, split_dim=1, stack_dim=1)
    shape_ = list(x.shape)
    shape_[0] = -1
    shape_[1] = split_sizes
    tt = t.view(shape_)
    return tt
