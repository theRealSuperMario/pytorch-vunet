import pytest
from src import vunet_org
import torch
import numpy as np
from src import modules


class Test_VUnet:
    def test_forward(self):
        spatial_size = 256
        bottleneck_factor = 2
        box_factor = 2
        n_stages = 3
        n_latent_stages = 2
        conv_layer_type = modules.CONV_LAYER_TYPES.l1_norm.value
        n_channels_start = 64
        n_channels_max = 256
        n_channels_x = 3 * 2 * 8  # (cropping to 8 body parts)
        net = vunet_org.VunetOrg(
            spatial_size=spatial_size,
            bottleneck_factor=bottleneck_factor,
            box_factor=box_factor,
            stages=n_stages,
            latent_stages=n_latent_stages,
            conv_layer_type=conv_layer_type,
            start_channels=n_channels_start,
            max_channels=n_channels_max,
            x_channels=n_channels_x,
        )

        conditioning = torch.ones(
            (1, n_channels_x, 256 // (2 ** box_factor), 256 // (2 ** box_factor))
        )
        stickman = torch.ones((1, 3, 256, 256))

        imgs, qs, ps, activations = net(conditioning, stickman)
        assert imgs.shape == stickman.shape

    # def test_with_cropping(self):
    #     crop_size = 64
    #     n_keypoints = 12
    #     app_image = torch.ones((1, 3 * n_keypoints * crop_size, crop_size, crop_size))
    #     shape_image = torch.ones((1, 3, 256, 256))

    #     net = vunet_org.VunetOrg("cpu", 256, False, 3, 128)
    #     mode = "train"
    #     out_img, posterior_params, prior_params = net(app_image, shape_image, mode=mode)


class Test_EncUp:
    def test_forward(self):
        spatial_size = 256
        box_factor = 2
        import math

        n_scales = 7 - box_factor
        n_channels = 32 * 2 ** box_factor
        max_filters = 128
        nf_in = 3

        t = torch.zeros(
            (
                1,
                nf_in,
                spatial_size // (2 ** box_factor),
                spatial_size // (2 ** box_factor),
            )
        )

        encoder = vunet_org.EncUp(n_scales, n_channels, max_filters, nf_in)
        h = encoder(t)
        assert len(h) == 10
        assert h[-1].shape == (1, 128, 4, 4)

        encoder = vunet_org.EncUp(n_scales, n_channels, math.inf, nf_in)
        h = encoder(t)
        assert len(h) == 10
        assert h[-1].shape == (1, 2048, 4, 4)


class Test_EncDown:
    def test_forward(self):
        spatial_size = 256
        box_factor = 2
        import math

        n_scales = 5
        n_latent_scales = 2
        n_channels = 32 * 2 ** box_factor
        max_filters = 128
        nf_in = 3

        t = torch.zeros(
            (
                1,
                nf_in,
                spatial_size // (2 ** box_factor),
                spatial_size // (2 ** box_factor),
            )
        )

        encoder = vunet_org.EncUp(n_scales, n_channels, max_filters, nf_in)
        decoder = vunet_org.EncDown(
            n_channels, in_channels=max_filters, stages=n_latent_scales
        )
        gs = encoder(t)

        hs, qs, zs = decoder(gs)
        assert len(hs) == 5
        assert hs[-1].shape == (1, 128, 16, 16)
