import torch
from torch import nn
import numpy as np
from src.modules import *
from torch.nn import ModuleList
from functools import partial


class VunetOrg(nn.Module):
    def __init__(
        self,
        spatial_size,
        bottleneck_factor,
        box_factor,
        stages,
        latent_stages,
        conv_layer_type,
        start_channels,
        max_channels,
        x_channels=3,
        init_fn=None,
    ):
        super().__init__()
        self.spatial_size = spatial_size
        self.bottleneck_factor = bottleneck_factor
        self.box_factor = box_factor
        self.n_scales = stages
        self.n_latent_stages = latent_stages
        self.conv_layer_type = conv_layer_type
        self.n_channels_start = start_channels
        self.n_channels_max = max_channels

        stages = (
            (
                1
                + int(np.round(np.log2(spatial_size)))
                - bottleneck_factor  # bottleneck factor = 2
            )
            if stages < 6
            else stages
        )
        n_stages_x = stages - box_factor if x_channels > 3 else stages
        latent_stages = latent_stages
        if conv_layer_type == "l1":
            conv_layer = NormConv2d
            conv_t = "L1NormConv2d"
        elif conv_layer_type == "l2":
            conv_layer = partial(L2NormConv2d, init=init_fn, bias=False)
            conv_t = "L2NormConv2d"
        else:
            conv_layer = LayerNormConv2d
            conv_t = "LayerNormConv2d"

        print("Vunet using " + conv_t + " as conv layers.")
        self.eu = EncUp(
            n_stages_x,
            channels=start_channels,
            max_channels=max_channels,
            conv_layer=conv_layer,
            in_channels=x_channels,
        )
        self.ed = EncDown(
            channels=max_channels,
            in_channels=max_channels,
            conv_layer=conv_layer,
            stages=latent_stages,
        )
        self.du = DecUp(
            stages,
            n_channels_out=start_channels,
            n_channels_max=max_channels,
            conv_layer=conv_layer,
        )
        self.dd = DecDown(
            stages,
            max_channels,
            start_channels,
            out_channels=3,
            conv_layer=conv_layer,
            n_latent_stages=latent_stages,
        )

    def forward(self, x, c):
        # x: shape image
        # c: stickman
        hs = self.eu(x)
        es, qs, zs_posterior = self.ed(hs)

        gs = self.du(c)
        imgs, ds, ps, zs_prior = self.dd(gs, zs_posterior, training=True)

        activations = hs, qs, gs, ds
        return imgs, qs, ps, activations

    def test_forward(self, c):
        # sample appearance
        gs = self.du(c)
        imgs, ds, ps, zs_prior = self.dd(gs, [], training=False)
        return imgs

    def transfer(self, x, c):
        hs = self.eu(x)
        es, qs, zs_posterior = self.ed(hs)
        zs_mean = list(qs)

        gs = self.du(c)
        imgs, _, _, _ = self.dd(gs, zs_mean, training=True)
        return imgs


class EncUp(nn.Module):
    def __init__(
        self, stages, channels, max_channels, in_channels=3, conv_layer=NormConv2d,
    ):
        super().__init__()
        # number of residual block per scale
        self.n_resnet_blocks = 2
        self.n_stages = stages
        self.nin = conv_layer(
            in_channels=in_channels, out_channels=channels, kernel_size=1
        )

        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(self.n_stages):
            for n in range(self.n_resnet_blocks):
                self.blocks.append(VunetRNB(channels=channels, conv_layer=conv_layer))

            if i + 1 < self.n_stages:
                n_channels_out = min(2 * channels, max_channels)
                self.downs.append(Downsample(channels, n_channels_out))
                channels = n_channels_out

    def forward(self, x, **kwargs):
        # x is an image, which defines the appearance of a person
        hs = []

        h = self.nin(x)

        for i in range(self.n_stages):

            for n in range(self.n_resnet_blocks):
                h = self.blocks[2 * i + n](h)
                hs.append(h)

            if i + 1 < self.n_stages:
                h = self.downs[i](h)

        return hs


def latent_sample(p):
    mean = p
    stddev = 1.0
    eps = torch.randn_like(mean)

    return mean + stddev * eps


class EncDown(nn.Module):
    def __init__(
        self,
        channels,
        in_channels,
        stages=2,
        subpixel_upsampling=False,
        conv_layer=NormConv2d,
    ):
        super().__init__()
        self.nin = conv_layer(in_channels, channels, kernel_size=1)
        self.n_stages = stages
        self.n_resent_blocks = 2
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.make_latent_params = nn.ModuleList()
        for i in range(self.n_stages):

            for n in range(self.n_resent_blocks // 2):
                self.blocks.append(
                    VunetRNB(channels=channels, a_channels=channels, residual=True)
                )

            self.make_latent_params.append(
                conv_layer(channels, channels, kernel_size=3, padding=1)
            )

            for n in range(self.n_resent_blocks // 2):
                self.blocks.append(
                    VunetRNB(channels=channels, a_channels=2 * channels, residual=True)
                )

            self.ups.append(Upsample(channels, channels, subpixel=subpixel_upsampling))

        self.fin_block = VunetRNB(channels=channels, a_channels=channels, residual=True)

    def forward(self, gs):
        hs = []  # hidden units
        qs = []  # posteriors
        zs = []  # samples from posterior

        h = self.nin(gs[-1])
        for i in range(self.n_stages):

            h = self.blocks[2 * i](h, gs.pop())
            hs.append(h)

            # post params
            q = self.make_latent_params[i](h)
            qs.append(q)

            # post sample
            z = latent_sample(q)
            zs.append(z)

            gz = torch.cat([gs.pop(), z], dim=1)
            h = self.blocks[2 * i + 1](h, gz)
            hs.append(h)

            h = self.ups[i](h)

        # last resnet_block
        h = self.fin_block(h, gs.pop())
        hs.append(h)
        return hs, qs, zs


class DecUp(nn.Module):
    def __init__(
        self,
        stages,
        n_channels_out,
        n_channels_max,
        n_channels_in=3,
        conv_layer=NormConv2d,
    ):
        super().__init__()
        # number of residual block per scale
        self.n_resnet_blocks = 2
        self.n_stages = stages
        self.nin = conv_layer(
            in_channels=n_channels_in, out_channels=n_channels_out, kernel_size=1
        )

        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(self.n_stages):
            for n in range(self.n_resnet_blocks):
                self.blocks.append(
                    VunetRNB(channels=n_channels_out, conv_layer=conv_layer)
                )

            if i + 1 < self.n_stages:
                out_c = min(2 * n_channels_out, n_channels_max)
                self.downs.append(Downsample(n_channels_out, out_c))
                n_channels_out = out_c

    def forward(self, c):
        # x is an image, which defines the shape and body pose of the person
        hs = []

        h = self.nin(c)

        for i in range(self.n_stages):

            for n in range(self.n_resnet_blocks):
                h = self.blocks[2 * i + n](h)
                hs.append(h)

            if i + 1 < self.n_stages:
                h = self.downs[i](h)

        return hs


class DecDown(nn.Module):
    def __init__(
        self,
        stages,
        in_channels,
        last_channels,
        out_channels,
        conv_layer=NormConv2d,
        subpixel_upsampling=False,
        n_latent_stages=2,
    ):
        super().__init__()
        self.n_rnb = 2
        self.n_stages = stages
        self.n_latent_stages = n_latent_stages
        self.nin = conv_layer(in_channels, in_channels, kernel_size=1)
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        # autoregressive stuff
        self.latent_nins = nn.ModuleDict()
        self.auto_lp = nn.ModuleDict()
        self.auto_blocks = nn.ModuleDict()
        # last conv
        self.out_conv = conv_layer(
            last_channels, out_channels, kernel_size=3, padding=1
        )
        # for reordering
        self.depth_to_space = DepthToSpace(block_size=2)
        self.space_to_depth = SpaceToDepth(block_size=2)

        n_latent_channels_in = in_channels

        in_channels = in_channels
        for i in range(self.n_stages):

            for n in range(self.n_rnb // 2):
                self.blocks.append(
                    VunetRNB(
                        channels=in_channels,
                        a_channels=in_channels,
                        residual=True,
                        conv_layer=conv_layer,
                    )
                )

            if i < self.n_latent_stages:
                scale = f"l_{i}"
                self.latent_nins.update(
                    {
                        scale: conv_layer(
                            n_latent_channels_in * 2,
                            n_latent_channels_in,
                            kernel_size=1,
                        )
                    }
                )

                # autoregressive_stuff
                clp = ModuleList()
                cb = ModuleList()
                for l in range(4):

                    clp.append(
                        conv_layer(
                            4 * n_latent_channels_in,
                            n_latent_channels_in,
                            kernel_size=3,
                            padding=1,
                        )
                    )
                    if l == 0:
                        cb.append(VunetRNB(channels=n_latent_channels_in))
                    else:
                        cb.append(
                            VunetRNB(
                                channels=4 * n_latent_channels_in,
                                a_channels=n_latent_channels_in,
                                residual=True,
                            )
                        )

                self.auto_lp.update({scale: clp})
                self.auto_blocks.update({scale: cb})

            for n in range(self.n_rnb // 2):
                self.blocks.append(
                    VunetRNB(
                        channels=in_channels,
                        a_channels=in_channels,
                        residual=True,
                        conv_layer=conv_layer,
                    )
                )

            if i + 1 < self.n_stages:
                out_c = min(in_channels, last_channels * 2 ** (stages - (i + 2)))
                self.ups.append(
                    Upsample(
                        in_channels,
                        out_c,
                        subpixel=subpixel_upsampling
                        if i < self.n_latent_stages
                        else False,
                    )
                )
                in_channels = out_c

    def forward(self, gs, zs_posterior, training):
        gs = list(gs)
        zs_posterior = list(zs_posterior)

        hs = []
        ps = []
        zs = []

        h = self.nin(gs[-1])
        for i in range(self.n_stages):

            h = self.blocks[2 * i](h, gs.pop())
            hs.append(h)

            if i < self.n_latent_stages:
                scale = f"l_{i}"
                if training:
                    zs_posterior_groups = self.__split_groups(zs_posterior[0])
                p_groups = []
                z_groups = []
                pre = self.auto_blocks[scale][0](h)
                p_features = self.space_to_depth(pre)

                for l in range(4):
                    p_group = self.auto_lp[scale][l](p_features)
                    p_groups.append(p_group)
                    z_group = latent_sample(p_group)
                    z_groups.append(z_group)

                    if training:
                        feedback = zs_posterior_groups.pop(0)
                    else:
                        feedback = z_group

                    if l + 1 < 4:
                        p_features = self.auto_blocks[scale][l + 1](
                            p_features, feedback
                        )
                if training:
                    assert not zs_posterior_groups

                p = self.__merge_groups(p_groups)
                ps.append(p)

                z_prior = self.__merge_groups(z_groups)
                zs.append(z_prior)

                if training:
                    z = zs_posterior.pop(0)
                else:
                    z = z_prior

                h = torch.cat([h, z], dim=1)
                h = self.latent_nins[scale](h)
                h = self.blocks[2 * i + 1](h, gs.pop())
                hs.append(h)
            else:
                h = self.blocks[2 * i + 1](h, gs.pop())
                hs.append(h)

            if i + 1 < self.n_stages:
                h = self.ups[i](h)

        assert not gs
        if training:
            assert not zs_posterior

        params = self.out_conv(hs[-1])

        # returns imgs, activations, prior params and samples
        return params, hs, ps, zs

    def __split_groups(self, x):
        # split along channel axis
        sec_size = x.shape[1]
        return list(torch.split(self.space_to_depth(x), sec_size, dim=1))

    def __merge_groups(self, x):
        # merge groups along channel axis
        return self.depth_to_space(torch.cat(x, dim=1))
