import torch

from functools import partial

from torch import nn

from .attention import Attention
from .modules import (
    groupby_prefix_and_trim,
    equals,
    Residual,
    FeedForward,
    LayerIntermediates
)
from torch import Tensor

class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1568, 512, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            torch.nn.Sigmoid()
        )

    def forward(self, image_features):
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        raw_features = []
        gen_volumes = []

        for features in image_features:
            gen_volume = features.view(-1, 1568, 2, 2, 2)
            # print(gen_volume.size())   # torch.Size([batch_size, 1568, 2, 2, 2])
            gen_volume = self.layer1(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
            gen_volume = self.layer2(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 128, 8, 8, 8])
            gen_volume = self.layer3(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 32, 16, 16, 16])
            gen_volume = self.layer4(gen_volume)
            raw_feature = gen_volume
            # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])
            gen_volume = self.layer5(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
            raw_feature = torch.cat((raw_feature, gen_volume), dim=1)
            # print(raw_feature.size())  # torch.Size([batch_size, 9, 32, 32, 32])
            gen_volumes.append(torch.squeeze(gen_volume, dim=1))
            raw_features.append(raw_feature)

        gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        # print(gen_volumes.size())      # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(raw_features.size())      # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_features, gen_volumes

##########################################################Decoer from 3DR
from einops import rearrange
from .losses import DiceLoss, CEDiceLoss, FocalLoss

class VoxelDecoderTr(nn.Module):
    def __init__(
            self,
            patch_num: int = 4,
            num_cnn_layers: int = 3,
            num_resnet_blocks: int = 2,
            cnn_hidden_dim: int = 64,
            voxel_size: int = 32,
            dim: int = 768, #512
            depth: int = 6,
            heads: int = 8,
            dim_head: int = 64,
            attn_dropout: float = 0.0,
            ff_dropout: float = 0.0,
    ):
        super().__init__()

        if voxel_size % patch_num != 0:
            raise ValueError('voxel_size must be dividable by patch_num')

        self.patch_num = patch_num
        self.voxel_size = voxel_size
        self.patch_size = voxel_size // patch_num
        self.emb = nn.Embedding(patch_num ** 3, dim)
        self.transformer = AttentionLayers(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=False,
            cross_attend=True
        )

        has_resblocks = num_resnet_blocks > 0
        dec_chans = [cnn_hidden_dim] * num_cnn_layers
        dec_init_chan = dim if not has_resblocks else dec_chans[0]
        dec_chans = [dec_init_chan, *dec_chans]

        dec_chans_io = list(zip(dec_chans[:-1], dec_chans[1:]))

        dec_layers = []

        for (dec_in, dec_out) in dec_chans_io:
            dec_layers.append(nn.Sequential(nn.ConvTranspose3d(dec_in, dec_out, 4, stride=2, padding=1), nn.ReLU()))

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(dec_chans[1]))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, nn.Conv3d(dim, dec_chans[1], 1))

        dec_layers.append(nn.Conv3d(dec_chans[-1], 1, 1))

        self.decoder = nn.Sequential(*dec_layers)

        self.layer_norm = nn.LayerNorm(dim)
        self.to_patch = nn.Linear(dim, self.patch_size ** 3)

    def generate(
            self,
            context: Tensor,
            context_mask: Tensor = None,
            **kwargs
    ):
        out = self(context, context_mask)
        return torch.sigmoid(out)

    def forward(
            self,
            context: Tensor,
            context_mask: Tensor = None
    ) -> Tensor:
        x = self.emb(torch.arange(self.patch_num ** 3, device=context.device))
        x = x.unsqueeze(0).repeat(context.shape[0], 1, 1)
        out = self.transformer(x=x, context=context, context_mask=context_mask)
        out = self.layer_norm(out)
        out = rearrange(out, 'b (h w c) d -> b d h w c', h=self.patch_num, w=self.patch_num, c=self.patch_num)
        out = self.decoder(out)
        # print("Inner output ofr decoder shape---------------------------:", out.shape)
        return out

    def get_loss(
            self,
            x: Tensor,
            context: Tensor,
            context_mask: Tensor = None,
            loss_type='dice'
    ):
        out = self(context, context_mask)
        out = out.view(out.size(0), -1)
        x = x.view(out.size(0), -1)

        if loss_type == 'ce':
            loss_fn = F.binary_cross_entropy_with_logits
        elif loss_type == 'dice':
            loss_fn = DiceLoss()
        elif loss_type == 'ce_dice':
            loss_fn = CEDiceLoss()
        elif loss_type == 'focal':
            loss_fn = FocalLoss()
        else:
            raise ValueError(f'Unsupported loss type "{loss_type}"')
        print("out Before the loss_fn shapes-------------------:", out.shape)
        print("x Before the loss_fn shapes---------------------:", x.shape)
        return loss_fn(out, x)




class ResBlock(nn.Module):
    def __init__(self, chan):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(chan, chan, 3, padding=1),
            nn.ReLU(),
            nn.Conv3d(chan, chan, 1)
        )

    def forward(self, x):
        return self.net(x) + x

class AttentionLayers(nn.Module):
    def __init__(
            self,
            dim: int,
            depth: int,
            heads: int,
            causal: bool = False,
            cross_attend: bool = False,
            only_cross: bool = False,
            **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])

        norm_fn = partial(nn.LayerNorm, dim)

        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        for layer_type in self.layer_types:
            if layer_type == 'a':
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            residual_fn = Residual()

            self.layers.append(nn.ModuleList([
                norm_fn(),
                layer,
                residual_fn
            ]))

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None,
            return_hiddens=False
    ):
        hiddens = []
        intermediates = []

        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            is_last = ind == (len(self.layers) - 1)

            if layer_type == 'a':
                hiddens.append(x)

            residual = x
            x = norm(x)

            if layer_type == 'a':
                out, inter = block(x, mask=mask)
            elif layer_type == 'c':
                out, inter = block(x, context=context, mask=mask, context_mask=context_mask)
            elif layer_type == 'f':
                out = block(x)

            x = residual_fn(out, residual)

            if layer_type in ('a', 'c'):
                intermediates.append(inter)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens,
                attn_intermediates=intermediates
            )

            return x, intermediates

        return x
