import torch
import torch.nn as nn
import torch.nn.functional as F

class Fusion(nn.Module):
    def __init__(self, cfg):
        super(Fusion, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = nn.Sequential(
            nn.Conv3d(9, 9, kernel_size=3, padding=1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer2 = nn.Sequential(
            nn.Conv3d(9, 9, kernel_size=3, padding=1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(9, 9, kernel_size=3, padding=1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(9, 9, kernel_size=3, padding=1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(36, 9, kernel_size=3, padding=1),
            nn.BatchNorm3d(9),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer6 = nn.Sequential(
            nn.Conv3d(9, 1, kernel_size=3, padding=1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )

    def forward(self, volume_vl, raw_vl, volume_dino, raw_dino ):
        # volume_weights = []
        raw_dino = raw_dino.squeeze(1)
        volume_vl = volume_vl.squeeze(1)
        volume_dino = volume_dino.squeeze(1)
        raw_dino = raw_dino.squeeze(1)
        
        # print("raw_dino shape", raw_dino.shape)
        volume_weight1 = self.layer1(raw_dino)
        volume_weight2 = self.layer2(volume_weight1)
        volume_weight3 = self.layer3(volume_weight2)
        volume_weight4 = self.layer4(volume_weight3)
        volume_weight = self.layer5(torch.cat([
            volume_weight1, volume_weight2, volume_weight3, volume_weight4
        ], dim=1))
        volume_weight = self.layer6(volume_weight)

        # print("Before the softmax", volume_weight.shape)
        volume_weight = F.softmax(volume_weight, dim=1)
        volume_vl = volume_vl * volume_weight
        # print("volume_vl--------------", volume_vl.shape)
        volume_vl = torch.sum(volume_vl, dim=1)
        
        volume_vl = volume_vl*volume_dino
        return torch.clamp(volume_vl, min=0, max=1)



class ConvFusion(nn.Module):
    def __init__(self, cfg):
        super(ConvFusion, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer_dino_1 = nn.Sequential(
            nn.Conv3d(1, 36, kernel_size=3, padding=1),
            nn.BatchNorm3d(36),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer_dino_2 = nn.Sequential(
            nn.Conv3d(36, 1, kernel_size=3, padding=1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        
        self.layer_flava_1 = nn.Sequential(
            nn.Conv3d(1, 36, kernel_size=3, padding=1),
            nn.BatchNorm3d(36),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )
        self.layer_flava_2 = nn.Sequential(
            nn.Conv3d(36, 1, kernel_size=3, padding=1),
            nn.BatchNorm3d(1),
            nn.LeakyReLU(cfg.NETWORK.LEAKY_VALUE)
        )


    def forward(self, volume_vl, volume_dino ):

        # volume_vl = volume_vl.squeeze(1)
        # volume_dino = volume_dino.squeeze(1)
        # print("volume_dino ----------------------:", volume_dino.shape)
        # print("volume_vl ------------------------:", volume_dino.shape)
        volume_dino_weight = self.layer_dino_1(volume_dino)
        volume_dino_weight = self.layer_dino_2(volume_dino_weight)
        
        volume_flava_weight = self.layer_flava_1(volume_vl)
        volume_flava_weight = self.layer_flava_2(volume_flava_weight)
        total_weights = volume_flava_weight + volume_dino_weight
        return torch.clamp(total_weights, min=0, max=1)
