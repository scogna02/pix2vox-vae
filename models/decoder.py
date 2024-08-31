# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>

import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, cfg, latent_dim=128):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim

        # Fully connected layer to transform latent vector
        self.fc = nn.Linear(latent_dim, 2048 * 2 * 2 * 2)

        # Layer Definitions
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(2048, 512, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.size(0)
        n_views = self.cfg.CONST.N_VIEWS_RENDERING  # Assuming this is defined in your config

        # Transform the latent vector z into the shape suitable for the first ConvTranspose3d layer
        x = self.fc(z)
        x = x.view(batch_size, 2048, 2, 2, 2)

        # Replicate the features for each view
        x = x.unsqueeze(1).repeat(1, n_views, 1, 1, 1, 1)
        x = x.view(batch_size * n_views, 2048, 2, 2, 2)

        # Pass through deconvolution layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        raw_feature = x
        x = self.layer5(x)

        # Combine raw_feature and generated volume
        raw_feature = torch.cat((raw_feature, x), dim=1)

        # Reshape to match the original output format
        gen_volumes = x.view(batch_size, n_views, 32, 32, 32)
        raw_features = raw_feature.view(batch_size, n_views, 9, 32, 32, 32)

        return raw_features, gen_volumes
    
class Decoder_or(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder_or, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(2048, 512, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
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
        gen_volumes = []
        raw_features = []

        for features in image_features:
            gen_volume = features.view(-1, 2048, 2, 2, 2)
            # print(gen_volume.size())   # torch.Size([batch_size, 2048, 2, 2, 2])
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
        # print(raw_features.size())     # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_features, gen_volumes



"""
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, cfg, latent_dim=128):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim

        # Fully connected layer to transform latent vector to match the first deconv layer input
        self.fc = nn.Linear(latent_dim, 2048 * 2 * 2 * 2)  # Adjust dimensions to fit ConvTranspose3d input

        # Layer Definitions
        self.layer1 = nn.Sequential(
            nn.ConvTranspose3d(2048, 512, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, bias=cfg.NETWORK.TCONV_USE_BIAS, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )
        self.layer5 = nn.Sequential(
            nn.ConvTranspose3d(8, 1, kernel_size=1, bias=cfg.NETWORK.TCONV_USE_BIAS),
            nn.Sigmoid()
        )

    def forward(self, z):
        # Transform the latent vector z into the shape suitable for the first ConvTranspose3d layer
        z = self.fc(z)  # Fully connected layer
        z = z.view(-1, 2048, 2, 2, 2)  # Reshape to (batch_size, 2048, 2, 2, 2)

        # Pass through deconvolution layers
        gen_volume = self.layer1(z)
        gen_volume = self.layer2(gen_volume)
        gen_volume = self.layer3(gen_volume)
        gen_volume = self.layer4(gen_volume)
        raw_feature = gen_volume
        gen_volume = self.layer5(gen_volume)
        raw_feature = torch.cat((raw_feature, gen_volume), dim=1)

        # Squeeze and stack outputs
        gen_volume = torch.squeeze(gen_volume, dim=1)
        return raw_feature, gen_volume
"""