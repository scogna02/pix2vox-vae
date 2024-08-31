# -*- coding: utf-8 -*-
#
# Developed by Haozhe Xie <cshzxie@gmail.com>
#
# References:
# - https://github.com/shawnxu1318/MVCNN-Multi-View-Convolutional-Neural-Networks/blob/master/mvcnn.py

import torch
import torchvision.models

import torch
import torch.nn as nn
import torchvision

class Encoder(nn.Module):
    def __init__(self, cfg, latent_dim=128):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim

        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = nn.Sequential(*list(vgg16_bn.features.children())[:27])
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ELU()
        )

        # Fully connected layers for mu and log_sigma
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_log_sigma = nn.Linear(256 * 8 * 8, latent_dim)

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, rendering_images):
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.vgg(img.squeeze(dim=0))
            features = self.layer1(features)
            features = self.layer2(features)
            features = self.layer3(features)
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        batch_size, n_views, c, h, w = image_features.size()
        image_features = image_features.view(batch_size, n_views, -1)  # Flatten
        image_features = image_features.mean(dim=1)  # Aggregate features over views

        # Get mu and log_sigma
        mu = self.fc_mu(image_features)
        log_sigma = self.fc_log_sigma(image_features)

        # Reparameterize to get z
        z = self.reparameterize(mu, log_sigma)

        return mu, log_sigma, z

    

class Encoder_or(torch.nn.Module):
    def __init__(self, cfg):
        super(Encoder_or, self).__init__()
        self.cfg = cfg

        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.vgg(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 26, 26])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 512, 24, 24])
            features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 256, 8, 8])
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        #print(image_features.size())  # torch.Size([batch_size, n_views, 256, 8, 8])
        return image_features


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class Encoder(nn.Module):
    def __init__(self, cfg, latent_dim=128):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.latent_dim = latent_dim

        # Layer Definition
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=True)
        self.vgg = nn.Sequential(*list(vgg16_bn.features.children())[:27])
        self.layer1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ELU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ELU()
        )

        # Fully connected layers for mu and log_sigma
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_log_sigma = nn.Linear(256 * 8 * 8, latent_dim)

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = False

    def reparameterize(self, mu, log_sigma):
        std = torch.exp(0.5 * log_sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.vgg(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 26, 26])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 512, 24, 24])
            features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 256, 8, 8])
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # Flatten the features to pass through fully connected layers
        batch_size, n_views, c, h, w = image_features.size()
        image_features = image_features.view(batch_size, n_views, -1)  # Flatten
        image_features = image_features.mean(dim=1)  # Aggregate features over views

        # Get mu and log_sigma
        mu = self.fc_mu(image_features)
        log_sigma = self.fc_log_sigma(image_features)

        # Reparameterize to get z
        z = self.reparameterize(mu, log_sigma)

        return mu, log_sigma, z
        """