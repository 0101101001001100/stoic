# Variational autoencoder with ResNet backbone

import numpy as np
from collections import OrderedDict
from typing import Tuple, Optional, Callable, List, Type, Any, Union
import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchvision.models.video.resnet import Conv3DSimple, BasicBlock, Bottleneck, VideoResNet
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from .convnets import ResNetStem
from .utils import reconstruction_loss, vae_loss


class STOICAutoEncoder(LightningModule):
    def __init__(self, 
        latent_dim: int=512, 
        shape: Tuple[int]=(512, 8, 8, 8), 
        batch_size: int=8, 
        max_epochs: int=200,
        train: bool=True
    ) -> None:
        '''
        Lightning module for a deep convolutional autoencoder.

        Args:
            `latent_dim`: size of the latent vector $z$
            `shape`: shape of the deepest Conv3D feature map given to flatten/unflatten
        '''
        super(STOICAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = ResEncoder(
            block=BasicBlock,
            conv_makers=[Conv3DSimple] * 4,
            layers=[2, 2, 2, 2],
            stem=ResNetStem,
            latent_dim=self.latent_dim
        )
        if train:
            self.encoder.load_state_dict(torch.load('models/r18_k400_encoder.pt'))
        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            shape=shape
        )
        self.loss_function = reconstruction_loss
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        x = batch['image']['data']
        x_hat = self.forward(x)
        loss = self.loss_function(x, x_hat)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']['data']
        x_hat = self.forward(x)
        loss = self.loss_function(x, x_hat)
        self.log("val_loss", loss, batch_size=self.batch_size)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.encoder.stem.parameters()},
            {'params': self.encoder.layer1.parameters(), 'lr': 5e-5},
            {'params': self.encoder.layer2.parameters(), 'lr': 5e-5},
            {'params': self.encoder.layer3.parameters(), 'lr': 5e-5},
            {'params': self.encoder.layer4.parameters()},
            {'params': self.encoder.fc.parameters()},
            {'params': self.decoder.parameters()}
            ], lr=5e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=1, max_epochs=self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class STOICVarAutoEncoder(LightningModule):
    def __init__(self, 
        latent_dim: int=512, 
        shape: Tuple[int]=(512, 8, 8, 8), 
        batch_size: int=8, 
        max_epochs: int=200,
        train: bool=True
    ) -> None:
        '''
        Lightning module for a variational autoencoder.

        Args:
            `latent_dim`: size of the latent vector $z$
            `shape`: shape of the deepest Conv3D feature map given to flatten/unflatten
        '''
        super(STOICVarAutoEncoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = VarResEncoder(
            block=BasicBlock,
            conv_makers=[Conv3DSimple] * 4,
            layers=[2, 2, 2, 2],
            stem=ResNetStem,
            latent_dim=self.latent_dim
        )
        # if train:
        #     self.encoder.load_state_dict(torch.load('models/r18_k400_varencoder.pt'))
        self.decoder = ConvDecoder(
            latent_dim=latent_dim,
            shape=shape
        )
        self.criterion = vae_loss
        self.batch_size = batch_size
        self.max_epochs = max_epochs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, mu, logvar = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def training_step(self, batch, batch_idx):
        x = batch['image']['data']
        x_hat, mu, logvar = self.forward(x)
        loss, rec, kld = self.criterion(x, x_hat, mu, logvar)
        self.log_dict({
                "train_loss": loss, 
                "train_rec_loss": rec,
                "train_kld_loss": kld,
            }, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch['image']['data']
        x_hat, mu, logvar = self.forward(x)
        loss, rec, kld = self.criterion(x, x_hat, mu, logvar)
        self.log_dict({
                "val_loss": loss, 
                "val_rec_loss": rec,
                "val_kld_loss": kld,
            }, batch_size=self.batch_size)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam([
        #     {'params': self.encoder.stem.parameters()},
        #     {'params': self.encoder.layer1.parameters(), 'lr': 5e-5},
        #     {'params': self.encoder.layer2.parameters(), 'lr': 5e-5},
        #     {'params': self.encoder.layer3.parameters(), 'lr': 5e-5},
        #     {'params': self.encoder.layer4.parameters()},
        #     {'params': self.encoder.fc.parameters()},
        #     {'params': self.encoder.fc_mu.parameters()},
        #     {'params': self.encoder.fc_logvar.parameters()},
        #     {'params': self.decoder.parameters()}
        #     ], lr=1e-4)
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=1, max_epochs=self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


class ResEncoder(VideoResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        conv_makers: Conv3DSimple,
        layers: List[int],
        stem: Callable[..., nn.Module],
        latent_dim: int = 512,
        shape: Tuple[int] = (512, 8, 8, 8),
        zero_init_residual: bool = False
    ) -> None:
        """
        3D ResNet encoder modified from `torchvision.models.video.resnet.VideoResNet`
        """
        super(ResEncoder, self).__init__(block, conv_makers, layers, stem)
        self.fc = nn.Sequential(nn.Linear(in_features=np.prod(shape), out_features=latent_dim), nn.ReLU(True))
        self._initialize_weights()

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(x.flatten(1))
        return x


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, shape: Tuple[int]) -> None:
        super(ConvDecoder, self).__init__()
        self.delatent = nn.Sequential(
            nn.Linear(latent_dim, np.prod(shape)),
            nn.Unflatten(1, shape)
        )
        self.deconv = nn.Sequential(OrderedDict([
            ['conv_in', nn.Sequential(
                nn.Conv3d(512, 256, kernel_size=3, padding=1),
                nn.BatchNorm3d(256),
                nn.LeakyReLU(True)
            )],
            # (1, 512, 8, 8, 8) 
            ['tconv1', nn.Sequential(
                nn.ConvTranspose3d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm3d(256),
                nn.LeakyReLU(inplace=True)
            )],
            # (1, 256, 16, 16, 16)
            ['tconv2', nn.Sequential(
                nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm3d(128),
                nn.LeakyReLU(inplace=True)
            )],
            # (1, 128, 32, 32, 32)
            ['tconv3', nn.Sequential(
                nn.ConvTranspose3d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm3d(64),
                nn.LeakyReLU(inplace=True)
            )],
            # (1, 64, 64, 64, 64)
            ['tconv4', nn.Sequential(
                nn.ConvTranspose3d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm3d(64),
                nn.LeakyReLU(inplace=True)
            )],
            # (1, 64, 128, 128, 128)
            ['conv_out', nn.Sequential(
                nn.Conv3d(64, 1, kernel_size=3, padding=1),
                nn.BatchNorm3d(1),
                nn.LeakyReLU(inplace=True)
            )]
        ]))
        self.upscale = Interpolate(scale_factor=2)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.delatent(x)
        x = self.deconv(x)
        x = self.upscale(x)
        return x


class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode='trilinear')


class DecoderBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConvTranspose3d(inplanes, inplanes, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(inplanes),
            nn.ReLU(inplace=True)
            )
        self.conv2 = nn.Sequential(
            nn.Conv3d(inplanes, planes, 3),
            nn.BatchNorm2d(inplanes)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        out = self.relu(out)
        return out


class VarResEncoder(VideoResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        conv_makers: Conv3DSimple,
        layers: List[int],
        stem: Callable[..., nn.Module],
        latent_dim: int = 512,
        zero_init_residual: bool = False
    ) -> None:
        """
        3D ResNet variational encoder modified from `torchvision.models.video.resnet.VideoResNet`
        """
        super(VarResEncoder, self).__init__(block, conv_makers, layers, stem)
        self.fc = nn.Linear(in_features=1 * 512 * 8 ** 3, out_features=512)
        self.fc_mu = torch.nn.Linear(in_features=512, out_features=latent_dim)
        self.fc_logvar = torch.nn.Linear(in_features=512, out_features=latent_dim)
        # initialize weights
        self._initialize_weights()

    def sample(self, mu, logvar):
        # reparameterize
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z

    def forward(self, x: torch.Tensor):
        # convolutional encoding
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.flatten(1)
        # MLP
        x = self.fc(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.sample(mu, logvar)
        return z, mu, logvar