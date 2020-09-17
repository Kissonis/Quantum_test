"""
UNet semantic segmentation with PyTorch
"""

import argparse
import os
from multiprocessing.dummy import freeze_support

import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torch import multiprocessing
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from unet_models_1 import *
from datasets_img_mask import *

import torch.nn as nn
import torch.nn.functional as F
import torch

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1
        num = targets.size(0)
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

if __name__ == '__main__':
    #multiprocessing.freeze_support()
    path_out_1_class = "images_1class/"
    path_test_images = "images_test_1class/"
    path_saved_models = "saved_models_1class/"
    os.makedirs(path_out_1_class, exist_ok=True)
    os.makedirs(path_test_images, exist_ok=True)
    os.makedirs(path_saved_models, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=10, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=1, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="data", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="size of the  batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=2, help="number of cpu threads to use during batch generation")
    parser.add_argument("--hr_height", type=int, default=128, help="image height")
    parser.add_argument("--hr_width", type=int, default=128, help="image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
    parser.add_argument("--mask_treshold", type=float, default=0.78, help="the trshold must match to image_3to1_chennel.py converter")
    opt = parser.parse_args()
    print(opt)

    cuda = torch.cuda.is_available()

    hr_shape = (opt.hr_height, opt.hr_width)

    # Initialize generator
    num_classes = 1
    num_channels = 3
    generator = UNet(num_channels, num_classes)

    # Losses
    # dice coefficient as loss function
    criterion_UNet = SoftDiceLoss()

    if cuda:
        generator = generator.cuda()
        criterion_UNet = criterion_UNet.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load(path_saved_models + "generator_%d.pth" % opt.epoch))

    # Optimizers
    optimizer_UNet = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    trainloader = DataLoader(
        TrainImageDataset("././%s" %opt.dataset_name, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    testloader = DataLoader(
        TestImageDataset("././%s" %opt.dataset_name, hr_shape=hr_shape),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    # ----------
    #  Training
    # ----------

    print('Training process evaluation ...')
    for epoch in range(opt.epoch, opt.n_epochs):
        print()
        for i, batch in enumerate(trainloader):

            # Configure model input
            imgs = Variable(batch["images"].type(Tensor))
            #masks = Variable(masks["masks"].type(Tensor))
            masks = batch["masks"].type(Tensor)
            target = masks

            # ------------------
            #  Train Generator-UNet
            # ------------------

            optimizer_UNet.zero_grad()

            # Generate the masks
            gen_masks = generator(imgs)

            # Loss
            imgs_batch = len(batch["images"])
            loss_UNet = criterion_UNet(gen_masks, target)

            loss_UNet.backward()
            optimizer_UNet.step()

            # # --------------
            # #  Log Progress
            # # --------------

            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [UNet loss: %f]"
                % (epoch, opt.n_epochs, i, len(trainloader), loss_UNet.item())
            )

            batches_done = epoch * len(trainloader) + i
            if batches_done % opt.sample_interval == 0:
                # Save image grid with training images, inputed and generated masks
                gen_masks[gen_masks > opt.mask_treshold] = 1
                gen_masks[gen_masks <= opt.mask_treshold] = 0
                imgs_lr = nn.functional.interpolate(imgs, scale_factor=4)
                imgs = make_grid(imgs, nrow=1, normalize=True)
                gen_masks = make_grid(gen_masks, nrow=1, normalize=True)
                masks = make_grid(masks, nrow=1, normalize=True)
                img_grid = torch.cat((imgs, masks, gen_masks), -1)
                save_image(img_grid, path_out_1_class + "%d.png" % batches_done, normalize=False)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), path_saved_models + "generator_%d.pth" % epoch)

    # Test
    print()
    print('Testing process evaluation ...')
    generator.eval()
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            imgs = batch["images"].type(Tensor)
            gen_masks = generator(imgs)
            gen_masks[gen_masks > opt.mask_treshold] = 1
            gen_masks[gen_masks <= opt.mask_treshold] = 0
            # Save image grid with testing images and generated masks
            imgs_lr = nn.functional.interpolate(imgs, scale_factor=4)
            imgs = make_grid(imgs, nrow=1, normalize=True)
            gen_masks = make_grid(gen_masks, nrow=1, normalize=True)
            img_grid = torch.cat((imgs, gen_masks), -1)
            save_image(img_grid, path_test_images + "%d.png" % i, normalize=False)
    print('Testing process finished!')

