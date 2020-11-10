import argparse
import os
import os.path as osp
import numpy as np
import math
import itertools
import copy
import pickle
from sys import exit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, LeakyReLU
from torch.autograd import Variable
from torch.distributions import normal, kl

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import NNConv, BatchNorm, EdgePooling, TopKPooling, global_add_pool
from torch_geometric.utils import get_laplacian, to_dense_adj

import matplotlib.pyplot as plt

from data_utils import MRDataset, create_edge_index_attribute, swap, cross_val_indices, MRDataset2
from model import Generator, Discriminator
from plot import plot, plot_matrix

torch.manual_seed(0)  # To get the same results across experiments

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('running on GPU')
else:
    device = torch.device("cpu")
    print('running on CPU')

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--lr_g', type=float, default=0.01, help='Generator learning rate')
parser.add_argument('--lr_d', type=float, default=0.0002, help='Discriminator learning rate')
parser.add_argument('--loss', type=str, default='BCE', help='Which loss to use for training',
                    choices=['BCE', 'LS'])
parser.add_argument('--batch', type=int, default=1, help='Batch Size')
parser.add_argument('--epoch', type=int, default=500, help='How many epochs to train')
parser.add_argument('--folds', type=int, default=3, help='How many folds for CV')
parser.add_argument('--tr_st', type=str, default='same', help='Training strategy',
                    choices=['same', 'turns', 'idle'])
parser.add_argument('--id_e', type=int, default=2, help='If training strategy is idle, for how many epochs')
parser.add_argument('--exp', type=int, default=0, help='Which experiment are you running')
parser.add_argument('--tp_c', type=float, default=0.0, help='Coefficient of topology loss')
parser.add_argument('--g_c', type=float, default=2.0, help='Coefficient of adversarial loss')
parser.add_argument('--i_c', type=float, default=2.0, help='Coefficient of identity loss')
parser.add_argument('--kl_c', type=float, default=0.001, help='Coefficient of KL loss')
parser.add_argument('--decay', type=float, default=0.0, help='Weight Decay')
opt = parser.parse_args()

# Datasets

h_data = MRDataset2("../data", "lh", subs=989)

# Parameters

batch_size = opt.batch
lr_G = opt.lr_g
lr_D = opt.lr_d
num_epochs = opt.epoch
folds = opt.folds

connectomes = 1
train_generator = 1

# Coefficients for loss
i_coeff = opt.i_c
g_coeff = opt.g_c
kl_coeff = opt.kl_c
tp_coeff = opt.tp_c

if opt.tr_st != 'idle':
    opt.id_e = 0

# Training

loss_dict = {"BCE": torch.nn.BCELoss().to(device),
             "LS": torch.nn.MSELoss().to(device)}


adversarial_loss = loss_dict[opt.loss.upper()]
identity_loss = torch.nn.L1Loss().to(device)  # Will be used in training
msel = torch.nn.MSELoss().to(device)
mael = torch.nn.L1Loss().to(device)  # Not to be used in training (Measure generator success)
counter_g, counter_d = 0, 0
tp = torch.nn.MSELoss().to(device) # Used for node strength

train_ind, val_ind = cross_val_indices(folds, len(h_data))

# Saving the losses for the future
gen_mae_losses_tr = None
disc_real_losses_tr = None
disc_fake_losses_tr = None
gen_mae_losses_val = None
disc_real_losses_val = None
disc_fake_losses_val = None
gen_mae_losses_tr2 = None
disc_real_losses_tr2 = None
disc_fake_losses_tr2 = None
gen_mae_losses_val2 = None
disc_real_losses_val2 = None
disc_fake_losses_val2 = None
k1_train_s = None
k2_train_s = None
k1_val_s = None
k2_val_s = None
tp1_train_s = None
tp2_train_s = None
tp1_val_s = None
tp2_val_s = None
gan1_train_s = None
gan2_train_s = None
gan1_val_s = None
gan2_val_s = None

# Cross Validation
for fold in range(folds):
    train_set, val_set = h_data[list(train_ind[fold])], h_data[list(val_ind[fold])]
    h_data_train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    h_data_test_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    val_step = len(h_data_test_loader)

    for data in h_data_train_loader:  # Determine the maximum number of samples in a batch
        data_size = data.x.size(0)
        break

    # Create generators and discriminators
    generator = Generator().to(device)
    generator2 = Generator().to(device)
    discriminator = Discriminator().to(device)
    discriminator2 = Discriminator().to(device)

    optimizer_G = torch.optim.AdamW(generator.parameters(), lr=lr_G, betas=(0.5, 0.999), weight_decay=opt.decay)
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=lr_D, betas=(0.5, 0.999), weight_decay=opt.decay)
    optimizer_G2 = torch.optim.AdamW(generator2.parameters(), lr=lr_G, betas=(0.5, 0.999), weight_decay=opt.decay)
    optimizer_D2 = torch.optim.AdamW(discriminator2.parameters(), lr=lr_D, betas=(0.5, 0.999), weight_decay=opt.decay)

    total_step = len(h_data_train_loader)
    real_label = torch.ones((data_size, 1)).to(device)
    fake_label = torch.zeros((data_size, 1)).to(device)

    
    # Will be used for reporting
    real_losses, fake_losses, mse_losses, mae_losses = list(), list(), list(), list()
    real_losses_val, fake_losses_val, mse_losses_val, mae_losses_val = list(), list(), list(), list()

    real_losses2, fake_losses2, mse_losses2, mae_losses2 = list(), list(), list(), list()
    real_losses_val2, fake_losses_val2, mse_losses_val2, mae_losses_val2 = list(), list(), list(), list()

    k1_losses, k2_losses, k1_losses_val, k2_losses_val = list(), list(), list(), list()
    tp_losses_1_tr, tp_losses_1_val, tp_losses_2_tr, tp_losses_2_val = list(), list(), list(), list()
    gan_losses_1_tr, gan_losses_1_val, gan_losses_2_tr, gan_losses_2_val = list(), list(), list(), list()


    for epoch in range(num_epochs):
        # Reporting
        r, f, d, g, mse_l, mae_l = 0, 0, 0, 0, 0, 0
        r_val, f_val, d_val, g_val, mse_l_val, mae_l_val = 0, 0, 0, 0, 0, 0
        k1_train, k2_train, k1_val, k2_val = 0.0, 0.0, 0.0, 0.0
        r2, f2, d2, g2, mse_l2, mae_l2 = 0, 0, 0, 0, 0, 0
        r_val2, f_val2, d_val2, g_val2, mse_l_val2, mae_l_val2 = 0, 0, 0, 0, 0, 0
        tp1_tr, tp1_val, tp2_tr, tp2_val = 0.0, 0.0, 0.0, 0.0
        gan1_tr, gan1_val, gan2_tr, gan2_val = 0.0, 0.0, 0.0, 0.0

        # Train
        generator.train()
        discriminator.train()
        generator2.train()
        discriminator2.train()
        for i, data in enumerate(h_data_train_loader):
            data = data.to(device)

            optimizer_D.zero_grad()

            # Train the discriminator
            # Create fake data
            fake_y = generator(data).detach()
            edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y)
            fake_data = Data(x=fake_y, edge_attr=edge_a, edge_index=edge_i).to(device)
            swapped_data = Data(x=data.y, edge_attr=data.y_edge_attr, edge_index=data.y_edge_index).to(device)

            # data: Real source and target
            # fake_data: Real source and generated target
            real_loss = adversarial_loss(discriminator(swapped_data, data), real_label[:data.x.size(0), :])
            fake_loss = adversarial_loss(discriminator(fake_data, data), fake_label[:data.x.size(0), :])
            loss_D = torch.mean(real_loss + fake_loss) / 2
            r += real_loss.item()
            f += fake_loss.item()
            d += loss_D.item()

            # Depending on the chosen training method, we might update the parameters of the discriminator
            if (epoch % 2 == 1 and opt.tr_st == "turns") or opt.tr_st == "same" or counter_d >= opt.id_e:
                loss_D.backward(retain_graph=True)
                optimizer_D.step()

            # Train the generator
            optimizer_G.zero_grad()

            # Adversarial Loss
            fake_data.x = generator(data)
            gan_loss = torch.mean(adversarial_loss(discriminator(fake_data, data), real_label[:data.x.size(0), :]))
            gan1_tr += gan_loss.item()

            # KL Loss
            kl_loss = kl.kl_divergence(normal.Normal(fake_data.x.mean(dim=1), fake_data.x.std(dim=1)),
                                       normal.Normal(data.y.mean(dim=1), data.y.std(dim=1))).sum()

            # Topology Loss
            tp_loss = tp(fake_data.x.sum(dim=-1), data.y.sum(dim=-1))
            tp1_tr += tp_loss.item()

            # Identity Loss is included in the end
            loss_G = i_coeff * identity_loss(generator(swapped_data), data.y) + g_coeff * gan_loss + kl_coeff * kl_loss + tp_coeff * tp_loss
            g += loss_G.item()
            if (epoch % 2 == 0 and opt.tr_st == "turns") or opt.tr_st == "same" or counter_g < opt.id_e:
                loss_G.backward(retain_graph=True)
                optimizer_G.step()
            k1_train += kl_loss.item()
            mse_l += msel(generator(data), data.y).item()
            mae_l += mael(generator(data), data.y).item()

            # Training of the second part !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            optimizer_D2.zero_grad()

            # Train the discriminator2

            # Create fake data for t2 from fake data for t1
            fake_data.x = fake_data.x.detach()
            fake_y2 = generator2(fake_data).detach()
            edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y2)
            fake_data2 = Data(x=fake_y2, edge_attr=edge_a, edge_index=edge_i).to(device)
            swapped_data2 = Data(x=data.y2, edge_attr=data.y2_edge_attr, edge_index=data.y2_edge_index).to(device)

            # fake_data: Data generated for t1
            # fake_data2: Data generated for t2 using generated data for t1
            # swapped_data2: Real t2 data
            real_loss = adversarial_loss(discriminator2(swapped_data2, fake_data), real_label[:data.x.size(0), :])
            fake_loss = adversarial_loss(discriminator2(fake_data2, fake_data), fake_label[:data.x.size(0), :])
            loss_D = torch.mean(real_loss + fake_loss) / 2
            r2 += real_loss.item()
            f2 += fake_loss.item()
            d2 += loss_D.item()

            if (epoch % 2 == 1 and opt.tr_st == "turns") or opt.tr_st == "same" or counter_d >= opt.id_e:
                loss_D.backward(retain_graph=True)
                optimizer_D2.step()

            # Train generator2
            optimizer_G2.zero_grad()

            # Adversarial Loss
            fake_data2.x = generator2(fake_data)
            gan_loss = torch.mean(adversarial_loss(discriminator2(fake_data2, fake_data), real_label[:data.x.size(0), :]))
            gan2_tr += gan_loss.item()

            # Topology Loss
            tp_loss = tp(fake_data2.x.sum(dim=-1), data.y2.sum(dim=-1))
            tp2_tr += tp_loss.item()

            # KL Loss
            kl_loss = kl.kl_divergence(normal.Normal(fake_data2.x.mean(dim=1), fake_data2.x.std(dim=1)),
                                       normal.Normal(data.y2.mean(dim=1), data.y2.std(dim=1))).sum()

            # Identity Loss
            loss_G = i_coeff * identity_loss(generator(swapped_data2), data.y2) + g_coeff * gan_loss + kl_coeff * kl_loss + tp_coeff * tp_loss
            g2 += loss_G.item()
            if (epoch % 2 == 0 and opt.tr_st == "turns") or opt.tr_st == "same" or counter_g < opt.id_e:
                loss_G.backward(retain_graph=True)
                optimizer_G2.step()

            k2_train += kl_loss.item()
            mse_l2 += msel(generator2(fake_data), data.y2).item()
            mae_l2 += mael(generator2(fake_data), data.y2).item()

        # Validate
        generator.eval()
        discriminator.eval()
        generator2.eval()
        discriminator2.eval()

        for i, data in enumerate(h_data_test_loader):
            data = data.to(device)
            # Train the discriminator
            # Create fake data
            fake_y = generator(data).detach()
            edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y)
            fake_data = Data(x=fake_y, edge_attr=edge_a, edge_index=edge_i).to(device)
            swapped_data = Data(x=data.y, edge_attr=data.y_edge_attr, edge_index=data.y_edge_index).to(device)

            # data: Real source and target
            # fake_data: Real source and generated target
            real_loss = adversarial_loss(discriminator(swapped_data, data), real_label[:data.x.size(0), :])
            fake_loss = adversarial_loss(discriminator(fake_data, data), fake_label[:data.x.size(0), :])
            loss_D = torch.mean(real_loss + fake_loss) / 2
            r_val += real_loss.item()
            f_val += fake_loss.item()
            d_val += loss_D.item()

            # Adversarial Loss
            fake_data.x = generator(data)
            gan_loss = torch.mean(adversarial_loss(discriminator(fake_data, data), real_label[:data.x.size(0), :]))
            gan1_val += gan_loss.item()

            # Topology Loss
            tp_loss = tp(fake_data.x.sum(dim=-1), data.y.sum(dim=-1))
            tp1_val += tp_loss.item()

            kl_loss = kl.kl_divergence(normal.Normal(fake_data.x.mean(dim=1), fake_data.x.std(dim=1)),
                                       normal.Normal(data.y.mean(dim=1), data.y.std(dim=1))).sum()

            # Identity Loss

            loss_G = i_coeff * identity_loss(generator(swapped_data), data.y) + g_coeff * gan_loss * kl_coeff * kl_loss
            g_val += loss_G.item()
            mse_l_val += msel(generator(data), data.y).item()
            mae_l_val += mael(generator(data), data.y).item()
            k1_val += kl_loss.item()

            # Second GAN

            # Create fake data for t2 from fake data for t1
            fake_data.x = fake_data.x.detach()
            fake_y2 = generator2(fake_data)
            edge_i, edge_a, _, _ = create_edge_index_attribute(fake_y2)
            fake_data2 = Data(x=fake_y2, edge_attr=edge_a, edge_index=edge_i).to(device)
            swapped_data2 = Data(x=data.y2, edge_attr=data.y2_edge_attr, edge_index=data.y2_edge_index).to(device)

            # fake_data: Data generated for t1
            # fake_data2: Data generated for t2 using generated data for t1
            # swapped_data2: Real t2 data
            real_loss = adversarial_loss(discriminator2(swapped_data2, fake_data), real_label[:data.x.size(0), :])
            fake_loss = adversarial_loss(discriminator2(fake_data2, fake_data), fake_label[:data.x.size(0), :])
            loss_D = torch.mean(real_loss + fake_loss) / 2
            r_val2 += real_loss.item()
            f_val2 += fake_loss.item()
            d_val2 += loss_D.item()

            # Adversarial Loss
            fake_data2.x = generator2(fake_data)
            gan_loss = torch.mean(adversarial_loss(discriminator2(fake_data2, fake_data), real_label[:data.x.size(0), :]))
            gan2_val += gan_loss.item()

            # Topology Loss
            tp_loss = tp(fake_data2.x.sum(dim=-1), data.y2.sum(dim=-1))
            tp2_val += tp_loss.item()

            # KL Loss
            kl_loss = kl.kl_divergence(normal.Normal(fake_data2.x.mean(dim=1), fake_data2.x.std(dim=1)),
                                       normal.Normal(data.y2.mean(dim=1), data.y2.std(dim=1))).sum()
            k2_val += kl_loss.item()

            # Identity Loss
            loss_G = i_coeff * identity_loss(generator(swapped_data2), data.y2) + g_coeff * gan_loss + kl_coeff * kl_loss
            g_val2 += loss_G.item()
            mse_l_val2 += msel(generator2(fake_data), data.y2).item()
            mae_l_val2 += mael(generator2(fake_data), data.y2).item()

        if opt.tr_st == 'idle':
            counter_g += 1
            counter_d += 1
            if counter_g == 2 * opt.id_e:
                counter_g = 0
                counter_d = 0


        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'[Train]: D Loss: {d / total_step:.5f}, G Loss: {g / total_step:.5f} R Loss: {r / total_step:.5f}, F Loss: {f / total_step:.5f}, MSE: {mse_l / total_step:.5f}, MAE: {mae_l / total_step:.5f}')
        print(f'[Val]: D Loss: {d_val / val_step:.5f}, G Loss: {g_val / val_step:.5f} R Loss: {r_val / val_step:.5f}, F Loss: {f_val / val_step:.5f}, MSE: {mse_l_val / val_step:.5f}, MAE: {mae_l_val / val_step:.5f}')
        print(f'[Train]: D2 Loss: {d2 / total_step:.5f}, G2 Loss: {g2 / total_step:.5f} R2 Loss: {r2 / total_step:.5f}, F2 Loss: {f2 / total_step:.5f}, MSE: {mse_l2 / total_step:.5f}, MAE: {mae_l2 / total_step:.5f}')
        print(f'[Val]: D2 Loss: {d_val2 / val_step:.5f}, G2 Loss: {g_val2 / val_step:.5f} R2 Loss: {r_val2 / val_step:.5f}, F2 Loss: {f_val2 / val_step:.5f}, MSE: {mse_l_val2 / val_step:.5f}, MAE: {mae_l_val2 / val_step:.5f}')

        real_losses.append(r / total_step)
        fake_losses.append(f / total_step)
        mse_losses.append(mse_l / total_step)
        mae_losses.append(mae_l / total_step)
        real_losses_val.append(r_val / val_step)
        fake_losses_val.append(f_val / val_step)
        mse_losses_val.append(mse_l_val / val_step)
        mae_losses_val.append(mae_l_val / val_step)
        real_losses2.append(r2 / total_step)
        fake_losses2.append(f2 / total_step)
        mse_losses2.append(mse_l2 / total_step)
        mae_losses2.append(mae_l2 / total_step)
        real_losses_val2.append(r_val2 / val_step)
        fake_losses_val2.append(f_val2 / val_step)
        mse_losses_val2.append(mse_l_val2 / val_step)
        mae_losses_val2.append(mae_l_val2 / val_step)
        k1_losses.append(k1_train / total_step)
        k2_losses.append(k2_train / total_step)
        k1_losses_val.append(k1_val / val_step)
        k2_losses_val.append(k2_val / val_step)
        tp_losses_1_tr.append(tp1_tr / total_step)
        tp_losses_1_val.append(tp1_val / val_step)
        tp_losses_2_tr.append(tp2_tr / total_step)
        tp_losses_2_val.append(tp2_val / val_step)
        gan_losses_1_tr.append(gan1_tr / total_step)
        gan_losses_1_val.append(gan1_val / val_step)
        gan_losses_2_tr.append(gan2_tr / total_step)
        gan_losses_2_val.append(gan2_val / val_step)

    # Plot losses
    plot("BCE", "DiscriminatorRealLossTrainSet" + str(fold) + "_exp" + str(opt.exp), real_losses)
    plot("BCE", "DiscriminatorRealLossValSet" + str(fold) + "_exp" + str(opt.exp), real_losses_val)
    plot("BCE", "DiscriminatorFakeLossTrainSet" + str(fold) + "_exp" + str(opt.exp), fake_losses)
    plot("BCE", "DiscriminatorFakeLossValSet" + str(fold) + "_exp" + str(opt.exp), fake_losses_val)
    plot("MSE", "GeneratorMSELossTrainSet" + str(fold) + "_exp" + str(opt.exp), mse_losses)
    plot("MSE", "GeneratorMSELossValSet" + str(fold) + "_exp" + str(opt.exp), mse_losses_val)
    plot("MAE", "GeneratorMAELossTrainSet" + str(fold) + "_exp" + str(opt.exp), mae_losses)
    plot("MAE", "GeneratorMAELossValSet" + str(fold) + "_exp" + str(opt.exp), mae_losses_val)
    plot("BCE", "Discriminator2RealLossTrainSet" + str(fold) + "_exp" + str(opt.exp), real_losses2)
    plot("BCE", "Discriminator2RealLossValSet" + str(fold) + "_exp" + str(opt.exp), real_losses_val2)
    plot("BCE", "Discriminator2FakeLossTrainSet" + str(fold) + "_exp" + str(opt.exp), fake_losses2)
    plot("BCE", "Discriminator2FakeLossValSet" + str(fold) + "_exp" + str(opt.exp), fake_losses_val2)
    plot("MSE", "Generator2MSELossTrainSet" + str(fold) + "_exp" + str(opt.exp), mse_losses2)
    plot("MSE", "Generator2MSELossValSet" + str(fold) + "_exp" + str(opt.exp), mse_losses_val2)
    plot("MAE", "Generator2MAELossTrainSet" + str(fold) + "_exp" + str(opt.exp), mae_losses2)
    plot("MAE", "Generator2MAELossValSet" + str(fold) + "_exp" + str(opt.exp), mae_losses_val2)
    plot("KL Loss", "KL_Loss_1_TrainSet" + str(fold) + "_exp" + str(opt.exp), k1_losses)
    plot("KL Loss", "KL_Loss_1_ValSet" + str(fold) + "_exp" + str(opt.exp), k1_losses_val)
    plot("KL Loss", "KL_Loss_2_TrainSet" + str(fold) + "_exp" + str(opt.exp), k2_losses)
    plot("KL Loss", "KL_Loss_2_ValSet" + str(fold) + "_exp" + str(opt.exp), k2_losses_val)
    plot("TP Loss", "TP_Loss_1_TrainSet" + str(fold) + "_exp" + str(opt.exp), tp_losses_1_tr)
    plot("TP Loss", "TP_Loss_1_ValSet" + str(fold) + "_exp" + str(opt.exp), tp_losses_1_val)
    plot("TP Loss", "TP_Loss_2_TrainSet" + str(fold) + "_exp" + str(opt.exp), tp_losses_2_tr)
    plot("TP Loss", "TP_Loss_2_ValSet" + str(fold) + "_exp" + str(opt.exp), tp_losses_2_val)
    plot("BCE", "GAN_Loss_1_TrainSet" + str(fold) + "_exp" + str(opt.exp), gan_losses_1_tr)
    plot("BCE", "GAN_Loss_1_ValSet" + str(fold) + "_exp" + str(opt.exp), gan_losses_1_val)
    plot("BCE", "GAN_Loss_2_TrainSet" + str(fold) + "_exp" + str(opt.exp), gan_losses_2_tr)
    plot("BCE", "GAN_Loss_2_ValSet" + str(fold) + "_exp" + str(opt.exp), gan_losses_2_val)

    # Save the losses
    if gen_mae_losses_tr is None:
        gen_mae_losses_tr = mae_losses
        disc_real_losses_tr = real_losses
        disc_fake_losses_tr = fake_losses
        gen_mae_losses_val = mae_losses_val
        disc_real_losses_val = real_losses_val
        disc_fake_losses_val = fake_losses_val
        gen_mae_losses_tr2 = mae_losses2
        disc_real_losses_tr2 = real_losses2
        disc_fake_losses_tr2 = fake_losses2
        gen_mae_losses_val2 = mae_losses_val2
        disc_real_losses_val2 = real_losses_val2
        disc_fake_losses_val2 = fake_losses_val2
        k1_train_s = k1_losses
        k2_train_s = k2_losses
        k1_val_s = k1_losses_val
        k2_val_s = k2_losses_val
        tp1_train_s = tp_losses_1_tr
        tp2_train_s = tp_losses_2_tr
        tp1_val_s = tp_losses_1_val
        tp2_val_s = tp_losses_2_val
        gan1_train_s = gan_losses_1_tr
        gan2_train_s = gan_losses_2_tr
        gan1_val_s = gan_losses_1_val
        gan2_val_s = gan_losses_2_val
    else:
        gen_mae_losses_tr = np.vstack([gen_mae_losses_tr, mae_losses])
        disc_real_losses_tr = np.vstack([disc_real_losses_tr, real_losses])
        disc_fake_losses_tr = np.vstack([disc_fake_losses_tr, fake_losses])
        gen_mae_losses_val = np.vstack([gen_mae_losses_val, mae_losses_val])
        disc_real_losses_val = np.vstack([disc_real_losses_val, real_losses_val])
        disc_fake_losses_val = np.vstack([disc_fake_losses_val, fake_losses_val])
        gen_mae_losses_tr2 = np.vstack([gen_mae_losses_tr2, mae_losses2])
        disc_real_losses_tr2 = np.vstack([disc_real_losses_tr2, real_losses2])
        disc_fake_losses_tr2 = np.vstack([disc_fake_losses_tr2, fake_losses2])
        gen_mae_losses_val2 = np.vstack([gen_mae_losses_val2, mae_losses_val2])
        disc_real_losses_val2 = np.vstack([disc_real_losses_val2, real_losses_val2])
        disc_fake_losses_val2 = np.vstack([disc_fake_losses_val2, fake_losses_val2])
        k1_train_s = np.vstack([k1_train_s, k1_losses])
        k2_train_s = np.vstack([k2_train_s, k2_losses])
        k1_val_s = np.vstack([k1_val_s, k1_losses_val])
        k2_val_s = np.vstack([k2_val_s, k2_losses_val])
        tp1_train_s = np.vstack([tp1_train_s, tp_losses_1_tr])
        tp2_train_s = np.vstack([tp2_train_s, tp_losses_2_tr])
        tp1_val_s = np.vstack([tp1_val_s, tp_losses_1_val])
        tp2_val_s = np.vstack([tp2_val_s, tp_losses_2_val])
        gan1_train_s = np.vstack([gan1_train_s, gan_losses_1_tr])
        gan2_train_s = np.vstack([gan2_train_s, gan_losses_2_tr])
        gan1_val_s = np.vstack([gan1_val_s, gan_losses_1_val])
        gan2_val_s = np.vstack([gan2_val_s, gan_losses_2_val])

        # Save the models
        torch.save(generator.state_dict(), "../weights/generator_" + str(fold) + "_" + str(epoch) + "_" + str(opt.exp))
        torch.save(discriminator.state_dict(), "../weights/discriminator_" + str(fold) + "_" + str(epoch) + "_" + str(opt.exp))
        torch.save(generator2.state_dict(),
                   "../weights/generator2_" + str(fold) + "_" + str(epoch) + "_" + str(opt.exp))
        torch.save(discriminator2.state_dict(),
                   "../weights/discriminator2_" + str(fold) + "_" + str(epoch) + "_" + str(opt.exp))

    del generator
    del discriminator

    del generator2
    del discriminator2

# Save losses
with open("../losses/G_TrainLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(gen_mae_losses_tr, f)
with open("../losses/G_ValLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(gen_mae_losses_val, f)
with open("../losses/D_TrainRealLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(disc_real_losses_tr, f)
with open("../losses/D_TrainFakeLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(disc_fake_losses_tr, f)
with open("../losses/D_ValRealLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(disc_real_losses_val, f)
with open("../losses/D_ValFakeLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(disc_fake_losses_val, f)
with open("../losses/G2_TrainLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(gen_mae_losses_tr2, f)
with open("../losses/G2_ValLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(gen_mae_losses_val2, f)
with open("../losses/D2_TrainRealLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(disc_real_losses_tr2, f)
with open("../losses/D2_TrainFakeLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(disc_fake_losses_tr2, f)
with open("../losses/D2_ValRealLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(disc_real_losses_val2, f)
with open("../losses/D2_ValFakeLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(disc_fake_losses_val2, f)
with open("../losses/GenTotal_Train_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(gen_mae_losses_tr + gen_mae_losses_tr2, f)
with open("../losses/GenTotal_Val_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(gen_mae_losses_val + gen_mae_losses_val2, f)
with open("../losses/K1_TrainLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(k1_train_s, f)
with open("../losses/K1_ValLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(k2_train_s, f)
with open("../losses/K2_TrainLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(k1_val_s, f)
with open("../losses/K2_ValLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(k2_val_s, f)
with open("../losses/TP1_TrainLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(tp1_train_s, f)
with open("../losses/TP1_ValLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(tp2_train_s, f)
with open("../losses/TP2_TrainLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(tp1_val_s, f)
with open("../losses/TP2_ValLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(tp2_val_s, f)
with open("../losses/GAN1_TrainLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(gan1_train_s, f)
with open("../losses/GAN1_ValLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(gan2_train_s, f)
with open("../losses/GAN2_TrainLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(gan1_val_s, f)
with open("../losses/GAN2_ValLoss_exp_" + str(opt.exp), "wb") as f:
    pickle.dump(gan2_val_s, f)

print(f"Training Complete for experiment {opt.exp}!")

