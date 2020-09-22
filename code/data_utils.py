import argparse
import os
import os.path as osp
import numpy as np
import math
import itertools
import copy
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Tanh, Dropout, LeakyReLU
from torch.autograd import Variable
from torch.distributions import normal

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from torch_geometric.data import Data, InMemoryDataset, DataLoader
from torch_geometric.nn import NNConv, BatchNorm, EdgePooling, TopKPooling, global_add_pool
from torch_geometric.utils import get_laplacian, to_dense_adj

import matplotlib.pyplot as plt


class MRDataset(InMemoryDataset):

    def __init__(self, root, src, dest, h, connectomes=1, subs=1000, transform=None, pre_transform=None):

        """
        src: Input to the model
        dest: Target output of the model
        h: Load LH or RH data
        subs: Maximum number of subjects

        Note: Since we do not reprocess the data if it is already processed, processed files should be
        deleted if there is any change in the data we are reading.
        """

        self.src, self.dest, self.h, self.subs, self.connectomes = src, dest, h, subs, connectomes
        super(MRDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def data_read(self, h="lh", nbr_of_subs=1000, connectomes=1):

        """
        Takes the (maximum) number of subjects and hemisphere we are working on
        as arguments, returns t0, t1, t2's of the connectomes for each subject
        in a single torch.FloatTensor.
        """

        subs = None  # Subjects

        data_path = "../data"

        for i in range(1, nbr_of_subs):
            s = data_path + "/cortical." + h.lower() + ".ShapeConnectivityTensor_OAS2_"
            if i < 10:
                s += "0"
            s += "00" + str(i) + "_"

            for mr in ["MR1", "MR2"]:
                try:  # Sometimes subject we are looking for does not exist
                    t0 = np.loadtxt(s + mr + "_t0.txt")
                    t1 = np.loadtxt(s + mr + "_t1.txt")
                    t2 = np.loadtxt(s + mr + "_t2.txt")
                except:
                    continue

                # Read the connectomes at t0, t1 and t2, then stack them
                read_limit = (connectomes * 35)
                t_stacked = np.vstack((t0[:read_limit, :], t1[:read_limit, :], t2[:read_limit, :]))
                tsr = t_stacked.reshape(3, connectomes * 35, 35)

                if subs is None:  # If first subject
                    subs = tsr
                else:
                    subs = np.vstack((subs, tsr))

        # Then, reshape to match the shape of the model's expected input shape
        # final_views should be a torch tensor or Pytorch Geometric complains
        final_views = torch.tensor(np.moveaxis(subs.reshape(-1, 3, (connectomes * 35), 35), 1, -1), dtype=torch.float)

        return final_views

    @property
    def processed_file_names(self):
        return [
            "data_" + str(self.connectomes) + "_" + self.h.lower() + "_" + str(self.subs) + "_" + str(self.src) + str(
                self.dest) + ".pt"]

    def process(self):

        """
        Prepares the data for PyTorch Geometric.
        """

        unprocessed = self.data_read(self.h, self.subs)
        num_samples, timestamps = unprocessed.shape[0], unprocessed.shape[-1]
        assert 0 <= self.dest <= timestamps
        assert 0 <= self.src <= timestamps

        # Turn the data into PyTorch Geometric Graphs
        data_list = list()

        for sample in range(num_samples):
            x = unprocessed[sample, :, :, self.src]
            y = unprocessed[sample, :, :, self.dest]

            edge_index, edge_attr, rows, cols = create_edge_index_attribute(x)
            y_edge_index, y_edge_attr, _, _ = create_edge_index_attribute(y)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        y=y, y_edge_index=y_edge_index, y_edge_attr=y_edge_attr)

            data.num_nodes = rows
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class MRDataset2(InMemoryDataset):

    def __init__(self, root, h, connectomes=1, subs=1000, transform=None, pre_transform=None):

        """
        src: Input to the model
        dest: Target output of the model
        h: Load LH or RH data
        subs: Maximum number of subjects

        Note: Since we do not reprocess the data if it is already processed, processed files should be
        deleted if there is any change in the data we are reading.
        """

        self.h, self.subs, self.connectomes = h, subs, connectomes
        super(MRDataset2, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def data_read(self, h="lh", nbr_of_subs=1000, connectomes=1):

        """
        Takes the (maximum) number of subjects and hemisphere we are working on
        as arguments, returns t0, t1, t2's of the connectomes for each subject
        in a single torch.FloatTensor.
        """

        subs = None  # Subjects

        data_path = "../data"

        for i in range(1, nbr_of_subs):
            s = data_path + "/cortical." + h.lower() + ".ShapeConnectivityTensor_OAS2_"
            if i < 10:
                s += "0"
            s += "00" + str(i) + "_"

            for mr in ["MR1", "MR2"]:
                try:  # Sometimes subject we are looking for does not exist
                    t0 = np.loadtxt(s + mr + "_t0.txt")
                    t1 = np.loadtxt(s + mr + "_t1.txt")
                    t2 = np.loadtxt(s + mr + "_t2.txt")
                except:
                    continue

                # Read the connectomes at t0, t1 and t2, then stack them
                read_limit = (connectomes * 35)
                t_stacked = np.vstack((t0[:read_limit, :], t1[:read_limit, :], t2[:read_limit, :]))
                tsr = t_stacked.reshape(3, connectomes * 35, 35)

                if subs is None:  # If first subject
                    subs = tsr
                else:
                    subs = np.vstack((subs, tsr))

        # Then, reshape to match the shape of the model's expected input shape
        # final_views should be a torch tensor or Pytorch Geometric complains
        final_views = torch.tensor(np.moveaxis(subs.reshape(-1, 3, (connectomes * 35), 35), 1, -1), dtype=torch.float)

        return final_views

    @property
    def processed_file_names(self):
        return [
            "2data_" + str(self.connectomes) + "_" + self.h.lower() + "_" + str(self.subs) + "_" + ".pt"]

    def process(self):

        """
        Prepares the data for PyTorch Geometric.
        """

        unprocessed = self.data_read(self.h, self.subs)
        num_samples, timestamps = unprocessed.shape[0], unprocessed.shape[-1]

        # Turn the data into PyTorch Geometric Graphs
        data_list = list()

        for sample in range(num_samples):
            x = unprocessed[sample, :, :, 0]
            y = unprocessed[sample, :, :, 1]
            y2 = unprocessed[sample, :, :, 2]

            edge_index, edge_attr, rows, cols = create_edge_index_attribute(x)
            y_edge_index, y_edge_attr, _, _ = create_edge_index_attribute(y)
            y2_edge_index, y2_edge_attr, _, _ = create_edge_index_attribute(y2)
            y_distr = normal.Normal(y.mean(dim=1), y.std(dim=1))
            y2_distr = normal.Normal(y2.mean(dim=1), y2.std(dim=1))
            y_lap_ei, y_lap_ea = get_laplacian(y_edge_index, y_edge_attr)
            y2_lap_ei, y2_lap_ea = get_laplacian(y2_edge_index, y2_edge_attr)
            y_lap = to_dense_adj(y_lap_ei, edge_attr=y_lap_ea)
            y2_lap = to_dense_adj(y2_lap_ei, edge_attr=y2_lap_ea)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                        y=y, y_edge_index=y_edge_index, y_edge_attr=y_edge_attr, y_distr=y_distr,
                        y2=y2, y2_edge_index=y2_edge_index, y2_edge_attr=y2_edge_attr, y2_distr=y2_distr,
                        y_lap=y_lap, y2_lap=y2_lap)

            data.num_nodes = rows
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def create_edge_index_attribute(adj_matrix):
    """
    Given an adjacency matrix, this function creates the edge index and edge attribute matrix
    suitable to graph representation in PyTorch Geometric.
    """

    rows, cols = adj_matrix.shape[0], adj_matrix.shape[1]
    edge_index = torch.zeros((2, rows * cols), dtype=torch.long)
    edge_attr = torch.zeros((rows * cols, 1), dtype=torch.float)
    counter = 0

    for src, attrs in enumerate(adj_matrix):
        for dest, attr in enumerate(attrs):
            edge_index[0][counter], edge_index[1][counter] = src, dest
            edge_attr[counter] = attr
            counter += 1

    return edge_index, edge_attr, rows, cols


def swap(data):
    # Swaps the x & y values of the given graph
    edge_i, edge_attr, _, _ = create_edge_index_attribute(data.y)
    data_s = Data(x=data.y, edge_index=edge_i, edge_attr=edge_attr, y=data.x)
    return data_s


def cross_val_indices(folds, num_samples, new=False):
    """
    Takes the number of inputs and number of folds.
    Determines indices to go into validation split in each turn.
    Saves the indices on a file for experimental reproducibility and does not overwrite
    the already determined indices unless new=True.
    """

    kf = KFold(n_splits=folds, shuffle=True)
    train_indices = list()
    val_indices = list()

    try:
        if new == True:
            raise IOError
        with open("../data/" + str(folds) + "_" + str(num_samples) + "cv_train", "rb") as f:
            train_indices = pickle.load(f)
        with open("../data/" + str(folds) + "_" + str(num_samples) + "cv_val", "rb") as f:
            val_indices = pickle.load(f)
    except IOError:
        for tr_index, val_index in kf.split(np.zeros((num_samples, 1))):
            train_indices.append(tr_index)
            val_indices.append(val_index)
        with open("../data/" + str(folds) + "_" + str(num_samples) + "cv_train", "wb") as f:
            pickle.dump(train_indices, f)
        with open("../data/" + str(folds) + "_" + str(num_samples) + "cv_val", "wb") as f:
            pickle.dump(val_indices, f)

    return train_indices, val_indices
