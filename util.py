import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import h5py
import os
import sklearn.metrics as metrics
from imblearn.metrics import geometric_mean_score


def get_dataset_dir(dataset):
    dir_data_base = "[BASE_PATH]"
    if dataset == "bcp":
        dir_data = "[PATH_TO_BCP]"
    elif dataset == "hcp":
        dir_data = "[PATH_TO_HCP]"
    elif dataset == "mb6":
        dir_data = "[PATH_TO_MB6]"
    elif dataset == "std":
        dir_data = "[PATH_TO_STD]"
    else:
        raise ValueError("Wrong dataset : {}".format(dataset))

    return os.path.join(dir_data_base, dir_data)

def accuracy(out, label):
    out = np.array(out)
    label = np.array(label)
    total = out.shape[0]
    correct = (out == label).sum().item() / total
    return correct

def sensitivity(out, label):
    out = np.array(out)
    label = np.array(label)
    mask = (label > 1-1e-5)
    sens = np.sum(out[mask]) / np.sum(mask)

    return sens

def specificity(out, label):
    out = np.array(out)
    label = np.array(label)
    mask = (label <= 1e-5)
    total = np.sum(mask)
    spec = (total - np.sum(out[mask])) / total

    return spec

def f1_score(out, label):
    out = np.array(out)
    label = np.array(label)
    f1 = metrics.f1_score(y_true=label, y_pred=out)

    return f1

def init_weights(m):
    init = "he" # config().parse_args().init
    if type(m) == nn.Conv3d or type(m) == nn.Conv1d:
        try:
            if init == "xavier":
                torch.nn.init.xavier_normal_(m.weight)
            elif init == "he":
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # msra
            elif init == "default":
                pass
            else:
                raise ValueError("unknown init")
            m.bias.data.fill_(0)
        except:
            pass

    elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm1d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class Hdf5:
    def __init__(self, num_source=1, label_percentage=100, datasets=None):
        assert num_source in range(1,6), f"num_source '{num_source}' must be between 1 and 5, 6th cv is used as test set"
        assert label_percentage in [(i+1)*10 for i in range(10)], f"labeled percentage '{label_percentage}' must be in 10, 20, ..., 100 %"

        self.cv_full = np.arange(1,6)
        self.cv_train = np.delete(np.arange(1,6), num_source-1)
        self.cv_test = np.array([num_source])
        self.num_source = num_source
        self.data_keys = ["data", "label", "tdata"]
        self.keys = ["data", "label", "tdata", "dlabel", "nlabel"]
        self.max_shape = {key:0 for key in self.keys}
        self.max_shape["tdata"] = 1200
        self.metadata = pd.read_csv("./data/metadata.csv", index_col=0)
        if datasets is None or "all" in datasets:
            self.datasets = ["hcp", "bcp", "mb6", "std"]
        else:
            if isinstance(datasets, str):
                datasets = datasets.split()
            self.datasets = datasets

        self.sampling_freq = {"hcp":1/0.73, "bcp":1/0.8, "mb6":1/1.3, "std":1/3}
        self.metadata = self.metadata[(self.metadata["dataset"].isin(self.datasets))]
        self.label_percentage = label_percentage
        # if only_labeled:
        #     self.metadata = self.metadata[(self.metadata["ten_fold"] <= self.label_percentage//10)]
        if len(datasets) >= 2:
            target = datasets[-1]
        else:
            raise NotImplementedError("Datasets should be larger than 2")

        sources = datasets[:-1]

        self.query_s = []
        self.support_s = []
        if label_percentage != 100:
            self.support_t = [self.metadata[(self.metadata["cv"].isin(self.cv_train))
                                           & (self.metadata["dataset"] == target)
                                           & (self.metadata["ten_fold"] > self.label_percentage // 10)].index.to_numpy() for j in range(len(sources))]
            self.query_t = [self.metadata[(self.metadata["cv"].isin(self.cv_train))
                                           & (self.metadata["dataset"] == target)
                                           & (self.metadata["ten_fold"] <= self.label_percentage // 10)].index.to_numpy() for j in range(len(sources))]
        else:
            self.query_t = [self.metadata[(self.metadata["cv"] == self.cv_train[0]) & (self.metadata["dataset"] == target)].index.to_numpy() for cv in range(len(sources))]
            self.support_t = [self.metadata[(self.metadata["cv"].isin(self.cv_train[1:])) & (self.metadata["dataset"] == target)].index.to_numpy() for cv in range(len(sources))]
          
        self.target_unlabeled_idx = self.metadata[(self.metadata["cv"].isin(self.cv_train))
                                                  & (self.metadata["dataset"] == target)
                                                  & (self.metadata["ten_fold"] > self.label_percentage // 10)].index.to_numpy()

        for source in sources:
            self.query_s.append(self.metadata[(self.metadata["cv"]==self.cv_train[0]) & (self.metadata["dataset"] == source)].index.to_numpy())
            self.support_s.append(self.metadata[(self.metadata["cv"].isin(self.cv_train[1:])) & (self.metadata["dataset"] == source)].index.to_numpy())

        self.test = self.metadata[(self.metadata["cv"].isin(self.cv_test))].index.to_numpy()

        print(f"Setting a dataloader for datasets {datasets}, {len(self.query_s)} tasks")
        print(f"[Meta train] : all {sources} datasets and target dataset {target} fold {self.cv_train} : query sets source {[len(q) for q in self.query_s]} target {[len(q) for q in self.query_t]} | support sets source {[len(s) for s in self.support_s]} target {[len(s) for s in self.support_t]}")

        num_sample_labeled = self.metadata[(self.metadata["cv"].isin(self.cv_train)) & (self.metadata["dataset"] == target) & (self.metadata["ten_fold"] <= self.label_percentage // 10)]["sample"].nunique()
        num_sample_total = self.metadata[(self.metadata["cv"].isin(self.cv_train)) & (self.metadata["dataset"] == target)]["sample"].nunique()
        print(f"[Meta train] : target label selected {label_percentage}% : {num_sample_labeled}/{num_sample_total} samples")
        print(f"[Meta test] : source & target dataset {target} fold {self.cv_test} number [{len(self.test)}]")

    def __getitem__(self, index):
        return self.getDataDicByIndex(index)

    def getMetadata(self):
        return self.metadata

    def getLenDatasets(self):
        return self.datasets.__len__()

    def getDatasetLabelByIndex(self, index):
        dataset = self.metadata.loc[index]["dataset"]
        return self.datasets.index(dataset)

    def getPretrainData(self, mode):
        if mode == "train":
            a = self.metadata[(self.metadata["cv"].isin(self.cv_train))
                              & (self.metadata["dataset"] == self.datasets[-1])
                              & (self.metadata["ten_fold"] <= self.label_percentage // 10)].index.to_numpy()
            if len(self.datasets) >= 2:
                b = self.metadata[(self.metadata["cv"].isin(self.cv_train))
                              & (self.metadata["dataset"].isin(self.datasets[:-1]))].index.to_numpy()
                return np.concatenate((a, b), axis=0)
            else:
                return a
        elif mode == "source_only":
            return self.metadata[(self.metadata["cv"].isin(self.cv_train))
                              & (self.metadata["dataset"].isin(self.datasets[:-1]))].index.to_numpy()
        elif mode == "test":
            return self.test
        else:
            raise ValueError(f"Mode {mode} is not available")

    def getFinetuneData(self):
        return self.getDAData()

    def getQuerySet(self):
        return self.query_s, self.query_t
    
    def getDAData(self):
        sources = self.metadata[(self.metadata["cv"].isin(self.cv_train)) & (self.metadata["dataset"].isin(self.datasets[:-1]))].index.to_numpy()
        target = self.metadata[(self.metadata["cv"].isin(self.cv_train)) & (self.metadata["dataset"] == self.datasets[-1])].index.to_numpy()
        return sources, target

    def getTestData(self):
        sources = self.metadata[(self.metadata["cv"].isin(self.cv_test)) & (self.metadata["dataset"].isin(self.datasets[:-1]))].index.to_numpy()
        target = self.metadata[(self.metadata["cv"].isin(self.cv_test)) & (self.metadata["dataset"] == self.datasets[-1])].index.to_numpy()
        return sources, target

    def getTargetData(self, label_only=True):
        if label_only:
            target = self.metadata[(self.metadata["cv"].isin(self.cv_train)) & (self.metadata["dataset"] == self.datasets[-1]) & (self.metadata["ten_fold"] <= self.label_percentage // 10)].index.to_numpy()
        else:
            target = self.metadata[(self.metadata["cv"].isin(self.cv_train)) & (self.metadata["dataset"] == self.datasets[-1])].index.to_numpy()
        return target

    def getIndexes(self, mode):
        mode = mode.lower()
        assert mode in ["train", "test", "source_only"], f"num_source '{mode}' must be one of [\"train\", \"test\", \"source_only\"]."
        if mode == "train":
            return self.query_s, self.support_s, self.query_t, self.support_t
        elif mode == "source_only":
            return self.query_s, self.support_s
        else:
            return self.test

    def getSeriesByIndex(self, index):
        series = self.metadata.iloc[index]
        return series

    def getFileNameByIndex(self, index):
        series = self.metadata.loc[index]
        return os.path.join(get_dataset_dir(series["dataset"]), f"Sample{series['sample']}/Comp{series['comp']:03d}.hdf5")

    def getDataDicByIndex(self, index):
        file = self.getFileNameByIndex(index)
        data = {}
        with h5py.File(file, "r") as hf:
            for k in self.data_keys:
                if k not in hf.keys():
                    raise Exception("\"{}\" is Invaild key, choose one of {}".format(k, tuple(hf.keys())))
                data[k] = np.array(hf[k])
                
        data["nlabel"] = data["label"].copy()
        if self.label_percentage != 100 and index in self.target_unlabeled_idx:
            data["nlabel"] = np.array([[-1]])
            
        data["dlabel"] = np.array([[self.getDatasetLabelByIndex(index)]])

        return data

    def getBatchDicByIndexes(self, indexes):
        ndim = {}
        for i, index in enumerate(indexes):
            if i == 0:
                batch = self.getDataDicByIndex(index)
                for k in self.keys:
                    ndim[k] = batch[k].ndim
                    if self.max_shape[k] >= batch[k].shape[-1]:
                        batch[k] = np.pad(batch[k], [[0, 0]] * (ndim[k] - 1) + [[0, self.max_shape[k] - batch[k].shape[-1]]])
                    else:
                        self.max_shape[k] = batch[k].shape[-1]
            else:
                data = self.getDataDicByIndex(index)
                for k in self.keys:
                    if self.max_shape[k] >= data[k].shape[-1]:
                        data[k] = np.pad(data[k],
                                         [[0, 0]] * (ndim[k] - 1) + [[0, self.max_shape[k] - data[k].shape[-1]]])
                    else:
                        self.max_shape[k] = data[k].shape[-1]
                        batch[k] = np.pad(batch[k],
                                          [[0, 0]] * (ndim[k] - 1) + [[0, self.max_shape[k] - batch[k].shape[-1]]])
                    batch[k] = np.concatenate([batch[k], data[k]], axis=0)

        return batch
