import os
import time
import argparse
from datetime import datetime
from collections import OrderedDict
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import higher

from util import *
from model import get_weight, CWConv, resnet_mmtm, ConHead
from loss import SupConLoss

class FeatureExtractor(nn.Module):
    def __init__(self, cwconv_kernel=7):
        super(FeatureExtractor, self).__init__()
        self.wavelet = CWConv(kernel_size=cwconv_kernel)
        self.net_tw = resnet_mmtm(num_class=32)

    def forward(self, data, tdata):
        w = self.wavelet(tdata)

        w = w.unsqueeze(1)
        x, y, z = self.net_tw(tdata, w, data)
        _out = torch.cat((x, y, z), dim=1).clone()

        return _out
    
class ClassifierNoisyLabel(nn.Module):
    def __init__(self, num_task, bias=True):
        super(ClassifierNoisyLabel, self).__init__()
        self.num_task = num_task
        self.layer_x = nn.ModuleList()
        self.layer_y = nn.ModuleList()
        self.layer_z = nn.ModuleList()
        for i in range(num_task+1):
            self.layer_x.append(nn.Linear(in_features=32, out_features=2, bias=bias))
            self.layer_y.append(nn.Linear(in_features=32, out_features=2, bias=bias))
            self.layer_z.append(nn.Linear(in_features=32, out_features=2, bias=bias))

    def forward(self, feature, latent_dims=32, log_softmax=True, idx_task=-1):
        x, y, z = feature[:,:latent_dims], feature[:,latent_dims:2*latent_dims], feature[:,2*latent_dims:]

        
        assert idx_task in [-1] + list(range(self.num_task+1)), f"arg idx_task {idx_task} should be -1 or range of num_task (-1, 0, ..., {self.num_task+1})"
        x = self.layer_x[idx_task](F.relu(x))
        y = self.layer_y[idx_task](F.relu(y))
        z = self.layer_z[idx_task](F.relu(z))
            
        x = F.softmax(x, dim=1)
        y = F.softmax(y, dim=1)
        z = F.softmax(z, dim=1)
        
        a = (x + y + z) / 3

        if self.training and log_softmax:
            a = torch.log(a)
            
        return a
 
class Trainer:
    def __init__(self, config):
        self.parser = config.parse_args()
        self.model_save_timestamp = self.parser.model_save_timestamp

    def train(self, num_source):
        parser = self.parser

        # Define Device
        device = parser.device

        ## Train Config
        batch_size = parser.batch_size
        iter_size = parser.meta_cycle_size
        iter_log_ratio = parser.iter_log_ratio

        ## Saving config
        model_save_path = parser.path_model
        model_checkpoint_ratio = parser.model_checkpoint_ratio
        if model_checkpoint_ratio != 0:
            model_checkpoint = int(iter_size * model_checkpoint_ratio)  # 0 for false
        else:
            model_checkpoint = 0
        if not (os.path.isdir(model_save_path)):
            os.makedirs(os.path.join(model_save_path))
        model_base_path = model_save_path + os.path.sep
        load_model = parser.train_load_model
        load_model_name = parser.train_load_model_name
        test_model = parser.test_model

        ## Hyperparam
        lr = parser.lr
        weight_decay = parser.weight_decay
        momentum = parser.momentum

        # Open train_data
        datasets = parser.datasets.split()
        self.hdf5 = Hdf5(num_source=num_source, datasets=datasets, label_percentage=parser.label_percentage)

        path_save_info = parser.path_save_info
        if not (os.path.isdir(path_save_info)):
            os.makedirs(os.path.join(path_save_info))
        path_save_info = path_save_info + os.path.sep + "train_info{}_{}.csv".format(self.model_save_timestamp, num_source)

        pd.Series(vars(parser)).to_csv(path_save_info.replace(".csv", "_hyperparams.csv"))
        with open(path_save_info.replace(".csv", "_test.csv"), "w") as f:
            f.write("acc,sen,spec,f1\n")

        ## define & init model
        self.net_f = nn.DataParallel(FeatureExtractor()).to(device)
        self.net_c = nn.DataParallel(ClassifierNoisyLabel(3)).to(device)

        load_model_name = load_model_name.replace(".pth", f"_{num_source}.pth")
        ## Load Model
        if load_model:
            self.net_f.load_state_dict(get_weight(load_model_name, self.net_f.state_dict()), strict=True)
            self.net_c.load_state_dict(get_weight(load_model_name, self.net_c.state_dict()), strict=True)

        ## train op
        n_loss = nn.NLLLoss().to(device)

        #########################################    1. Pretraining     ###################################################
        if parser.pretrain:
            ## optimizers
            idx_train = self.hdf5.getPretrainData("train")
            optimizer = torch.optim.SGD(list(self.net_f.parameters())+list(self.net_c.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)

            pretrain_iter_size = parser.pretrain_iter_size
            batch_size = parser.batch_size
            step_size = len(idx_train) // batch_size
            
            step_log = step_size * iter_log_ratio // 1 + 1
            print(f"Pretraining with {pretrain_iter_size} epoch")
            for epoch in range(pretrain_iter_size):
                start_time = time.time()
                eval_list = ["acc_1", "sen_1", "spec_1", "f1_1"]
                loss_list = ["n"]
                epoch_cost = {key1: 0 for key1 in loss_list}
                step_cost = {key1: 0 for key1 in loss_list}
                step_info = {key1: 0 for key1 in eval_list}
                out_stack1 = []
                label_stack1 = []
                self.net_f.train()
                self.net_c.train()
                optimizer.zero_grad()

                np.random.shuffle(idx_train)
                for i in range(step_size):
                    batch_mask = range(i * batch_size, (i + 1) * batch_size)
                    train_batch_dic1 = self.hdf5.getBatchDicByIndexes(idx_train[batch_mask])

                    data1 = Variable(torch.tensor(train_batch_dic1['data'], dtype=torch.float)).to(device)
                    tdata1 = Variable(torch.tensor(train_batch_dic1['tdata'], dtype=torch.float)).to(device)
                    label1 = Variable(torch.tensor(train_batch_dic1['label'].squeeze(), dtype=torch.long)).to(device)

                    # stage1 : extractor + classifiers
                    feature = self.net_f(data1, tdata1)
                    re_out = self.net_c(feature)
                    
                    n_cost = n_loss(re_out, label1)
                    optimizer.zero_grad()
                    n_cost.backward()
                    optimizer.step()

                    epoch_cost["n"] += n_cost.item()
                    step_cost["n"] += n_cost.item()

                    # save the info. of training step
                    out_stack1.extend(np.argmax(re_out.cpu().detach().tolist(), axis=1))
                    label_stack1.extend(label1.cpu().detach().tolist())

                    ## evaluate each model
                    step_info['acc_1'] = accuracy(out_stack1, label_stack1)
                    step_info['sen_1'] = sensitivity(out_stack1, label_stack1)
                    step_info['spec_1'] = specificity(out_stack1, label_stack1)
                    step_info['f1_1'] = f1_score(out_stack1, label_stack1)

                    if step_log != 0 and (i + 1) % step_log == 0:
                        log = f"epoch[{epoch + 1}] step [{i + 1:3}/{step_size}] time [{time.time() - start_time:.1f}s]" + \
                            f"| noise/signal CEloss [{step_cost['n'] / step_log:.5f}] | Training set acc [{step_info['acc_1']:.3f}] sen [{step_info['sen_1']:.3f}] spec [{step_info['spec_1']:.3f}] f1 [{step_info['f1_1']:.3f}] |"
                        print(log)

                        for key in step_cost.keys():
                            step_cost[key] = 0
                            
            ## saving model
            print("pretrained model finished!!")
            print("saving pretrain learning model....")
            model_name = model_base_path + test_model
            torch.save(self.net_f.state_dict(), model_name.replace(".pth", "_{}_pretrain_f.pth".format(num_source)))
        else:
            print("skip pretraining....")

        #########################################    2. Meta learning     #################################################
        # Sampling source & target in 50/50
        batch_size = parser.batch_size // 2
        
        # Initialize classifiers
        self.net_c = nn.DataParallel(ClassifierNoisyLabel(3)).to(device)
        
        # Load data
        idx_query_s, idx_support_s, idx_query_t, idx_support_t = self.hdf5.getIndexes("train")
        num_task = len(idx_query_s)

        # setting for contrastive learning 
        parameters = list(self.net_f.parameters())
        self.con_head = ConHead(96, 32, True).to(device)
        con_loss = SupConLoss().to(device)
        parameters += list(self.con_head.parameters())
            
        ## optimizers
        meta_optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
        inner_optimizer = torch.optim.SGD(self.net_f.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        classifier_optimizer = torch.optim.SGD(self.net_c.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        ## data settings
        support_max_step_size = parser.meta_step_size # seperate for memory effiency
        query_max_step_size = parser.meta_step_size
        support_data_split = [len(s) // (batch_size * support_max_step_size) for s in idx_support_s]
        query_data_split = [len(q) // (batch_size * query_max_step_size) for q in idx_query_s]
        idx_support_data_split = [0 for i in range(num_task)]
        idx_query_data_split = [0 for i in range(num_task)]
        idx_inner_shuffle = [np.arange(len(s)) for s in idx_support_s]
        idx_outer_shuffle = [np.arange(len(q)) for q in idx_query_s]

        support_data_split_target = [len(s) // (batch_size * support_max_step_size) for s in idx_support_t]
        query_data_split_target = [len(q) // (batch_size * query_max_step_size) for q in idx_query_t]
        idx_support_data_split_target = [0 for i in range(num_task)]
        idx_query_data_split_target = [0 for i in range(num_task)]
        idx_inner_shuffle_target = [np.arange(len(s)) for s in idx_support_t]
        idx_outer_shuffle_target = [np.arange(len(q)) for q in idx_query_t]

        for a, b, c, d in zip(idx_inner_shuffle, idx_outer_shuffle, idx_inner_shuffle_target, idx_outer_shuffle_target):
            np.random.shuffle(a)
            np.random.shuffle(b)
            np.random.shuffle(c)
            np.random.shuffle(d)

        print(f"[Data split] support {support_data_split} splits for {support_max_step_size} steps, query {query_data_split} splits for {query_max_step_size} steps")
        print(f"Meta Learning with {iter_size} epoch")
        for epoch in range(iter_size):
            start_time = time.time()
            eval_list = [metric+f"_{task_idx+1}_{b}"for task_idx in range(num_task) for metric in ["acc", "sen", "spec", "f1"] for b in ["source", "target"]]
            loss_list = ["outer", "inner"]
            epoch_cost = {key1: 0 for key1 in loss_list}

            self.net_f.train()
            self.net_c.train()
            self.con_head.train()

            meta_optimizer.zero_grad()
            inner_optimizer.zero_grad()
            total_loss = torch.tensor(0.).to(device)
            task_sort = np.arange(num_task)
            np.random.shuffle(task_sort)

            for j, idx_task in enumerate(task_sort):
                with higher.innerloop_ctx(self.net_f, inner_optimizer, copy_initial_weights=False) as (fnet, diffopt):
                    query_set_s = idx_query_s[idx_task]
                    support_set_s = idx_support_s[idx_task]
                    query_set_t = idx_query_t[idx_task]
                    support_set_t = idx_support_t[idx_task]

                    ## 1) inner loop
                    inner_step_size = len(support_set_s) // batch_size
                    inner_step_size = inner_step_size // support_data_split[idx_task]
                    step_info = {key1: 0 for key1 in eval_list}
                    out_stack1 = []
                    label_stack1 = []
                    out_stack2 = []
                    label_stack2 = []
                    for i in range(inner_step_size):
                        # load batch data
                        batch_mask = range(idx_support_data_split[idx_task] * inner_step_size * batch_size + i * batch_size, idx_support_data_split[idx_task] * inner_step_size * batch_size + (i + 1) * batch_size)
                        train_batch_dic_s = self.hdf5.getBatchDicByIndexes(support_set_s[idx_inner_shuffle[idx_task][batch_mask]])
                        support_data_s = Variable(torch.tensor(train_batch_dic_s['data'], dtype=torch.float)).to(device)
                        support_tdata_s = Variable(torch.tensor(train_batch_dic_s['tdata'], dtype=torch.float)).to(device)
                        support_label_s = Variable(torch.tensor(train_batch_dic_s['label'].squeeze(), dtype=torch.long)).to(device)

                        batch_mask_t = [b % len(idx_inner_shuffle_target[idx_task]) for b in batch_mask] # over/undersampling
                        train_batch_dic_t = self.hdf5.getBatchDicByIndexes(support_set_t[idx_inner_shuffle_target[idx_task][batch_mask_t]])
                        support_data_t = Variable(torch.tensor(train_batch_dic_t['data'], dtype=torch.float)).to(device)
                        support_tdata_t = Variable(torch.tensor(train_batch_dic_t['tdata'], dtype=torch.float)).to(device)
                        support_label_t = Variable(torch.tensor(train_batch_dic_t['label'].squeeze(), dtype=torch.long)).to(device)
                        support_unlabeled = Variable(torch.tensor(train_batch_dic_t['nlabel'].squeeze(), dtype=torch.long)).to(device)
                        
                        # forwarding
                        support_modulated_s = fnet(support_data_s, support_tdata_s)
                        support_modulated_t = fnet(support_data_t, support_tdata_t)

                        support_out_s = self.net_c(support_modulated_s, idx_task=idx_task)
                        support_out_t = self.net_c(support_modulated_t)

                        support_modulated_s = self.con_head(support_modulated_s)
                        support_modulated_t = self.con_head(support_modulated_t)
                        da_cost = con_loss(support_modulated_s[:, None, ...], support_label_s)
                        if (support_unlabeled != -1).cpu().detach().numpy().sum() > 0:
                            labeled_features = support_modulated_t[support_unlabeled != -1]
                            da_cost += con_loss(labeled_features[:, None, ...], support_label_t[support_unlabeled != -1])
                        
                        # compute loss
                        if parser.label_percentage == 100:
                            support_cost = n_loss(support_out_s / parser.temperature, support_label_s) + n_loss(support_out_t / parser.temperature, support_label_t) + 0.1 * da_cost
                        else:
                            support_cost = n_loss(support_out_s / parser.temperature, support_label_s) + 0.1 * da_cost
                        diffopt.step(support_cost)

                        with torch.no_grad():
                            epoch_cost["inner"] += support_cost.item()

                        out_stack1.extend(np.argmax(support_out_s.cpu().detach().tolist(), axis=1))
                        label_stack1.extend(support_label_s.cpu().detach().tolist())
                        out_stack2.extend(np.argmax(support_out_t.cpu().detach().tolist(), axis=1))
                        label_stack2.extend(support_label_t.cpu().detach().tolist())

                    step_info[f'acc_task{idx_task + 1}_source_inner'] = accuracy(out_stack1, label_stack1)
                    step_info[f'sen_task{idx_task + 1}_source_inner'] = sensitivity(out_stack1, label_stack1)
                    step_info[f'spec_task{idx_task + 1}_source_inner'] = specificity(out_stack1, label_stack1)
                    step_info[f'f1_task{idx_task + 1}_source_inner'] = f1_score(out_stack1, label_stack1)
                    step_info[f'acc_task{idx_task + 1}_target_inner'] = accuracy(out_stack2, label_stack2)
                    step_info[f'sen_task{idx_task + 1}_target_inner'] = sensitivity(out_stack2, label_stack2)
                    step_info[f'spec_task{idx_task + 1}_target_inner'] = specificity(out_stack2, label_stack2)
                    step_info[f'f1_task{idx_task + 1}_target_inner'] = f1_score(out_stack2, label_stack2)

                    ## 2) outer loop
                    outer_step_size = len(query_set_s) // batch_size
                    outer_step_size = outer_step_size // query_data_split[idx_task]
                    step_log = outer_step_size
                    
                    out_stack1 = []
                    label_stack1 = []
                    out_stack2 = []
                    label_stack2 = []
                    for i in range(outer_step_size):
                        batch_mask = range(idx_query_data_split[idx_task] * outer_step_size * batch_size + i * batch_size,
                                           idx_query_data_split[idx_task] * outer_step_size * batch_size + (i + 1) * batch_size)
                        train_batch_dic_s = self.hdf5.getBatchDicByIndexes(query_set_s[idx_outer_shuffle[idx_task][batch_mask]])
                        query_data_s = Variable(torch.tensor(train_batch_dic_s['data'], dtype=torch.float)).to(device)
                        query_tdata_s = Variable(torch.tensor(train_batch_dic_s['tdata'], dtype=torch.float)).to(device)
                        query_label_s = Variable(torch.tensor(train_batch_dic_s['label'].squeeze(), dtype=torch.long)).to(device)
                        batch_mask_t = [b % len(idx_outer_shuffle_target[idx_task]) for b in batch_mask]
                        train_batch_dic_t = self.hdf5.getBatchDicByIndexes(query_set_t[idx_outer_shuffle_target[idx_task][batch_mask_t]])
                        query_data_t = Variable(torch.tensor(train_batch_dic_t['data'], dtype=torch.float)).to(device)
                        query_tdata_t = Variable(torch.tensor(train_batch_dic_t['tdata'], dtype=torch.float)).to(device)
                        query_label_t = Variable(torch.tensor(train_batch_dic_t['label'].squeeze(), dtype=torch.long)).to(device)
                        query_unlabeled = Variable(torch.tensor(train_batch_dic_t['nlabel'].squeeze(), dtype=torch.long)).to(device)
                        # query_labeled_mask = (query_unlabeled != -1)

                        assert -1 not in query_unlabeled, "Unlabeled data are not allowed in outer loop"

                        query_modulated_s = fnet(query_data_s, query_tdata_s)
                        query_modulated_t = fnet(query_data_t, query_tdata_t)
                        
                        query_out_s = self.net_c(query_modulated_s, idx_task=idx_task)
                        query_out_t = self.net_c(query_modulated_t)

                        query_modulated_s = self.con_head(query_modulated_s)
                        query_modulated_t = self.con_head(query_modulated_t)
                        da_cost = con_loss(query_modulated_s[:, None, ...], query_label_s) + con_loss(query_modulated_t[:, None, ...], query_label_t)
                        
                        meta_coeff = 0.1
                        source_coeff = 1

                        outer_loss = source_coeff * n_loss(query_out_s / parser.temperature, query_label_s) + n_loss(query_out_t / parser.temperature, query_label_t) + meta_coeff * da_cost
                        total_loss += outer_loss
                        
                    
                        with torch.no_grad():
                            # save the info. of training step
                            epoch_cost["outer"] += outer_loss.item()
                            # step_cost["outer"] += outer_loss.item()

                        out_stack1.extend(np.argmax(query_out_s.cpu().detach().tolist(), axis=1))
                        label_stack1.extend(query_label_s.cpu().detach().tolist())
                        out_stack2.extend(np.argmax(query_out_t.cpu().detach().tolist(), axis=1))
                        label_stack2.extend(query_label_t.cpu().detach().tolist())

                    ## evaluate each model
                    step_info[f'acc_task{idx_task+1}_source'] = accuracy(out_stack1, label_stack1)
                    step_info[f'sen_task{idx_task+1}_source'] = sensitivity(out_stack1, label_stack1)
                    step_info[f'spec_task{idx_task+1}_source'] = specificity(out_stack1, label_stack1)
                    step_info[f'f1_task{idx_task+1}_source'] = f1_score(out_stack1, label_stack1)
                    step_info[f'acc_task{idx_task+1}_target'] = accuracy(out_stack2, label_stack2)
                    step_info[f'sen_task{idx_task+1}_target'] = sensitivity(out_stack2, label_stack2)
                    step_info[f'spec_task{idx_task+1}_target'] = specificity(out_stack2, label_stack2)
                    step_info[f'f1_task{idx_task+1}_target'] = f1_score(out_stack2, label_stack2)

                    # print epoch info & saving
                    log = f"Epoch [{epoch + 1}] task [{j + 1}/{num_task}][#{idx_task + 1}] | query_split [{idx_query_data_split[idx_task] + 1}/{query_data_split[idx_task]}] support_split [{idx_support_data_split[idx_task] + 1}/{support_data_split[idx_task]}] | time [{time.time() - start_time:.1f}s]" + \
                          f"| noise/signal outer_loss [{epoch_cost['outer'] / outer_step_size:.5f}] inner_loss [{epoch_cost['inner'] / inner_step_size:.5f}] | \n" + \
                          f" Support source acc [{step_info[f'acc_task{idx_task + 1}_source_inner']:.3f}] sen [{step_info[f'sen_task{idx_task + 1}_source_inner']:.3f}] spec [{step_info[f'spec_task{idx_task + 1}_source_inner']:.3f}] f1 [{step_info[f'f1_task{idx_task + 1}_source_inner']:.3f}] | target acc [{step_info[f'acc_task{idx_task + 1}_target_inner']:.3f}] sen [{step_info[f'sen_task{idx_task + 1}_target_inner']:.3f}] spec [{step_info[f'spec_task{idx_task + 1}_target_inner']:.3f}] f1 [{step_info[f'f1_task{idx_task + 1}_target_inner']:.3f}] | \n" + \
                          f" Query source acc [{step_info[f'acc_task{idx_task + 1}_source']:.3f}] sen [{step_info[f'sen_task{idx_task + 1}_source']:.3f}] spec [{step_info[f'spec_task{idx_task + 1}_source']:.3f}] f1 [{step_info[f'f1_task{idx_task + 1}_source']:.3f}] | target acc [{step_info[f'acc_task{idx_task + 1}_target']:.3f}] sen [{step_info[f'sen_task{idx_task + 1}_target']:.3f}] spec [{step_info[f'spec_task{idx_task + 1}_target']:.3f}] f1 [{step_info[f'f1_task{idx_task + 1}_target']:.3f}] | "
                    print(log)
            
            # Meta optimization   
            meta_optimizer.zero_grad()
            total_loss.div_(num_task)
            total_loss.backward()
            meta_optimizer.step()
            total_loss = torch.tensor(0.).to(device)
            
            ## 3) Decoupling training
            t_cost = 0
            for _idx_task in range(num_task):
                _query_set_s = idx_query_s[_idx_task]
                _query_set_t = idx_query_t[_idx_task]

                outer_step_size = len(_query_set_s) // batch_size
                outer_step_size = outer_step_size // query_data_split[_idx_task]
                for i in range(outer_step_size):
                    batch_mask = range(idx_query_data_split[_idx_task] * outer_step_size * batch_size + i * batch_size,
                                        idx_query_data_split[_idx_task] * outer_step_size * batch_size + (i + 1) * batch_size)
                    train_batch_dic_s = self.hdf5.getBatchDicByIndexes(_query_set_s[idx_outer_shuffle[_idx_task][batch_mask]])
                    trans_data_s = Variable(torch.tensor(train_batch_dic_s['data'], dtype=torch.float)).to(device)
                    trans_tdata_s = Variable(torch.tensor(train_batch_dic_s['tdata'], dtype=torch.float)).to(device)
                    trans_label_s = Variable(torch.tensor(train_batch_dic_s['label'].squeeze(), dtype=torch.long)).to(device)
                    batch_mask_t = [b % len(idx_outer_shuffle_target[_idx_task]) for b in batch_mask]
                    train_batch_dic_t = self.hdf5.getBatchDicByIndexes(_query_set_t[idx_outer_shuffle_target[_idx_task][batch_mask_t]])
                    trans_data_t = Variable(torch.tensor(train_batch_dic_t['data'], dtype=torch.float)).to(device)
                    trans_tdata_t = Variable(torch.tensor(train_batch_dic_t['tdata'], dtype=torch.float)).to(device)
                    trans_label_t = Variable(torch.tensor(train_batch_dic_t['label'].squeeze(), dtype=torch.long)).to(device)
                    trans_unlabeled = Variable(torch.tensor(train_batch_dic_t['nlabel'].squeeze(), dtype=torch.long)).to(device)

                    assert -1 not in trans_unlabeled, "Unlabeled data are not allowed in outer loop"

                    with torch.no_grad():
                        trans_features_s = self.net_f(trans_data_s, trans_tdata_s)
                        trans_features_t = self.net_f(trans_data_t, trans_tdata_t)

                    trans_out_s = self.net_c(trans_features_s, idx_task=idx_task)
                    trans_out_t = self.net_c(trans_features_t)


                    transition_loss = n_loss(trans_out_s / parser.temperature, trans_label_s) + n_loss(trans_out_t / parser.temperature, trans_label_t)

                    classifier_optimizer.zero_grad()
                    transition_loss.backward()
                    classifier_optimizer.step()
                    
                    with torch.no_grad():
                        t_cost += transition_loss.item()

            
            # shuffling
            if idx_query_data_split[idx_task]+1 == query_data_split[idx_task]:
                np.random.shuffle(idx_outer_shuffle[idx_task])
                np.random.shuffle(idx_outer_shuffle_target[idx_task])

            if idx_support_data_split[idx_task]+1 == support_data_split[idx_task]:
                np.random.shuffle(idx_inner_shuffle[idx_task])
                np.random.shuffle(idx_inner_shuffle_target[idx_task])

            idx_query_data_split[idx_task] = (idx_query_data_split[idx_task]+1) % query_data_split[idx_task]
            idx_support_data_split[idx_task] = (idx_support_data_split[idx_task]+1) % support_data_split[idx_task]
            idx_query_data_split_target[idx_task] = (idx_query_data_split_target[idx_task]+1) % query_data_split_target[idx_task]
            idx_support_data_split_target[idx_task] = (idx_support_data_split_target[idx_task]+1) % support_data_split_target[idx_task]

            log = f"Epoch [{epoch + 1}/{iter_size}]| time [{time.time() - start_time:.1f}s]" + \
                  f"| noise/signal outer_loss [{epoch_cost['outer'] / outer_step_size:.5f}]| inner_loss [{epoch_cost['inner'] / inner_step_size:.5f}]  "
            print(log)

            if not os.path.exists(path_save_info.replace(".csv","_meta.csv")):
                with open(path_save_info.replace(".csv","_meta.csv"), "w") as f:
                    f.write(",".join([*[f"loss_{k}" for k in loss_list], *eval_list]) + "\n")
            with open(path_save_info.replace(".csv","_meta.csv"), "a") as f:
                f.write(",".join([str(i) for i in [*[epoch_cost[k] / step_log if k not in ["filtered", "thre"] else epoch_cost[k] for k in loss_list],
                                                   *[step_info[key] for key in eval_list]]]) + "\n")

            ## saving checkpoint
            if model_checkpoint != 0 and (epoch + 1) % model_checkpoint == 0:
                print("checkpoint saving..")
                model_name = model_base_path + 'model_{}_{:04}.pth'.format(num_source, epoch + 1)
                torch.save(self.net_f.state_dict(), model_name.replace(".pth", "_meta_f.pth".format(num_source)))
                torch.save(self.net_c.state_dict(), model_name.replace(".pth", "_meta_c.pth".format(num_source)))
                torch.save(self.con_head.state_dict(), model_name.replace(".pth", "_meta_con.pth".format(num_source)))
            
        print("Meta learning finished!!")

        ## saving model
        print("Saving meta learning model....")
        model_name = model_base_path + test_model
        torch.save(self.net_f.state_dict(), model_name.replace(".pth", "_{}_meta_f.pth".format(num_source)))
        torch.save(self.net_c.state_dict(), model_name.replace(".pth", "_{}_meta_c.pth".format(num_source)))
        torch.save(self.con_head.state_dict(), model_name.replace(".pth", "_{}_meta_con.pth".format(num_source)))

        ## 4. Quick Finetuning 

        # load data
        idx_train_t = self.hdf5.getTargetData()

        # optimizers
        optimizer = torch.optim.SGD(list(self.net_f.parameters())+list(self.net_c.parameters()), lr=lr, momentum=momentum, weight_decay=weight_decay)

        finetune_iter_size = parser.finetune_iter_size
        batch_size = parser.batch_size
        step_size = len(idx_train_t) // batch_size
        
        step_log = step_size * iter_log_ratio // 1 + 1
        print(f"Finetuning with {finetune_iter_size} epoch")
        for epoch in range(finetune_iter_size):
            start_time = time.time()
            eval_list = ["acc_1", "sen_1", "spec_1", "f1_1"]
            loss_list = ["n"]
            epoch_cost = {key1: 0 for key1 in loss_list}
            step_cost = {key1: 0 for key1 in loss_list}
            step_info = {key1: 0 for key1 in eval_list}
            out_stack1 = []
            label_stack1 = []
            self.net_f.train()
            self.net_c.train()
            optimizer.zero_grad()

            np.random.shuffle(idx_train_t)
            for i in range(step_size):
                batch_mask = range(i * batch_size, (i + 1) * batch_size)
                train_batch_dic1 = self.hdf5.getBatchDicByIndexes(idx_train_t[batch_mask])

                data1 = Variable(torch.tensor(train_batch_dic1['data'], dtype=torch.float)).to(device)
                tdata1 = Variable(torch.tensor(train_batch_dic1['tdata'], dtype=torch.float)).to(device)
                label1 = Variable(torch.tensor(train_batch_dic1['label'].squeeze(), dtype=torch.long)).to(device)

                # stage1 : extractor + classifiers
                feature = self.net_f(data1, tdata1)
                
                re_out = self.net_c(feature)
                n_cost = n_loss(re_out, label1)
                optimizer.zero_grad()
                n_cost.backward()
                optimizer.step()

                epoch_cost["n"] += n_cost.item()
                step_cost["n"] += n_cost.item()

                # save the info. of training step
                out_stack1.extend(np.argmax(re_out.cpu().detach().tolist(), axis=1))
                label_stack1.extend(label1.cpu().detach().tolist())

                ## evaluate each model
                step_info['acc_1'] = accuracy(out_stack1, label_stack1)
                step_info['sen_1'] = sensitivity(out_stack1, label_stack1)
                step_info['spec_1'] = specificity(out_stack1, label_stack1)
                step_info['f1_1'] = f1_score(out_stack1, label_stack1)

                if step_log != 0 and (i + 1) % step_log == 0:
                    log = f"epoch[{epoch + 1}] step [{i + 1:3}/{step_size}] time [{time.time() - start_time:.1f}s]" + \
                          f"| noise/signal loss [{step_cost['n'] / step_log:.5f}] | Domain 1 acc [{step_info['acc_1']:.3f}] sen [{step_info['sen_1']:.3f}] spec [{step_info['spec_1']:.3f}] f1 [{step_info['f1_1']:.3f}] |"
                    print(log)

                    for key in step_cost.keys():
                        step_cost[key] = 0

            log = f"[=] epoch [{epoch + 1}/{finetune_iter_size}] time [{time.time() - start_time:.1f}s]" + \
                  f"| noise/signal loss [{epoch_cost['n'] / step_log:.5f}] | Domain 1 acc [{step_info['acc_1']:.3f}] sen [{step_info['sen_1']:.3f}] spec [{step_info['spec_1']:.3f}] f1 [{step_info['f1_1']:.3f}] |"
            print(log)

            if not os.path.exists(path_save_info.replace(".csv", "_finetune.csv")):
                with open(path_save_info.replace(".csv", "_finetune.csv"), "w") as f:
                    f.write(",".join([*[f"loss_{k}" for k in loss_list], *eval_list]) + "\n")
            with open(path_save_info.replace(".csv", "_finetune.csv"), "a") as f:
                f.write(",".join([str(i) for i in [
                    *[epoch_cost[k] / step_log for k in loss_list],
                    *[step_info[key] for key in eval_list]]]) + "\n")

        ## testing checkpoint
        self.eval(num_source, mode="test")

        print("training finished!!")

        ## saving model
        print("saving model....")
        model_name = model_base_path + test_model
        torch.save(self.net_f.state_dict(), model_name.replace(".pth", "_{}_final_f.pth".format(num_source)))
        torch.save(self.net_c.state_dict(), model_name.replace(".pth", "_{}_final_c.pth".format(num_source)))

        torch.cuda.empty_cache()
        print("[!!] Training finished")


    def eval(self, num_source, mode="test"):
        assert mode in ["test"], f"mode {mode} should be one of [\"test\"]"

        # Define Device
        parser = self.parser
        device = parser.device
        iter_log_ratio = parser.iter_log_ratio

        # load test data
        eval_idx = self.hdf5.getIndexes(mode)
        meta_data = self.hdf5.getMetadata()

        # load model
        self.net_f.eval()
        self.net_c.eval()

        # testing
        print("Device :", device)
        print("Testing...")

        step_size = len(eval_idx)  
        step_log = int(step_size * iter_log_ratio)
        start_time = time.time()

        df = None
        n_out_stack = []
        n_label_stack = []
        d_label_stack = []
        with torch.no_grad():
            for step in range(step_size):
                train_batch_dic = self.hdf5.getBatchDicByIndexes([eval_idx[step]])
                data = torch.tensor(train_batch_dic['data'], dtype=torch.float).to(device)
                tdata = torch.tensor(train_batch_dic["tdata"], dtype=torch.float).to(device)
                n_label = torch.tensor(train_batch_dic['label'].squeeze(), dtype=torch.long).to(device)
                d_label = torch.tensor(train_batch_dic['dlabel'].squeeze(), dtype=torch.long).to(device)

                f_out = self.net_f(data, tdata)
                n_out = self.net_c(f_out, idx_task=d_label.item())
                n_out_stack.extend(np.argmax(n_out.cpu().detach().tolist(), axis=1))
                n_label_stack.append(n_label.cpu().detach().tolist())
                d_label_stack.append(d_label.cpu().detach().tolist())

                if step_log != 0 and (step + 1) % step_log == 0:
                    log = "step [{:3}/{}] time [{:.1f}s]".format(step + 1, step_size, time.time() - start_time)
                    print(log)

                _dic = OrderedDict(
                    [("dataset", meta_data.loc[eval_idx[step], "dataset"]),
                     ("cv", meta_data.loc[eval_idx[step], "cv"]),
                     ("sample", meta_data.loc[eval_idx[step], "sample"]),
                     ("comp", meta_data.loc[eval_idx[step], "comp"]),
                     ("label", train_batch_dic['label'].squeeze()),
                     ("n_out", np.argmax(n_out.cpu().detach().tolist(), axis=1)),
                     ("n_prob", np.max(n_out.cpu().detach().tolist(), axis=1))])
                if df is None:
                    df = pd.DataFrame(_dic, columns=_dic.keys(), index=[0])
                else:
                    _df = pd.DataFrame(_dic, columns=_dic.keys(), index=[0])
                    df = pd.concat([df, _df], ignore_index=True)

        df = df.sort_values(by=["dataset", "sample", "comp"], ascending=True)
        df.reset_index(inplace=True, drop=True)
        timestamp = parser.model_save_timestamp
        base_path = f"test/result{timestamp}"
        if not os.path.exists(base_path):
            os.mkdir(base_path)
        csv_file = f"result{timestamp}.csv"
        if os.path.exists(os.path.join(base_path, csv_file)):
            exist_result = pd.read_csv(os.path.join(base_path, csv_file), index_col=0)
            exist_result = exist_result[exist_result["cv"] != num_source]
            df = pd.concat([exist_result, df], ignore_index=True)
            df = df.drop_duplicates()
        df.to_csv(os.path.join(base_path, csv_file))


    def cross(self):
        # cross validation
        print("5-fold cross validation start...")
        
        for i in range(1, 6):
            print("[!] [{}/5] fold validation...".format(i))
            self.train(i)
            torch.cuda.empty_cache()

        print("[!!] cross validation successfully complete...")
        print("Validation results saved : [test/result{}]".format(self.model_save_timestamp))
        


def config():
    config = argparse.ArgumentParser()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.today().strftime("_%Y%m%d%H%M%S")
                
    # pytorch base
    config.add_argument("--device", default=device)

    # training mode
    config.add_argument("--mode", default="cross", type=str, help="one, cross")
    config.add_argument("--init", default="he", type=str, help="[xavier, he, default]")
    config.add_argument("--datasets", default="hcp bcp mb6 std", type=str, help="list datasets with blank, make sure to choose within [bcp, hcp, mb6, std], place target dataset to the last (e.g., 'hcp bcp mb6': source hcp, bcp & target mb6)")
    config.add_argument("--label_percentage", default=100, type=int,
                        help="Label percentage of target domain for training choose in [10, 20, ..., 100], metadata annotation required")

    # model
    config.add_argument("--network", default="cbam", type=str, help="network, one of [resnet, cbam]")
    config.add_argument("--alpha", default=0.9, type=float, help="exponential moving average decay")

    # training hyperparams
    config.add_argument("--lr", default=0.01, type=float, help="learning rate")
    config.add_argument("--batch_size", default=12, type=int, help="batch size")
    config.add_argument("--pretrain", default=True, type=bool, help="pretrain")
    config.add_argument("--pretrain_iter_size", default=10, type=int, help="pretrain iteration size of maximum iter_size")
    config.add_argument("--meta_cycle_size", default=300, type=int, help="cycles of meta-learning")
    config.add_argument("--finetune_iter_size", default=10, type=int, help="iteration for finetuning")
    config.add_argument("--meta_step_size", default=2, type=int, help="step size for each meta optimization cycle, this option increases memory use")
    config.add_argument("--weight_decay", default=1e-4, type=float, help="weight decay for regularization ")
    config.add_argument("--momentum", default=0.9, type=float, help="optimizer momentum")
    config.add_argument("--beta1", default=0.9, type=float, help="adam optimizer beta1")
    config.add_argument("--beta2", default=0.999, type=float, help="adam optimizer beta2")
    config.add_argument("--temperature", default=0.5, type=float, help="prediction ensemble update step")
    config.add_argument("--sampling", default="under", type=str, help="oversampling or undersampling")
    config.add_argument("--reduction", default=32, type=int, help="reduction of decomposer")
    config.add_argument("--iter_log_ratio", default=0.2, type=float, help="ratio to print log step by step")

    # save & load
    config.add_argument("--model_save_timestamp", default=timestamp, type=str, help="")
    config.add_argument("--path_model", default="model{}".format(timestamp), type=str, help="path to save and to load model")
    config.add_argument("--model_checkpoint_ratio", default=0.1, type=float, help="0 for False")
    config.add_argument("--test_model", default="model.pth", type=str, help="name of saving model")
    config.add_argument("--path_save_info", default="info{}".format(timestamp), type=str, help="path of saving model")
    config.add_argument("--train_load_model", default=False, type=bool,
                        help="true or false to load model starting to train")
    config.add_argument("--train_load_model_name", default=f"", type=str,
                        help="pth file name to load model starting to train")

    return config


if __name__ == "__main__":
    config = config()
    trainer = Trainer(config)
    print("GPU : ", torch.cuda.is_available())

    trainer.cross()
