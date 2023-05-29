from cgi import test
import os
import sys
import copy
from ..utils.experiment_utils import note_taking
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from functorch import make_functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from continual.utils.fsd_utils import linearize, l2_norm, kl_div, lin_act_conv, tasks_fsd
from continual.utils.data_utils import compute_data_moments, compute_k_moments, compute_classwise_moments, SplitCIFAR100
from continual.utils.model_utils import parameters_to_vector, CifarNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda = True if torch.cuda.is_available() else False
import wandb
import matplotlib.pyplot as plt
from os.path import join as pjoin


class FSDSplitCifar(object):

    def __init__(self, hparams):
        self.hparams = hparams
        self.results_dict = {}
        self.results_back = {}
        if hparams.train.ber_momentum == -1:
            hparams.train.ber_momentum = 1. / hparams.dataset.batch_size
        self.fsd_done = []
        self.download_flag = (True if self.hparams.checkpoint_path else False)
        self.output_metric = kl_div if self.hparams.train.kl_div else l2_norm

    def compute_fsd(self, params0, params1, data_moments, mus, fisher, model, fsd_type,
                    val_batch, label_set):
        if self.hparams.cl.classwise or self.hparams.cl.k == 5:
            total_fsd = 0
            for i in range(len(mus)):
                cur_mean, cur_var, cur_cov = data_moments[0][i], data_moments[
                    1][i], data_moments[2][i]
                total_fsd += self.compute_fsd_single(
                    params0, params1, (cur_mean, cur_var, cur_cov), mus[i], fisher,
                    model, fsd_type, val_batch[i], label_set)
            total_fsd /= len(mus)
            return total_fsd
        else:
            return self.compute_fsd_single(params0, params1, data_moments, mus, fisher,
                                           model, fsd_type, val_batch,
                                           label_set)

    def compute_fsd_single(self, params0, params1, data_moments, mus, fisher, model,
                           fsd_type, val_batch, label_set):
        MEAN, VAR, COV = data_moments
        MEAN, VAR, COV = MEAN.to(device), VAR.to(device), COV.to(device)
        val_batch = val_batch.to(device)
        f, _ = make_functional(model)
        if self.hparams.cl.classwise:
            num_samples = self.hparams.train.num_samples // 10  # num_classes=10
        elif self.hparams.cl.k == 5:
            num_samples = self.hparams.train.num_samples // 5
        else:
            num_samples = self.hparams.train.num_samples

        if self.hparams.cl.use_var:
            COV = VAR

        if fsd_type == 'None':
            fsd = 0
        elif fsd_type == 'lin_act_ber':
            fsd = lin_act_conv(
                model,
                params0,
                params1,
                mus,
                MEAN,
                COV,
                num_samples,
                self.output_metric,
                label_set=label_set,
                val_batch=val_batch if self.hparams.cl.use_coreset else None,
                do_dropout=self.hparams.cl.do_dropout,
                stoch_act=True)
            if self.hparams.train.fisher:
                fsd *= fisher
        elif fsd_type == 'lin_act':
            fsd = lin_act_conv(model,
                               params0,
                               params1,
                               mus,
                               MEAN,
                               COV,
                               num_samples,
                               self.output_metric,
                               label_set=label_set,
                               val_batch=val_batch if self.hparams.cl.use_coreset else None,
                               do_dropout=self.hparams.cl.do_dropout,
                               stoch_act=False)
            if self.hparams.train.fisher:
                fsd *= fisher
        elif fsd_type == 'linear':
            f_lin = linearize(f, params0)
            fsd = self.output_metric(f(params0, val_batch, label_set),
                          f_lin(params1, val_batch, label_set))
        elif fsd_type == 'subset':
            fsd = self.output_metric(f(params0, val_batch, label_set),
                          f(params1, val_batch, label_set))
        return fsd

    def run_splitcifar(self, fsd_type):

        num_tasks = self.hparams.cl.num_tasks
        num_classes_per_task = 10
        data_transforms = transforms.ToTensor()

        cifar10_train = datasets.CIFAR10('~/datasets/cifar10/',
                                         train=True,
                                         transform=data_transforms,
                                         download=self.download_flag)
        cifar10_test = datasets.CIFAR10('~/datasets/cifar10/',
                                        train=False,
                                        transform=data_transforms,
                                        download=self.download_flag)
        cifar100_train = datasets.CIFAR100('~/datasets/cifar100/',
                                           train=True,
                                           transform=data_transforms,
                                           download=self.download_flag)
        cifar100_test = datasets.CIFAR100('~/datasets/cifar100/',
                                          train=False,
                                          transform=data_transforms,
                                          download=self.download_flag)

        datagen = SplitCIFAR100(cifar100_train, cifar100_test)

        model = CifarNet(3, num_tasks * num_classes_per_task)
        self.model = model
        criterion = nn.CrossEntropyLoss()
        if use_cuda:
            criterion.cuda()
            model.cuda()

        opt = optim.Adam(model.parameters(),
                         lr=self.hparams.train.learning_rate)
        self.opt = opt
        
        testloaders = []
        label_sets = []
        test_acc_list = []
        task_params = []
        data_moments = []
        val_batches = []
        all_mus = []
        all_fisher = []
        self.tid_loaded = None
        note_taking("starting a new CL experiment. ")

        for tid in range(num_tasks):
            if self.tid_loaded:
                if tid < self.tid_loaded + 1:
                    note_taking(f"skipping tid={tid}")
                    if tid != 0:
                        datagen.next_task()
                    continue
                else:
                    note_taking(f"running tid={tid}")
            if tid == 0:
                itrain, itest = cifar10_train, cifar10_test
                ilabel_set = list(range(10))
                if self.hparams.cl.task0epoch is None:
                    num_epochs = 200
                else:
                    num_epochs = self.hparams.cl.task0epoch
            else:
                itrain, itest, ilabel_set = datagen.next_task()
                num_epochs = self.hparams.train.train_iter
            label_sets.append(ilabel_set)
            itrainloader = DataLoader(
                dataset=itrain,
                batch_size=self.hparams.dataset.batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False)
            itestloader = DataLoader(
                dataset=itest,
                batch_size=self.hparams.dataset.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=False)

            testloaders.append(itestloader)

            if self.hparams.cl.classwise:
                MEAN, VAR, COV, val_batch = compute_classwise_moments(
                    itrainloader, list(range(num_classes_per_task)))
            elif self.hparams.cl.k == 5:
                MEAN, VAR, COV, val_batch = compute_k_moments(
                    itrainloader, [[0,1], [2,3], [4,5], [6,7], [8,9]])
            else:
                MEAN, VAR, COV, val_batch = compute_data_moments(itrainloader)
            data_moments.append((MEAN, VAR, COV))
            val_batches.append(val_batch)

            # Train on current task
            model.train()
            label_set_cur = label_sets[tid]
            for epoch in range(num_epochs):
                last_epoch = epoch == num_epochs - 1
                if last_epoch:
                    mus_avg = []
                for inputs, labels in itrainloader:
                    if use_cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    opt.zero_grad()
                    logits, mus = model.forward(inputs,
                                                label_set_cur,
                                                return_mus=True)
                    loss = criterion(logits, labels)

                    if tid > 0:
                        fsd = 0
                        for prev_tid in range(tid):
                            fsd += self.compute_fsd(
                                task_params[prev_tid],
                                parameters_to_vector(model.parameters(),
                                                     clone=False),
                                data_moments[prev_tid], all_mus[prev_tid], all_fisher[prev_tid],
                                model, fsd_type, val_batches[prev_tid],
                                label_sets[prev_tid])
                        fsd /= tid
                        loss += self.hparams.train.fsd_scale * fsd
                    loss.backward()
                    opt.step()
                    if last_epoch:
                        if len(mus_avg) == 0:
                            mus_avg = mus
                        else:
                            mus_avg = [
                                (1 - self.hparams.train.ber_momentum) *
                                mus_avg[count] +
                                self.hparams.train.ber_momentum * mus[count]
                                for count in range(len(mus))
                            ]

            if self.hparams.cl.classwise:
                all_mus.append(
                    model.get_classwise_mus(itrainloader,
                                            self.hparams.train.ber_momentum,
                                            list(range(num_classes_per_task))))
            elif self.hparams.cl.k == 5:
                all_mus.append(
                    model.get_k_mus(itrainloader,
                                self.hparams.train.ber_momentum,
                                [[0,1], [2,3], [4,5], [6,7], [8,9]]))
            else:
                all_mus.append(mus_avg)
            task_params.append(
                parameters_to_vector(model.parameters(), clone=True))

            # get fisher 
            fisher_model = copy.deepcopy(self.model)
            for inputs, labels in itrainloader:
                labels = F.one_hot(labels, num_classes=num_classes_per_task).to(torch.float32) 
                if use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                fisher_loss = criterion(fisher_model.forward(inputs, label_set_cur), labels)
                fisher_loss.backward()
            for name, param in fisher_model.named_parameters():
                if name == "out_block.weight":
                    fisher = param.grad.data.clone().pow(2) / len(itrainloader)
            all_fisher.append(torch.sqrt(fisher.sum()))

            # Test on all tasks seen so far
            model.eval()
            print('Begin testing...')
            test_accuracy = []
            for test_tid, testdata in enumerate(testloaders):
                total = 0
                correct = 0
                for inputs, labels in testdata:
                    if use_cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    label_set_t = label_sets[test_tid]
                    logits = model.forward(inputs, label_set_t)
                    predict_label = torch.argmax(logits, dim=-1)
                    total += inputs.shape[0]
                    if use_cuda:
                        correct += torch.sum(
                            predict_label == labels).cpu().item()
                    else:
                        correct += torch.sum(predict_label == labels).item()
                test_accuracy.append(correct / total)

            test_acc_list.append(test_accuracy)
            print(test_accuracy)
            print('Mean accuracy after task %d: %f' %
                  (tid, sum(test_accuracy) / len(test_accuracy)))
            tasks, mean = tasks_fsd(test_acc_list)
            self.results_dict[fsd_type] = [tasks, mean]

        # Backward Transfer Metric
        bwt = 0
        if self.hparams.checkpoint_path:
            tasks = self.results_dict[fsd_type][0]
        for i in range(len(tasks)):
            bwt += 100 * (tasks[i][-1] - tasks[i][0])
        bwt = bwt / (len(tasks) - 1)
        self.results_back[fsd_type] = bwt
        print("Backward Transfer: ", bwt)
        
        with torch.no_grad():
            if self.hparams.use_wandb:
                wandb_logging_dict = dict()
                for key, value in self.results_dict.items():
                    print(f"test {value}")
                    wandb_logging_dict[f'{key}_task0'] = value[0][0][num_tasks-1]
                    wandb_logging_dict[f'{key}_all'] = value[1][num_tasks-1]
                    wandb.log(wandb_logging_dict)

                for key, value in self.results_back.items():
                    wandb_logging_dict[f'{key}_back'] = value
                    wandb.log(wandb_logging_dict)

        return test_acc_list
