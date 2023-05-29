from cgi import test
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from functorch import make_functional
import torch.optim as optim
from torch.utils.data import DataLoader

from continual.utils.fsd_utils import linearize, l2_norm, kl_div, lin_act, determ_fsd, ewc, tasks_fsd
from continual.utils.data_utils import compute_data_moments, compute_classwise_moments, PermutedMnistGenerator
from continual.utils.model_utils import parameters_to_vector, MLP

device = 'cuda' if torch.cuda.is_available() else 'cpu'
use_cuda = True if torch.cuda.is_available() else False
import wandb
import matplotlib.pyplot as plt


class FSDPermMnist(object):

    def __init__(self, hparams):
        self.hparams = hparams
        self.results_dict = {}
        self.results_back = {}
        self.fsd_values = {}
        self.output_metric = kl_div if self.hparams.train.kl_div else l2_norm

    def compute_fsd(self, params0, params1, data_moments, mus, model, fsd_type,
                    val_batch, optpar_dict, fisher_dict):
        if fsd_type == 'ewc':
            return ewc(model, fisher_dict, optpar_dict)
        if self.hparams.cl.classwise:
            total_fsd = 0
            for i in range(len(mus)):
                cur_mean, cur_var, cur_cov = data_moments[0][i], data_moments[
                    1][i], data_moments[2][i]
                total_fsd += self.compute_fsd_single(
                    params0, params1, (cur_mean, cur_var, cur_cov), mus[i],
                    model, fsd_type, val_batch[i])
            total_fsd /= len(mus)
            return total_fsd
        else:
            return self.compute_fsd_single(params0, params1, data_moments, mus,
                                           model, fsd_type, val_batch)

    def compute_fsd_single(self, params0, params1, data_moments, mus, model,
                           fsd_type, val_batch):
        MEAN, VAR, COV = data_moments
        MEAN, VAR, COV = MEAN.to(device), VAR.to(device), COV.to(device)
        val_batch = val_batch.to(device)
        f, _ = make_functional(model)
        if self.hparams.cl.classwise:
            num_samples = self.hparams.train.num_samples // 10  # num_classes=10
        else:
            num_samples = self.hparams.train.num_samples

        if self.hparams.cl.use_var:
            COV = VAR

        if fsd_type == 'None':
            fsd = 0
        elif fsd_type == 'lin_act_ber':
            fsd = lin_act(
                params0,
                params1,
                mus,
                MEAN,
                COV,
                num_samples,
                self.output_metric,
                val_batch=val_batch if self.hparams.cl.use_coreset else None,
                stoch_act=True)
        elif fsd_type == 'lin_act':
            fsd = lin_act(params0,
                          params1,
                          mus,
                          MEAN,
                          COV,
                          num_samples,
                          self.output_metric,
                          stoch_act=False)
        elif fsd_type == 'determ':
            fsd = determ_fsd(params0, params1, mus, MEAN, COV)
        elif fsd_type == 'linear':
            f_lin = linearize(f, params0)
            fsd = self.output_metric(f(params0, val_batch), f_lin(params1, val_batch))
        elif fsd_type == 'subset':
            fsd = self.output_metric(f(params0, val_batch), f(params1, val_batch))
        return fsd

    def run_permmnist(self, fsd_type):

        num_tasks = 10
        num_classes = 10
        datagen = PermutedMnistGenerator(max_iter=num_tasks)

        layer_size = [
            784, self.hparams.model.hidden_size,
            self.hparams.model.hidden_size, num_classes
        ]
        model = MLP(layer_size, act='relu')

        criterion = nn.CrossEntropyLoss()
        if use_cuda:
            criterion.cuda()
            model.cuda()

        opt = optim.Adam(model.parameters(),
                         lr=self.hparams.train.learning_rate)

        testloaders = []
        test_acc_list = []
        task_params = []
        data_moments = []
        val_batches = []
        all_mus = []
        fisher_dict, optpar_dict = [dict() for _ in range(num_tasks)], [dict() for _ in range(num_tasks)]
        for tid in range(num_tasks):

            itrain, itest = datagen.next_task()
            itrainloader = DataLoader(
                dataset=itrain,
                batch_size=self.hparams.dataset.batch_size,
                shuffle=True)
            #num_workers=3)
            itestloader = DataLoader(
                dataset=itest,
                batch_size=self.hparams.dataset.batch_size,
                shuffle=False)
            #num_workers=3)
            if tid == 0:
                first_loader = DataLoader(
                    dataset=itrain,
                    batch_size=self.hparams.dataset.batch_size,
                    shuffle=True)

            testloaders.append(itestloader)

            if self.hparams.cl.classwise:
                MEAN, VAR, COV, val_batch = compute_classwise_moments(
                    itrainloader, list(range(num_classes)))
            else:
                MEAN, VAR, COV, val_batch = compute_data_moments(itrainloader)
            data_moments.append((MEAN, VAR, COV))
            val_batches.append(val_batch)

            # Train on current task
            model.train()
            for epoch in range(self.hparams.train.train_iter):
                last_epoch = epoch == self.hparams.train.train_iter - 1
                if last_epoch:
                    mus_avg = []
                total_loss, count = 0, 0
                for inputs, labels in itrainloader:
                    if use_cuda:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    opt.zero_grad()
                    logits, mus = model.forward(inputs, return_mus=True)
                    loss = criterion(logits, labels)
                    total_loss += loss.item()
                    count += 1

                    if tid > 0:
                        fsd = 0
                        for prev_tid in range(tid):
                            fsd += self.compute_fsd(
                                task_params[prev_tid],
                                parameters_to_vector(model.parameters(),
                                                     clone=False),
                                data_moments[prev_tid], all_mus[prev_tid],
                                model, fsd_type, val_batches[prev_tid],
                                optpar_dict[prev_tid], fisher_dict[prev_tid])
                        fsd /= tid
                        loss += self.hparams.train.fsd_scale * fsd
                    loss.backward()
                    opt.step()
                    if last_epoch:
                        mus_avg.append(mus)

            if self.hparams.cl.classwise:
                all_mus.append(
                    model.get_classwise_mus(itrainloader,
                                            list(range(num_classes))))
            else:
                all_mus.append(torch.mean(torch.stack(mus_avg), dim=0))
            task_params.append(
                parameters_to_vector(model.parameters(), clone=True))
            if fsd_type == "ewc":
                for name, param in model.named_parameters():
                    optpar_dict[tid][name] = param.data.clone()
                    fisher_dict[tid][name] = param.grad.data.clone().pow(2)

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
                    logits = model.forward(inputs)
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

        # Backward Transfer Metric
        bwt = 0
        for i in range(len(tasks)):
            bwt += 100 * (tasks[i][-1] - tasks[i][0])
        bwt = bwt / (len(tasks) - 1)
        self.results_back[fsd_type] = bwt
        print("Backward Transfer: ", bwt)

        tasks, mean = self.get_fromp_data()
        self.results_dict['fromp'] = [tasks, mean]
        if self.hparams.use_wandb:
            wandb_logging_dict = dict()
            for key, value in self.results_dict.items():
                wandb_logging_dict[f'{key}_task0'] = value[0][0][num_tasks-1]
                wandb_logging_dict[f'{key}_all'] = value[1][num_tasks-1]
                wandb.log(wandb_logging_dict)

            for key, value in self.results_back.items():
                wandb_logging_dict[f'{key}_back'] = value
                wandb.log(wandb_logging_dict)


        return test_acc_list
