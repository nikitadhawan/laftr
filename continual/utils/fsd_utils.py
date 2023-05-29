import numpy as np
import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map
from torch.distributions.multivariate_normal import MultivariateNormal
from functorch import vjp, jvp, make_functional


def _sub(x, y):
    return [param1 - param2 for param1, param2 in zip(x, y)]


def my_jvp(f, w, R_w, *args, **kwargs):
    out, f_vjp = vjp(lambda param: f(param, *args, **kwargs), w)
    _, f_vjp_vjp = vjp(f_vjp, torch.zeros_like(out))
    return f(w), f_vjp_vjp((R_w,))[0]


def linearize(f, params):

    def f_lin(p, *args, **kwargs):
        dparams = _sub(p, params)
        f_params_x, proj = my_jvp(lambda param: f(param, *args, **kwargs), params, dparams)
        # f_params_x, proj = jvp(lambda param: f(param, *args, **kwargs),
        #                        (params, ), (dparams, ))
        return f_params_x + proj

    return f_lin


def l2_norm(x1, x2, reduce='mean', normalize=True):
    if normalize:
        x1, x2 = F.softmax(x1, dim=1), F.softmax(x2, dim=1)
    if reduce == 'mean':
        return torch.mean(0.5 * torch.sum((x2 - x1)**2, dim=1))
    elif reduce == 'sum':
        return torch.sum(0.5 * torch.sum((x2 - x1)**2, dim=1))


def kl_div(x1, x2, reduce='mean'):
    if reduce == 'mean':
        return F.kl_div(F.log_softmax(x2, dim=1), F.softmax(x1, dim=1), reduction="batchmean").clamp(min=0.)
    elif reduce == 'sum':
        return F.kl_div(F.log_softmax(x2, dim=1), F.softmax(x1, dim=1), reduction="sum").clamp(min=0.)


def apply_dense(params, inputs, bias=True):
    W, b = params
    outputs = inputs @ W.t()
    if bias:
        outputs += b
    return outputs


def apply_dense_var(params, inputs):
    W, b = params
    outputs = W @ inputs @ W.t()
    return outputs


def get_mus(params, inputs):
    output = apply_dense(params, inputs)
    mus = torch.mean((output >= 0).float(), dim=0)
    return mus


def get_params_pert(params, sigma=0.1):
    perturbed_params = []
    for param in params:
        new_param = param + torch.normal(0, sigma, size=param.shape).to('cuda')
        perturbed_params.append(new_param)
    return perturbed_params


def lin_act(params0, params1, mus, inputs_mean, inputs_cov, batch_size, output_metric, val_batch=None, label_set=None, stoch_act=True):
  
    if val_batch is None:
        inputs = MultivariateNormal(inputs_mean, inputs_cov).sample(
            (batch_size, )).to('cuda')
    else: 
        inputs = val_batch.to('cuda')[:batch_size]
    num_layers = len(params0) // 2 - 1
    out0 = out1 = inputs
    for n in range(num_layers):
        out0 = apply_dense(params0[2 * n:2 * n + 2], out0)
        out1 = apply_dense(params1[2 * n:2 * n + 2], out1)
        delta_s = out1 - out0
        if stoch_act:
            ber_samples = torch.bernoulli(mus[n])
            delta_a = ber_samples * delta_s
            out0 = ber_samples * out0
        else:
            out0 = torch.maximum(out0, torch.zeros_like(out0))
            delta_a = torch.where(out0 > 0, delta_s, torch.zeros_like(out0))
        out1 = out0 + delta_a
    out0 = apply_dense(params0[-2:], out0)
    out1 = apply_dense(params1[-2:], out1)
    if label_set:
        out0 = out0[:, label_set]
        out1 = out1[:, label_set]
    return output_metric(out0, out1)


def determ_fsd(params0, params1, mus, mean, cov):

    first = moment1(params0, params1, mus, mean)
    second = moment2(params0, params1, mus, cov)
    return 0.5 * (first + second)


def moment1(params0, params1, mus, mean):
    ea = mean
    delta_a = ea - ea
    num_layers = len(params0) // 2 - 1
    for n in range(num_layers):
        es = apply_dense(params0[2 * n:2 * n + 2], ea)
        delta_params = (params1[2 * n] - params0[2 * n],
                        params1[2 * n + 1] - params0[2 * n + 1])
        delta_s = apply_dense(delta_params, ea) + apply_dense(
            params1[2 * n:2 * n + 2], delta_a, bias=False)
        ea = mus[n] * es
        delta_a = mus[n] * delta_s
    es = apply_dense(params0[-2:], ea)
    delta_params = (params1[-2] - params0[-2], params1[-1] - params0[-1])
    delta_s = apply_dense(delta_params, ea) + apply_dense(
        params1[-2:], delta_a, bias=False)
    return torch.sum(torch.pow(delta_s, 2))


def moment2(params0, params1, mus, cov):
    cov_a = cov
    delta_a = cov_a - cov_a
    num_layers = len(params0) // 2 - 1
    for n in range(num_layers):
        cov_mu = torch.outer(mus[n], mus[n])  
        cov_s = apply_dense_var(params0[2 * n:2 * n + 2], cov_a)
        delta_params = (params1[2 * n] - params0[2 * n],
                        params1[2 * n + 1] - params0[2 * n + 1])
        delta_s = apply_dense_var(delta_params, cov_a) + apply_dense_var(
            params1[2 * n:2 * n + 2], delta_a)
        cov_a = torch.mul(cov_mu, cov_s)
        delta_a = torch.mul(cov_mu, delta_s)
    cov_s = apply_dense_var(params0[-2:], cov_a)
    delta_params = (params1[-2] - params0[-2], params1[-1] - params0[-1])
    delta_s = apply_dense_var(delta_params, cov_a) + apply_dense_var(
        params1[-2:], delta_a)
    return torch.trace(delta_s)



def lin_act_conv(model, params0, params1, mus, inputs_mean, inputs_cov, batch_size, output_metric, label_set=None, val_batch=None, do_dropout=False, stoch_act=True):
    if val_batch is None:
        inputs = MultivariateNormal(inputs_mean, inputs_cov).sample((batch_size, )).to('cuda') 
        inputs = inputs.reshape((-1, 3, 32, 32))
    else:
        inputs = val_batch[:batch_size].to('cuda')
    out0 = out1 = inputs
    count, relu_count = 0, 0
    for block in [model.conv_modules, model.linear_modules, [model.out_block]]:
        for layer in block:
            if isinstance(layer, nn.Conv2d):
                out0 = layer._conv_forward(out0, params0[2*count], params0[2*count+1])
                out1 = layer(out1) 
                count += 1
            elif isinstance(layer, nn.ReLU):
                delta_s = out1 - out0
                if stoch_act:
                    ber_samples = torch.bernoulli(mus[relu_count])
                    delta_a = ber_samples * delta_s
                    out0 = ber_samples * out0
                else:
                    out0 = torch.maximum(out0, torch.zeros_like(out0))
                    delta_a = torch.where(out0 > 0, delta_s, torch.zeros_like(out0))
                out1 = out0 + delta_a
                relu_count += 1
            elif isinstance(layer, nn.Linear):
                out0 = torch.flatten(out0, 1)
                out1 = torch.flatten(out1, 1)
                out0 = apply_dense(params0[2 * count:2 * count + 2], out0)
                out1 = layer(out1) 
                count += 1
            elif isinstance(layer, nn.Dropout):
                if do_dropout:
                    mask = torch.empty(out0.size()[1:]).uniform_(0, 1) >= layer.p
                    mask = mask.to('cuda')
                    out0 = out0.mul(mask) * (1 / (1 - layer.p))
                    out1 = out1.mul(mask) * (1 / (1 - layer.p))
                else:
                    pass
            else:
                out0 = layer(out0)
                out1 = layer(out1)
    out0 = out0[:, label_set]
    out1 = out1[:, label_set]
    return output_metric(out0, out1) 


def tasks_fsd(acc_list_of_lists):
    num_tasks = len(acc_list_of_lists)
    all_tasks = []
    for task_id in range(num_tasks):
        all_tasks.append(
            [after[task_id] for after in acc_list_of_lists[task_id:]])
    means = [np.mean(after) for after in acc_list_of_lists]
    return all_tasks, means


def ewc(model, fisher_dict, optpar_dict):
    model.train()
    total = 0
    for name, param in model.named_parameters():
        fisher = fisher_dict[name]
        optpar = optpar_dict[name]
        total += (fisher * (optpar - param).pow(2)).sum()
    return total
