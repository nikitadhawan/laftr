from tkinter import FALSE
from continual.registry import register
from continual.hparams.hparam import Hparam as hp
from continual.hparams.base_hparams.defaults import test_fsd0413, test_fsd0414, test_fsd0415


def default_cl():
    """ model_train """
    return hp(
        num_tasks=10,
        num_points=200,
        classwise=False,
        k=10,
        use_var=False,
        use_coreset=False)


@register
def cl_smnist():
    hparams = test_fsd0413()
    hparams.cl = default_cl()
    hparams.train.train_iter = 15
    hparams.train.num_samples = 40
    hparams.train.learning_rate = 1e-3
    hparams.model.hidden_size = 512
    hparams.dataset.batch_size = 32
    hparams.random_seed = 42
    hparams.train.fsd_scale = 1
    hparams.cl.fsd_list = ["linact"]
    hparams.cl.type = "split_mnist"
    hparams.wandb_project_name = "laftr"
    return hparams


@register
def cl_pmnist_cw():
    hparams = test_fsd0414()
    hparams.cl = default_cl()
    hparams.train.train_iter = 15
    hparams.train.num_samples = 40
    hparams.train.learning_rate = 1e-4
    hparams.train.fsd_scale = 1
    hparams.model.hidden_size = 512
    hparams.dataset.batch_size = 128
    hparams.random_seed = 42
    hparams.cl.type = "permute_mnist"
    hparams.wandb_project_name = "laftr"
    hparams.cl.fsd_list = ['lin_act']
    return hparams


@register
def cl_scifar_kl_base():
    hparams = test_fsd0415()
    hparams.cl = default_cl()
    hparams.cl.do_dropout = True
    hparams.cl.num_tasks = 6
    hparams.cl.task0epoch = 100
    hparams.train.train_iter = 50
    hparams.train.learning_rate = 1e-3
    hparams.train.fsd_scale = 1e-1
    hparams.train.ber_momentum = -1
    hparams.train.num_samples = 200 
    hparams.dataset.batch_size = 512
    hparams.random_seed = 42
    hparams.cl.type = "split_cifar"
    hparams.cl.task0epoch = 1
    hparams.wandb_project_name = "laftr"
    hparams.cl.fsd_list = ["lin_act"]
    hparams.train.kl_div = True
    hparams.cl.use_coreset = False
    hparams.cl.classwise = False
    return hparams

@register
def kl_gauss_ber_cw():
    hparams = cl_scifar_kl_base()
    hparams.cl.use_coreset = False
    hparams.cl.classwise = True
    hparams.cl.fsd_list = ["lin_act_ber"]
    hparams.cl.task0epoch = 200
    hparams.train.train_iter = 50
    hparams.train.fsd_scale = 10
    return hparams

@register
def kl_core_ber_cw():
    hparams = cl_scifar_kl_base()
    hparams.cl.use_coreset = True
    hparams.cl.classwise = True
    hparams.cl.fsd_list = ["lin_act_ber"]
    hparams.cl.task0epoch = 100
    hparams.train.train_iter = 50
    hparams.train.fsd_scale = 5
    return hparams

@register
def kl_gauss_relu_cw():
    hparams = cl_scifar_kl_base()
    hparams.cl.use_coreset = False
    hparams.cl.classwise = True
    hparams.cl.fsd_list = ["lin_act"]
    hparams.cl.task0epoch = 100
    hparams.train.train_iter = 50
    hparams.train.fsd_scale = 5
    return hparams

@register
def kl_core_ber_ncw():
    hparams = cl_scifar_kl_base()
    hparams.cl.use_coreset = True
    hparams.cl.classwise = False
    hparams.cl.fsd_list = ["lin_act_ber"]
    hparams.cl.task0epoch = 100
    hparams.train.train_iter = 50
    hparams.train.fsd_scale = 1
    return hparams

@register
def kl_gauss_ber_ncw():
    hparams = cl_scifar_kl_base()
    hparams.cl.use_coreset = False
    hparams.cl.classwise = False
    hparams.cl.fsd_list = ["lin_act_ber"]
    hparams.cl.task0epoch = 100
    hparams.train.train_iter = 80
    hparams.train.fsd_scale = 5
    return hparams

@register
def kl_core_relu_ncw():
    hparams = cl_scifar_kl_base()
    hparams.cl.use_coreset = True
    hparams.cl.classwise = False
    hparams.cl.fsd_list = ["lin_act"]
    hparams.cl.num_tasks = 11
    hparams.cl.task0epoch = 200
    hparams.train.train_iter = 50
    hparams.train.fsd_scale = 6
    return hparams


@register
def kl_gauss_relu_ncw():
    hparams = cl_scifar_kl_base()
    hparams.cl.use_coreset = False
    hparams.cl.classwise = False
    hparams.cl.fsd_list = ["lin_act"]
    hparams.cl.task0epoch = 200
    hparams.train.train_iter = 50
    hparams.train.fsd_scale = 5
    return hparams


@register
def euc_gauss_ber_cw():
    hparams = cl_scifar_kl_base()
    hparams.train.kl_div = False
    hparams.cl.use_coreset = False
    hparams.cl.classwise = True
    hparams.cl.fsd_list = ["lin_act_ber"]
    hparams.cl.task0epoch = 100
    hparams.train.train_iter = 80
    hparams.train.fsd_scale = 5
    return hparams


@register
def euc_gauss_ber_ncw():
    hparams = cl_scifar_kl_base()
    hparams.train.kl_div = False
    hparams.cl.use_coreset = False
    hparams.cl.classwise = False
    hparams.cl.fsd_list = ["lin_act_ber"]
    hparams.cl.task0epoch = 100
    hparams.train.train_iter = 100
    hparams.train.fsd_scale = 6
    return hparams


@register
def euc_gauss_relu_ncw():
    hparams = cl_scifar_kl_base()
    hparams.train.kl_div = False
    hparams.cl.use_coreset = False
    hparams.cl.classwise = False
    hparams.cl.fsd_list = ["lin_act"]
    hparams.cl.task0epoch = 100
    hparams.train.train_iter = 50
    hparams.train.fsd_scale = 10
    return hparams

@register
def euc_core_relu_ncw():
    hparams = cl_scifar_kl_base()
    hparams.train.kl_div = False
    hparams.cl.use_coreset = True
    hparams.cl.classwise = False
    hparams.cl.fsd_list = ["lin_act"]
    hparams.cl.task0epoch = 200
    hparams.train.train_iter = 50
    hparams.train.fsd_scale = 20
    return hparams
