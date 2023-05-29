# # Copyright (c) 2018-present, Royal Bank of Canada.
# # All rights reserved.
# #
# # This source code is licensed under the license found in the
# # LICENSE file in the root directory of this source tree.
# # Author: Sicong Huang

# #!/usr/bin/env python3
from continual.registry import register
from continual.hparams.hparam import Hparam as hp


def default_train():
    """ model_train """
    return hp(
        learning_rate=1e0,
        train_iter=15,
    )


def default_dataset():
    """ model_train """
    return hp(
        batch_size=32,
        input_dim=784,
        output_dim=10,
    )

def default_model():
    """ model_train """
    return hp(hidden_size=512)


def default_experiment():
    Hparam = hp(
        output_root_dir="./runoutputs",
        checkpoint_root_dir="./runoutputs/checkpoints/",
        data_dir="./datasets",
        dataset=default_dataset(),
        train=default_train(),
        model=default_model(),
        cuda=True,
        verbose=True,
        random_seed=42,
        chkt_epoch=-1,
        use_wandb=True)
    return Hparam


@register
def test_fsd0413():
    hparams = default_experiment()
    hparams.wandb_project_name = "fsd_test_splitm"
    hparams.dataset.batch_size = 128
    hparams.model.hidden_size = 256
    hparams.train.learning_rate = 1e-3
    hparams.train.train_iter = 15
    hparams.train.fsd_scale = 1
    return hparams


@register
def test_fsd0414():
    hparams = default_experiment()
    hparams.wandb_project_name = "fsd_test_permm"
    hparams.dataset.batch_size = 128
    hparams.model.hidden_size = 100
    hparams.train.learning_rate = 1e-3
    hparams.train.train_iter = 10
    hparams.train.fsd_scale = 1
    return hparams


@register
def test_fsd0415():
    hparams = default_experiment()
    hparams.wandb_project_name = "fsd_test_splitc"
    hparams.dataset.batch_size = 256
    hparams.train.learning_rate = 1e-3
    hparams.train.train_iter = 80
    hparams.train.fsd_scale = 0.01
    hparams.train.ber_momentum = -1
    return hparams
