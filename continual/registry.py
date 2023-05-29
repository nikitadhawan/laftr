# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Author: Sicong Huang

#!/usr/bin/env python3
_REGISTERS = dict()
_COLLECTOR_REGISTERS = dict()


def register(fn):

    global _REGISTERS
    name = fn.__name__
    _REGISTERS[name] = fn
    return fn


def register_overwrite(fn):
    """Does the same thing as the register function but if the
        key already exists then overwrite it
    """

    global _REGISTERS
    name = fn.__name__
    _REGISTERS[name] = fn
    return fn


def register_with_name(fn, name):
    global _REGISTERS
    if name in _REGISTERS:
        raise ValueError(
            "Name conflict detected for {} in the registry".format(name))
    _REGISTERS[name] = fn
    return fn


def register_collector(fn):
    global _COLLECTOR_REGISTERS
    name = fn.__name__
    if name in _COLLECTOR_REGISTERS:
        raise ValueError(
            "Name conflict detected for {} in the registry".format(name))
    _COLLECTOR_REGISTERS[name] = fn
    return fn


def has_hparams(hparams):
    return hparams in _REGISTERS


def get_hparams(hparams):
    return _REGISTERS[hparams]()


def get_collector_hparams(hparams):
    return _COLLECTOR_REGISTERS[hparams]()


def get_dset(dataset_name, hparams, transforms, train):
    if (dataset_name in ["constant", "noise", "simulate"
                         ]) or (hparams.dataset.augmentation):
        return _REGISTERS["custom_dset"](hparams, transforms, train)
    return _REGISTERS[dataset_name](hparams, transforms, train)


def get_model(hparams):
    return _REGISTERS[hparams.model_name](hparams)


def get_G(hparams):
    return _REGISTERS[hparams.G_name](hparams)


def get_D(hparams):
    return _REGISTERS[hparams.D_name](hparams)


def get_vp_model(model_name, hparams):
    return _REGISTERS[model_name](hparams)


def get_cnn_model(hparams):
    return _REGISTERS[hparams.cnn_model_name](hparams)


def get_encoder(hparams):
    return _REGISTERS[hparams.encoder_name](hparams)


def get_decoder(hparams):
    return _REGISTERS[hparams.decoder_name](hparams)


def get_plots(hparams):
    return _REGISTERS[hparams]()


def get_augment(aug_type, hparams):
    return _REGISTERS[aug_type](hparams)


def get_register():
    return _REGISTERS


def get_pipeline(augmentation, hparams):
    if (hparams.dataset.source_dataset.type in [
            "constant", "noise", "simulate"
    ]) or (hparams.dataset.augmentation):
        return _REGISTERS[hparams.data_pipeline](augmentation,
                                                 hparams,
                                                 saved_dataset=True)
    return _REGISTERS[hparams.data_pipeline](augmentation, hparams, False)
