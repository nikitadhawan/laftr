import os
from os.path import dirname
from os.path import join as pjoin
from pathlib import Path

from copy import deepcopy
from ..hparams.hparam import Hparam as container
import torch
# import torch.nn as nn
import glob
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from datetime import datetime
import random
import logging as python_log
import copy
import shutil
import re
# from torchvision.utils import save_image
from ..registry import get_hparams, get_model
from copy import deepcopy
import wandb

# import torch.multiprocessing


class EvaluateModel:
    """
    A wrapper to disable grad and set model to eval mode:
        sample usage:
        with EvaluateModel(model):
            code that doesn't require grad and wants model to be in eval mode
    """

    def __init__(self, model):
        self.model = model
        self.is_training = True
        self.prev = False

    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        self.is_training = self.model.training
        self.model.eval()
        torch.set_grad_enabled(False)

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.train(self.is_training)
        torch.set_grad_enabled(self.prev)


def note_taking(message, print_=True):
    if print_:
        print(message)
    python_log.info(message)


def compute_duration(start, end):
    duration = (end - start).total_seconds()
    days, _dreminder = divmod(duration, 86400)
    hours, _hreminder = divmod(_dreminder, 3600)
    minutes, seconds = divmod(_hreminder, 60)
    note_taking(
        "It's been {} days, {} hours, {} minutes and {} seconds.".format(
            days, hours, minutes, seconds))
    hour_summery = 24 * days + hours + minutes / 60.
    return hour_summery


def set_random_seed(i):
    random.seed(i)
    np.random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)
    torch.cuda.manual_seed_all(i)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_data_stats(loader):
    data_list = list()
    for i, (data, _) in enumerate(loader):
        data_list.append(data)
    whole_data = torch.cat(data_list, dim=0)
    note_taking("size of whole data: {}".format(whole_data.size()))

    mean = torch.mean(whole_data, dim=0)
    note_taking("size of mean: {}".format(mean.size()))
    std = torch.std(whole_data, dim=0)
    data_max = torch.max(whole_data)
    data_min = torch.min(whole_data)
    return mean, std, data_max, data_min


def compute_data_stats_nolog(loader):
    data_list = list()
    for i, (data, _) in enumerate(loader):
        data_list.append(data)
    whole_data = torch.cat(data_list, dim=0)
    print("size of whole data: {}".format(whole_data.size()))

    mean = torch.mean(whole_data, dim=0)
    print("size of mean: {}".format(mean.size()))
    std = torch.std(whole_data, dim=0)
    data_max = torch.max(whole_data)
    data_min = torch.min(whole_data)
    return mean, std, data_max, data_min


# def compute_data_stats(loader, data):
#     data_list = list()
#     for i, (data, _) in enumerate(loader):
#         data_list.append(data)
#     whole_data = torch.cat(data_list, dim=0)
#     note_taking("size of whole data: {}".format(whole_data.size()))

#     mean = torch.mean(whole_data, dim=0)
#     note_taking("size of mean: {}".format(mean.size()))
#     std = torch.std(whole_data, dim=0)
#     data_max = torch.max(whole_data)
#     data_min = torch.min(whole_data)
#     return mean, std, data_max, data_min


def print_hparams(hparams_dict, name):
    """Prints the values of all hyper parameters.
    """
    if name is None:
        print("Sanity check: hyper parameters:")
    print('=' * 80)
    print('Hparams'.center(80) if name is None else str(name).center(80))
    print('-' * 80)
    for key, value in hparams_dict.items():
        if isinstance(value, dict):
            print_hparams(value, key)
        else:
            if "msg" in key:
                print('=' * 80)
                print(key.center(80))

                print('-' * 80)
                print(value)

            else:
                print('{}: {}'.format(key, value).center(80))
    print('*' * 80)


def log_hparams(file, hparams_dict, name):
    """Log down the values of all hyper parameters.
    """
    file.write('=' * 80)
    file.write("\n")
    file.write('Hparams'.center(80) if name is None else str(name).center(80))
    file.write("\n")
    file.write('-' * 80)
    file.write("\n")
    for key, value in hparams_dict.items():
        file.write("\n")
        if isinstance(value, dict):
            log_hparams(file, value, key)
        else:
            file.write('{}: {}'.format(key, value).center(80))
    file.write("\n")
    file.write('=' * 80)
    file.write("\n")


def logging(args_dict, hparams_dict, log_dir, dir_path, stage="init"):
    if stage == "init":
        log_path = pjoin(log_dir, "init_log.txt")
    else:
        log_path = pjoin(log_dir, str(stage) + "_log.txt")

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    with open(log_path, 'a') as file:
        file.write("The current time when logging  is:" + current_time + "\n")
        file.write("The Current directory is:" + str(dir_path))
        file.write("\n Args for this experiment are:\n")
        for key, value in args_dict.items():
            file.write('%s:%s\n' % (key, value))
        file.write("\n")
        file.write("\nlogging hparam and results recursively...:\n")
        log_hparams(file, hparams_dict, args_dict["hparam_set"])
        file.close()


def init_dir(dir, overwrite=False):
    if os.path.exists(dir) and overwrite:
        shutil.rmtree(dir)
    if not os.path.exists(dir):
        print("Initializing directory: {}".format(dir))
        os.makedirs(dir, 0o777, exist_ok=True)


def add_groups(group_list, dir_to_append):
    for subgroup in group_list:
        dir_to_append = pjoin(dir_to_append, subgroup)
    return dir_to_append


def initialize_run(hparams, args):
    """
    Args:
    hparams: The hyper parameter object
    args: dictionary of keyword arguments.
    """

    if args.hparam_set is None:
        args.hparam_set = hparams.hparam_set

    loadable_dir = pjoin(hparams.output_root_dir, "loadable",
                         hparams.wandb_project_name, args.hparam_set)
    readable_dir = pjoin(hparams.output_root_dir, "readable",
                         hparams.wandb_project_name, args.hparam_set)
    init_dir(loadable_dir, overwrite=hparams.overwrite)
    init_dir(readable_dir, overwrite=hparams.overwrite)
    hparams.save(pjoin(loadable_dir, 'init_hparams.json'))
    hparams.save(pjoin(readable_dir, 'init_hparams.json'))
    log_path = pjoin(readable_dir, "RT_log.txt")
    python_log.basicConfig(filename=log_path,
                           filemode='a',
                           level=python_log.INFO,
                           format='%(message)s')

    # checkpoint_dir = hparams.checkpoint_root_dir + (
    #     args.hparam_set if (hparams.load_checkpoint_name is None) else
    #     hparams.load_checkpoint_name) + "/"

    checkpoint_dir = loadable_dir

    # hparams.cuda = hparams.cuda and torch.cuda.is_available()
    # hparams.device = torch.device("cuda" if hparams.cuda else "cpu")

    # torch.multiprocessing.set_sharing_strategy('file_system')

    # if hparams.double_precision:
    #     hparams.add_hparam("dtype", torch.DoubleTensor)
    #     hparams.add_hparam("tensor_type", torch.float64)
    # else:
    #     hparams.add_hparam("dtype", torch.FloatTensor)
    #     hparams.add_hparam("tensor_type", torch.float32)
    # if hparams.cuda:
    #     note_taking("torch.cuda.current_device() {}".format(
    #         torch.cuda.current_device()))
    if hparams.verbose:
        note_taking("hparams.dtype {}".format(hparams.dtype))
        note_taking("hparams.tensor_type {}".format(hparams.tensor_type))
    if hparams.dataset.input_dims:
        input_vector_length = 1
        for dim in hparams.dataset.input_dims:
            input_vector_length *= dim
        hparams.dataset.input_vector_length = input_vector_length
    # torch.set_default_tensor_type(hparams.dtype)

    # init hparams
    dir_path = os.path.dirname(os.path.realpath(__file__))

    # messenger is a Hparam object, it can be regarded as the messenger acting as
    # the global info buffer for miscellaneous bits of information
    hparams.messenger = container()
    # hparams.messenger.results_dir = results_dir
    # hparams.messenger.image_dir = image_dir
    # hparams.messenger.arxiv_dir = arxiv_dir
    # hparams.messenger.backup_dir = backup_dir
    hparams.messenger.checkpoint_dir = checkpoint_dir
    hparams.messenger.loadable_dir = loadable_dir
    hparams.messenger.readable_dir = readable_dir
    hparams.messenger.result_dict = dict()
    hparams.messenger.dir_path = dir_path
    hparams.messenger.log_path = log_path
    hparams.messenger.save_data = True

    if hasattr(args, "jobid"):
        hparams.jobid = args.jobid

    # initial logging
    hparams.hparam_set = args.hparam_set
    logging(vars(args), hparams.to_dict(), readable_dir, dir_path)
    if os.path.exists(checkpoint_dir) and not hparams.overwrite:
        if hparams.specific_model_path:
            load_path = hparams.specific_model_path
        else:
            load_path = get_checkpoint_path(checkpoint_dir)
            # load_path = get_checkpoint_path(checkpoint_dir, hparams.chkt_epoch)
        hparams.checkpoint_path = load_path
    else:
        init_dir(checkpoint_dir, overwrite=hparams.overwrite)

    if hparams.verbose:
        print_hparams(hparams.to_dict(), None)

    hparams.absolute_readable_dir = os.path.abspath(
        hparams.messenger.readable_dir)
    hparams.absolute_loadable_dir = os.path.abspath(
        hparams.messenger.loadable_dir)
    if hparams.use_wandb:
        hparam_dict = deepcopy(hparams.to_dict())
        del hparam_dict["messenger"]

        if hparams.wandb:
            wandb_project_name = hparams.wandb.project
            wandb_group = hparams.wandb.group
            wandb_job_type = hparams.wandb.job_type
            wandb_name = hparams.wandb.name
            wandb_tags = hparams.wandb.tags
            wandb_id_file_path = (Path(hparams.messenger.loadable_dir) /
                                  "_wandb_id.txt")
            if wandb_id_file_path.exists():
                resume_id = wandb_id_file_path.read_text()
                wandb.init(project=wandb_project_name,
                           tags=wandb_tags,
                           name=wandb_name,
                           job_type=wandb_job_type,
                           group=wandb_group,
                           resume="allow",
                           id=resume_id,
                           config=hparam_dict)
                note_taking(f"wandb run resumed from pk: {wandb_name}")
            else:
                run = wandb.init(project=wandb_project_name,
                                 tags=wandb_tags,
                                 name=wandb_name,
                                 job_type=wandb_job_type,
                                 group=wandb_group,
                                 config=hparam_dict)
                wandb_id_file_path.write_text(str(run.id))
                note_taking(
                    "loaded pk and initialized new wandb run with project {} and run name {}"
                    .format(wandb_project_name, wandb_name))
        else:
            wandb_project_name = hparams.wandb_project_name
            if wandb_project_name is None:
                wandb_project_name = "unnamed"
            # wandb_project_name = f"dml/{wandb_project_name}"
            wandb_id_file_path = (Path(hparams.messenger.loadable_dir) /
                                  "_wandb_id.txt")
            if wandb_id_file_path.exists():
                resume_id = wandb_id_file_path.read_text()
                wandb.init(project=wandb_project_name,
                           entity="dml",
                           settings=wandb.Settings(start_method='fork'),
                           name=args.hparam_set,
                           resume="allow",
                           id=resume_id,
                           config=hparam_dict)
                # wandb.init(project=wandb_project_name,
                #            entity="dml",
                #            name=args.hparam_set,
                #            resume="allow",
                #            id=resume_id,
                #            config=hparam_dict)
                note_taking("wandb run resumed")
            else:
                run = wandb.init(project=wandb_project_name,
                                 entity="dml",
                                 settings=wandb.Settings(start_method='fork'),
                                 name=args.hparam_set,
                                 config=hparam_dict)
                wandb_id_file_path.write_text(str(run.id))
                note_taking(
                    "initialized new wandb run with project {} and run name {}"
                    .format(wandb_project_name, args.hparam_set))


# def initialize_run_torch(hparams, args):
#     """
#     Args:
#     hparams: The hyper parameter object
#     args: dictionary of keyword arguments.
#     """

#     if args.hparam_set is None:
#         args.hparam_set = hparams.hparam_set

#     loadable_dir = pjoin(hparams.output_root_dir, "loadable",
#                          hparams.wandb_project_name, args.hparam_set)
#     readable_dir = pjoin(hparams.output_root_dir, "readable",
#                          hparams.wandb_project_name, args.hparam_set)
#     init_dir(loadable_dir, overwrite=hparams.overwrite)
#     init_dir(readable_dir, overwrite=hparams.overwrite)
#     hparams.save(pjoin(loadable_dir, 'init_hparams.json'))
#     hparams.save(pjoin(readable_dir, 'init_hparams.json'))
#     log_path = pjoin(readable_dir, "RT_log.txt")
#     python_log.basicConfig(filename=log_path,
#                            filemode='a',
#                            level=python_log.INFO,
#                            format='%(message)s')

#     checkpoint_dir = hparams.checkpoint_root_dir + (
#         args.hparam_set if (hparams.load_checkpoint_name is None) else
#         hparams.load_checkpoint_name) + "/"

#     hparams.cuda = hparams.cuda and torch.cuda.is_available()
#     hparams.device = torch.device("cuda" if hparams.cuda else "cpu")

#     torch.multiprocessing.set_sharing_strategy('file_system')

#     if hparams.double_precision:
#         hparams.add_hparam("dtype", torch.DoubleTensor)
#         hparams.add_hparam("tensor_type", torch.float64)
#     else:
#         hparams.add_hparam("dtype", torch.FloatTensor)
#         hparams.add_hparam("tensor_type", torch.float32)
#     if hparams.cuda:
#         note_taking("torch.cuda.current_device() {}".format(
#             torch.cuda.current_device()))
#     if hparams.verbose:
#         note_taking("hparams.dtype {}".format(hparams.dtype))
#         note_taking("hparams.tensor_type {}".format(hparams.tensor_type))
#     if hparams.dataset.input_dims:
#         input_vector_length = 1
#         for dim in hparams.dataset.input_dims:
#             input_vector_length *= dim
#         hparams.dataset.input_vector_length = input_vector_length
#     torch.set_default_tensor_type(hparams.dtype)

#     # init hparams
#     dir_path = os.path.dirname(os.path.realpath(__file__))

#     # messenger is a Hparam object, it can be regarded as the messenger acting as
#     # the global info buffer for miscellaneous bits of information
#     hparams.messenger = container()
#     # hparams.messenger.results_dir = results_dir
#     # hparams.messenger.image_dir = image_dir
#     # hparams.messenger.arxiv_dir = arxiv_dir
#     # hparams.messenger.backup_dir = backup_dir
#     hparams.messenger.checkpoint_dir = checkpoint_dir
#     hparams.messenger.loadable_dir = loadable_dir
#     hparams.messenger.readable_dir = readable_dir
#     hparams.messenger.result_dict = dict()
#     hparams.messenger.dir_path = dir_path
#     hparams.messenger.log_path = log_path
#     hparams.messenger.save_data = True

#     if hasattr(args, "jobid"):
#         hparams.jobid = args.jobid

#     # initial logging
#     hparams.hparam_set = args.hparam_set
#     logging(vars(args), hparams.to_dict(), readable_dir, dir_path)
#     if os.path.exists(checkpoint_dir) and not hparams.overwrite:
#         if hparams.specific_model_path:
#             load_path = hparams.specific_model_path
#         else:
#             load_path = get_checkpoint_path(checkpoint_dir, hparams.chkt_epoch)
#         hparams.checkpoint_path = load_path
#     else:
#         init_dir(checkpoint_dir, overwrite=hparams.overwrite)

#     if hparams.verbose:
#         print_hparams(hparams.to_dict(), None)

#     hparams.absolute_readable_dir = os.path.abspath(
#         hparams.messenger.readable_dir)
#     hparams.absolute_loadable_dir = os.path.abspath(
#         hparams.messenger.loadable_dir)
#     if hparams.use_wandb:
#         hparam_dict = deepcopy(hparams.to_dict())
#         del hparam_dict["messenger"]

#         if hparams.wandb:
#             wandb_project_name = hparams.wandb.project
#             wandb_group = hparams.wandb.group
#             wandb_job_type = hparams.wandb.job_type
#             wandb_name = hparams.wandb.name
#             wandb_tags = hparams.wandb.tags
#             wandb_id_file_path = (Path(hparams.messenger.loadable_dir) /
#                                   "_wandb_id.txt")
#             if wandb_id_file_path.exists():
#                 resume_id = wandb_id_file_path.read_text()
#                 wandb.init(project=wandb_project_name,
#                            tags=wandb_tags,
#                            name=wandb_name,
#                            job_type=wandb_job_type,
#                            group=wandb_group,
#                            resume="allow",
#                            id=resume_id,
#                            config=hparam_dict)
#                 note_taking(f"wandb run resumed from pk: {wandb_name}")
#             else:
#                 run = wandb.init(project=wandb_project_name,
#                                  tags=wandb_tags,
#                                  name=wandb_name,
#                                  job_type=wandb_job_type,
#                                  group=wandb_group,
#                                  config=hparam_dict)
#                 wandb_id_file_path.write_text(str(run.id))
#                 note_taking(
#                     "loaded pk and initialized new wandb run with project {} and run name {}"
#                     .format(wandb_project_name, wandb_name))
#         else:
#             wandb_project_name = hparams.wandb_project_name
#             if wandb_project_name is None:
#                 wandb_project_name = "unnamed"
#             # wandb_project_name = f"dml/{wandb_project_name}"
#             wandb_id_file_path = (Path(hparams.messenger.loadable_dir) /
#                                   "_wandb_id.txt")
#             if wandb_id_file_path.exists():
#                 resume_id = wandb_id_file_path.read_text()
#                 wandb.init(project=wandb_project_name,
#                            entity="dml",
#                            settings=wandb.Settings(start_method='fork'),
#                            name=args.hparam_set,
#                            resume="allow",
#                            id=resume_id,
#                            config=hparam_dict)
#                 # wandb.init(project=wandb_project_name,
#                 #            entity="dml",
#                 #            name=args.hparam_set,
#                 #            resume="allow",
#                 #            id=resume_id,
#                 #            config=hparam_dict)
#                 note_taking("wandb run resumed")
#             else:
#                 run = wandb.init(project=wandb_project_name,
#                                  entity="dml",
#                                  settings=wandb.Settings(start_method='fork'),
#                                  name=args.hparam_set,
#                                  config=hparam_dict)
#                 wandb_id_file_path.write_text(str(run.id))
#                 note_taking(
#                     "initialized new wandb run with project {} and run name {}"
#                     .format(wandb_project_name, args.hparam_set))

# def get_checkpoint_path(to_search, chkt_epoch):
#     """
#     generate appropriate checkpointing path based on checkpoint step.
#         if chkt_epoch is -2: load best
#         if chkt_epoch is -1: load last
#         if chkt_epoch >= 0, load the epoch specified
#     returns None if checkpoint_path is not found
#     """
#     max_epoch, best_ckpt, latest_ckpt, epoch_ckpt = 0, None, None, None
#     for f in glob.glob(os.path.join(to_search, "*.pth")):
#         if os.path.getsize(f) > 0:
#             regex_match = re.match(".*epoch([0-9]+).pth", f)
#             if regex_match:
#                 current_epoch = int(regex_match.group(1))
#                 if current_epoch == chkt_epoch:
#                     epoch_ckpt = f
#                     break
#                 if current_epoch > max_epoch:
#                     max_epoch = current_epoch
#                     latest_ckpt = f
#             elif "best" in f:
#                 best_ckpt = f
#         else:
#             note_taking(f"WARNING:{f} is not valid as it has size 0")
#     checkpoint_path, print_str = None, ""
#     if chkt_epoch == -2 and best_ckpt:
#         checkpoint_path = best_ckpt
#         print_str = "best"
#     elif chkt_epoch == -1 and latest_ckpt:
#         checkpoint_path = latest_ckpt
#         print_str = "latest"
#     elif chkt_epoch >= 0 and epoch_ckpt:
#         checkpoint_path = epoch_ckpt
#         print_str = f"{chkt_epoch}th epoch"
#     if checkpoint_path:
#         note_taking("the {} Checkpoint detected: {}".format(
#             print_str, checkpoint_path).center(80))
#     return checkpoint_path


def get_checkpoint_path(to_search):
    """
    generate appropriate checkpointing path.
    returns None if checkpoint_path is not found
    """
    # max_epoch, best_ckpt, latest_ckpt, epoch_ckpt = 0, None, None, None
    checkpoint_path = None
    for f in glob.glob(os.path.join(to_search, "*.pth")):
        if os.path.getsize(f) > 0:
            checkpoint_path = f
            note_taking(f"Checkpoint detected: {checkpoint_path}")
        else:
            note_taking(f"WARNING:{f} is not valid as it has size 0")
    # checkpoint_path, print_str = None, ""
    # if chkt_epoch == -2 and best_ckpt:
    #     checkpoint_path = best_ckpt
    #     print_str = "best"
    # elif chkt_epoch == -1 and latest_ckpt:
    #     checkpoint_path = latest_ckpt
    #     print_str = "latest"
    # elif chkt_epoch >= 0 and epoch_ckpt:
    #     checkpoint_path = epoch_ckpt
    #     print_str = f"{chkt_epoch}th epoch"
    # if checkpoint_path:
    # note_taking("the {} Checkpoint detected: {}".format(
    #     print_str, checkpoint_path).center(80))
    note_taking(f"final checkpoint_path: {checkpoint_path}")
    return checkpoint_path


# def get_path_from_hparam(hparam_name, chkt_epoch):
#     """Get checkpoint path from hparam
#     """
#     Hparam = get_hparams(hparam_name)
#     base_path = Hparam.checkpoint_root_dir
#     for item in Hparam.group_list:
#         base_path = os.path.join(base_path, item)
#     full_path = os.path.join(base_path, hparam_name)
#     path = get_checkpoint_path(full_path, chkt_epoch)
#     return path

# def load_checkpoint(path,
#                     optimizer,
#                     reset_optimizer,
#                     model,
#                     load_rng=True,
#                     device='cuda'):
#     '''
#     Load model from checkpoint with optimizer loading support
#     '''
#     note_taking("Loading checkpoint from: {}".format(path))
#     if device.type == 'cpu':
#         checkpoint = torch.load(path, map_location=torch.device('cpu'))
#     else:
#         checkpoint = torch.load(path)
#     model.load_state_dict(checkpoint["state_dict"])
#     epoch = checkpoint["global_epoch"]
#     validation_loss_list = checkpoint["validation_loss_list"]
#     if "rng" in checkpoint and load_rng:
#         torch.random.set_rng_state(checkpoint["rng"])
#     if optimizer is not None:
#         if not reset_optimizer:
#             optimizer_state = checkpoint["optimizer"]
#             if optimizer_state is not None:
#                 note_taking("Loading optimizer state from {}".format(path))
#                 optimizer.load_state_dict(checkpoint["optimizer"])
#     note_taking(
#         "Loaded checkpoint from {} at epoch {}, and the validation loss was {}"
#         .format(path, epoch, validation_loss_list[-1]))
#     return epoch, validation_loss_list

# def save_checkpoint(checkpoint_path, optimizer, save_optimizer_state, model,
#                     epoch, validation_loss_list):
#     '''
#     Save the pytorch model with optimizer saving support
#     '''
#     optimizer_state = optimizer.state_dict() if save_optimizer_state else None
#     temp_file = os.path.join(dirname(checkpoint_path),
#                              "_you_got_preempted.pth")
#     torch.save(
#         {
#             "state_dict": model.state_dict(),
#             "optimizer": optimizer_state,
#             "global_epoch": epoch,
#             "validation_loss_list": validation_loss_list,
#             "rng": torch.random.get_rng_state(),
#         }, temp_file)
#     if not os.path.exists(checkpoint_path):
#         Path(checkpoint_path).touch(exist_ok=True)
#     os.rename(temp_file, checkpoint_path)
