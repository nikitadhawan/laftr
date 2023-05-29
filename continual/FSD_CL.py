import imp
import os, numpy as np
import matplotlib

matplotlib.use("Agg")

from continual.utils.experiment_utils import note_taking, initialize_run, logging, compute_duration, set_random_seed
from datetime import datetime
import sys
import argparse
from continual.registry import get_hparams
import subprocess

from continual.algorithms.fsd_splitmnist import FSDSplitMnist
from continual.algorithms.fsd_permutedmnist import FSDPermMnist
from continual.algorithms.fsd_splitcifar import FSDSplitCifar

sys.stdout.flush()
parser = argparse.ArgumentParser()
parser.add_argument("--hparam_set", default=None, type=str)
parser.add_argument("--hparam_from_pickle", default=None, type=str)
parser.add_argument("--overwrite", default=False, type=bool)
parser.add_argument("--jobid", default=None, type=str)
args = parser.parse_args()
args_dict = vars(args)
if args.hparam_set:
    hparams = get_hparams(args.hparam_set)

hparams.overwrite = args.overwrite
set_random_seed(hparams.random_seed)


def main(hparams):

    if hparams.cl.type == "permute_mnist":
        experiment = FSDPermMnist(hparams)
        for fsd_type in hparams.cl.fsd_list:
            print("############ ", fsd_type, " ############")
            experiment.run_permmnist(fsd_type)
            print("####################################")

    elif hparams.cl.type == "split_mnist":
        experiment = FSDSplitMnist(hparams)
        print("############ ", fsd_type, " ############")
        experiment.run_splitmnist(fsd_type)
        print("####################################")

    elif hparams.cl.type == "split_cifar":
        experiment = FSDSplitCifar(hparams)
        print("############ ", fsd_type, " ############")
        experiment.run_splitcifar(fsd_type)
        print("####################################")

    logging(args_dict,
            hparams.to_dict(),
            hparams.messenger.readable_dir,
            hparams.messenger.dir_path,
            stage="final")


if __name__ == '__main__':

    initialize_run(hparams, args)
    start_time = datetime.now()
    hparams.messenger.start_time = start_time
    try:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        git_label = subprocess.check_output(
            ["cd " + dir_path + " && git describe --always && cd .."],
            shell=True).strip()
        if hparams.verbose:
            note_taking("The git label is {}".format(git_label))
    except:
        note_taking(
            "WARNING! Encountered unknwon error recording git label...")
    main(hparams)
    end_time = datetime.now()
    hour_summery = compute_duration(start_time, end_time)
    note_taking(
        f"{hparams.cl.type} finished, results written at: {hparams.messenger.readable_dir}.  Took {hour_summery} hours"
    )
    if hparams.use_wandb:
        note_taking(f"wandb project name: {hparams.wandb_project_name}")
    sys.exit(0)
