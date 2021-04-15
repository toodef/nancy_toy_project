import argparse
import os
import sys

import torch
import numpy as np
from piepline.builtin.monitors.tensorboard import TensorboardMonitor
from piepline.monitoring.monitors import FileLogMonitor, ConsoleLossMonitor
from piepline.train import Trainer
from piepline.utils.checkpoints_manager import CheckpointsManager
from piepline.utils.fsm import FileStructManager

from pietoolbelt.metrics.torch.segmentation import SegmentationMetricsProcessor
from pietoolbelt.steps.common.train import FoldedTrainer
from piepline.monitoring.hub import MonitorHub

from config import folds_indices_files, TRAIN_DIR
from train_config.config import epochs_num, folds_num
from train_config.train_config import TrainConfig, ResNet18TrainConfig


def init_trainer(config_type: type(TrainConfig), folds: dict, fsm: FileStructManager) -> Trainer:
    config = config_type(folds)

    train_metrics_proc = SegmentationMetricsProcessor(stage_name='train').subscribe_to_stage(config.train_stage)
    val_metrics_proc = SegmentationMetricsProcessor(stage_name='validation').subscribe_to_stage(config.val_stage)

    trainer = Trainer(config, fsm, device=torch.device('cuda'))

    file_log_monitor = FileLogMonitor(fsm).write_final_metrics()
    console_monitor = ConsoleLossMonitor()
    tensorboard_monitor = TensorboardMonitor(fsm, is_continue=False)
    mh = MonitorHub(trainer).subscribe2metrics_processor(train_metrics_proc) \
        .subscribe2metrics_processor(val_metrics_proc) \
        .add_monitor(file_log_monitor).add_monitor(console_monitor).add_monitor(tensorboard_monitor)

    config.train_stage._stage_end_event.add_callback(lambda stage: mh.update_losses({'train': config.train_stage.get_losses()}))
    config.val_stage._stage_end_event.add_callback(lambda stage: mh.update_losses({'validation': config.val_stage.get_losses()}))

    def get_m():
        return np.mean(val_metrics_proc.get_metrics()['groups'][0].metrics()[1].get_values())

    trainer.set_epoch_num(epochs_num)
    trainer.enable_lr_decaying(coeff=0.5, patience=10, target_val_clbk=get_m)

    CheckpointsManager(fsm=fsm).subscribe2trainer(trainer)

    return trainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('-m', '--model', type=str, help='Train one model', required=True, choices=['resnet18', 'resnet34'])
    parser.add_argument('-f', '--fold_num', type=int, help='Fold num', required=True)

    if len(sys.argv) < 2:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    folds_dict = {'fold_{}.npy'.format(i): 'fold_{}'.format(i) for i in range(folds_num)}

    if args.model == "resnet18":
        cur_dir = os.path.join(TRAIN_DIR, args.model, 'fold_{}'.format(args.fold_num))
        if not os.path.exists(cur_dir):
            os.makedirs(cur_dir)

        with open(os.path.join(cur_dir, "{}_fold.txt".format(args.fold_num)), 'w') as out_file:
            out_file.write('done')
        # folded_trainer = FoldedTrainer(folds=list(folds_indices_files.keys()))
        # folded_trainer.train_fold(init_trainer=lambda fsm, folds: init_trainer(ResNet18TrainConfig, folds, fsm),
        #                           model_name='resnet18', out_dir=os.path.join('artifacts', 'train'), fold_num=args.fold_num)
