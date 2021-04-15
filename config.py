import os
from train_config.config import folds_num

ARTIFACTS_DIR = 'artifacts'
DATASET_ROOT = os.path.join(ARTIFACTS_DIR, 'dataset')
DATASET_LABELS = os.path.join(DATASET_ROOT, 'labels.json')
TRAIN_DIR = os.path.join(ARTIFACTS_DIR, 'train')
INDICES_DIR = os.path.join(ARTIFACTS_DIR, 'indices')
PREDICTS_PATH = os.path.join(ARTIFACTS_DIR, 'predicts')
THRESHOLDS_PATH = os.path.join(ARTIFACTS_DIR, 'thresholds')

folds_indices_files = {'fold_{}'.format(i): 'fold_{}.npy'.format(i) for i in range(folds_num)}
