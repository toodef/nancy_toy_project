import os
from pietoolbelt.steps.stratification import DatasetStratification

from train_config.config import folds_num
from train_config.dataset import create_dataset
from config import INDICES_DIR


def calc_label(x):
    return int(x[0])


if __name__ == '__main__':
    if not os.path.exists(INDICES_DIR):
        os.makedirs(INDICES_DIR)

    test_part = 0.1
    folds_dict = {'fold_{}.npy'.format(i): (1 - test_part) / folds_num for i in range(folds_num)}

    strat = DatasetStratification(create_dataset(), calc_label)
    strat.run(dict(folds_dict, **{'test.npy': test_part}), INDICES_DIR)
