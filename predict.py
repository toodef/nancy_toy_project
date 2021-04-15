import argparse
import json
import os
import sys
from random import random

from config import PREDICTS_PATH

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calc metrics')
    parser.add_argument('-m', '--model', type=str, help='Train one model', required=True, choices=['resnet18', 'resnet34'])

    if len(sys.argv) < 2:
        print('Bad arguments passed', file=sys.stderr)
        parser.print_help(file=sys.stderr)
        exit(2)
    args = parser.parse_args()

    cur_dir = os.path.join(PREDICTS_PATH, args.model)
    if not os.path.exists(cur_dir):
        os.makedirs(cur_dir)

    with open(os.path.join(cur_dir, "metrics.json".format(args.fold_num)), 'w') as out_file:
        json.dump({'metric1': random(), 'metric2': random()}, out_file)
