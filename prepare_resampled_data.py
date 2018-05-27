import argparse
import algorithms
import datasets
import logging
import numpy as np
import pandas as pd

from collections import Counter
from pathlib import Path


DEFAULT_ROOT_OUTPUT_PATH = Path(__file__).parent / 'results'
DEFAULT_DATA_PATH = Path(__file__).parent / 'data'

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('-mode', choices=['OVA', 'OVO'], default='OVA')
parser.add_argument('-output_path', type=str, default=DEFAULT_ROOT_OUTPUT_PATH)
parser.add_argument('-energy', type=float, default=0.25)
parser.add_argument('-cleaning_strategy', type=str, choices=['ignore', 'translate', 'remove'], default='translate')
parser.add_argument('-selection_strategy', type=str, choices=['proportional', 'random'], default='proportional')
parser.add_argument('-p_norm', type=float, default=1.0)
parser.add_argument('-method', choices=['sampling', 'complete'], default='sampling')

args = parser.parse_args()

for dataset in datasets.names():
    for partition in [1, 2, 3, 4, 5]:
        for fold in [1, 2]:
            logging.info('Processing fold %dx%d of dataset "%s"...' % (partition, fold, dataset))

            output_path = Path(args.output_path) / dataset
            output_path.mkdir(parents=True, exist_ok=True)

            (X_train, y_train), (X_test, y_test) = datasets.load(dataset, partition, fold)

            header = pd.read_csv(
                DEFAULT_DATA_PATH / 'folds' / dataset / ('%s.%d.%d.train.csv' % (dataset, partition, fold))
            ).columns

            if args.mode == 'OVA':
                logging.info('Training distribution before resampling: %s.' % Counter(y_train))

                X_train, y_train = algorithms.MultiClassCCR(
                    energy=args.energy, cleaning_strategy=args.cleaning_strategy,
                    selection_strategy=args.selection_strategy, p_norm=args.p_norm, method=args.method
                ).fit_sample(X_train, y_train)

                logging.info('Training distribution after resampling: %s.' % Counter(y_train))

                csv_path = output_path / ('%s.%d.%d.train.oversampled.csv' % (dataset, partition, fold))

                pd.DataFrame(np.c_[X_train, y_train]).to_csv(csv_path, index=False, header=header)
            elif args.mode == 'OVO':
                classes = np.unique(np.concatenate([y_train, y_test]))

                for i in range(len(classes)):
                    for j in range(i + 1, len(classes)):
                        logging.info('Resampling class %s vs. class %s.' % (classes[i], classes[j]))

                        indices = ((y_train == classes[i]) | (y_train == classes[j]))

                        X, y = X_train[indices].copy(), y_train[indices].copy()

                        logging.info('Training distribution before resampling: %s.' % Counter(y))

                        class_distribution = {cls: len(y[y == cls]) for cls in [classes[i], classes[j]]}
                        minority_class = min(class_distribution, key=class_distribution.get)

                        X, y = algorithms.CCR(
                            energy=args.energy, cleaning_strategy=args.cleaning_strategy,
                            selection_strategy=args.selection_strategy, p_norm=args.p_norm
                        ).fit_sample(X, y)

                        logging.info('Training distribution after resampling: %s.' % Counter(y))

                        csv_path = output_path / ('%s.%d.%d.train.oversampled.%dv%d.csv' %
                                                  (dataset, partition, fold, classes[i], classes[j]))

                        pd.DataFrame(np.c_[X, y]).to_csv(csv_path, index=False, header=header)
            else:
                raise NotImplementedError
