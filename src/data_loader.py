####################
# 2019-n2c2-MCN dataset loader:
####################

import os
import random
import torch
from torch.utils.data import DataLoader
from requests.exceptions import HTTPError

def biocreative_loader(dataset_path):
        dataset = dict()
        with open(dataset_path, 'r') as f:
                for idx, line in enumerate(f):
                        dataset[idx] = dict()
                        line = line.split('||')
                        span = line[1]
                        hpo = line[2]
                        dataset[idx]['span'] = span
                        dataset[idx]['hpo'] = hpo.replace("\n", "")
        return dataset
                        