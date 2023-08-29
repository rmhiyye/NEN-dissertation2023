import os
import time
import argparse

import src.config as config
from src.norm import HPO_MAPPING
from src.data_loader import biocreative_loader
from src.utils import eval_map, eval_map_unseen, eval_distance

import os
os.chdir("/home/yangye/BioCreative")

def main(args, config):
        
        start_time = time.time()

        # Load data
        biocreative_path = 'dataset/val_annotation.txt'
        dataset = biocreative_loader(biocreative_path)

        path = args.path

        trainer = HPO_MAPPING(config, path)
        predictions = trainer.inference()

        inference_time_minutes = (time.time() - start_time) / 60

        print("Finish inference...")
        print("Inference time: {:.2f} minutes".format(inference_time_minutes))

        MAP_k1 = eval_map(dataset, predictions)
        print('MAP', MAP_k1)
        MAP_k5 = eval_map(dataset, predictions, k=5)
        print('MAP@5', MAP_k5)

        train = '/home/yangye/BioCreative/dataset/train_annotation.txt'
        MAP_seen, MAP_unseen = eval_map_unseen(dataset, predictions, train)
        print('MAP_seen', MAP_seen)
        print('MAP_unseen', MAP_unseen)


if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--path', type=str, default='/home/yangye/BioCreative/embedding_matrix/bert/30epoch')
        args = parser.parse_args()

        main(args, config)