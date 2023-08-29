# do the embedding by BERT

import sys
sys.path.append('/mnt/data/yangye/MCN-baseline/src')

import numpy as np
import csv
import torch
from tqdm import tqdm
import pickle
from transformers import AutoTokenizer
import argparse

from src import config
from src.model import Bertembedding
from src.data_loader import biocreative_loader


import os

def main(args, config):
        bert = Bertembedding(args.model_name).to(config.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        print('Load model from {}'.format(args.model_name))

        file_path = '/home/yangye/BioCreative/BioCreative VIII Track 3/Evaluation script/HP2Terms.tsv'
        unobserved_path = '/home/yangye/BioCreative/BioCreative VIII Track 3/List of unobservable HPO terms/UnobservableHPOTerms.tsv'
        dataset_path = '/home/yangye/BioCreative/dataset/val_annotation.txt'

        save_path = args.save_path
        if not os.path.exists(save_path):
                os.makedirs(save_path)
        hpo_encode_path = os.path.join(save_path, 'hpo_encode.pickle')
        hpo_id_path = os.path.join(save_path, 'hpo_id.pickle')
        spans_path = os.path.join(save_path, 'spans_encode.pickle')

        bert.eval()

        os.makedirs('embedding_matrix', exist_ok=True)

        if args.embedding_target == 'hpo' or args.embedding_target == 'both':
                HPO_id, HPO_term = HPO_term_collect(file_path, unobserved_path)
                hpo_encode = map_cui(HPO_term, bert, tokenizer)

                # Save cui_encode dictionary
                with open(hpo_encode_path, 'wb') as f:
                        pickle.dump(hpo_encode, f)
                
                with open(hpo_id_path, 'wb') as f:
                        pickle.dump(HPO_id, f)

        if args.embedding_target == 'spans' or args.embedding_target == 'both':
                dataset = biocreative_loader(dataset_path)
                spans_embeded = map_spans(dataset, bert, tokenizer)

                # Save spans_encode dictionary
                with open(spans_path, 'wb') as f:
                        pickle.dump(spans_embeded, f)


def HPO_term_collect(file_path, unobserved_path):
        hpo_id = []
        hpo_name = []
        unob_hpo_id = []
        unob_hpo_name = []

        with open(unobserved_path, 'r', newline='') as tsv_file:
                tsv_reader = csv.reader(tsv_file, delimiter='\t')
                for row in tsv_reader:
                        unob_hpo_id.append(row[0])
                        unob_hpo_name.append(row[1])
        with open(file_path, 'r', newline='') as tsv_file:
                tsv_reader = csv.reader(tsv_file, delimiter='\t')
                for row in tsv_reader:
                        if row[0] not in unob_hpo_id:
                                hpo_id.append(row[0])
                                hpo_name.append(row[1])

        return hpo_id, hpo_name

# Mapping ontology concepts
def map_cui(ref, bert, tokenizer):
        hpo_encode = np.zeros((len(ref), config.embbed_size))
        with torch.no_grad():
                for idx, hpo in tqdm(enumerate(ref), total=len(ref)):
                        encode =  tokenizer.encode_plus(hpo, padding="max_length", max_length=config.max_length, truncation=True, add_special_tokens=True, return_tensors="pt").to(config.device) # Tokenize input into ids.
                        hpo_encode[idx] = bert(encode).cpu().numpy()
        return hpo_encode # Returns a dictionnary containing the embedding of each concept in the ontology.

def map_spans(dataset, bert, tokenizer):
        spans_embeded = np.zeros((len(dataset.keys()), config.embbed_size))
        with torch.no_grad():
                for i, idx in tqdm(enumerate(dataset.keys()), total=len(dataset)):
                        encode = tokenizer.encode_plus(dataset[idx]['span'], padding="max_length", max_length=config.max_length, truncation=True, add_special_tokens=True, return_tensors="pt").to(config.device) # Tokenize input into ids.
                        spans_embeded[i] = bert(encode).cpu().numpy()

        return spans_embeded

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--embedding_target', type=str, default='both', help='hpo, spans or both')
        parser.add_argument('--model_name', type=str, default='/home/yangye/BioCreative/model/bert_hpo_50epoch/checkpoint_30', help='model name')
        parser.add_argument('--save_path', type=str, default='/home/yangye/BioCreative/embedding_matrix/bert/30epoch', help='path to save the embedding matrix')
        args = parser.parse_args()
# '/home/yangye/BioCreative/model/biobert_hpo_50epoch/checkpoint_5'
# "dmis-lab/biobert-base-cased-v1.2"
        main(args, config)
