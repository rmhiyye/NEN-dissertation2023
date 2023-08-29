from transformers import AutoTokenizer
import torch
from sklearn.metrics.pairwise import cosine_distances
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

import pickle

from src.model import Bertembedding

class HPO_MAPPING(object):
        def __init__(self, config, path):
                self.config = config

                # Load models
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                print(path)
                self.hpo_encode_path = os.path.join(path, 'hpo_encode.pickle')
                self.hpo_id_path = os.path.join(path, 'hpo_id.pickle')
                self.spans_path = os.path.join(path, 'spans_encode.pickle')

        def inference(self):

                # Load hpo_encode dictionary
                with open(self.hpo_encode_path, 'rb') as f:
                        hpo_encode = pickle.load(f)

                with open(self.hpo_id_path, 'rb') as f:
                        hpo_id = pickle.load(f)

                with open(self.spans_path, 'rb') as f:
                        span_encode = pickle.load(f)

                # Returns a dictionary of candidates ranked by their cosine similarity.
                dd_predictions = self.candidates_rank(span_encode, hpo_id, hpo_encode)
                print("Done.\n")

                del hpo_encode
                return dd_predictions
        
        # create a function to output a dictionary of candidates ranked by their cosine similarity
        def candidates_rank(self, span_encode, hpo_id, hpo_encode):

                # Calculate distance matrix
                scoreMatrix = cosine_distances(span_encode, hpo_encode)

                # Prepare prediction dictionary
                dd_predictions = {id: {'first candidate': [], 'top 5 candidates': []} for id in range(len(span_encode))}

                # For each mention, find back the nearest hpo vector, then attribute the associated hpo:
                for i, id in enumerate(dd_predictions.keys()):
                        min_indices_10 = np.argpartition(scoreMatrix[i], 5)[:5]
                        min_indices_10 = min_indices_10[np.argsort(scoreMatrix[i][min_indices_10])]
                        min_indices = np.argmin(scoreMatrix[i])
                        
                        # Store the closest hpo in the predictions dictionary.
                        dd_predictions[id]['first candidate'] = [hpo_id[min_indices]]
                        dd_predictions[id]['top 5 candidates'] = [hpo_id[idx] for idx in min_indices_10]
                        
                return dd_predictions