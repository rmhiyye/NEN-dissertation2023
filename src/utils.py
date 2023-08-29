####################
# id_combination, lowercaser_mentions
####################
from sklearn.metrics import f1_score
import src.config as config
from requests.exceptions import HTTPError

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import networkx
import obonet



def id_combination(norm_dict):
    '''
    input:
        {"0034":
            {"N000":
                {"cui": .. ,
                 "mention", ..}
            }
        }
    output:
        {"0034_N000":
            {"cui": .. ,
             "mention", ..}
        }
    '''
    combin_dict = dict()
    for file_id in norm_dict.keys():
        for norm_id in norm_dict[file_id].keys():
            combin_id = file_id + "_" +norm_id
            combin_dict[combin_id] = norm_dict[file_id][norm_id]
    return combin_dict

def lowercaser_mentions(train_dict):
    for key in train_dict.keys():
        train_dict[key]["mention"] = train_dict[key]["mention"].lower()
    return train_dict

def eval_accuracy(actual, predicted):
    acc = 0
    for i, key in enumerate(actual.keys()):
        if isinstance(actual[key], dict):
            true_cui = actual[key]['cui']
        else:
            true_cui = actual[key]
        pred_cui = predicted[i]['first candidate'][0]
        if true_cui == pred_cui:
            acc += 1
    return acc / len(actual.keys())

# calculate mean average precision (k=1, 5)
def eval_map(actual, predicted, k=1):

    aps = []

    for i, key in enumerate(actual.keys()):
        if isinstance(actual[key], dict):
            true_cui = actual[key]['hpo']
        else:
            true_cui = actual[key]
        pred_cui = predicted[i]['first candidate'] if k == 1 else predicted[i]['top 5 candidates']

        num_relevant_items = 0
        sum_precisions = 0

        for j, pred in enumerate(pred_cui, start=1):
            if pred == true_cui:
                num_relevant_items += 1
                precision_at_j = num_relevant_items / j
                sum_precisions += precision_at_j

        ap = sum_precisions / num_relevant_items if num_relevant_items > 0 else 0
        aps.append(ap)

    map = sum(aps) / len(aps)
    
    return map

def eval_distance(actual, predicted):
    graph = obonet.read_obo('/home/yangye/BioCreative/dataset/hp.obo')

    G = networkx.Graph(graph)

    spl = dict(networkx.all_pairs_shortest_path_length(G))

    distance = dict()

    for i, key in enumerate(actual.keys()):
        if isinstance(actual[key], dict):
            true_cui = actual[key]['hpo']
        else:
            true_cui = actual[key]
        pred_cui = predicted[i]['first candidate']

        for j, pred in enumerate(pred_cui, start=1):
            if pred != true_cui:
                if true_cui != 'NA':
                    if pred in spl and true_cui in spl[pred]:
                        distance_ = spl[pred][true_cui]
                        if distance_ not in distance.keys():
                            distance[distance_] = 1
                        distance[distance_] += 1

    return distance

def eval_map_unseen(actual, predicted, train):
    aps = {'seen': [], 'unseen': []}

    num_relevant_items = {'seen': 0, 'unseen': 0}
    sum_precisions = {'seen': 0, 'unseen': 0}
    
    test_hpo_list = []

    test_in_train = []

    train_cui_list = []
    with open(train, 'r') as f:
        for line in f:
            train_cui_list.append(line.split('||')[-1].strip())

    for i, key in enumerate(actual.keys()):
        true_cui = actual[key]['hpo'] if isinstance(actual[key], dict) else actual[key]
        pred_cui = predicted[i]['first candidate']
        pred = pred_cui[0]
        test_hpo_list.append(true_cui)

        type_ = 'seen' if true_cui in train_cui_list else 'unseen'
        num_relevant_items[type_] += 1

        if pred == true_cui:
            sum_precisions[type_] += 1

        if true_cui not in train_cui_list:
            test_in_train.append(true_cui)

    map = [sum_precisions[type_] / num_relevant_items[type_] if num_relevant_items[type_] > 0 else 0 for type_ in ['seen', 'unseen']]
    return map[0], map[1]

def get_cui_name(cui):
    key = config.api_key
    try:
        api = umls_api.API(api_key=key)
        name = api.get_cui(cui)['result']['name']
    except HTTPError:
        print(f"HTTPError occurred for CUI: {cui}")
        name = 'NAME-less'
    return name

def text_preprocessing(text):

    tokens = word_tokenize(text)

    stopwords_list = stopwords.words('english')
    stemmer = PorterStemmer()

    tokens_filtered = []

    for token in tokens:
        if token not in stopwords_list:
            token = stemmer.stem(token)
            tokens_filtered.append(token.lower())

    