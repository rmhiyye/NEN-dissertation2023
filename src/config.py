cuda_visible_devices = '0'
device = 'cuda'
# device = 'cpu'
# model_name = 'cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
# model_name = '/home/yangye/BioCreative/model/sapbert_hpo_5epoch'
model_name = '/home/yangye/BioCreative/model/sapbert_hpo_30epoch/checkpoint_26'
api_key = '64a17e8c-3406-4dbc-bb2e-1afbe350bc37'
no_cuda = False
embbed_size = 768
max_length = 16
batch_size = 64
epochs = 10
lr = 1e-5
checkpoint_dir = './checkpoint(SapBERT)_5.30_CLS'
data_path = './dataset'
seed = 999
show_step = 1
save_step = 100