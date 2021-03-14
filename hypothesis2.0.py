#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
import sentence_transformers

from util import parse_db, Document, read_markup_tsv
from torch.utils.data import Dataset, DataLoader

import random

records = parse_db("data_/0525_parsed.db")


# In[3]:


from sentence_transformers import InputExample

random.seed(0)

def separate_samples(markup, train_percent=0.85):

    count_samples = len(markup)
    shuffled_ids = list(range(count_samples))
    random.shuffle(shuffled_ids)
    train_len = round(count_samples * train_percent)

    train_markup, test_markup = [], []

    for i, id in enumerate(shuffled_ids):
        if i < train_len:
            train_markup.append(markup[id])
        else:
            test_markup.append(markup[id])        
           
    return train_markup, test_markup


def get_data(markup, records):
    input_samples, other_samples, qualities = [], [], []

    for mrkp in markup:
        first_url, second_url, quality = mrkp.values()

        input_samples.append(url2record[first_url]['title'] + ' ' + url2record[first_url]['text'])
        other_samples.append(url2record[second_url]['title'] + ' ' + url2record[second_url]['text'])
        if quality == 'OK':
            qualities.append(1)
        else:
            qualities.append(0)
    return input_samples, other_samples, qualities

url2record = dict()
    
for i, record in enumerate(records):
    url2record[record["url"]] = record

markup = read_markup_tsv("data_/ru_clustering_0525_urls.tsv")

train_markup, test_markup = separate_samples(markup)
test_inputs, test_others, test_qualities  = get_data(test_markup, records)


# In[4]:


from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader

model_name = 'bert-base-multilingual-uncased'

ensemble_models_predicts = []

for model_no in range(5):
    model = CrossEncoder(f'{model_name}UNC/{model_no}')
    ensemble_models_predicts.append(model.predict(list(zip(test_inputs, test_others))))


# In[5]:


import numpy as np

test_qualities = np.array(test_qualities, dtype = float)


# In[55]:


def entropy(preds):
    entropies = []
    
    for pred in preds:
        entropy_per_sample = - pred * np.log(pred)  - (1 - pred) * np.log(1 - pred)
        entropies.append(entropy_per_sample)
    
    entropies = np.array(entropies)
    
    return entropies


# In[7]:


total = sum(ensemble_models_predicts)/len(ensemble_models_predicts)

total_uncertainty = entropy(total)

individual_entropy = []

for model_predict in ensemble_models_predicts:
    
    individual_entropy.append(entropy(model_predict))

expected_data_uncertainty = sum(individual_entropy)/len(individual_entropy)


# In[12]:


test_inputs = np.array(test_inputs)
test_others = np.array(test_others)

uncert_inp = test_inputs[total_uncertainty > 0.5]
uncert_oth = test_others[total_uncertainty > 0.5]
uncert_qual = test_qualities[total_uncertainty > 0.5]


# In[13]:


import pickle

with open('pretrain','rb') as fp:
    pretrain_data = pickle.load(fp)


# In[14]:


from tqdm.notebook import tqdm
from hnswlib import Index as HnswIndex

def model_embed_text(model, text):
    return model.encode([text],show_progress_bar = False)

class Hnsw:
    def __init__(self, model):
        self.model = model
        self.vector_dim = 512
        self.hnsw = HnswIndex(space='l2', dim=self.vector_dim)

    def build_hnsw(self, texts):
        n = len(texts)
        self.hnsw.init_index(max_elements=n, ef_construction=100, M=16)
        embeddings = np.zeros((n, self.vector_dim))
        for i, text in enumerate(tqdm(texts)):
            embeddings[i] = model_embed_text(self.model, text)
        self.hnsw.add_items(embeddings)


# In[75]:


model = sentence_transformers.SentenceTransformer('distiluse-base-multilingual-cased-v2')


# In[20]:


use_hnsw = Hnsw(model)
pretrain_data.sort(key=lambda x: x.get("date")) 


# In[21]:


use_hnsw.build_hnsw([r["title"] + ' ' + r["text"] for r in pretrain_data[:120000]])


# In[22]:


cos = torch.nn.CosineSimilarity(dim=1, eps=1e-08)


# In[73]:


uncert_data = np.append(uncert_inp, uncert_oth)


# In[76]:


samples = []

for row in tqdm(uncert_data):
    vector = model_embed_text(model, row)
                                 
    labels = list(use_hnsw.hnsw.knn_query(vector, k=20)[0][0])
    samples.append(labels[0])
    


# In[78]:


print(len(samples))


# In[82]:


test_new_pair = []
for sample in samples:
    test_new_pair.append(pretrain_data[sample]['title'] + ' ' + pretrain_data[sample]['text'])
    


# In[83]:


new_qual = np.zeros_like(samples)


# In[84]:


model_name = 'bert-base-multilingual-uncased'

ensemble_models_predicts = []

for model_no in range(5):
    model = CrossEncoder(f'{model_name}UNC/{model_no}')
    ensemble_models_predicts.append(model.predict(list(zip(uncert_data, test_new_pair))))


# In[85]:


total = sum(ensemble_models_predicts)/len(ensemble_models_predicts)

total_uncertainty = entropy(total)

individual_entropy = []

for model_predict in ensemble_models_predicts:
    
    individual_entropy.append(entropy(model_predict))

expected_data_uncertainty = sum(individual_entropy)/len(individual_entropy)


# In[68]:


len(total_uncertainty[total_uncertainty > 0.5])


# In[69]:


len(total[(total_uncertainty > 0.5) & ((total > 0.5) == new_qual)])


# In[90]:


len(total[(total>0.45) == new_qual])/len(total)


# In[ ]:




