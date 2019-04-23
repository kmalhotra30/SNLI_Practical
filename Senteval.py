#########################################################################################
# Note : The following code has been adapted from https://github.com/facebookresearch/SentEval/blob/master/examples/bow.py. 


# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##########################################################################################



from __future__ import absolute_import, division, unicode_literals
import sys

import argparse
path_to_Models = './Models'
path_to_Modules = './Modules'
sys.path.append(path_to_Models)
sys.path.append(path_to_Modules)

import io
import logging
#Loading methods
from helper_functions import *
from train_model import *

#Loading all models
from BaselineModel import BaselineModel
from Classification_Model import Classification_Model
from Lstms import lstm_class
from Model import Model
# path to senteval
PATH_TO_SENTEVAL = './SentEval/'
# path to the NLP datasets 
PATH_TO_DATA = './SentEval/data/'
# path to glove embeddings
PATH_TO_VEC = './pretrained/glove.840B.300d.txt'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

#tokenize = True
# Create dictionary
def create_dictionary(sentences, threshold=0):
    words = {}
    for s in sentences:
      for word in s:
        words[word] = words.get(word, 0) + 1

    if threshold > 0:
        newwords = {}
        for word in words:
            if words[word] >= threshold:
                newwords[word] = words[word]
        words = newwords
    words['<s>'] = 1e9 + 4
    words['</s>'] = 1e9 + 3
    words['<p>'] = 1e9 + 2

    sorted_words = sorted(words.items(), key=lambda x: -x[1])  # inverse sort
    id2word = []
    word2id = {}
    for i, (w, _) in enumerate(sorted_words):
        id2word.append(w)
        word2id[w] = i

    return id2word, word2id

# Get word vectors from vocabulary (glove, word2vec, fasttext ..)
def get_wordvec(path_to_vec, word2id):
    word_vec = {}

    with io.open(path_to_vec, 'r', encoding='utf-8') as f:
        # if word2vec or fasttext file : skip first line "next(f)"
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word2id:
                word_vec[word] = np.fromstring(vec, sep=' ')

    logging.info('Found {0} words with word vectors, out of \
        {1} words'.format(len(word_vec), len(word2id)))
    
    return word_vec


# SentEval prepare and batcher

def prepare_padded_batch(batch):
 
  #Creating batch 
  batch_matrix = [batch[b] for b in range(len(batch))]
  batch_sent_len = np.array([len(batch[b]) for b in range(len(batch))])
  
  max_length = max(batch_sent_len)
  
  for sent in batch_matrix:
    sent.extend(['<PAD>'] * (max_length - len(sent)))
     
  return batch_matrix,batch_sent_len,max_length
    
  
  
def prepare(params, samples):
    _, params.word2id = create_dictionary(samples)
    params.word_vec = get_wordvec(PATH_TO_VEC, params.word2id)
    params.wvec_dim = 300
    #Adding embedding for <PAD>
    params.word_vec['<PAD>'] = np.zeros(params.wvec_dim)
    params.no_of_words_in_vocab = len(params.word_vec)
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    
    batch_s = len(batch)
    model_type = params.model_type
    
    padded_batch ,batch_sent_length,max_length = prepare_padded_batch(batch)
    
    if model_type == 1:
      encoder_model = BaselineModel()
    elif model_type == 2:
      encoder_model = lstm_class(params.no_of_words_in_vocab,params.wvec_dim,lstm_units = 2048,batch_size=batch_s,bidirectional=False)
    elif model_type == 3:
      encoder_model = lstm_class(params.no_of_words_in_vocab,params.wvec_dim,lstm_units = 2048,batch_size=batch_s,bidirectional=True)
    elif model_type == 4:
      encoder_model = lstm_class(params.no_of_words_in_vocab,params.wvec_dim,lstm_units = 2048,batch_size=batch_s,bidirectional=True,pool=True
      )
    
    loaded_checkpoint = torch.load(params['checkpoint_path'] + '/Model_best_' + str(model_type) +'.tar')
    encoder_model.load_state_dict(loaded_checkpoint["encoder_state_dict"])
    encoder_model.to(device)
    embeddings = []

    glove_embeddings_for_batch_tensor = []
    for s,sent in enumerate(padded_batch):
      
      sentvec = []
      for word in sent:
        
        if word in params.word_vec:
          sentvec.append(params.word_vec[word])
        else:
          #Ignoring unknown words
          sentvec.append(params.word_vec['<PAD>'])    
          
        
      sentvec = np.array(sentvec)
      
      glove_embeddings_for_batch_tensor.append(sentvec) 
      
    glove_embeddings_for_batch_tensor = torch.FloatTensor(glove_embeddings_for_batch_tensor)
    glove_embeddings_for_batch_tensor = glove_embeddings_for_batch_tensor.to(torch.device(device))

    batch_sent_length = torch.LongTensor(batch_sent_length)
    batch_sent_length = batch_sent_length.to(torch.device(device))

    embeddings_by_encoder = encoder_model(glove_embeddings_for_batch_tensor,batch_sent_length)
    
    #Shape of embeddings_by_encoder = Batch_size x Dimension of encoding
    
    return embeddings_by_encoder.data.cpu().numpy()

# Set params for SentEval

def run_senteval(model_type,checkpoint_path):


  params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
  params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,'tenacity': 5, 'epoch_size': 4}
  params['model_type'] = model_type
  params['checkpoint_path'] = checkpoint_path
  

  # Set up logger
  logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

  # if __name__ == "__main__":
  se = senteval.engine.SE(params, batcher, prepare)
  transfer_tasks = ['SST2','SST5','STS14','SUBJ','MR', 'CR', 'MPQA','TREC', 'MRPC','SICKEntailment', 'SICKRelatedness', 'STSBenchmark']
  
  
  results = se.eval(transfer_tasks)
  print(results)