from imports import *
from config import *
#Loading all models
from BaselineModel import BaselineModel
from Classification_Model import Classification_Model
from Lstms import lstm_class
class Model(nn.Module):
  
  def __init__(self,no_of_words_in_vocab,dim_embedding,model_type,batch_size):
    
    super(Model,self).__init__()
    self.no_of_words_in_vocab = no_of_words_in_vocab
    self.dim_embedding = dim_embedding
    self.embed = nn.Embedding(no_of_words_in_vocab, dim_embedding)
    self.embed.weight.requires_grad = False
    
    
    no_of_lstm_layers = 1
    embedding_size_for_encoder = 300 
    
    
    if model_type == 1:
      
      self.encoder = BaselineModel()
      
    if model_type == 2:
      
      self.encoder = lstm_class(self.no_of_words_in_vocab,self.dim_embedding,lstm_units = 2048,batch_size=batch_size,bidirectional=False)
      embedding_size_for_encoder = 2048

    if model_type == 3:
      
      self.encoder = lstm_class(self.no_of_words_in_vocab,self.dim_embedding,lstm_units = 2048,batch_size=batch_size,bidirectional=True)
      embedding_size_for_encoder = 2048
      no_of_lstm_layers = 2
      
    if model_type == 4:
      
      
      self.encoder = lstm_class(self.no_of_words_in_vocab,self.dim_embedding,lstm_units = 2048,batch_size=batch_size,bidirectional=True,pool=True)
      embedding_size_for_encoder = 2048
      no_of_lstm_layers = 2
      
    self.encoder = self.encoder.to(torch.device(device))
      
    self.classification_model= Classification_Model(self.no_of_words_in_vocab,embedding_size_for_encoder * no_of_lstm_layers)
    self.classification_model = self.classification_model.to(torch.device(device))
    
  def forward(self,input_premise,input_hypothesis,model_type,**kwargs):
    
    embeddings_premise = self.embed(input_premise)
    embeddings_hypothesis = self.embed(input_hypothesis)
    
    premise_lenghts = kwargs['premise_lengths']
    hypothesis_lenghts = kwargs['hypothesis_lengths']
      
    premise_encoding = self.encoder(embeddings_premise,premise_lenghts)
    hypothesis_encoding = self.encoder(embeddings_hypothesis,hypothesis_lenghts)

    output = self.classification_model(premise_encoding,hypothesis_encoding)
    return output
