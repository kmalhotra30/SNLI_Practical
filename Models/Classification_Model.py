from imports import *
from config import *
class Classification_Model(nn.Module):

  def __init__(self, no_of_words_in_vocab,dim_embedding):


    super(Classification_Model, self).__init__()
   
    self.seq_layers = nn.Sequential(nn.Linear(4 * dim_embedding,512),nn.Tanh(),nn.Linear(512,3))

    
  def forward(self,input_premise_embeddings,input_hypothesis_embeddings):

    #embeddings_premise = input_premise_embeddings.mean(1)

    #embeddings_hypothesis = input_hypothesis_embeddings.mean(1)
    embeddings_premise = input_premise_embeddings
    embeddings_hypothesis = input_hypothesis_embeddings
    concat = torch.cat((embeddings_premise,embeddings_hypothesis),1)

    abs_t = torch.abs(embeddings_premise - embeddings_hypothesis)
    mul = embeddings_premise * embeddings_hypothesis
    final_tensor = torch.cat((concat,abs_t,mul),1)
    output = self.seq_layers(final_tensor)

    return output
    #Shape of output = Batch X Length of Sentence
      