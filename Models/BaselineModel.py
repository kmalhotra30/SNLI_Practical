from imports import *
from config import *
class BaselineModel(nn.Module):
  
  def __init__(self):
    
    super(BaselineModel,self).__init__()
    
  def forward(self,input_embedding,input_lengths):
    
    return torch.div(input_embedding.sum(1),input_lengths.view(-1,1).to(torch.float))