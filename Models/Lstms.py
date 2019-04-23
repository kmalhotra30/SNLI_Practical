from imports import *
from config import *
class lstm_class(nn.Module):
  
    def __init__(self,no_of_words_in_vocab,dim_embedding,**kwargs):

      super(lstm_class,self).__init__()
      self.lstm_units = kwargs['lstm_units']
      self.batch_size = kwargs['batch_size']
      self.bidrectional = kwargs['bidirectional']
      self.pool = False
      if 'pool' in kwargs:
        self.pool = True
      self.no_of_layers = 1
      
      if self.bidrectional == True:
        self.no_of_layers = 2
      
      
      self.no_of_words_in_vocab = no_of_words_in_vocab

      self.lstm_cell = nn.LSTM(dim_embedding,self.lstm_units,1,batch_first = True,bidirectional = self.bidrectional)
      
             
    def init_hidden_lstm(self):
      
      self.hidden_input_cell = Variable(torch.zeros(self.no_of_layers,self.batch_size, self.lstm_units)).to(device)
      self.hidden_input_hidden = Variable(torch.zeros(self.no_of_layers,self.batch_size, self.lstm_units)).to(device)
       
    def forward(self,input_embeddings,input_lenghts):

      
      self.init_hidden_lstm() # Clearing hidden_state for new batch
      embeddings_input = input_embeddings
      
      hidden_input= (self.hidden_input_hidden,self.hidden_input_cell)
      
      #Sorting input by lenghts:
      
      idx_input_sort = torch.argsort(-1 * input_lenghts)#[::-1 ] not supported yet
      input_lenghts = input_lenghts[idx_input_sort]
      embeddings_input = embeddings_input[idx_input_sort]
      idx_input_unsort = torch.argsort(idx_input_sort)
 
      
      embeddings_input_packed = torch.nn.utils.rnn.pack_padded_sequence(embeddings_input, input_lenghts,batch_first=True)
      
      input_out_from_lstm , hidden_input = self.lstm_cell(embeddings_input_packed, hidden_input)
      
      input_hidden_all,_ = torch.nn.utils.rnn.pad_packed_sequence(input_out_from_lstm,batch_first=True)
      
      shape_sent_input = input_hidden_all.shape[1]
      
      if self.bidrectional == True:
        
        #Dealing with BiLstms
        seperated_hidden_states = input_hidden_all.view(self.batch_size,shape_sent_input,2,self.lstm_units)
        rev_indexes = torch.arange(shape_sent_input-1,-1,-1,dtype=torch.long).to(device)
      
        fwd_pass = seperated_hidden_states[:,:,0,:]
        backward_pass = seperated_hidden_states[:,:,1,:]
        
        reversed_hidden_states_for_backward_direction = backward_pass.index_select(1,rev_indexes)
        input_hidden_all = torch.cat((fwd_pass,reversed_hidden_states_for_backward_direction),2)
        
      input_hidden_all = input_hidden_all[idx_input_unsort]
      
      #Reshaping input
      
      
      input_hidden_all = input_hidden_all.view(-1, input_hidden_all.shape[2])
      
      
      dummy_input_length_tensor = (shape_sent_input * torch.arange(0,self.batch_size,dtype=torch.long)).to(device)
      
      dummy_input_length_tensor2 = dummy_input_length_tensor.view(-1) + input_lenghts[idx_input_unsort].view(-1) - 1
      
      input_hidden_all_last = input_hidden_all[dummy_input_length_tensor2]
      
     
      if self.pool == True:
     
        input_len = dummy_input_length_tensor.data.cpu().numpy()
        input_len2 = dummy_input_length_tensor2.data.cpu().numpy()
        
        
        max_tensor = torch.empty(self.batch_size, self.lstm_units * self.no_of_layers).to(device)
        #dummy_input_length_tensor3 = np.array([])
        
        iterator_end = self.batch_size
        for i in range(iterator_end):
        
          start = input_len[i]
          end = input_len2[i]
           
          dummy_tensor = input_hidden_all[start:end+1,:]
          dummy_tensor = torch.max(dummy_tensor,0)[0].view(1,-1)
          max_tensor[i] = dummy_tensor
          
        input_hidden_all = max_tensor
        input_hidden_all.to(device)
        
#         input_hidden_all = input_hidden_all.view(self.batch_size,shape_sent_input,self.lstm_units * self.no_of_layers)
#         input_hidden_all[input_hidden_all == 0.0] = -1e12
#         input_hidden_all = torch.max(input_hidden_all,1)[0]
        
      else:
        input_hidden_all =  input_hidden_all_last
        
      return input_hidden_all
                                         