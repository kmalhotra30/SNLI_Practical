from imports import *
from config import *
# from BaselineModel import BaselineModel
# from Classification_Model import Classification_Model
# from Lstms import lstm_class
# from Model import Model
data_pickle_in = open("./data/data_pickle.p","rb")
data_dict_from_pickle = pickle.load(data_pickle_in)
vocab = data_dict_from_pickle['vocab']

def create_word_to_idx(vocab):

  word_to_idx = {}
  idx = 0
  for word in vocab:
    
    word_to_idx[word] = idx
    idx = idx + 1
    
  return word_to_idx

def create_word_to_idx(vocab):

  word_to_idx = {}
  idx = 0
  for word in vocab:
    
    word_to_idx[word] = idx
    idx = idx + 1
    
  return word_to_idx

def prune_data_set(data_set,data_set_size):
  
  pruned_data_set = data_set[:data_set_size]
  return pruned_data_set

def sent_to_idx_list(sentence):

  sentence_word_idx_list = []
  for word in nltk.word_tokenize(sentence):
    
    if word in word_to_idx:
      word_idx = word_to_idx[word]
      sentence_word_idx_list.append(word_to_idx[word])
      
  
  return sentence_word_idx_list

def prepare_target_dict():
  
  target_dict = {}
  target_dict['neutral'] = 0
  target_dict['entailment'] = 1
  target_dict['contradiction'] = 2
  
  return target_dict


def prepare_for_pytorch(inp):
  
  inp = torch.LongTensor(inp)
  inp = inp.to(torch.device(device))

  return inp

target_dict = prepare_target_dict()
word_to_idx = create_word_to_idx(vocab)
def prepare_batch_with_idx(start_idx,data_set,batch_size):
  
  
  end_idx = min(len(data_set),start_idx + batch_size)
  batch_premise = []
  batch_hypothesis = []
  batch_label = []
  current_size = 0
  
  for i in range(start_idx,end_idx):
    
    sample = data_set[i]
   
    while(len(sent_to_idx_list(sample['premise'])) == 0 or len(sent_to_idx_list(sample['hypothesis']))==0):
      
      sample = np.random.choice(data_set)
      
    batch_premise.append(sent_to_idx_list(sample['premise']))
    batch_hypothesis.append(sent_to_idx_list(sample['hypothesis']))
    batch_label.append(target_dict[sample['label']])

  #Padding bacthes with random samples from the data set 
  
  if end_idx - start_idx < batch_size:
    for i in range(batch_size - end_idx + start_idx):

      random_sample = np.random.choice(data_set) 
      
      while(len(sent_to_idx_list(random_sample['premise'])) == 0 or len(sent_to_idx_list(random_sample['hypothesis']))==0):
      
        random_sample = np.random.choice(data_set)
     
      batch_premise.append(sent_to_idx_list(random_sample['premise']))
      batch_hypothesis.append(sent_to_idx_list(random_sample['hypothesis']))
      batch_label.append(target_dict[random_sample['label']])

  #Padding with word_to_idx['<PAD>'], since not all sentences have equal length
  
  lengths_premise = [len(batch_p) for batch_p in batch_premise]
  lengths_hypothesis = [len(batch_h) for batch_h in batch_hypothesis]
  for l in lengths_hypothesis:
    if l == 0:
      
      print("Zero length sentence found")
      print(lengths_hypothesis,start_idx,end_idx,batch_size)
  
  max_len_premise = np.max(lengths_premise)
  max_len_hypothesis = np.max(lengths_hypothesis)
  
  for batch_p in batch_premise:
    batch_p.extend([word_to_idx["<PAD>"]] * (max_len_premise - len(batch_p)))
    
  for batch_h in batch_hypothesis:
    batch_h.extend([word_to_idx["<PAD>"]] * (max_len_hypothesis - len(batch_h)))
    
    
  batch_premise = np.array([np.array(batch_p) for batch_p in batch_premise])
  batch_hypothesis = np.array([np.array(batch_h) for batch_h in batch_hypothesis])
  batch_label = np.array(batch_label)
  
  return batch_premise,lengths_premise,batch_hypothesis,lengths_hypothesis,batch_label

def create_vocab_matrix(vocab):
  
  vocab_vectors_list = np.array([vocab[w] for w in vocab])
  vocab_matrix = np.array(vocab_vectors_list)
  
  return vocab_matrix

def compute_accuracy(model,model_type,data_set,batch_size,**kwargs):
  
  current_batch_start_index = 0
  number_of_batches_that_fit_in = math.ceil(len(data_set) / batch_size)
  
  no_of_correct = 0
  for i in range(number_of_batches_that_fit_in):

    #Loading batches 
    batch_premise,lengths_premise,batch_hypothesis,lengths_hypothesis,batch_label = prepare_batch_with_idx(current_batch_start_index,data_set,batch_size)
    current_batch_start_index += batch_size


    batch_premise = prepare_for_pytorch(batch_premise)
    lengths_premise = prepare_for_pytorch(lengths_premise)
    batch_hypothesis = prepare_for_pytorch(batch_hypothesis)
    lengths_hypothesis = prepare_for_pytorch(lengths_hypothesis)
    batch_label = prepare_for_pytorch(batch_label)



    #Forward pass

    with torch.no_grad(): 
      
      if model_type == 1:
        output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis)
      elif model_type == 2:
        output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis,bidrectional=False)
      elif model_type == 3:
        output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis,bidirectional=True)      
      else:
        output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis,bidirectional=True,pool=True)
      

    predictions = output.argmax(1)
    no_of_correct += (predictions.view(-1) == batch_label.view(-1)).sum().item()
    
  return no_of_correct / max(len(data_set),batch_size)
   
def compute_batch_accuracy(model,model_type,data_set,batch_size,**kwargs):
  
  no_of_correct = 0

  batch_premise = data_set[0]
  lengths_premise = data_set[1]
  batch_hypothesis = data_set[2]
  lengths_hypothesis = data_set[3]
  batch_label = data_set[4] 
  
  #Forward pass

  with torch.no_grad(): 

    if model_type == 1:
      output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis)
    elif model_type == 2:
      output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis,bidrectional=False)
    elif model_type == 3:
      output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis,bidirectional=True)      
    else:
      output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis,bidirectional=True,pool=True)


    predictions = output.argmax(1)
    no_of_correct += (predictions.view(-1) == batch_label.view(-1)).sum().item()
    
  return no_of_correct / max(len(data_set),batch_size)

def print_model_params(model):
  total = 0
  for name, parameter in model.named_parameters():
    total += np.prod(parameter.shape)
    print(name,parameter.shape,"requires gradient = " + str(parameter.requires_grad))
  
  print("Total parameters: ",total)

def create_checkpoint(model,encoder,model_type,optimizer,train_avg_accuracy,train_avg_loss,epoch_no,ckpt_path):

  
  checkpoint_path_Model = ckpt_path + "/Model_" + str(model_type) + "_" + str(epoch_no) + ".tar"
  checkpoint_Model = {
              "model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "train_avg_accuracy": train_avg_accuracy,
              "train_avg_loss":train_avg_loss,
              "encoder_state_dict":encoder.state_dict(),
              "epoch":epoch_no
          }
  
  
  torch.save(checkpoint_Model, checkpoint_path_Model)


  
