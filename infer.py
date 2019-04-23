import sys
import argparse
path_to_Models = './Models'
path_to_Modules = './Modules'
sys.path.append(path_to_Models)
sys.path.append(path_to_Modules)
batch_size = 1

inverse_target_dict = {0:'Neutral',1:'Entailment',2:'Contradiction'}

from imports import * 
nltk.download('punkt') #Nltk tokenizer

#Loading methods
from helper_functions import *
from train_model import *

#Loading all models
from BaselineModel import BaselineModel
from Classification_Model import Classification_Model
from Lstms import lstm_class
from Model import Model

#Flags for command line arguements
FLAGS = None

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type = int, default = 1)
parser.add_argument('--checkpoint_path', type = str,default = "./Checkpoints")
parser.add_argument('--use_full_embedding',type = bool,default = False)
FLAGS, unparsed = parser.parse_known_args()

data_pickle_in = open("./data/data_pickle.p","rb")
data_dict_from_pickle = pickle.load(data_pickle_in)
vocab = data_dict_from_pickle['vocab']

vocab_matrix = create_vocab_matrix(vocab)

model_type = FLAGS.model_type
model = Model(len(vocab),300,model_type,batch_size)
model.embed.weight.data.copy_(torch.from_numpy(vocab_matrix))
model.embed.weight.requires_grad = False
model = model.to(device)

loaded_checkpoint = torch.load(FLAGS.checkpoint_path + '/Model_best_' + str(model_type) + '.tar')
model.load_state_dict(loaded_checkpoint["model_state_dict"])

print("Enter 0 exit at any point of time")
while(True):

    premise = input("Enter a premise\n")
    if premise == "0":
        break
    hypothesis = input("ENter a hypothesis\n")
    if hypothesis == "0":
        break

    batch_premise = [sent_to_idx_list(premise)]
    batch_premise = np.array(batch_premise)
    lengths_premise = np.array([len(batch_premise[0])])
    
    batch_hypothesis = [sent_to_idx_list(hypothesis)]
    batch_hypothesis = np.array(batch_hypothesis)
    lengths_hypothesis = np.array([len(batch_hypothesis[0])])
    
    batch_premise = prepare_for_pytorch(batch_premise)
    lengths_premise = prepare_for_pytorch(lengths_premise)

    batch_hypothesis = prepare_for_pytorch(batch_hypothesis)
    lengths_hypothesis = prepare_for_pytorch(lengths_hypothesis)

    model.eval()
    
    with torch.no_grad(): 
      
        if model_type == 1:
            output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis)
        elif model_type == 2:
            output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis,bidrectional=False)
        elif model_type == 3:
            output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis,bidirectional=True)      
        else:
            output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis,bidirectional=True,pool=True)
      

    predictions = output.argmax(1).item()

    print("The prediction is : " + inverse_target_dict[predictions])
  

