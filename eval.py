import sys
import argparse
path_to_Models = './Models'
path_to_Modules = './Modules'
sys.path.append(path_to_Models)
sys.path.append(path_to_Modules)
batch_size = 64


from imports import * 
from config import *

#Loading methods
from helper_functions import *
from train_model import *

#Loading all models
from BaselineModel import BaselineModel
from Classification_Model import Classification_Model
from Lstms import lstm_class
from Model import Model

#Loading senteval
from Senteval import *
#Flags for command line arguements
FLAGS = None

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type = int, default = 1)
parser.add_argument('--checkpoint_path', type = str,default = "./Checkpoints")
parser.add_argument('--dev_snli',type=int,default=1)
parser.add_argument('--test_snli',type=int,default=0)
parser.add_argument('--senteval',type=int,default=0)

FLAGS, unparsed = parser.parse_known_args()

data_pickle_in = open("./data/data_pickle.p","rb")
data_dict_from_pickle = pickle.load(data_pickle_in)
test = data_dict_from_pickle['test']
dev = data_dict_from_pickle['dev']
    
vocab = data_dict_from_pickle['vocab']

vocab_matrix = create_vocab_matrix(vocab)

model_type = FLAGS.model_type
model = Model(len(vocab),300,model_type,batch_size)
model.embed.weight.data.copy_(torch.from_numpy(vocab_matrix))
model.embed.weight.requires_grad = False
model = model.to(device)

loaded_checkpoint = torch.load(FLAGS.checkpoint_path + '/Model_best_' + str(FLAGS.model_type) + '.tar')
model.load_state_dict(loaded_checkpoint["model_state_dict"])

if FLAGS.dev_snli == 1:

    dev_acc = compute_accuracy(model,model_type,dev,batch_size)
    print("Accuracy on Development Set : ",dev_acc)

if FLAGS.test_snli == 1:

    test_acc = compute_accuracy(model,model_type,test,batch_size)
    print("Accuracy on Test Set : ",test_acc)

if FLAGS.senteval == 1 :
    run_senteval(model_type,FLAGS.checkpoint_path)

