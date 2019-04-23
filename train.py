#Adding path to sys for loading python files
import sys
import argparse
path_to_Models = './Models'
path_to_Modules = './Modules'
sys.path.append(path_to_Models)
sys.path.append(path_to_Modules)


from imports import *
from config import *
 
nltk.download('punkt') #Nltk tokenizer

#Loading dataset and glove embeddings - Prepreocessed / cleaned 
data_pickle_in = open("./data/data_pickle.p","rb")
data_dict_from_pickle = pickle.load(data_pickle_in)
train = data_dict_from_pickle['train']
dev = data_dict_from_pickle['dev']
test = data_dict_from_pickle['test']
vocab = data_dict_from_pickle['vocab']

#Loading methods
from helper_functions import *
from train_model import *

#Loading all models
from BaselineModel import BaselineModel
from Classification_Model import Classification_Model
from Lstms import lstm_class
from Model import Model

np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)


#Flags for command line arguements
FLAGS = None

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type = int, default = 1)
parser.add_argument('--data_prune_size',type = int)
parser.add_argument('--checkpoint_path', type = str,default = "./Checkpoints")

FLAGS, unparsed = parser.parse_known_args()

model_name = ""
model_type = FLAGS.model_type

if model_type == 1:
	model_name = "baseline"
elif model_type == 2:
	model_name = "lstm"
elif model_type == 3:
	model_name = "BI-LSTM"
else:
	model_name = "bilstm_max_pooling"
#SummaryWriter encapsulates everything
writer = SummaryWriter('runs/' + model_name)
vocab_matrix = create_vocab_matrix(vocab)
batch_size = 64


if FLAGS.data_prune_size:

	data_prune_size = FLAGS.data_prune_size
	train_prune = prune_data_set(train,data_prune_size)
	dev_prune = prune_data_set(dev,data_prune_size)
	test_prune = prune_data_set(test,data_prune_size)


model = Model(len(vocab),300,model_type,batch_size)
model.embed.weight.data.copy_(torch.from_numpy(vocab_matrix))
model.embed.weight.requires_grad = False
model = model.to(device)

print_model_params(model)

if FLAGS.data_prune_size:

	train_model(model,model_type,batch_size,1e-5,train_prune,dev_prune,test_prune,tb_writer=writer,ckpt_path=FLAGS.checkpoint_path)
else:
	train_model(model,model_type,batch_size,1e-5,train,dev,test,tb_writer=writer,ckpt_path=FLAGS.checkpoint_path)
