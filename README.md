# SNLI_Practical
This repository contains the code for the SNLI Practical assignment of the course Statistical Methods for Natural Language Semantics. The goal of this assignment was to replicate the Infersent paper by Facebook AI Research.

# Preprocessing Data :

The `data` folder contains a zip file which consists of `data_pickle.p`. This pickle file coomprises of the processed dataset and glove embeddings of the words in the dataset. Ensure that the pickle file is extracted and kept in the `data` folder before running any other tasks (train / infer / eval). 

# Executing the code :

### 
- train.py : This file accepts 3 flag i.e --model_type (Default 1) , --data_prune_size and --checkpoint_path.The flags are of integer , integer and string type respectively. `--model_type` indicates one of the four model types (i.e 1 - Baseline , 2 - LSTM , 3 - BILSTM , 4 - BILSTM with max pooling). `--data_prune_size` flag can be used to train / test on a smaller dataset. The size of train data can be specified as the argument.`--checkpoint_path` is a string indicating the path to the folder where the checkpoints will be saved. By default it points to a folder named 'Checkpoints'. Example for running this file :-
  
    ```sh
    $ python train.py --model_type 2 
 - eval.py : This file accepts 5 flags (`--model_type`,`--checkpoint_path`,`--dev_snli`,`--test_snli`, `--senteval`). The desciption of `--model_type` stays the same. `--checkpoint_path` is the name of the folder where checkpoints are saved. `--dev_snli`,`--test_snli`, `--senteval` are integers (1 or 0 as booleans) indicating the data to be evalued on. Example :-
    ```sh
    $ python eval.py --model_type 2 --dev_snli 1
Note : eval.py was intended for evaluation of the best model of each type (1,2,3,4) only. Hence the models in the checkpoints folder are expected to be named as 'Model_best_{model_type}.tar'. Eg - 'Model_best_1.tar' (for the baseline model). Evaluation of any checkpoint is possible by making minor changes to the code.

To perform transfer learning tests (senteval) , please ensure that the following steps are carried out in prior :-

        $ git clone https://github.com/facebookresearch/SentEval.git
        $ cd SentEval/
        $ python setup.py install
        $ cd data/downstream/
        $ ./get_transfer_data.bash
        $ cd ../../../
        $ mkdir pretrained
        $ cd pretrained
        $ wget http://nlp.stanford.edu/data/glove.840B.300d.zip
        $ unzip glove.840B.300d.zip
        $ cd ../
- infer.py : This is an interactive file. It asks the user to enter a premise and hypothesis and further displays the predictions. It accpets two flags i.e `--model_type` and `--checkpoint_path`.The `--checkpoint_path` is the path to the folder containing the best models of each type (1,2,3,4). The best models should be named as :'Model_best_{model_type}.tar'. Eg - 'Model_best_1.tar' (for the baseline model).
