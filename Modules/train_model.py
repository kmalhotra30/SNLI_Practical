from imports import *
from config import *
from helper_functions import *
def train_model(model,model_type,batch_size,threshold,train,dev,test,**kwargs):
  
  if 'tb_writer' in kwargs:
    writer = kwargs['tb_writer']

  ckpt_path = kwargs['ckpt_path']

  epoch_no = 1
  learn_rate = 0.1
  weight_decay = 1e-3
  dev_set_acc_max = -1.0
  epoch_no = 1
  best_model_epoch = 1
  
  
  batch_train_accuracies = list()
  batch_train_losses = list()
  validation_accuracies = list()
  
  optimizer = optim.SGD(model.parameters(),lr=learn_rate,weight_decay=weight_decay)
    
  
  while(optimizer.param_groups[0]['lr']
     >= threshold):
  
    batch_train_accuracies.append([])
    batch_train_losses.append([])
    
    
    model.train()  
    shuffle(train) #Shuffling data set for gradient descent
    
    number_of_batches_that_fit_in_train = math.ceil(len(train)/ batch_size) # This is equal to number of iterations - 1
    
    #Load batch , forward pass , backward pass
    
    current_batch_start_index = 0
      
    for i in range(number_of_batches_that_fit_in_train):
      
      #Loading batches 
      batch_premise,lengths_premise,batch_hypothesis,lengths_hypothesis,batch_label = prepare_batch_with_idx(current_batch_start_index,train,batch_size)
      current_batch_start_index += batch_size
      
      batch_premise = prepare_for_pytorch(batch_premise)
      batch_hypothesis = prepare_for_pytorch(batch_hypothesis)
      batch_label = prepare_for_pytorch(batch_label)
      lengths_premise = prepare_for_pytorch(lengths_premise)
      lengths_hypothesis = prepare_for_pytorch(lengths_hypothesis)

      #Forward pass
      
      if model_type == 1:
        output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis)
      elif model_type == 2:
        
        output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis,bidirectional=False)
      elif model_type == 3:
        output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis,bidirectional=True)      
      else:
        output = model(batch_premise,batch_hypothesis,model_type,premise_lengths=lengths_premise,hypothesis_lengths=lengths_hypothesis,bidirectional=True,pool=True)
      
      
      #Compute loss
      no_of_sample_in_batch = batch_premise.size(0)
      criterion = nn.CrossEntropyLoss()
      loss = criterion(output.view([no_of_sample_in_batch,-1]),batch_label.view(-1)) # Does the softmax implicitly
      
      batch_data_set_for_accuracy = [batch_premise,lengths_premise,batch_hypothesis,lengths_hypothesis,batch_label]
      batch_accuracy = compute_batch_accuracy(model,model_type,batch_data_set_for_accuracy,batch_size)
    
      batch_train_accuracies[epoch_no - 1].append(batch_accuracy)
      
      batch_train_losses[epoch_no - 1].append(loss.item())
      optimizer.zero_grad()
      loss.backward()
          
      optimizer.step()
      
    train_avg_accuracy = np.mean(batch_train_accuracies[epoch_no - 1])
    train_avg_loss = np.mean(batch_train_losses[epoch_no - 1])
    print("Average Train Accuracy at Epoch " + str(epoch_no)+ ": ",train_avg_accuracy)
    print("Average Train Loss at Epoch " + str(epoch_no)+ ": " ,train_avg_loss)
    create_checkpoint(model,model.encoder,model_type,optimizer,train_avg_accuracy,train_avg_loss,epoch_no,ckpt_path)
    
    writer.add_scalar('Average Train Acccuracy', train_avg_accuracy, epoch_no)
    writer.add_scalar('Average Train Loss', train_avg_loss, epoch_no)
    #Evaluating on dev set
    model.eval()
    dev_acc = compute_accuracy(model,model_type,dev,batch_size)
    
    if dev_acc > dev_set_acc_max:
      
      best_model_epoch = epoch_no  
      dev_set_acc_max = dev_acc
    
    else:
      optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr']/ 5.0
         
    print("Validation Accuracy at Epoch " + str(epoch_no) + ": ",dev_acc)
    writer.add_scalar('Validation Accuracy', dev_acc, epoch_no)
    
    validation_accuracies.append(dev_acc)  
    epoch_no = epoch_no + 1
        


  #Evaluating on test set
  
  print("Best Model found at epoch : ",best_model_epoch)
  path = "Checkpoints/Model_" + str(model_type) + "_" + str(best_model_epoch) + ".tar"        
  loaded_checkpoint = torch.load(path)
  model.load_state_dict(loaded_checkpoint["model_state_dict"])

  test_acc = compute_accuracy(model,model_type,test,batch_size)
  print("Test Accuracy : ",test_acc)
  
  model_pickle_file = open('./Pickle/Model_' + str(model_type) + '_pickle.p','wb')
  data_dict = {'batch_train_accuracies': batch_train_accuracies,
               'batch_train_losses':batch_train_losses,
               'validation_accuracies':validation_accuracies,
               'test_acc':test_acc}

  
  pickle.dump(data_dict,model_pickle_file)
  model_pickle_file.close()

        
  
  