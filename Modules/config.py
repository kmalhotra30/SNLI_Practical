from imports import *
#Setting device
if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'