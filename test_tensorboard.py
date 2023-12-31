import torch
from torch.utils.tensorboard import SummaryWriter
import time
writer = SummaryWriter(log_dir='Learning-PyTorch/runs')
global_step = 400
for epoch in range(400,600):
    # ... your training code ...
    train_loss = torch.randint(1000, 2000, (1,))/(epoch+1)
    val_loss = torch.randint(100, 200, (1,))/(epoch+1)
    # Add scalar values to TensorBoard
    writer.add_scalar('Loss/Train', train_loss, global_step=global_step)
    writer.add_scalar('Loss/Validation', val_loss, global_step=global_step)
    print(global_step)
    time.sleep(1)
    # Add other information as needed (e.g., images, histograms, etc.)

    global_step += 1

writer.close()

#    tensorboard --logdir=./Learning-PyTorch/runs
# terminal :  F:\shreeradha> 
#how to run?
'''
create the directory /Learning-PyTorch/runs its empty now 
run pyton file and run python file in dedicated terminal  2 dummy terminals clear them
now in python file terminal run   tensorboard --logdir=./Learning-PyTorch/runs
go to localhost:6006 in browser
run python file in dedicated terminal
use reload button in tensorboard to load the incoming data
'''