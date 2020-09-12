

print("Collect Inputs...")

# Batch Size for training and testing
batch_size = 32

# Number of additional worker processes for dataloading
workers = 2

# Number of epochs to train for
num_epochs = 2

# Starting Learning Rate
starting_lr = 0.1

# Number of distributed processes
world_size = 4

# Distributed backend type
dist_backend = 'nccl'

# Url used to setup distributed training
dist_url = "tcp://172.31.22.234:23456"