import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from utils.dataset import BirdDataset
import torch.distributed as dist
from torchvision import transforms
from torch.utils.data import DataLoader
from models.main_stage_1 import G_Stage1
from models.main_stage_2 import G_Stage2, D_Stage2
from utils.helpers import weights_init
from train_stage2 import train_epoch
from torch.utils.data.distributed import DistributedSampler

device = torch.device('cpu')

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
dist_url = "tcp://localhost:12355/"

print("Initialize Process Group...")
#os.environ['MASTER_ADDR'] = 'localhost'
#os.environ['MASTER_PORT'] = '12355'
dist.init_process_group(backend=dist_backend, init_method=dist_url, rank=int(sys.argv[1]), world_size=world_size)

local_rank = int(sys.argv[2])
dp_device_ids = [local_rank]
torch.cuda.set_device(local_rank)

transform = transforms.Compose([
                transforms.RandomCrop(256),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = BirdDataset(dataDir = './data/bird_stack/', split='train', transform=transform, imgSize=256)

print('in')
train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
print('out')
tr_loader = DataLoader(dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                        num_workers=workers, pin_memory=False, sampler=train_sampler)

G1 = G_Stage1()
G1.load_state_dict(torch.load('./Result_stage1/netG_epoch_600.pth', map_location=torch.device('cpu')))
G1.eval()
netG = G_Stage2(G1)
netG.apply(weights_init)
netG = netG.cuda()
netD = D_Stage2()
netD.apply(weights_init)
netD = netD.cuda()

lr = 0.0002
optD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))
# remove the parameter from Stage-I generator
netG_param = []
for p in netG.parameters():
    if p.requires_grad:
        netG_param.append(p)
optG = optim.Adam(netG_param, lr=lr, betas=(0.5, 0.999))

fixed_noise = torch.rand(batch_size, 100, 1, 1).to(device)

if not (os.path.isdir('./Result_stage2/')):
    os.makedirs('Result_stage2')
num_epochs = 600

print("Initialize Models...")
netG = torch.nn.parallel.DistributedDataParallel(netG, device_ids=dp_device_ids, output_device=local_rank)
netD = torch.nn.parallel.DistributedDataParallel(netD, device_ids=dp_device_ids, output_device=local_rank)

# define loss function (criterion) and optimizer
# criterion = nn.CrossEntropyLoss().cuda()
# optimizer = torch.optim.SGD(model.parameters(), starting_lr, momentum=0.9, weight_decay=1e-4)


for epoch in range(num_epochs):
    # Set epoch count for DistributedSampler
    train_sampler.set_epoch(epoch)

    if epoch % 100 == 0 and epoch > 0:
        lr = lr*0.5
        for param_group in optG.param_groups:
            param_group['lr'] = lr
        for param_group in optD.param_groups:
            param_group['lr'] = lr

    # train for one epoch
    train_epoch(epoch, batch_size, tr_loader, netG, netD, fixed_noise, optD, optG, device)