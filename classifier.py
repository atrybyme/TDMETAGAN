import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from itertools import cycle
from tensorboardX import SummaryWriter
writer = SummaryWriter('runs/exp-2')

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        loader = DataLoader(dataset)
        self.labels_list = []
        for _, label in loader:
            self.labels_list.append(label)
        self.labels = torch.LongTensor(self.labels_list)
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


img_size = 28
n_classes = 10
task_iters = 10
class_batch = 32
meta_iters = 100000
transform = transforms.Compose([torchvision.transforms.Resize(img_size),torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(root="C:/Users/shubh/Desktop/Work/Dataset", train=True,download=True,transform=transform)
balanced_batch_sampler = BalancedBatchSampler(dataset, n_classes, class_batch*task_iters)


dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=balanced_batch_sampler)
dataiter = cycle(dataloader)
##parameters

dim = 64


class netD(nn.Module):
    def __init__(self):
        super(netD,self).__init__()
        self.conv1 = nn.Conv2d(1,dim,5,stride=2,padding=2)
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(dim,dim*2,5,stride=2,padding=2)
        self.act2 = nn.ReLU(True)
        self.conv3 = nn.Conv2d(dim*2,dim*4,5,stride=2,padding=2)
        self.act3 = nn.ReLU(True)
        self.fc1 = nn.Linear(4*4*4*dim,1)

    def forward(self,input):
        #input = input.view(-1,1,28,28)
        l1_1 = self.conv1(input)
        l1_2 = self.act1(l1_1)
        l2_1 = self.conv2(l1_2)
        l2_2 = self.act2(l2_1)
        l3_1 = self.conv3(l2_2)
        l3_2 = self.act3(l3_1)

        l4_1 = self.fc1(l3_2.view(-1,4*4*4*dim))
        return l4_1.view(-1)




criterion = nn.BCEWithLogitsLoss()
metanet = netD().to(device)
tasknet = netD().to(device)
taskoptim = optim.Adam(tasknet.parameters(),lr=0.0001)
metaoptim = optim.Adam(metanet.parameters(),lr=0.00001)

logits = torch.cat((torch.ones(class_batch),torch.zeros(class_batch))).to(device)
##training


for i in range(100000):
    tasknet.train()
    tasknet.load_state_dict(metanet.state_dict())
    tasknet.zero_grad()
    train_batch,_ = next(dataiter)
    train_batch = train_batch.to(device)
    #print("alfa")
    class_0 = np.random.randint(0,10)
    class_1 = np.random.randint(0,10)
    while class_0==class_1:
        class_1 = np.random.randint(1,10)
    train_batch_1 = train_batch[class_0*task_iters*class_batch:class_0*task_iters*class_batch+task_iters*class_batch].view(-1,class_batch,1,img_size,img_size)
    train_batch_2 = train_batch[class_1*task_iters*class_batch:class_1*task_iters*class_batch+task_iters*class_batch].view(-1,class_batch,1,img_size,img_size)
    batch_loss = 0
    for j in range(task_iters):
        taskoptim.zero_grad()
        x_train = torch.cat((train_batch_1[j],train_batch_2[j]),0).to(device)
        output = tasknet(x_train)
        task_loss = criterion(output,logits)
        task_loss.backward()
        taskoptim.step()
        batch_loss = batch_loss + task_loss.item()
        writer.add_scalar('instantaneous_loss',task_loss.item(),i*1000+j)
    ##training of meta net
    for meta_w,w in zip(metanet.parameters(),tasknet.parameters()):
        diff = meta_w - w
        meta_w.grad = diff
    writer.add_scalar('meta_loss',batch_loss,i)
    metaoptim.step()
    if i%1000==0:
        torch.save(metanet.state_dict(),"./checkpoints/alfa_exp2"+str(i))
        print(str(i)+ " iter and loss    : "+ str(batch_loss))


    






