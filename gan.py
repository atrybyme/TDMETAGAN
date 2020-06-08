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
import torch.autograd as autograd


writer = SummaryWriter('runs_gan/exp-gan_1')

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples,custom_classes =None):
        loader = DataLoader(dataset)
        self.labels_list = []
        self.custom_classes = custom_classes
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
            if self.custom_classes!=None:
                classes = self.custom_classes
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
##Simulataneous training of different classes
n_classes = 1
#custom_classes = np.random.randint(0,10,n_classes)
custom_classes = np.array([9])
class_batch = 32
transform = transforms.Compose([torchvision.transforms.Resize(img_size),torchvision.transforms.ToTensor()])
dataset = torchvision.datasets.MNIST(root="C:/Users/shubh/Desktop/Work/Dataset", train=True,download=True,transform=transform)
balanced_batch_sampler = BalancedBatchSampler(dataset, n_classes, class_batch,custom_classes)


dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=balanced_batch_sampler)
##parameters

dim = 64
LAMBDA = 10
epochs= 1000

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


class netG(nn.Module):
    def __init__(self):
        super(netG,self).__init__()
        self.convt1 = nn.Linear(128,4*4*4*dim)
        self.act1 = nn.ReLU(True)
        self.convt2 = nn.ConvTranspose2d(4*dim,dim*2,5)
        self.act2 = nn.ReLU(True)
        self.convt3 = nn.ConvTranspose2d(dim*2,dim,5)
        self.act3 = nn.ReLU(True)
        self.convt4 = nn.ConvTranspose2d(dim,1,8,stride=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self,input):
        #input = input.view(-1,1,28,28)
        l1_1 = self.convt1(input)
        l1_2 = self.act1(l1_1)
        l1_2 = l1_2.view(-1,4*dim,4,4)
        l2_1 = self.convt2(l1_2)
        l2_2 = self.act2(l2_1)
        l2_2 = l2_2[:,:,:7,:7]
        l3_1 = self.convt3(l2_2)
        l3_2 = self.act3(l3_1)
        l4_1 = self.convt4(l3_2)
        output = self.sigmoid(l4_1)
        return output.view(-1,1,img_size,img_size)

fixed_batch = torch.randn(class_batch,128,device=device)

def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.randn(class_batch, 1).to(device)
    alpha = alpha.expand(real_data.size()[0],real_data.size()[2]*real_data.size()[3]).to(device)
    alpha = alpha.view(real_data.size())
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty
#criterion = nn.BCEWithLogitsLoss()
#metanet = netD().to(device)
#tasknet = netD().to(device)
#taskoptim = optim.Adam(tasknet.parameters(),lr=0.0001)
#metaoptim = optim.Adam(metanet.parameters(),lr=0.00001)

##training

generator = netG().to(device)
discriminator = netD().to(device)
optimizerD = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizerG = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.9))
iters=0
for epoch in range(epochs):
    for x_real,_ in iter(dataloader):
        discriminator.train()
        generator.zero_grad()
        discriminator.zero_grad()
        x_real = x_real.to(device)
        x_real = autograd.Variable(x_real)
        D_real = discriminator(x_real)
        D_real = D_real.mean()

        noise = torch.randn(class_batch,128).to(device)
        x_fake = generator(noise).detach()
        D_fake = discriminator(x_fake)
        D_fake = D_fake.mean()
        optimizerD.zero_grad()
        D_loss = D_fake - D_real + calc_gradient_penalty(discriminator,x_real,x_fake)
        D_loss.backward()
        optimizerD.step()


        generator.zero_grad()
        noise = torch.randn(class_batch,128).to(device)
        x_fake = generator(noise)
        G_fake = discriminator(x_fake)
        G_loss = -G_fake.mean()
        G_loss.backward()
        optimizerG.step()
        writer.add_scalar('D_loss',D_loss.item(),iters)
        writer.add_scalar('G_loss',G_loss.item(),iters)
        iters = iters+1
        

    if epoch%10==0:
        sample_batch = generator(fixed_batch).detach().cpu()
        grid = torchvision.utils.make_grid(sample_batch)
        writer.add_image('sample output',grid,epoch)
        torch.save(generator.state_dict(),'./checkpoints/gan_generator_'+str(epoch))
        torch.save(discriminator.state_dict(),'./checkpoints/gan_discriminator_'+str(epoch))


    






