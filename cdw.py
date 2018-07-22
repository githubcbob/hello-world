import matplotlib
matplotlib.use('Agg')
    
import os 
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH=1
BATCH_SIZE=50
LR=0.001
DOWNLOAD_MNIST=False

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
        DOWNLOAD_MNIST=True

train_data=torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),

        download=DOWNLOAD_MNIST,
        )        

print(train_data.train_data.size())
print(train_data.train_labels.size())
plt.imshow(train_data.train_data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.train_labels[0])
plt.show()

train_loader=Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data=torchvision.datasets.MNIST(root='./mnist/', train=False)
test_x=torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000]/255.
test_y=test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
                nn.Con2d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=5,
                    stride=1,
                    padding=2,
                    ),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),

                )

        self.conv2=nn.Sequential(
                nn.Conv2d(16,32,5,1,2),
                nn.ReLU(),
                nn.MaxPoo12d(2),

                )
        self.out=nn.Linear(32*7*7,10)





print("finish")
