############ Import Libraries ############
from torch.nn import Module, ReLU, Conv2d, Linear, MaxPool2d, LogSoftmax, NLLLoss, Dropout, BatchNorm2d, LeakyReLU, GELU, SELU, Mish
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import flatten, float, no_grad
from torch.optim import Adam
import torch
import wandb
import math
import matplotlib.pyplot as plt
import numpy as np
import sys, argparse


################################# Parse Arguments #################################

def parse_arguments():
	parser = argparse.ArgumentParser(description='Training Arguments')
	parser.add_argument('-wp', '--wandb_project', type=str, default='dl_assignment_2', help='Project name used to track experiments in Weights & Biases dashboard')
	parser.add_argument('-we', '--wandb_entity', type=str, default='dl_assignment_2', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
	parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train neural network.')
	parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size used to train neural network.')
	parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate used to optimize model parameters')
	parser.add_argument('-a', '--activation', type=str, default='relu', choices=["selu", "mish", "relu"], help='Activation function choice: ["selu", "mish", "relu"]')
	parser.add_argument('-f', '--filters', type=int, default=64, help='Number of filters in 1st layer')
	parser.add_argument('-fo', '--filter_org', type=str, default='half', choices=["same", "half", "double"], help='Number of Filters down the network')
	parser.add_argument('-da', '--data_augmentation', type=bool, default=True, choices=[True, False], help='Augment data randomly.')
	parser.add_argument('-dr', '--dropout', type=float, default=0.2, help='Dropout probability to be used.')
	parser.add_argument('-bn', '--batch_normalization', type=bool, default=False, choices=[True, False], help='whether to use Batch Normalization or not')
	parser.add_argument('-dn', '--dense_neurons', type=int, default=1024, help='Number of neurons in last fc layer')
	parser.add_argument('-ks', '--kernel_size', type=int, default=3, help='Size of convolution kernel filter')
	parser.add_argument('-dim', '--dim', type=int, default=256, help='Resize dimension of input images')
	parser.add_argument('-trd', '--train_data_dir', type=str, default='./data/train', help='Relative path of training directory')
	parser.add_argument('-ted', '--test_data_dir', type=str, default='./data/val', help='Relative path of test directory')
	
	return parser.parse_args()





################################# Load Data #################################

def get_data(param, type):
    if(type.lower() == 'train'):
        if param['data_augmentation']:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomRotation(degrees=12),
                transforms.Resize((param['dim'],param['dim'])),
                transforms.ToTensor(), 
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((param['dim'],param['dim'])),
                transforms.ToTensor(), 
                transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  
            ])

        tdataset = datasets.ImageFolder(root=param['train_data_dir'], transform=transform)
        total = len(tdataset)
        train_sample = math.ceil(total*(0.8))
        val_sample = total-train_sample
        train_dataset, validation_dataset = torch.utils.data.random_split(tdataset, [train_sample, val_sample])
        train_dataloader = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=param['batch_size'], shuffle=False)
        return train_dataloader, validation_dataloader
    
    else:
        transform = transforms.Compose([
            transforms.Resize((param['dim'],param['dim'])),
            transforms.ToTensor(), 
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  
        ])
        test_dataset = datasets.ImageFolder(root=param['test_data_dir'], transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=param['batch_size'])
        return test_dataloader

    


################################# Convolution Network Training Class #################################

class CNN(Module):
    def __init__(self, param):
        super(CNN, self).__init__()
        self.param=param
        self.data_augmentation = param['data_augmentation']
        self.dropout = param['dropout']
        self.act = self.getActivation(param['activation'])
        self.filters = self.filter_logic(param['filters'], param['filter_org'])
        self.conv_ks = param['conv_kernel_size']
        self.dim = param['dim']
        self.bn = param['batch_normalization']
        self.dense_neurons = param['dense_neurons']


        ####### Layer 1 #######
        curr_dim = self.dim
        self.conv1 = Conv2d(kernel_size=(self.conv_ks,self.conv_ks), in_channels=3, out_channels=self.filters[0])
        curr_dim -= (self.conv_ks-1)
        self.act1 = self.act
        if(self.bn): self.bn1 = BatchNorm2d(self.filters[0])
        self.pool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        curr_dim //= 2
        self.dropout1 = Dropout(p=self.dropout)

        ####### Layer 2 #######
        self.conv2 = Conv2d(kernel_size=(self.conv_ks,self.conv_ks), in_channels=self.filters[0], out_channels=self.filters[1])
        curr_dim -= (self.conv_ks-1)
        self.act2 = self.act
        if(self.bn): self.bn2 = BatchNorm2d(self.filters[1])
        self.pool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        curr_dim //= 2
        self.dropout2 = Dropout(p=self.dropout)

        ####### Layer 3 #######
        self.conv3 = Conv2d(kernel_size=(self.conv_ks,self.conv_ks), in_channels=self.filters[1], out_channels=self.filters[2])
        curr_dim -= (self.conv_ks-1)
        self.act3 = self.act
        if(self.bn): self.bn3 = BatchNorm2d(self.filters[2])
        self.pool3 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        curr_dim //= 2
        self.dropout3 = Dropout(p=self.dropout)

        ####### Layer 4 #######
        self.conv4 = Conv2d(kernel_size=(self.conv_ks,self.conv_ks), in_channels=self.filters[2], out_channels=self.filters[3])
        curr_dim -= (self.conv_ks-1)
        self.act4 = self.act
        if(self.bn): self.bn4 = BatchNorm2d(self.filters[3])
        self.pool4 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        curr_dim //= 2
        self.dropout4 = Dropout(p=self.dropout)


        ####### Layer 5 #######
        self.conv5 = Conv2d(kernel_size=(self.conv_ks,self.conv_ks), in_channels=self.filters[3], out_channels=self.filters[4])
        curr_dim -= (self.conv_ks-1)
        self.act5 = self.act
        if(self.bn): self.bn5 = BatchNorm2d(self.filters[4])
        self.pool5 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        curr_dim //= 2
        self.dropout5 = Dropout(p=self.dropout)

    
        ####### Fully Connected Layer #######
        self.dense_neurons = curr_dim * curr_dim * self.filters[4]
        self.fc1 = Linear(in_features=(curr_dim * curr_dim * self.filters[4]), out_features=self.dense_neurons)  # How to calculate dimension of filters at previous level
        self.act6 = self.act
        self.dropout6 = Dropout(p=0.5)
        

        ####### Output Layer #######
        self.out = Linear(in_features=self.dense_neurons, out_features=10)
        self.act7 = LogSoftmax(dim=1)


    def getActivation(self, act):
        act = act.lower()
        if(act == 'relu'):
            return ReLU()
        elif(act == 'leakyrelu'):
            return LeakyReLU()
        elif(act == 'gelu'):
            return GELU()
        elif(act == 'selu'):
            return SELU()
        elif(act == 'mish'):
            return Mish()
    

    def filter_logic(self, filter, org):
        level = []
        org = org.lower()
        if org == 'same':
            level = [filter for i in range(5)]
        elif org == 'double':
            level = [filter*pow(2,i) for i in range(5)]
        elif org == 'half':
            level = [max(filter//pow(2,i),1) for i in range(5)]
        return level

    

    def forward(self, r):

        r=self.conv1(r)
        r=self.act1(r)
        if(self.bn): r=self.bn1(r)
        r=self.pool1(r)
        r=self.dropout1(r)

        r=self.conv2(r)
        r=self.act2(r)
        if(self.bn): r=self.bn2(r)
        r=self.pool2(r)
        r=self.dropout2(r)

        r=self.conv3(r)
        r=self.act3(r)
        if(self.bn): r=self.bn3(r)
        r=self.pool3(r)
        r=self.dropout3(r)

        r=self.conv4(r)
        r=self.act4(r)
        if(self.bn): r=self.bn4(r)
        r=self.pool4(r)
        r=self.dropout4(r)

        r=self.conv5(r)
        r=self.act5(r)
        if(self.bn): r=self.bn5(r)
        r=self.pool5(r)
        r=self.dropout5(r)

        r=flatten(r,1)
        r=self.fc1(r)
        r=self.act6(r)
        r=self.dropout6(r)
        
        r=self.out(r)
        output=self.act7(r)

        return output
        




################################# Main Function  #################################
def train():
    wandb.init()
    param = wandb.config
    wandb.run.name = f'fltr_{param.filters}_fltrOrg_{param.filter_org}_dataAug_{param.data_augmentation}_batchNorm_{param.batch_normalization}_act_{param.activation}_batchSz_{param.batch_size}'

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") ## For Apple Silicon GPU.. use CUDA for Nvidia GPU
    model = CNN(param).to(device)
    optimizer = Adam(model.parameters(), lr=param['learning_rate'])
    loss_function = NLLLoss()
    train_data_loader, validation_data_loader = get_data(param, 'train')
    

    for epo in range(param['epochs']):
        model.train()
        totalTrainLoss = 0
        totalValLoss = 0
        trainCorrect = 0
        valCorrect = 0
        train_counter=0
        validation_counter=0
        for (image, label) in train_data_loader:
            (image, label) = (image.to(device), label.to(device))
            prediction = model(image)
            loss = loss_function(prediction, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            totalTrainLoss += loss
            trainCorrect += (prediction.argmax(1) == label).type(float).sum().item()
            train_counter+=1
            # print(train_counter)
        
        with no_grad():
            model.eval()
            for (image, label) in validation_data_loader:
                (image, label) = (image.to(device), label.to(device))
                pred = model(image)
                totalValLoss += loss_function(pred, label)
                valCorrect += (pred.argmax(1) == label).type(float).sum().item()
                validation_counter+=1

        tr_ls = (totalTrainLoss/train_counter).cpu().detach().numpy()
        tr_acc = trainCorrect/len(train_data_loader.dataset)
        val_ls = (totalValLoss/validation_counter).cpu().detach().numpy()
        val_acc = valCorrect/len(validation_data_loader.dataset)
        print(f"Epoch --> {epo}")
        print(f"Train Loss --> {tr_ls}")
        print(f"Train Accuracy --> {tr_acc}")
        print(f"Validation Loss --> {val_ls}")
        print(f"Validation Accuracy --> {val_acc}")
        print("###################################################################")
        
        lg={
            'epoch': epo,
            'tr_accuracy': tr_acc,
            'val_accuracy': val_acc,
            'tr_loss': tr_ls,
            'val_loss': val_ls
        }
        wandb.log(lg)




    # Testing Data after all epochs are completed ...
    test_data_loader = get_data(param, 'test')
    tstCorrect = 0
    tstCounter = 0
    y = []
    y_pred = []
    with no_grad():
        model.eval()
        for (image, label) in test_data_loader:
            (image, label) = (image.to(device), label.to(device))
            pred = model(image)
            ll = label.tolist()
            y.extend(ll)
            y_pred.extend(pred.argmax(1).tolist())
            # print(pred)
            tstCorrect += (pred.argmax(1) == label).type(float).sum().item()
            tstCounter+=len(ll)

    print(f"Total Testing Image --> {tstCounter}")
    print(f"Total Correctly Predicted --> {tstCorrect}")
    print(f"Testing Accuracy --> {(tstCorrect/tstCounter)*100}")
    




if __name__ == "__main__": 
    
    inp = vars(parse_arguments())
    sweep_config = {
        "method": "grid", 
        "name": "Q1 Sweep",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "data_augmentation":{"values": [inp['data_augmentation']]},
            "batch_normalization":{"values": [inp['batch_normalization']]},
            "filters":{"values": [inp['filters']]},
            "filter_org":{"values": [inp['filter_org']]},
            "dropout":{"values": [inp['dropout']]},
            "activation":{"values": [inp['activation']]},
            "batch_size":{"values": [inp['batch_size']]},
            "learning_rate":{"values": [inp['learning_rate']]},
            "epochs":{"values": [inp['epochs']]},
            "dim":{"values": [inp['dim']]},
            "conv_kernel_size":{"values": [inp['kernel_size']]},
            "dense_neurons":{"values": [inp['dense_neurons']]},
            "train_data_dir":{"values": [inp['train_data_dir']]},
            "test_data_dir":{"values": [inp['test_data_dir']]}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project=inp['wandb_project'])
    wandb.agent(sweep_id, function=train)
    wandb.finish()