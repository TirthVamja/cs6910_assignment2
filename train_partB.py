############ Import Libraries ############
from torch.nn import Sequential, Module, ReLU, Conv2d, Linear, MaxPool2d, LogSoftmax, NLLLoss, Dropout, BatchNorm2d, LeakyReLU, GELU, SELU, Mish, CrossEntropyLoss
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch import flatten, float, no_grad
from torch.optim import Adam
import torch
import wandb
import math
import sys, argparse



################################# Parse Arguments #################################

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Arguments')
    parser.add_argument('-wp', '--wandb_project', type=str, default='dl_assignment_2', help='Project name used to track experiments in Weights & Biases dashboard')
    parser.add_argument('-we', '--wandb_entity', type=str, default='dl_assignment_2', help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
    parser.add_argument('-e', '--epochs', type=int, default=10, help='Number of epochs to train neural network.')
    parser.add_argument('-b', '--batch_size', type=int, default=32, help='Batch size used to train neural network.')
    parser.add_argument('-s', '--strategy', type=str, default='k_freeze', choices=["all_freeze", "k_freeze", "no_freeze"], help='Strategy for freezing the layers (no weight update)')
    parser.add_argument('-lf', '--layers_to_freeze', type=int, default=15, help='Number of layers to freeze if k_freeze strategy is used.')
    parser.add_argument('-trd', '--train_data_dir', type=str, default='./data/train', help='Relative path of training directory')
    parser.add_argument('-ted', '--test_data_dir', type=str, default='./data/val', help='Relative path of test directory')
    return parser.parse_args()





################################# Plotting Confusion Matrix #################################
def get_data(param, type):
    if(type.lower() == 'train'):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=12),
            transforms.ColorJitter(),
            transforms.CenterCrop(224),
            transforms.ToTensor(), 
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  
        ])
        tdataset = datasets.ImageFolder(root=param['train_data_dir'], transform=transform)
        total = len(tdataset)
        train_sample = math.ceil(total*(0.8))
        val_sample = total-train_sample
        # print(total, train_sample, val_sample)
        train_dataset, validation_dataset = torch.utils.data.random_split(tdataset, [train_sample, val_sample])
        train_dataloader = DataLoader(train_dataset, batch_size=param['batch_size'], shuffle=True)
        validation_dataloader = DataLoader(validation_dataset, batch_size=param['batch_size'], shuffle=False)
        return train_dataloader, validation_dataloader
    
    else:
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(), 
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])  
        ])
        test_dataset = datasets.ImageFolder(root=param['test_data_dir'], transform=transform)
        test_dataloader = DataLoader(test_dataset, batch_size=param['batch_size'])
        return test_dataloader
    




################################# Main Function  #################################
def train():

    wandb.init()
    param = wandb.config
    wandb.run.name = f'GoogLeNet_strategy_{param.strategy}_batchSz_{param.batch_size}_epochs_{param.epochs}_layersToFreeze_{param.layers_to_freeze}'

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pmodel = models.googlenet(pretrained=True)

    if(param['strategy'] == 'all_freeze'):
        num_features = pmodel.fc.in_features
        pmodel.fc = Linear(num_features, 10)
        for name, par in pmodel.named_parameters():
            if name not in ['fc.weight', 'fc.bias']:
                par.requires_grad = False

    elif(param['strategy'] == 'k_freeze'):
        layers_to_freeze = list(pmodel.children())[:param['layers_to_freeze']]
        for x in layers_to_freeze:
            for y in x.parameters():
                y.requires_grad = False
        num_features = pmodel.fc.in_features
        pmodel.fc = Linear(num_features, 10)
    
    else:
        num_features = pmodel.fc.in_features
        pmodel.fc = Linear(num_features, 10)
    
    
    pmodel = pmodel.to(device)
    optimizer = Adam(pmodel.parameters())
    loss_function = CrossEntropyLoss()
    train_data_loader, validation_data_loader = get_data(param, 'train')

    for epo in range(param['epochs']):
        totalTrainLoss = 0
        totalValLoss = 0
        trainCorrect = 0
        valCorrect = 0
        train_counter=0
        validation_counter=0
        pmodel.train()
        for (image, label) in train_data_loader:
            (image, label) = (image.to(device), label.to(device))
            prediction = pmodel(image)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            totalTrainLoss += loss
            trainCorrect += (prediction.argmax(1) == label).type(float).sum().item()
            train_counter+=1

        ## Validation
        pmodel.eval()
        with no_grad():
            for (image, label) in validation_data_loader:
                (image, label) = (image.to(device), label.to(device))
                pred = pmodel(image)
                loss = loss_function(pred, label)
                totalValLoss += loss
                valCorrect += (pred.argmax(1) == label).type(float).sum().item()
                validation_counter += 1


        tr_ls = (totalTrainLoss/train_counter).cpu().detach().numpy()
        tr_acc = trainCorrect/len(train_data_loader.dataset)
        val_ls = (totalValLoss/validation_counter).cpu().detach().numpy()
        val_acc = valCorrect/len(validation_data_loader.dataset)


        print(f"Epoch --> {epo}")
        print(f"Train Loss --> {tr_ls}")
        print(f"Train Accuracy --> {tr_acc}")
        print(f"Validation Loss --> {val_ls}")
        print(f"Validation Accuracy --> {val_acc}")
        print("-----------------------------------------------------------")
        
        lg={
            'epoch': epo,
            'tr_accuracy': tr_acc,
            'val_accuracy': val_acc,
            'tr_loss': tr_ls,
            'val_loss': val_ls
        }
        wandb.log(lg)







if __name__ == "__main__": 
    
    inp = vars(parse_arguments())
    sweep_config = {
        "method": "grid",
        "name": "PartB GoogLeNet Sweep",
        "metric": {"goal": "maximize", "name": "val_accuracy"},
        "parameters": {
            "batch_size":{"values": [inp['batch_size']]},
            "epochs":{"values": [inp['epochs']]},
            "strategy":{"values": [inp['strategy']]},
            "layers_to_freeze": {"values": [inp['layers_to_freeze']]},
            "train_data_dir":{"values": [inp['train_data_dir']]},
            "test_data_dir":{"values": [inp['test_data_dir']]}
        }
    }


    sweep_id = wandb.sweep(sweep_config, project=inp['wandb_project'])
    wandb.agent(sweep_id, function=train)
    wandb.finish()