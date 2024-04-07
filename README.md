# CS6910 Assignment 2 - Convolutional Neural Network (CNN)

**WandB Report Link** : [Tirth_Vamja_CS23M070](https://wandb.ai/cs23m070/cs6910_assignment2/reports/Tirth-s-CS23M070-CS6910-Assignment-2--Vmlldzo3NDAwNDQ2)

**Task** : Develop a CNN based image classifier for [iNaturalist](https://storage.googleapis.com/wandb_datasets/nature_12K.zip) dataset from scratch using PyTorch.

**Meta Data about Dataset**
1. Total Images --> 12K
2. Training Images --> 10K
3. Testing Images --> 2K
4. No. of Classes --> 10 ('Amphibia','Animalia','Arachnida','Aves','Fungi','Insecta','Mammalia','Mollusca','Plantae','Reptilia')






# Part A - Training CNN Model from Scratch
In this part, I have trained a Convolution Neural Network from Scratch using PyTorch.
#### Related Files ####
```
train.py
assignment2.ipynb
```

 - **Command-line arguments**: Control training parameters through command-line arguments using argparse.
 - **Train/Validation/Test Split**: Splits the dataset into training, validation, and testing sets with data augmentation for the training set.
 - **Data Augmentation**: Augment data randomly at every epoch using horizontal_flip, rotate, blur functions.
 - **Filter Organization**: Strategy for filters at different layers (same number of filters, reduce filters by factor of 2 at every layers and increase filters by factor of 2).
 - **Weights & Biases Integration**: Tracks training progress and logs metrics (loss, accuracy) using Weights & Biases for visualization and comparison.


### Prerequisites:
- Python 3.6 or later
- PyTorch (https://pytorch.org/)
- torchvision (https://pypi.org/project/torchvision/)
- Weights & Biases account (https://wandb.ai/site)

### Installation:
1. Clone this repository.
2. Install required libraries.


### Running the Script

#### Bash
```
python train.py -e <epochs> -b <batch_size> ..... -wp <wandb_project> -we <wandb_entity> -trd <train_data_dir> -ted <test_data_dir>
```

#### Arguments:
```
-e: Number of epochs for training (default: 10)
-b: Batch size for training (default: 32)
-lr: Learning rate used to optimize model parameters (default=0.001)
-a: Activation function (default='relu', choices=["selu", "mish", "relu"])
-f: Number of filters in 1st layer (default=64)
-fo: Number of Filters down the network (default='half', choices=["same", "half", "double"])
-da: Augment data randomly (default=True, choices=[True, False])
-dr: Dropout probability (default=0.2)
-bn: Whether to use Batch Normalization or not (default=False, choices=[True, False])
-dn: Number of neurons in last fc layer (default=1024)
-ks: Size of convolution kernel filter (default=3)
-dim: Resize dimension of input images (default=256)
-wp: Weights & Biases project name (default: dl_assignment_2)
-we: Weights & Biases entity (default: dl_assignment_2)
-trd: Relative Path to training data directory (default: ./data/train)
-ted: Relative Path to test data directory (default: ./data/val)
```

### Sample Run
#### Bash
```
python train.py -e 5
```
#### Output
```
Epoch --> 1
Train Loss --> 2.2898917198181152
Train Accuracy --> 0.12075
Validation Loss --> 2.279803514480591
Validation Accuracy --> 0.1455727863931966
###################################################################
Epoch --> 2
Train Loss --> 2.2340314388275146
Train Accuracy --> 0.18075
Validation Loss --> 2.191962957382202
Validation Accuracy --> 0.2191095547773887
###################################################################
Epoch --> 3
Train Loss --> 2.161862373352051
Train Accuracy --> 0.208625
Validation Loss --> 2.135850667953491
Validation Accuracy --> 0.23411705852926462
###################################################################
Epoch --> 4
Train Loss --> 2.1284983158111572
Train Accuracy --> 0.228
Validation Loss --> 2.13728928565979
Validation Accuracy --> 0.22961480740370185
###################################################################
Epoch --> 5
Train Loss --> 2.123326063156128
Train Accuracy --> 0.22425
Validation Loss --> 2.1055619716644287
Validation Accuracy --> 0.2366183091545773
###################################################################
Total Testing Image --> 2000
Total Correctly Predicted --> 476.0
Testing Accuracy --> 23.799999999999997
```




# Part B - FineTuning GoogLeNet Model with WandB Integration
#### Related Files ####
```
train_partB.py
assignment2_partB.ipynb
```

This part implements fine-tuning of the pre-trained GoogLeNet model for image classification with the following functionalities:

 - **Command-line arguments**: Control training parameters through command-line arguments using argparse.
 - **Train/Validation/Test Split**: Splits the dataset into training, validation, and testing sets with data augmentation for the training set.
 - **Freezing Layers**: Freezes layers of the pre-trained model for transfer learning based on different strategies (all freeze, partial freeze based on layer number, no freeze).
 - **Weights & Biases Integration**: Tracks training progress and logs metrics (loss, accuracy) using Weights & Biases for visualization and comparison.


### Prerequisites:
- Python 3.6 or later
- PyTorch (https://pytorch.org/)
- torchvision (https://pypi.org/project/torchvision/)
- Weights & Biases account (https://wandb.ai/site)

### Installation:
1. Clone this repository.
2. Install required libraries.


### Running the Script

#### Bash
```
python train_partB.py -e <epochs> -b <batch_size> -s <strategy> -lf <layers_to_freeze> -wp <wandb_project> -we <wandb_entity> -trd <train_data_dir> -ted <test_data_dir>
```

#### Arguments:
```
-e: Number of epochs for training (default: 10)
-b: Batch size for training (default: 32)
-s: Freezing strategy (all_freeze, k_freeze, no_freeze) (default: k_freeze)
-lf: Number of layers to freeze for k_freeze strategy (default: 15)
-wp: Weights & Biases project name (default: dl_assignment_2)
-we: Weights & Biases entity (default: dl_assignment_2)
-trd: Relative Path to training data directory (default: ./data/train)
-ted: Relative Path to test data directory (default: ./data/val)
```

### Sample Run
#### Bash
```
python train_partB.py -e 5
```
#### Output
```
Epoch --> 1
Train Loss --> 3.175224781036377
Train Accuracy --> 0.404875
Validation Loss --> 2.38615345954895
Validation Accuracy --> 0.47573786893446723
-----------------------------------------------------------
Epoch --> 2
Train Loss --> 1.6527040004730225
Train Accuracy --> 0.499
Validation Loss --> 1.4728065729141235
Validation Accuracy --> 0.4822411205602801
-----------------------------------------------------------
Epoch --> 3
Train Loss --> 1.3008484840393066
Train Accuracy --> 0.559875
Validation Loss --> 1.1031557321548462
Validation Accuracy --> 0.6678339169584793
-----------------------------------------------------------
Epoch --> 4
Train Loss --> 1.168255090713501
Train Accuracy --> 0.65625
Validation Loss --> 1.161903977394104
Validation Accuracy --> 0.6743371685842922
-----------------------------------------------------------
Epoch --> 5
Train Loss --> 1.0856289863586426
Train Accuracy --> 0.673
Validation Loss --> 1.1095525026321411
Validation Accuracy --> 0.6828414207103551
-----------------------------------------------------------
Total Testing Image --> 2000
Total Correctly Predicted --> 1397.0
Testing Accuracy --> 69.85
```
