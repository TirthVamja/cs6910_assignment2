# Part B - FineTuning GoogLeNet Model with Weights & Biases Integration
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
-trd: Path to training data directory (relative path)
-ted: Path to test data directory (relative path)
```

### Sample Run
#### Input
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
