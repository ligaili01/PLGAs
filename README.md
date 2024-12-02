Accurately estimating the binding strength between proteins and ligands is fundamental in the field of pharmaceutical research and innovation. Previous research has largely concentrated on 1D or 2D molecular descriptors, often neglecting the pivotal 3D features of molecules that profoundly impact drug properties and target binding. This oversight has resulted in diminished predictive performance in molecule-related analyses. A comprehensive grasp of molecular properties necessitates the integration of both local and global molecular information. In this paper, we introduce a deep-learning model, termed PLGAs, which represents molecular systems as graphs based on the three-dimensional configurations of protein-ligand complexes. PLGAs consist of two components: Graph Convolution Networks (GCN) and a Global Attention Mechanism (GAM) network. Specifically, GCNs learn both the graph structure and node attribute information, capturing local and global information to better represent node features. GAM is then used to gather interactive edges by reducing information loss and amplifying global interactions. PLGAs were tested on the standard PDBbind refined set (v.2019) and core set (v.2016). The model demonstrated a Spearman's correlation coefficient of 0.823 on the refined set and an RMSE (Root Mean Square Error) of 1.211 kcal/mol between experimental and predicted affinities on the core set, surpassing several advanced contemporary binding affinity prediction methods. We further evaluated the efficacy of various components within our model, and the marked improvements in accuracy underscore the potential of PLGAs to significantly enhance the drug development process



Certainly! Below is a sample documentation for the provided code, which outlines its functionality, setup, and usage. This documentation assumes that the user has a basic understanding of Python and PyTorch.

---

# A spatial-temporal graph attention network]{A spatial-temporal graph attention network for protein-ligand binding affinity prediction  based
on molecular geometry

This project implements a Graph Neural Network (GNN) to predict binding affinities using the PyTorch Geometric library. The model is trained and evaluated using RMSE and correlation coefficients (Pearson and Spearman) as metrics.

## Code Structure

- **`train_loop`**: A function to train the GNN model on the training data.
- **`test`**: A function to evaluate the GNN model on validation or test data.
- **`plot_corr`**: A utility function to plot the correlation between true and predicted values.
- **`save_weights`**: Saves the model weights to a specified directory.
- **`train`**: A function to handle the training process, including loading data, training the model, and saving the best model weights.
- **`GNN_LBA`**: A class defining the GNN model architecture.
- **`GAM_Attention`**: A class implementing a Global Attention Mechanism used within the GNN model.

## Installation

Ensure that you have the following Python libraries installed:

- torch
- torch-geometric
- numpy
- pandas
- matplotlib
- seaborn
- scipy

You can install them using pip:

```bash
pip install torch torch-geometric numpy pandas matplotlib seaborn scipy
```

## Usage

### Command Line Arguments

The script can be executed with the following command line arguments:

- `--data_dir`: Directory containing the dataset.
- `--mode`: Mode of operation, either 'train' or 'test'.
- `--batch_size`: Batch size for training and evaluation.
- `--hidden_dim`: Hidden dimension size for the GNN layers.
- `--num_epochs`: Number of epochs for training.
- `--learning_rate`: Learning rate for the optimizer.
- `--log_dir`: Directory to save logs and model weights.
- `--seqid`: Sequence identity threshold.
- `--precomputed`: Flag to use precomputed datasets.

### Training the Model

To train the model, execute the script with the `--mode` set to 'train':

```bash
python script.py --data_dir path/to/data --mode train --log_dir my_log_dir
```

### Testing the Model

To test the model, execute the script with the `--mode` set to 'test':

```bash
python script.py --data_dir path/to/data --mode test --seqid 30
```

## Model Architecture

The `GNN_LBA` model consists of several graph convolutional layers with batch normalization, followed by a global attention mechanism (`GAM_Attention`) for feature aggregation. The final output is processed through two fully connected layers to predict binding affinity.

## Attention Mechanism

The `GAM_Attention` class implements a channel and spatial attention mechanism, enhancing the model's ability to focus on important features in the input data.

## Logging and Outputs

- Training and validation logs are printed to the console, showing RMSE, Pearson R, and Spearman R for each epoch.
- Best model weights are saved in the specified log directory.
- Predicted and true values are stored in text files during testing for further analysis.

## Contributions

This code was developed and maintained by [Your Name]. Contributions and feedback are welcome to improve the functionality and usability of the code.

---

This documentation provides a comprehensive overview of the code, its functionality, and how to use it for training and testing a GNN model for binding affinity prediction. Adjust the content as necessary to reflect any additional details specific to your implementation or dataset.
