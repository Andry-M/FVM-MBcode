# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Author: Andry Guillaume Jean-Marie Monlon
# Version: 1.0
# Creation date: 12/02/2025
# Context: Semester project - INCEPTION – Investigating New Convergence schEmes 
#          for Performance and Time Improvement Of fuel and Neutronics calculations
#
# Description: Machine learning models and methods to be used for acceleration techniques in MBCode
# -----------------------------------------------------------------------------

import numpy as np                                      # Array manipulation
import torch                                            # PyTorch
import torch.nn as nn                                   # Neural network module
from torch.utils.data import DataLoader, TensorDataset
torch.autograd.set_detect_anomaly(True)                 # Detects NaNs in the gradients
from sklearn.model_selection import train_test_split
from parameters import DEVICE                   # Import parameters

with torch.device(DEVICE):
    TORCH_DEVICE = torch.tensor([0]).device

class EarlyStopping:
    """
        Utility class to implement a patience mechanism for early stopping
    """
    def __init__(self, eps, patience):
        """
            Parameters:
                - eps (float) : Minimum difference between current loss and last loss to consider an improvement
                - patience (int) : Number of epochs to wait before stopping the training
        """
        self.patience = patience
        self.eps = eps
        self.counter = 0
        self.last_loss = 1e15 # Initial loss to consider for comparison
        self.stop_flag = False # Flag to stop the training

    def __call__(self, current_loss):
        if (current_loss - self.last_loss) > self.eps:
            self.counter +=1
        else:
            self.counter = 0
        if self.counter >= self.patience:
            self.stop_flag = True
        self.last_loss = current_loss

class CustomNN(torch.nn.Module):
    """
        Parent class for my custom Neural Networks
    """
    def __init__(self, len_layers = [], activation_func = nn.Identity, init_seed = None):
        super().__init__()
        self.init_seed = init_seed
        if self.init_seed != None:
            torch.manual_seed(self.init_seed)
        else :
            self.init_seed = torch.seed()
        if type(self)==CustomNN:
            self.wrappedLayers = nn.ParameterList([])
            if len(len_layers)!=0:
                for i in range(len(len_layers)-2):
                    self.wrappedLayers.append(nn.Sequential(*[nn.Linear(len_layers[i], len_layers[i+1]), activation_func()]))
                self.wrappedLayers.append(nn.Linear(len_layers[-2], len_layers[-1]))
                self._initialize_weights(activation_func)

    def _initialize_weights(self, activation_func):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                try:
                    gain = nn.init.calculate_gain(str.lower(activation_func.__name__))
                except ValueError:
                    gain = 1
                nn.init.xavier_uniform_(m.weight, gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        training_device = self.wrappedLayers[0][0].weight.get_device()
        if x.get_device() != training_device:
            x = x.to(training_device)
        for layer in self.wrappedLayers:
            x = layer(x)
        return x
  
class FCNN(CustomNN):
  """
    Fully connected Neural Network
  """
  def __init__(self, input_dim, output_dim, len_hidden_layers, 
               activation_func=nn.Identity, init_seed=None):
    """
        Parameters:
            - input_dim (int) : Dimension of the input
            - output_dim (int) : Dimension of the output
            - len_hidden_layers (list) : List containing the number of neurons in each hidden layer
            - activation_func (torch.nn.Module, default = nn.Identity) : Activation function to use
            - init_seed (int, default = None) : Seed for reproducibility
    """
    super().__init__(init_seed=init_seed)
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.inputLayer = nn.Sequential(*[nn.Linear(input_dim, len_hidden_layers[0]), activation_func()])
    self.hiddenLayers = nn.Sequential(*[nn.Linear(len_hidden_layers[i], len_hidden_layers[i+1], activation_func()) for i in range(len(len_hidden_layers)-1)])
    self.outputLayer = nn.Linear(len_hidden_layers[-1], output_dim)
    self.wrappedLayers = [self.inputLayer, self.hiddenLayers, self.outputLayer]
    self._initialize_weights(activation_func)
    
class TorchStandardScaler:
    """
        Standardize features by removing the mean and scaling to unit variance
    """
    def fit(self, T : torch.Tensor, exclude : list = []):
        self.exclude = exclude
        self.include = []
        if T.ndim>1:
            for i in range(T.shape[-1]):
                if i not in self.exclude:
                    self.include.append(i)
            T = T[..., self.include]
        self.mean = T.mean(0, keepdim=True)
        self.scale = T.std(0, keepdim=True)
    
    def transform(self, T : torch.Tensor):
        if T.ndim>1: 
            T_excluded = T[..., self.exclude]
            T = T[..., self.include]
        T -= self.mean
        T /= (self.scale + 1e-14)
        if T.ndim>1: 
            T = torch.cat((T, T_excluded), dim=-1)
        return T
  
    def fitTransform(self, T : torch.Tensor, exclude : list = []):
        self.fit(T, exclude)
        return self.transform(T)
  
    def inverseTransform(self, T : torch.Tensor):
        if T.ndim>1: 
            T_excluded = T[..., self.exclude]
            T = T[..., self.include]
        T *= (self.scale + 1e-14)
        T += self.mean
        if T.ndim>1: 
            T = torch.cat((T, T_excluded), dim=-1)
        return T
  
def fit(model, train_loader, test_loader, n_epochs, optimizer, early_stop_eps, early_stop_patience, 
        verbose=True, verb_one_out_of=10, loss_func = nn.MSELoss(), scheduler = None, target_loss = 0):
  
    hist = {'train' : [], 'test' : []}
  
    early_stopper = EarlyStopping(early_stop_eps, early_stop_patience)
  
    for epoch in range(n_epochs):
    # Learning on the train set
        for inputs, true_outputs in train_loader:
            train_loss = 0
            def closure():
                optimizer.zero_grad()
                pred_outputs = model(inputs)
                train_loss = loss_func(pred_outputs, true_outputs.to(pred_outputs.device))
                train_loss.backward()  
                if scheduler:
                    scheduler.step()
                return train_loss
      
        train_loss = optimizer.step(closure=closure).item()
        hist['train'].append(train_loss)
    
        # Evaluation on the test set
        for inputs, true_outputs in test_loader:
            test_loss = 0
            pred_outputs = model(inputs)
            test_loss = loss_func(pred_outputs, true_outputs.to(pred_outputs.device)).item()
            hist['test'].append(test_loss)
    
        # Print of the optimization step every verb_one_out_of step 
        if verbose and ((len(hist['train'])-1)%verb_one_out_of)==0:
            if test_loader:
                print("Ep", epoch+1, "- Optim step", len(hist['train']), "| Train loss:", hist['train'][-1], 
                        "| Test loss:", hist['test'][-1]) 
            else:
                print("Ep", epoch+1, "- Optim step", len(hist['train']), "| Train loss:", hist['train'][-1])
    
        if test_loader:
            early_stopper(hist['test'][-1])
            if early_stopper.stop_flag:
                print('Early stopper activated')
                break
        
        if hist['train'][-1] < target_loss:
            print('Target loss reached')
            break
  
    # Print of the last optimization step 
    if test_loader:
        print("Final Optim step", len(hist['train']), "| Train loss:", round(hist['train'][-1], 4), "| Test loss:", round(hist['test'][-1], 4))
    else:
        print("Final Optim step", len(hist['train']), "| Train loss:", round(hist['train'][-1], 4))
  
    return hist

def preprocess(hist_Ux, hist_Uy, n_train_iter : int = 20):
    """
        Preprocess the displacement fields to generate a datasets
        
        Parameters:
            - hist_Ux (list) : History of the displacement fields in the x-direction
            - hist_Uy (list) : History of the displacement fields in the y-direction
            - n_train_iter (int) : Number of training iterations to consider
    """
    hist_Ux = np.asarray(hist_Ux) # Convert to numpy array if needed
    hist_Uy = np.asarray(hist_Uy) # Convert to numpy array if needed
    n_cells = hist_Ux.shape[1]
    
    features = []
    targets = []

    for c in range(n_cells): # For each cell of the coarse mesh
        feature = []
        for i in range(n_train_iter): # For nb_iteration_per_input consecutive displacement fields
            feature.append(hist_Ux[i,c])
            feature.append(hist_Uy[i,c])
        features.append(feature)
        targets.append([hist_Ux[-1,c], hist_Uy[-1,c]])

    features = np.asarray(features, dtype=np.float64) # Convert to numpy array
    targets = np.asarray(targets, dtype=np.float64)   # Convert to numpy array

    torch_features = torch.from_numpy(features) # Convert to PyTorch tensor
    torch_targets = torch.from_numpy(targets)   # Convert to PyTorch tensor

    # Apply standardization for even importance on training
    scaler_features = TorchStandardScaler()
    torch_features = scaler_features.fitTransform(torch_features)
    scaler_targets = TorchStandardScaler()
    torch_targets  = scaler_targets.fitTransform(torch_targets)
    
    # Convert Double to Float for PyTorch compatibility
    torch_features = torch_features.to(torch.float32)
    torch_targets = torch_targets.to(torch.float32)
    
    return torch_features, torch_targets, scaler_features, scaler_targets

def train(n_epochs : int, model : torch.nn.Module, optimizer, features, targets, scheduler = None, loss_func = nn.MSELoss(),
          test_size = 0.1, early_stop_eps : float = 1e-12, early_stop_patience : int = 8, verb_one_out_of : int = 10,
          target_loss : float = 1e-3):
    """
        Generate the training and testing datasets and train the model
        
        Parameters:
            - n_epochs (int) : Number of epochs
            - model (torch.nn.Module) : Model to train
            - optimizer (torch.optim) : Optimizer to use
            - features (np.ndarray) : Features
            - targets (np.ndarray) : Targets
            - scheduler (torch.optim.lr_scheduler, default = None) : Learning rate scheduler
            - loss_func (torch.nn.Module, default = nn.MSELoss()) : Loss function
            - test_size (float, default = 0.1) : Proportion of the dataset to include in the test split
            - early_stop_eps (float, default = 1e-12) : Minimum difference between current loss and last loss to consider an improvement
            - early_stop_patience (int, default = 8) : Number of epochs to wait before stopping the training
            - verb_one_out_of (int, default = 10) : Verbosity of the training
            - target_loss (float, default = 1e-3) : Target loss to reach before stopping the training
    """
    # Split the dataset into training and testing sets
    if test_size != 0:
        train_features, test_features, train_targets, test_targets = train_test_split(
            features, targets, test_size=test_size)
    else:  
        train_features, test_features, train_targets, test_targets = features, features, targets, targets

    train_dataset = TensorDataset(train_features, train_targets)
    test_dataset = TensorDataset(test_features, test_targets)

    generator = torch.Generator(device=TORCH_DEVICE)
    batch_size = train_features.shape[0]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=generator)
    with torch.device(DEVICE):
        hist = fit(model, train_loader, test_loader, n_epochs, optimizer, early_stop_eps, early_stop_patience, 
                   verbose=(verb_one_out_of!=0), verb_one_out_of=verb_one_out_of, loss_func=loss_func, 
                   scheduler=scheduler, target_loss=target_loss)
    return hist 