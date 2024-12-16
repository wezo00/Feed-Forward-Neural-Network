import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset # Wrapper for dataset
from torch.utils.data import DataLoader # Responsible for maning batches

def train1(N=19,M=4000,L=3,W=25):
    
    # 1: Output layer dimension
    # M: Number of FE solutions
    # j: Node

    seed = 3
    np.random.seed(seed)  # Set a seed for reproducibility
    # mu1, mu2 in [-2,2] and mu3, mu4 in [2,4]
    mu1vec = np.random.uniform(low=-2, high=2, size=M).astype(np.double)
    mu2vec = np.full(M, 1., dtype=np.double)
    mu3vec = np.random.uniform(low=1, high=4, size=M).astype(np.double)
    mu4vec = np.random.uniform(low=1, high=4, size=M).astype(np.double)

    mu1 = 2.
    mu2 = 1.
    mu3 = 2.5
    mu4 = 3.

    # Float because the MODEL PARAMETERS (WEIGHTS AND BIASES) are in float format
    # You can also do model = model.double()

    h = 1/(N+1)
    x = np.linspace(h, 1 - h, N)

    x_toTest = torch.tensor([mu1, mu3, mu4, x[1]], dtype=torch.double,requires_grad=False).unsqueeze(0)

    u_real = mu1/(mu2*2)*np.sin(mu3*np.pi*x)*np.cos(mu4*np.pi*x) 

    mu = np.array([mu1vec, mu3vec ,mu4vec ]).T
    #print(mu)
    # Parametric analytical solution
    def u_real_f(*args, x):
        return args[0]/(args[1]*2)*np.sin(args[2]*np.pi*x)*np.cos(args[3]*np.pi*x) 
    def b_real_f(*args, x):
        return args[0]/(args[1]*2)*np.pi**2 * (
            (args[2]**2 + args[3]**2)*np.sin(args[2]*np.pi*x) * np.cos(args[3]*np.pi*x)
            + 2*args[2]*args[3]* np.sin(args[3]*np.pi*x) * np.cos(args[2]*np.pi*x))
    # Rows = Number dof / Columns = Number FD sol
    solutions_matrix = np.zeros((M, N))
    A = (N+1)**2 * (np.diag(2 * np.ones(N)) - np.diag(np.ones(N-1), k=1) - np.diag(np.ones(N-1), k=-1))
    for i in range(M):
        # Solve the system u = linalg.solve(A, b)
        b = b_real_f(mu1vec[i], mu2vec[i], mu3vec[i], mu4vec[i], x=x)
        b[-1] += (u_real_f(mu1vec[i], mu2vec[i], mu3vec[i], mu4vec[i], x=x[-1]+h) /h**2)
        f = b_real_f(mu1vec[i], mu2vec[i], mu3vec[i], mu4vec[i], x=x)
        u = linalg.solve(A, b)
        solutions_matrix[i,:] = u

    # FFNN
    #x_train, y_train = map(lambda x: torch.tensor(x, dtype=torch.double), ([[mu1, mu2, mu3, mu4]], [u]))
    output_vect = np.reshape(solutions_matrix,(N*M,))
    input_matrix = np.zeros((N*M,4))

    for i in range(N*M):
        node = np.array([x[i%N]])
        input_matrix[i,:] = np.concatenate((mu[i//N,:],node))

    x_train, y_train = map(lambda x: torch.tensor(x, dtype=torch.double), (input_matrix, output_vect))

    # Normalizing input and output
    x_mean, x_std = x_train.mean(dim=0), x_train.std(dim=0)
    y_mean, y_std = y_train.mean(dim=0), y_train.std(dim=0)
    print("Mean after normalization:", y_train.mean().item())
    print("Std after normalization:", y_train.std().item())

    x_train = (x_train - x_mean) / x_std
    y_train = (y_train - y_mean) / y_std
    print(x_train, x_train.size(), x_train.dtype)
    print(y_train, y_train.size(), y_train.dtype)

    lr = 1e-3  # learning rate 0.01 with one random parameter
    epochs = 50  # how many epochs to train for
    bs = 2048 # batch size

    train_ds = TensorDataset(x_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=bs)
    '''
    MODEL
    '''
    class FD_FFNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(FD_FFNN, self).__init__()
            # Define layers
            self.layer1 = nn.Linear(input_size, hidden_size).double()
            self.layer2 = nn.Linear(hidden_size, hidden_size).double()
            self.layer3 = nn.Linear(hidden_size, hidden_size).double()
            if L==4:
                self.layer4 = nn.Linear(hidden_size, hidden_size).double()
            self.layer5 = nn.Linear(hidden_size, output_size).double()

            self.relu = nn.ReLU() #nn.ReLU() nn.Softplus() 

        def forward(self, x):
            # Pass through the layers with ReLU activations
            x = self.relu(self.layer1(x))
            x = self.relu(self.layer2(x))
            x = self.relu(self.layer3(x))
            if L==4:
                x = self.relu(self.layer4(x))
            x = self.layer5(x)  # Output layer without activation (for regression or further softmax)
            return x

    # Example dimensions
    input_size = 4     # Number of input features
    #hidden_size1 = 30   # First hidden layer neurons
    #hidden_size2 = 45   # Second hidden layer neurons
    #hidden_size3 = 40   # Third hidden layer neurons
    hidden_size = W
    output_size = 1    # Output size

    model = FD_FFNN(input_size, hidden_size, output_size)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

    model.apply(init_weights)
    '''
    OPTIMIZER
    '''
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    '''
    LOSS FUNCTION
    '''
    loss_func = torch.nn.MSELoss(reduction='mean') 

    loss_values = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb.unsqueeze(1))        
            loss.backward()
            opt.step()
            opt.zero_grad()
            total_loss += loss.item()
        #scheduler.step()  # Adjust learning rate if using a scheduler
        loss_values.append(total_loss / len(train_dl))  # Save as float for tracking
        print(f"Epoch {epoch+1}, Loss: {loss_values[-1]}")
    '''
    plt.plot(loss_values, '-o', label="FFNN_train")
    plt.legend()
    plt.show()
    '''

    model.eval()
    '''
    pred_normalized1 = model(x_train[3,:])
    u_fd = y_train * y_std + y_mean
    pred1 = pred_normalized1 * y_std + y_mean
    print(pred1)
    # Convert to NumPy array for plotting
    pred_np1 = pred1.detach().numpy().reshape(-1)
    u_fd = u_fd.detach().numpy().reshape(-1)

    plt.plot(pred_np1, '-o', label="FFNN")
    plt.plot(u_fd, label="Real")
    plt.legend()
    plt.show()
    
    result = x_toTest
    print(result)
    x_test_normalized = (result - x_mean) / x_std
    pred_normalized = model(x_test_normalized)
    pred = pred_normalized * y_std + y_mean
    print(pred)
    # Convert to NumPy array for plotting
    pred_np = pred.detach().numpy().reshape(-1)
    print(pred_np)
    
    plt.plot(pred_np, '-o', label="FFNN")
    plt.plot(u_real, label="Real")
    plt.legend()
    plt.show()
    '''
    
    return model, mu, solutions_matrix, x_mean, x_std, y_mean, y_std

#train1(8000,19)