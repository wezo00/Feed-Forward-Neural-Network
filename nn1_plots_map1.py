import numpy as np
import torch
import matplotlib.pyplot as plt
from nn1_train_map1 import train1

N = 19
M_train = np.array([1000,2000,4000,8000,16000])
M_train = np.array([4000])

mu_1 = 2.
mu_2 = 1.
mu_3 = 2.5
mu_4 = 3.5
mu1 = np.array([mu_1,mu_2,mu_3,mu_4])
mutest = torch.tensor([mu_1, mu_2, mu_3, mu_4], dtype=torch.double,requires_grad=False).unsqueeze(0)


def diffin(N, mutens,model, x_mean, x_std, y_mean, y_std):

    h = 1/(N+1)
    # compute solution u with model prediction by neural network
    result = mutens
    x_test_normalized = (result - x_mean) / x_std
    u_normalized = model(x_test_normalized)
    u = u_normalized * y_std + y_mean
    #print(u)
    u = u.detach().numpy().reshape(-1)
   
    return u[0]

def main(muarr, u_exact, N, M, plotsol = 0, ploterr = 0):

    # Step size h
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1 - h, N)

    # Initialize error array
    error2 = np.zeros(len(M))
    error = np.zeros(N)

    u = np.zeros(N)

    # Compute the error and sol for each value of N (main loop)
    for m in range(len(M)):

        # create dataset and train neural network    
        model, mu, sols, x_mean, x_std, y_mean, y_std = train1(N,M[m])

        for n in range(N):
            inp = torch.tensor([muarr[0], muarr[2], muarr[3], x[n]], dtype=torch.double,requires_grad=False).unsqueeze(0)
            u[n] = diffin(N,inp,model,x_mean, x_std, y_mean, y_std)
            error[n] = u[n] - u_exact((n + 1) * h,muarr)
        
        error2[m] = h*np.linalg.norm(error)

    if plotsol == 1 :
        # Plot the solution
        x = np.linspace(h, 1 - h, N)
        plt.figure()
        plt.plot(x, u, '-o', label='sol')
        plt.plot(x, u_ex(x,muarr), label='exact sol')
        plt.legend()
        plt.show()

    if ploterr == 1 : 
        # Plot the results (errors)
        plt.figure()

        plt.loglog(M, error2, '-y', label='Error, L_2')

        # Add labels and legend
        plt.xlabel('N_train')
        plt.ylabel('Error')
        plt.legend()

        # Show plot
        plt.show()

    return error2, model, x_mean, x_std, y_mean, y_std

def f(x,mu_0):
    return mu_0[0]/(2 * mu_0[1]) * np.pi**2 * ((mu_0[2]**2+mu_0[3]**2) * (np.sin(mu_0[2] * np.pi * x) * np.cos(mu_0[3] * np.pi * x)) 
                                           + 2 * mu_0[2] * mu_0[3] * np.sin(mu_0[3] * np.pi * x) * np.cos(mu_0[2] * np.pi * x))

def u_ex(x,mu_0):
    return mu_0[0]/(2 * mu_0[1]) * np.sin(mu_0[2] * np.pi * x) * np.cos(mu_0[3] * np.pi * x)
def u_ex(x,mu_0):
    return mu_0[0]/2 * np.sin(mu_0[1] * np.pi * x) * np.cos(mu_0[2] * np.pi * x)

def MC(M,N,M_tr):
    seed = 3
    np.random.seed(seed)
    mu1vec = np.random.uniform(low=-2, high=2, size=M).astype(np.double)
    mu2vec = np.full(M, 1., dtype=np.double)
    mu3vec = np.random.uniform(low=1, high=4, size=M).astype(np.double)
    mu4vec = np.random.uniform(low=1, high=4, size=M).astype(np.double)
    
    error2 = np.zeros(M)

    # train nn
    model, mu, sols, x_mean, x_std, y_mean, y_std = train1(N,M_tr)
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1 - h, N)
    for i in range(M):
        error = np.zeros(N)
        u = np.zeros(N)
        mui = np.array([mu1vec[i],mu2vec[i],mu3vec[i],mu4vec[i]])
        for n in range(N):
            inp = torch.tensor([mui[0], mui[2], mui[3], x[n]], dtype=torch.double,requires_grad=False).unsqueeze(0)
            u[n] = diffin(N,inp,model,x_mean, x_std, y_mean, y_std)
            error[n] = u[n] - u_ex((n + 1) * h,mui)
        
        error2[i] = np.linalg.norm(error)
    
    error_M = sum(error2)/M
    print(error_M)

    return error_M

#error2, _, _, _, _, _= main(mu1,u_ex,N,M_train, plotsol=1, ploterr=1)
#print(error2)

####

def MC(M,N,N_tr,L,W):
    error2 = np.zeros(M)
    truerror = np.zeros(M)
    errorFD = np.zeros(M)
    model,mu,solutions, x_mean, x_std, y_mean, y_std = train1(N,N_tr,L,W)
    h = 1.0 / (N + 1)
    x = np.linspace(h, 1 - h, N)
    for i in range(M):
        error = np.zeros(N)
        errortrue = np.zeros(N)
        u = np.zeros(N)
        mui  = mu[i,:]
        u_true = u_ex(x,mui)
        for n in range(N):
            mutesti = torch.tensor([mui[0],mui[1],mui[2],x[n]], dtype=torch.double,requires_grad=False).unsqueeze(0)
            x_norm = (mutesti - x_mean) / x_std
            y_norm = model(x_norm)
            u = y_norm * y_std + y_mean
            u_N = u.detach().numpy().reshape(-1)
            u_N = u_N[0]
            u_h = solutions[i,n]
            error[n] = u_N-u_h
            errortrue[n] = u_N - u_true[n]
        error2[i] = np.linalg.norm(error)**2/(N+1)
        truerror[i] = np.linalg.norm(errortrue)**2/(N+1)
        errorFD[i] = np.linalg.norm(u_true-solutions[i,:])**2/(N+1)
            
    error_M = np.mean(error2)
    stderr = np.std(error2)
    truerrorM = np.mean(truerror)
    errorFDM = np.mean(errorFD)
    return error_M, stderr, stderr/np.sqrt(M), truerrorM, errorFDM
'''
N_train = np.array([4000,8000,16000])
M=4000

errorM = np.zeros(len(N_train))
for i in range(len(N_train)):
    errorM[i] = MC(M,N,N_train[i])

plt.figure()
plt.loglog(N_train, errorM, '-o', label='L=3, W=50')
plt.loglog(N_train, 10 * 1/N_train, '--b', label='Slope for N_train^(-1)')
plt.xlabel('N_train')
plt.ylabel('Average error')
plt.legend()
plt.show()
'''
def fig2():
    N = 19
    N_train = np.array([4000,8000,16000,32000])
    M = 4000
    plt.figure()
    L = [3,4]
    W = [25,50]

    for l in L:
        for w in W:
            errorM = np.zeros(len(N_train))
            for i in range(len(N_train)):
                errorM[i] = MC(M,N,N_train[i],l,w)
            plt.loglog(N_train, errorM, '-o', label=f"L={l}, W={w}")
    plt.loglog(N_train, 1/N_train, '--b', label='Slope for N_train^(-1)')
    plt.xlabel('N_train')
    plt.ylabel('(|||u_N-u_h|||_M)^2')
    plt.legend()     
    plt.show()


def table4():
    N = np.array([39,79])
    N_train = 8000
    M=4000
    L = [3,4]
    W = [25,50]
    for l in L:
        for w in W:
            for n in N:
                print(l,w,n+1)
                errorM,stderr,stdM,_,_ = MC(M,n,N_train,l,w)
                print(errorM,stderr,stdM)


def table9():
    N = np.array([19,39,79])
    N_train = 4000
    M = 2000
    L = [3,4]
    W = [25,50]
    for n in N:
        for l in L:
            for w in W:
                error_M, _, _, truerror, errorFD = MC(M,n,N_train,l,w)
                print(n+1,l,w)
                print(errorFD,error_M,truerror)


#fig2()
#table4()
table9()

