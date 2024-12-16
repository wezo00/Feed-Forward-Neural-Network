import numpy as np
import torch
import matplotlib.pyplot as plt
from nn_training import maintrain

N = 19
N_train = np.array([4000,8000,16000,32000])
N_train = np.array([4000])

mu_1 = 2.
mu_2 = 1.
mu_3 = 2.5
mu_4 = 3.5
mu1 = np.array([mu_1,mu_2,mu_3,mu_4])
mutest = torch.tensor([mu_1, mu_2, mu_3, mu_4], dtype=torch.double,requires_grad=False).unsqueeze(0)


def diffin(N,u_exact,mutens, muarr,model, x_mean, x_std, y_mean, y_std):

    h = 1/(N+1)
    # compute solution u with model prediction by neural network
    result = torch.cat((mutens[:, 0:1], mutens[:, -2:]), dim=1)
    x_test_normalized = (result - x_mean) / x_std
    u_normalized = model(x_test_normalized)
    u = u_normalized * y_std + y_mean
    print(u)
    u = u.detach().numpy().reshape(-1)
    
    error = np.zeros(N)
    max_err = -1.0

    for i in range(N):
        error[i] = u[i] - u_exact((i + 1) * h,muarr)
        max_err = max(max_err, np.abs(error[i]))

    err = max_err
    err2 = np.linalg.norm(error)
    return u, err, err2

def main(mutens, muarr, f1, u_exact, N, M, plotsol = 0, ploterr = 0):

    # Step size h
    h = 1.0 / (N + 1)

    # Initialize error array
    error = np.zeros(len(M))
    er_e = np.zeros(len(M))
    error2 = np.zeros(len(M))
    ei = np.zeros(len(M))

    # Compute the error and sol for each value of N (main loop)
    for n in range(len(M)):

        # create dataset and train neural network    
        model,_,_, x_mean, x_std, y_mean, y_std = maintrain(N,M[n],3,100)

        u, err, err2 = diffin(N,u_exact,mutens,muarr,model,x_mean, x_std, y_mean, y_std)
        error[n] = err
        er_e[n] = ee(muarr,N,f1)
        # print(f'ee  = {er_e[n]:e}')
        error2[n] = h*err2**2
        ei[n] = (er_e[n]**0.5)/err2
        # print(f'eff index  = {ei[n]:e}')

        if plotsol == 1 :
                # Plot the solution
                x = np.linspace(h, 1 - h, N)
                plt.figure()
                plt.plot(x, u, '-o', label='sol')
                plt.plot(x, u_ex(x,muarr), label='exact sol')
                plt.xlabel(f"N_train={M[n]}")
                plt.legend()
                plt.show()

    if ploterr == 1 : 
        # Plot the results (errors)
        plt.figure()

        # Plot error vs N in log-log scale
        #plt.loglog(M, error, 'b', label='Error, L_inf')
        #plt.loglog(M, er_e, 'r', label='Error estimate')
        plt.loglog(M, error2, '-o', label='Error, L_2')

        # Plot h^2 to compare the order of convergence
        #order2 = h**2
        #plt.loglog(M, order2, 'k', label='O(h^2)')
        # theoretical order of convergence for the L_inf norm: O(h^2)
        #order3 = h**(3)
        #plt.loglog(M, order3, 'g', label='O(h^3)')
        # theoretical order of convergence for the L_2 norm: O(h^(3/2)) -> L_2 norm squared: O(h^3)

        # Add labels and legend
        plt.xlabel('N_train')
        plt.ylabel('Error')
        plt.legend()

        # Show plot
        plt.show()

    return error2, model, x_mean, x_std, y_mean, y_std

def ee(mu_0,N,f1):
    h = 1/(N+1)
    sum = 0
    for i in range(N+1):
        sum = sum + f1((i + 1) * h,mu_0)**2 + f1(i * h,mu_0)**2
    sum = 0.5 * h**4 * sum
    return sum

def f(x,mu_0):
    return mu_0[0]/(2 * mu_0[1]) * np.pi**2 * ((mu_0[2]**2+mu_0[3]**2) * (np.sin(mu_0[2] * np.pi * x) * np.cos(mu_0[3] * np.pi * x)) 
                                           + 2 * mu_0[2] * mu_0[3] * np.sin(mu_0[3] * np.pi * x) * np.cos(mu_0[2] * np.pi * x))

def u_ex(x,mu_0):
    return mu_0[0]/(2 * mu_0[1]) * np.sin(mu_0[2] * np.pi * x) * np.cos(mu_0[3] * np.pi * x)
def u_ex(x,mu_0):
    return mu_0[0]/2 * np.sin(mu_0[1] * np.pi * x) * np.cos(mu_0[2] * np.pi * x)


def MC_old(M,N,M_tr):
    seed = 3
    np.random.seed(seed)
    mu1vec = np.random.uniform(low=-2, high=2, size=M).astype(np.double)
    mu2vec = np.full(M, 1., dtype=np.double)
    mu3vec = np.random.uniform(low=1, high=4, size=M).astype(np.double)
    mu4vec = np.random.uniform(low=1, high=4, size=M).astype(np.double)
    
    error2 = np.zeros(M)

    # train nn
    model, x_mean, x_std, y_mean, y_std = maintrain(N,M_tr)

    for i in range(M):
        mui = np.array([mu1vec[i],mu2vec[i],mu3vec[i],mu4vec[i]])
        print(mui)
        mutesti = torch.tensor([mu1vec[i],mu2vec[i],mu3vec[i],mu4vec[i]], dtype=torch.double,requires_grad=False).unsqueeze(0)
        _, _, error2[i] = diffin(N,u_ex,mutesti,mui, model, x_mean, x_std, y_mean, y_std)

    
    error_M = sum(error2)/M
    print(error_M)

    return error_M

## run code

#error2, _,_,_,_,_ = main(mutest,mu1,f,u_ex,N,N_train, plotsol=1, ploterr=1)
#print(error2)


####

def MC(M,N,N_tr,L,W):
    error2 = np.zeros(M)
    truerror = np.zeros(M)
    errorFD = np.zeros(M)
    model,mu,solutions, x_mean, x_std, y_mean, y_std = maintrain(N,N_tr,L,W)
    for i in range(M):
        mui  = mu[i,:]
        mutesti = torch.tensor(mui, dtype=torch.double,requires_grad=False).unsqueeze(0)
        x_norm = (mutesti - x_mean) / x_std
        y_norm = model(x_norm)
        u = y_norm * y_std + y_mean
        u_N = u.detach().numpy().reshape(-1)
        u_h = solutions[i,:]
        error2[i] = (np.linalg.norm(u_N-u_h)/((N+1)**(1/2)))**2
        x = np.linspace(1/(N+1),1-1/(N+1),N)
        u_true = u_ex(x,mui)
        truerror[i] = np.linalg.norm(u_true-u_N)**2/(N+1)
        errorFD[i] = np.linalg.norm(u_true-u_h)**2/(N+1)
            
    error_M = np.mean(error2)
    stderr = np.std(error2)
    truerrorM = np.mean(truerror)
    errorFDM = np.mean(errorFD)
    return error_M, stderr, stderr/np.sqrt(M), truerrorM, errorFDM

def fig2():
    N = 19
    N_train = np.array([4000,8000,16000,32000])
    M=4000
    plt.figure()
    L = [3,4]
    W = [25,50]

    for l in L:
        for w in W:
            errorM = np.zeros(len(N_train))
            for i in range(len(N_train)):
                errorM[i],_,_,_,_ = MC(M,N,N_train[i],l,w)
            plt.loglog(N_train, errorM, '-o', label=f"L={l}, W={w}")
    plt.loglog(N_train, 2*1/N_train, '--b', label='Slope for N_train^(-1)')
    plt.xlabel('N_train')
    plt.ylabel('(|||u_N-u_h|||_M)^2')
    plt.legend()
    plt.show()


'''
errorM = np.zeros(len(N_train))
for i in range(len(N_train)):
    errorM[i] = MC(M,N,N_train[i],L[0],W[0])
plt.loglog(N_train, errorM, '-o', label=f"L={L[0]}, W={W[0]}")
plt.loglog(N_train, 100*1/(N_train**(1/2)), '--b', label='Slope for N_train^(-1/2)')
plt.xlabel('N_train')
plt.ylabel('|||u_N-u_h|||_M')
plt.legend()     
plt.show()
'''

def table4():
    N = np.array([39,79])
    N_train = 8000
    M=4000
    L = [3,4]
    W = [50,100]
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
    W = [50,100]
    for n in N:
        for l in L:
            for w in W:
                error_M, _, _, truerror, errorFD = MC(M,n,N_train,l,w)
                print(n+1,l,w)
                print(errorFD,error_M,truerror)


fig2()
#table4()
#table9()