import math
from ncpol2sdpa import *
import numpy as np
import mosek
import sympy as sp #use for differentiation, say sp.diff(f,x)

def teneig_max4(A):
    '''
        Calculate the max tensor eigenvalue for a specific tensor - order 2 case

        Inputs:
        - A: input single tensor of shape(H,W) - order 2 case

        Return a number of max eigval
    '''
    n = A.shape[0] #number of variables, equals 3 in example 1 of the paper 
    x = generate_variables('x',n)
    m = len(A.shape) #order of input tensor, equals 4 in example 1 of the paper
    p = 2 #Z eigval order
    N0 = math.ceil((m+1)/2)
    N = N0

    ## Generate the objective function
    f_A = 0 #Initialize the objective
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                for l in range(A.shape[3]):
                    f_A += A[i,j,k,l] * x[i] * x[j] * x[k] * x[l]

    ## Generate the constraint function
    fc = []

    # First Constraint
    fc1 = 0
    for i in range(len(x)):
        fc1 += x[i]**2
    fc1 = fc1 - 1
    fc += [fc1]
    #print('first constraint is',fc)

    #Second Constraint
    provis = []
    for i in range(len(x)):
        provis += [-f_A * x[i]]

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                for l in range(A.shape[3]):
                    provis[i] += A[i,j,k,l] * x[j] * x[k] * x[l]
    #print(provis)

    fc += provis
    #print(fc)

    sdp = SdpRelaxation(x)
    sdp.get_relaxation(N,objective = -f_A,equalities = fc)
    sdp.solve()
    #print(type(sdp.status))
    while sdp.status != 'optimal':
        N = N+1
        sdp = SdpRelaxation(x)
        sdp.get_relaxation(N,objective = -f_A,equalities = fc)
        sdp.solve()
    return(-sdp.primal)

#Example from paper of Tensor Eigenval
A = np.array([ [[[25.1,0],[0,0]], [[0,25.6],[0,0]]],  [[[0,0],[24.8,0]], [[0,0],[0,23]]] ])
print(A[0,0,0,0])
print(A.shape)
print(teneig_max4(A))



def teneig_max3(A):
    '''
        Calculate the max tensor eigenvalue for a specific tensor - order 2 case

        Inputs:
        - A: input single tensor of shape(H,W) - order 2 case

        Return a number of max eigval
    '''
    n = A.shape[0] #number of variables, equals 3 in example 1 of the paper 
    x = generate_variables('x',n)
    m = len(A.shape) #order of input tensor, equals 4 in example 1 of the paper
    p = 2 #Z eigval order
    N0 = math.ceil((m+1)/2)
    N = N0

    ## Generate the objective function
    f_A = 0 #Initialize the objective
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                f_A += A[i,j,k] * x[i] * x[j] * x[k]

    ## Generate the constraint function
    fc = []

    # First Constraint
    fc1 = 0
    for i in range(len(x)):
        fc1 += x[i]**2
    fc1 = fc1 - 1
    fc += [fc1]
    #print('first constraint is',fc)

    #Second Constraint
    provis = []
    for i in range(len(x)):
        provis += [-f_A * x[i]]

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[2]):
                provis[i] += A[i,j,k] * x[j] * x[k] 
    #print(provis)

    fc += provis
    #print(fc)

    sdp = SdpRelaxation(x)
    sdp.get_relaxation(N,objective = -f_A,equalities = fc)
    sdp.solve()
    #print(type(sdp.status))
    while sdp.status != 'optimal':
        N = N+1
        sdp = SdpRelaxation(x)
        sdp.get_relaxation(N,objective = -f_A,equalities = fc)
        sdp.solve()
    return(-sdp.primal)


B = np.zeros((3,)*3) #second 3 is dimension
print(B.shape)

for i in range(3):
    for j in range(3):
        for k in range(3):
            text = input('please enter B'+str(i+1)+str(j+1)+str(k+1)+' ')
            B[i,j,k] = text

print(teneig_max3(B))

'''
B[1,1,1] = 0.4333
B[1,2,1] = 0.4278
B[1,3,1] = 0.4140
B[2,1,1] = 0.8154
B[2,2,1] = 0.0199

B[2,3,1] = 0.5598
B[3,1,1] = 0.0643
B[3,2,1] = 0.3815
B[3,3,1] = 0.8834
B[1,1,2] = 0.4866

B[1,2,2] = 0.8087
B[1,3,2] = 0.2073
B[2,1,2] = 0.7641
B[2,2,2] = 0.9924
B[2,3,2] = 0.8752

B[3,1,2] = 0.6708
'''
