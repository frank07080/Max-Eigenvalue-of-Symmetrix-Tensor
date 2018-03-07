# Max-Eigenvalue-of-Symmetrix-Tensor
This is a Python project to compute the max eigenvalue of a symmetric tensor using Mosek.

As people know, to compute a eigenvalue of a matrix is easy. However, to compute a eigenvalue of a tensor is mathematically hard. Chun-Feng Cui, Yu-Hong Dai, and Jiawang Nie has recently raised an iterative method to compute all eigenvalues of symmetric tensors based on semidefinite programming. The method is published on paper "ALL REAL EIGENVALUES OF SYMMETRIC TENSORS."

Therefore, this project is to implement the method using Python. We will implement some of the examples that the paper has. However, instead of computing all eigenvalues, we only compute the biggest one known as the one with the highest "saliency."
