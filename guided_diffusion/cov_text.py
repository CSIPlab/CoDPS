import sys

import scipy.linalg
import scipy.signal 

import copy
import matplotlib.pyplot as plt
import torch
import scipy 
import numpy as np
import cv2 

from guided_diffusion.measurements import GaussialBlurOperator 

# def roll(matrces):

d = 8
N = 16
M = N//d
M_squared = M**2
N_squared = N**2
X = torch.arange(N**2).view(N,N).float()

S = torch.zeros(M**2, N**2).float() 
# Generate the indices to set to 1
# indices = torch.arange(0, M**2) * d
# S[torch.arange(M**2), indices] = 1
row_indices = torch.arange(0, N, d).repeat(1, M)
column_indices = (torch.arange(M) * N * d).repeat(1, M)
column_indices = torch.arange(0, N_squared, M*d**2).view(-1, 1).repeat(1, N//d).reshape(-1)
indices = row_indices + column_indices

# Set the specified indices to 1
S[torch.arange(M_squared), indices] = 1
Mask = torch.zeros(N, N)

rows = torch.arange(0, N, d)
grid_x, grid_y = torch.meshgrid(rows, rows, indexing='ij')
Mask[grid_x, grid_y] = 1
print("Mask: ", Mask, rows)
# Mask[torch.arange(N), torch.arange(0, N, d).view(-1,1).repeat(1,d)] = 1



Y = S @ X.view(-1)
Y = Y.reshape(M,M)

assert torch.equal(S @ S.T, torch.eye(M_squared))
assert torch.equal(S.T@S, torch.diag(Mask.view(-1))), f"{torch.diag(Mask)} != {S.T@S}"

delta = torch.rand(N_squared).type(torch.complex64)
delta_conj = torch.conj(delta)
F = torch.fft.fft( torch.eye(N_squared), norm="ortho")
F_H = torch.fft.ifft( torch.eye(N_squared), norm="ortho")
H = torch.fft.ifft( F_H @ torch.diag(delta) @ F ).abs()

assert torch.equal(torch.eye(N_squared), torch.round(torch.abs(F@F_H), decimals=5)), f"{torch.eye(N_squared)} != {torch.abs(F@F_H)}"
assert torch.equal(torch.eye(N_squared), torch.round(torch.abs(F_H@F), decimals=5))
assert H.shape  == torch.Size([N_squared, N_squared])

S_complex = S.type(torch.complex64)
Mat_1 = torch.abs(F @ torch.conj(S_complex).T @ S_complex @ torch.conj(F).T)
Mat_1 = torch.round(Mat_1, decimals=4)
Identity_1 = 1 / d**2 * torch.kron(torch.ones(d**2,d**2), torch.eye(M_squared))


My_Mat = S_complex @ F_H @ torch.diag(delta * delta_conj) @ F @ S_complex.H

# print(My_Mat.abs(), delta.reshape(N,N).abs()/d**2)
# assert torch.equal(Mat_1, Identity_1), f"{Mat_1[0]} != {Identity_1[0]}"
print("F F@H", torch.equal(Mat_1, Identity_1))
print(torch.round(Mat_1, decimals=3) - Identity_1)

