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

def assert_cirulant(H, H_j):
    N = len(H_j)
    row, col = H_j[0].shape
    assert H.shape , torch.Size([row * N, col * N])
    for i in range(N):
        for j in range(N):
            h_j_idx = i-j
            assert torch.equal(H[i*row:i*row+row,j*col:j*col+col], H_j[h_j_idx]), f"falied {(i,j,h_j_idx)}, {torch.max(H[i*row:i*row+row,j*col:j*col+col]-H_j[h_j_idx])}"

def circular_stack(matrices):
    __matrices__ = copy.deepcopy(matrices)
    n = len(__matrices__)
    rows, cols = __matrices__[0].shape
    matrix = torch.vstack(__matrices__)
    
    result = torch.zeros(rows * n, cols *n).type(torch.float32)
    for i in range(0,n*cols,cols):
        result[:, i:i+cols] = matrix
        # Roll 
        __matrices__ = __matrices__[-1:] + __matrices__[:-1]
        matrix = torch.vstack(__matrices__)
    return result

m = torch.tensor([[1,2,1],[1,3,-1],[0,1,0]]).type(torch.float32)
h = torch.tensor([[1,-1,0],[1,0,0],[0,0,0]]).type(torch.float32)
H_j = []

for i in range(len(h)):
    H_j.append(torch.from_numpy(scipy.linalg.circulant(h[i,:])).type(torch.float32))
H = circular_stack(H_j)
G = (H@m.view(-1)).reshape(3,3)
g = torch.tensor([[0,2,-1],[3,4,-3],[1,4,-2]])
# a = torch.rand(64*64*64)
# b = [ a[i:i+64*64].reshape(64,64) for i in range(0,64*64*64,64*64) ]
# m = circular_stack(b)
# print("Test circ ", m.shape, len(b), b[0].shape )
# assert_cirulant(m,b)
# print("Passed circ ")
print(torch.equal(g,G),g, G)
# exit()
'''
    Deblurring
'''
N = 64
sample_image = torch.rand(N,N,3)
img = cv2.resize(cv2.imread("data/samples/00015.png"), (64,64)).astype(np.float32)[:,:,::-1] / 255
# print(img)
sample_image = torch.from_numpy(img) 
sigma_n = 0.05

k = 11
op = GaussialBlurOperator(k,1,'cpu')
Y = op.forward(sample_image.permute(2,0,1).unsqueeze(0)).squeeze(0).permute(1,2,0).detach().cpu()

kernel_scipy = op.kernel.clone()
kernel = op.kernel
# kernel = torch.zeros_like(kernel)
# kernel[k//2,k//2] = 1
# kernel = torch.fft.ifftshift(kernel)

N_FFT = (N)**2
pad_width = pad_height = N-k
padding = (pad_width//2+1, pad_width//2, pad_height//2+1, pad_height//2) 
padded_kernel = (torch.nn.functional.pad(kernel, padding))
# Flip padded kernel becuase convolution vs correaltion inpytorch 
padded_kernel = padded_kernel
padded_kernel = torch.fft.ifftshift(padded_kernel).type(torch.float32)
print("padded kernel sape ", padded_kernel.shape)
plt.imshow(padded_kernel)
plt.savefig("zzz_paddedkernel")

H_j = []

for i in range(len(padded_kernel)):
    H_j.append(torch.from_numpy(scipy.linalg.circulant(padded_kernel[i,:])).type(torch.float32))
H = circular_stack(H_j)
assert_cirulant(H,H_j)

print("First columnas are equal, ", H[:,0] - padded_kernel.reshape(-1), torch.equal(H[:,0], padded_kernel.reshape(-1)))
delta = torch.fft.fft2(padded_kernel.reshape(N,N)).type(torch.complex64)#.reshape(-1)
# delta = torch.diag(torch.fft.fftn(H[:,0])).type(torch.complex64)
img_fft = torch.fft.fft2(sample_image, dim=[0,1]).type(torch.complex64)
print("Deta ", delta.shape, img_fft.shape)
Y_hat = torch.fft.ifft2(delta.unsqueeze(-1) * img_fft, dim=[0,1])
Y_hat = torch.real(Y_hat).reshape(N,N,3)

Y_mat_mul = (H @ sample_image.reshape(-1,3)).reshape(N,N,3)

# Y_scipy = torch.stack([torch.from_numpy(scipy.signal.convolve2d(sample_image[i], kernel, boundary='wrap') for i in range(3)) ] )
Y_scipy = [torch.from_numpy(scipy.signal.convolve2d(sample_image[:,:,i], kernel_scipy, mode='same', boundary='wrap')) for i in range(3)]
Y_scipy = torch.stack(Y_scipy, dim=2)

# Using 2d fourier transform 
img_fft = torch.fft.fftn(sample_image, dim=[0,1])
kernel_fft = torch.fft.fftn(padded_kernel, s= [N,N], dim=[0,1]).unsqueeze(-1)
Y_FFT = torch.fft.ifftn(img_fft * kernel_fft, dim=[0,1]).abs()

print(Y_scipy.shape)
print("Error Y_scipy - Y_hat ", torch.max(Y_scipy - Y_hat))
print("Error Y_scipy - Y_mat_mul ", torch.max(Y_scipy - Y_mat_mul))
print("Error Y_scipy - Y_FFT ", torch.max(Y_scipy - Y_FFT))
print("Error Y_hat - Y_FFT ", torch.max(Y_hat - Y_FFT))
print("Error Y_mat_mul - Y_FFT ", torch.max(Y_mat_mul - Y_FFT))
print("Error Y_mat_mul - Y_hat ", torch.max(Y_mat_mul - Y_hat))

# print("Kernel shape:", padded_kernel.shape)
# print("Image FFT shape:", img_fft.shape)
# print("Delta shape:", delta.shape)
# print("Troch equal ", torch.equal(Y,Y_hat), (Y-Y_hat))

# Display the original image (Y)
plt.subplot(2, 3, 1)  # Subplot with 1 row, 2 columns, position 1
plt.imshow(Y_scipy)  # Display color image
plt.title('Y_scipy')
plt.axis('off')

# Display the reconstructed image (Y_hat)
plt.subplot(2, 3, 2)  # Subplot with 1 row, 2 columns, position 2
plt.imshow(Y_hat)  # Display color image
plt.title('Y_hat')
plt.axis('off')

# Display the reconstructed image (Y_hat)
# plt.subplot(2, 3, 3)  # Subplot with 1 row, 2 columns, position 2
# plt.imshow(Y_mat_mul)  # Display color image
# plt.title('Y_mat_mul')
# plt.axis('off')
plt.subplot(2, 3, 3)  # Subplot with 1 row, 2 columns, position 2
plt.imshow(Y_FFT)  # Display color image
plt.title('Y_FFT')
plt.axis('off')



# Display the reconstructed image (Y_hat)
plt.subplot(2, 3, 4)  # Subplot with 1 row, 2 columns, position 2
plt.imshow(torch.abs(Y-Y_hat))  # Display color image
plt.title('Y_hat res')
plt.axis('off')

# Display the reconstructed image (Y_hat)
plt.subplot(2, 3, 5)  # Subplot with 1 row, 2 columns, position 2
plt.imshow(torch.abs(Y-Y_mat_mul))  # Display color image
plt.title('Y_mat_mul')
plt.axis('off')

# Display the reconstructed image (Y_hat)
plt.subplot(2, 3, 6)  # Subplot with 1 row, 2 columns, position 2
plt.imshow(torch.abs(Y_hat-Y_mat_mul))  # Display color image
plt.title('Y_hat - Y_mat_mul')
plt.axis('off')

# Adjust layout to prevent overlapping
plt.tight_layout()

# Save the figure
plt.savefig('zzz_images_comparison.png')


plt.figure()
plt.hist(Y_mat_mul.reshape(-1))
plt.hist(Y_FFT.reshape(-1))
plt.savefig("ZZZ___PLT")

'''
    Test super resolution
'''