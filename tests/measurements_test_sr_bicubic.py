import sys

import scipy.linalg
import scipy.signal 

import copy
import matplotlib.pyplot as plt
import torch
import scipy 
import numpy as np
import cv2 

from guided_diffusion.sr_conv_test_only import SRConv, SuperResolution
from guided_diffusion.measurements import SuperResolutionOperatorV2

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


def calculate_delta_M_v1(d,  m, delta):    
    if len(delta) != m * d:
        raise ValueError("Length of delta must be m*d")

    sqrt_d, sqrt_m = int(d**0.5), int(m**0.5)
    i = torch.arange(m)

    r_offset = (i%sqrt_m + sqrt_d * (sqrt_m) * (i // sqrt_m)).long().unsqueeze(1).unsqueeze(2)
    offset = m * sqrt_d * torch.arange(sqrt_d).unsqueeze(0).unsqueeze(2)
    offset2 = sqrt_m * torch.arange(sqrt_d).unsqueeze(0).unsqueeze(1)
    all_offset = (r_offset + offset + offset2).reshape(-1)

    lambda_i = torch.gather(delta,0,all_offset).reshape(m,-1).sum(axis=1) / d
    return lambda_i.reshape(sqrt_m, sqrt_m)


'''
    Deblurring
'''
N = 128
d = 4 
M = N // d
sample_image = torch.rand(N,N,3)
img = cv2.resize(cv2.imread("data/samples/00015.png"), (N,N)).astype(np.float32)[:,:,::-1] / 255
# print(img)
sample_image = torch.from_numpy(img) 
sigma_n = 0.05

factor = 4
def bicubic_kernel(x, a=-0.5):
    if abs(x) <= 1:
        return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
    elif 1 < abs(x) and abs(x) < 2:
        return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
    else:
        return 0
k = np.zeros((factor * 4))
for i in range(factor * 4):
    x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
    k[i] = bicubic_kernel(x)
k = k / np.sum(k)
kernel = torch.from_numpy(k).float()

def get_gaussian_blur(kernel_size, std):
    n = np.zeros((kernel_size)).astype(np.float64)
    n[kernel_size // 2] = 1
    return scipy.ndimage.gaussian_filter(n, sigma=std)

k = get_gaussian_blur(9,3)
kernel = torch.from_numpy(k)


H_goal = SRConv(kernel / kernel.sum(), 3, N, "cpu", stride=factor)
H_goal_avg = SuperResolution(3, N, factor, "cpu")
print("image shape ", sample_image.permute(2,0,1).unsqueeze(0).shape)
img_ready_tensor = sample_image.permute(2,0,1).unsqueeze(0)
yy_g = H_goal.H(img_ready_tensor)
Y_goal = yy_g.squeeze().reshape(3,N//factor,N//factor).permute(1,2,0)
Y_goal_pinv = H_goal.H_pinv(yy_g).squeeze().reshape(3,N,N).permute(1,2,0)
Y_goal_avg = H_goal_avg.H(img_ready_tensor).squeeze().reshape(3,N//factor,N//factor).permute(1,2,0)

plt.imshow(Y_goal)
plt.savefig("zzz_sr_goal.png")
plt.imshow(Y_goal_avg)
plt.savefig("zzz_Y_goal_avg_goal.png")

kernel_2d = torch.outer(kernel, kernel)
# kernel_2d = torch.ones(factor,factor)
# kernel_2d /= kernel_2d.sum()
k = kernel_2d.numpy()


N_FFT = (N)**2
pad_width = pad_height = N-len(k)
# padding = (pad_width//2, pad_width//2, pad_height//2, pad_height//2) 
padding = (pad_width//2+1, pad_width//2, pad_height//2+1, pad_height//2) 

padded_kernel = (torch.nn.functional.pad(kernel_2d, padding))
# Flip padded kernel becuase convolution vs correaltion inpytorch 
padded_kernel = padded_kernel
plt.imshow(padded_kernel)
plt.savefig("zzz_paddedkernel_before_fft")
padded_kernel = torch.fft.ifftshift(padded_kernel).type(torch.float32)
print("padded kernel shape ", padded_kernel.shape)
plt.imshow(padded_kernel)
plt.savefig("zzz_paddedkernel")

H_j = []

for i in range(len(padded_kernel)):
    H_j.append(torch.from_numpy(scipy.linalg.circulant(padded_kernel[i,:])).type(torch.float32))
H = circular_stack(H_j)
assert_cirulant(H,H_j)

print("First columnas are equal, ", H[:,0] - padded_kernel.reshape(-1), torch.equal(H[:,0], padded_kernel.reshape(-1)))
delta = torch.fft.fft2(padded_kernel.reshape(N,N)).type(torch.complex64)#.reshape(-1)
delta_conj = torch.adjoint(delta)
# delta = torch.diag(torch.fft.fftn(H[:,0])).type(torch.complex64)
img_fft = torch.fft.fft2(sample_image, dim=[0,1]).type(torch.complex64)
print("Deta ", delta.shape, img_fft.shape)
Y_hat = torch.fft.ifft2(delta.unsqueeze(-1) * img_fft, dim=[0,1])
Y_hat = torch.real(Y_hat).reshape(N,N,3)
Y_hat_sr = Y_hat[::d,::d,:]
# Y_hat_sr = Y_hat.reshape((N // d, d, N // d, d,3))[:, d//2, :, d//2]
print("Y_hat_sr.shape ", Y_hat_sr.shape)
delta_squared = delta * delta_conj
delta_m = calculate_delta_M_v1(d**2, M**2, delta_squared.reshape(-1)).reshape(M,M)
# Y_hat_sr_mine = torch.fft.ifft2(delta_m.unsqueeze(-1) * img_fft, dim=[0,1])
# Y_hat_sr_mine = torch.real(Y_hat_sr_mine).reshape(M,M,3)

Y_mat_mul = (H @ sample_image.reshape(-1,3)).reshape(N,N,3)
S = torch.zeros(M**2, N**2).float()
row_indices = torch.arange(0, N, d).repeat(1, M)
column_indices = torch.arange(0, N**2, M*d**2).view(-1, 1).repeat(1, N//d).reshape(-1)
indices = row_indices + column_indices
S[torch.arange(M**2), indices] = 1

A =  S@H
AAT = A @ A.T 

A_pinv = A.T @ torch.linalg.inv(AAT)


DFT_M = torch.fft.fft(torch.eye(M), norm='ortho')
DFT_2_M = torch.kron(DFT_M, DFT_M)
DFT_2_M_ADJ = torch.adjoint(DFT_2_M)

Y_mat_mul_sr_orig = (A @ sample_image.reshape(-1,3)).reshape(M,M,3)
Y_mat_mul_sr = Y_mat_mul[0::d,0::d,:]

act_operator    = SuperResolutionOperatorV2(scale_factor = 4, degradation = 'gaussian', device = 'cpu')
st_op           = SuperResolutionOperatorV2(scale_factor = 4, degradation = 'avg_pool', device = 'cpu')
y_from_operator_tensor = act_operator.forward(img_ready_tensor)
y_from_operator = y_from_operator_tensor.squeeze().permute(1,2,0)
y_from_sr_avg_operator_tensor = st_op.forward(img_ready_tensor)
y_from_sr_avg_operator = y_from_sr_avg_operator_tensor.squeeze().permute(1,2,0)

y_p_inv = (A_pinv @ Y_mat_mul_sr.reshape(-1,3)).reshape(N,N,3)
y_p_inv_mine = act_operator.get_x_start(y_from_operator_tensor).squeeze().permute(1,2,0)


plt.imshow(y_p_inv.detach().cpu())
plt.savefig("zzz_y_p_inv_best.png")
plt.imshow(y_p_inv_mine.detach().cpu())
plt.savefig("zzz_y_p_inv_mine.png")
plt.imshow(Y_goal_pinv.detach().cpu())
plt.savefig("zzz_Y_goal_pinv.png")



print("-"*15, " errrs y_from_operator and y goal", torch.max(y_from_operator - Y_goal) )
print("-"*15, " errrs y_from_operator and y Y_mat_mul_sr", torch.max(y_from_operator - Y_mat_mul_sr) )
print("-"*15, " errrs Y_goal_avg and y y_from_sr_avg_operator", torch.max(Y_goal_avg - y_from_sr_avg_operator) )
print("-"*15, " errrs y_p_inv_mine and y y_p_inv", y_p_inv.shape , y_p_inv_mine.shape )
print("-"*15, " errrs y_p_inv_mine and y y_p_inv", torch.max(y_p_inv - y_p_inv_mine) )
print("-"*15, " errrs y_p_inv_mine and y y_p_inv", torch.max(y_p_inv - Y_goal_pinv) )

AAT_mine = DFT_2_M_ADJ @ torch.diag(delta_m.reshape(-1)) @ DFT_2_M

# Using 2d fourier transform 
img_fft = torch.fft.fftn(sample_image, dim=[0,1])
kernel_fft = torch.fft.fftn(padded_kernel, s= [N,N], dim=[0,1]).unsqueeze(-1)
Y_FFT = torch.fft.ifftn(img_fft * kernel_fft, dim=[0,1]).abs()

print("Error Y_hat - Y_FFT ", torch.max(Y_hat - Y_FFT))
print("Error Y_mat_mul - Y_FFT ", torch.max(Y_mat_mul - Y_FFT))
print("Error Y_mat_mul - Y_hat ", torch.max(Y_mat_mul - Y_hat))
# print("Error Y_goal - Y_hat ", torch.max(Y_goal - Y_hat))



print("-"*(25), "SR results ", "-"*25)
print("Error Y_mat_mul_sr_orig - Y_mat_mul_sr: ", torch.max(Y_mat_mul_sr_orig - Y_mat_mul_sr) )
print("Error AAT - AAT_mine: ", torch.allclose(AAT, AAT_mine.real, atol=1e-5), torch.max(AAT - AAT_mine.real))
print("Error Y_hat_sr - Y_goal: ", torch.max(Y_hat_sr - Y_goal) )
print("Error Y_mat_mul_sr_orig - Y_goal: ", torch.max(Y_mat_mul_sr_orig - Y_goal) )
print("Error y_from_sr_avg_operator - Y_mat_mul_sr: ", torch.max(y_from_sr_avg_operator - Y_mat_mul_sr) )
print("Error y_from_sr_avg_operator - Y_goal_avg: ", torch.max(y_from_sr_avg_operator - Y_goal_avg) )
# print("Kernel shape:", padded_kernel.shape)
# print("Image FFT shape:", img_fft.shape)
# print("Delta shape:", delta.shape)
# print("Troch equal ", torch.equal(Y,Y_hat), (Y-Y_hat))

# Display the reconstructed image (Y_hat)
plt.subplot(2, 3, 2)  # Subplot with 1 row, 2 columns, position 2
plt.imshow(Y_hat_sr)  # Display color image
plt.title('Y_hat_sr')
plt.axis('off')

# Display the reconstructed image (Y_hat)
# plt.subplot(2, 3, 3)  # Subplot with 1 row, 2 columns, position 2
# plt.imshow(Y_mat_mul)  # Display color image
# plt.title('Y_mat_mul')
# plt.axis('off')
plt.subplot(2, 3, 3)  # Subplot with 1 row, 2 columns, position 2
plt.imshow(Y_goal)  # Display color image
plt.title('Y_goal')
plt.axis('off')



# Display the reconstructed image (Y_hat)
plt.subplot(2, 3, 4)  # Subplot with 1 row, 2 columns, position 2
plt.imshow(torch.abs(Y_goal-Y_hat_sr))  # Display color image
plt.title('Y_hat_sr res')
plt.axis('off')

# Display the reconstructed image (Y_hat)
# plt.subplot(2, 3, 5)  # Subplot with 1 row, 2 columns, position 2
# plt.imshow(torch.abs(Y_goal-Y_mat_mul))  # Display color image
# plt.title('Y_mat_mul')
# plt.axis('off')

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