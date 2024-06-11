import torch 
import numpy as np 
import time

def calculate_gamma_v2(d, i, m, delta):    
    if len(delta) != m * d:
        raise ValueError("Length of delta must be m*d")
    N = int((m*d)**0.5)
    delta = delta.reshape(N, N)    
    sqrt_d, sqrt_m = int(d**0.5), int(np.sqrt(m))
    lambda_i = delta[::sqrt_d,::sqrt_d].sum() / d
    
    return lambda_i.float()


def calculate_gamma_v2_all(d, m, delta):
    if len(delta) != m * d:
        raise ValueError("Length of delta must be m*d")
    sqrt_d, sqrt_m = int(d**0.5), int(np.sqrt(m))
    
    lambda_i = float(0)
    i = torch.arange(m)
    r_offset = (i%sqrt_m + sqrt_d * (sqrt_m) * (i // sqrt_m)).long().unsqueeze(1).unsqueeze(2)
    offset = m * sqrt_d * torch.arange(sqrt_d).unsqueeze(0).unsqueeze(2)
    offset2 = sqrt_m * torch.arange(sqrt_d).unsqueeze(0).unsqueeze(1)

    all_offset = (r_offset + offset + offset2).reshape(m,-1)

    lambda_i = torch.zeros(m).float()
    for i in range(m):
        lambda_i[i] = torch.sum(torch.gather(delta,0,all_offset[i])).float()
      
    lambda_i /= d
    
    return lambda_i

def calculate_gamma_v3_all(d, m, delta):
    if len(delta) != m * d:
        raise ValueError("Length of delta must be m*d")
    sqrt_d, sqrt_m = int(d**0.5), int(np.sqrt(m))
    
    lambda_i = float(0)
    i = torch.arange(m)
    r_offset = (i%sqrt_m + sqrt_d * (sqrt_m) * (i // sqrt_m)).long().unsqueeze(1).unsqueeze(2)
    offset = m * sqrt_d * torch.arange(sqrt_d).unsqueeze(0).unsqueeze(2)
    offset2 = sqrt_m * torch.arange(sqrt_d).unsqueeze(0).unsqueeze(1)

    all_offset = (r_offset + offset + offset2).reshape(-1)

    lambda_i = torch.gather(delta,0,all_offset).reshape(m,-1).sum(axis=1).float()
      
    lambda_i /= d
    
    return lambda_i

def calculate_gamma_v1(d, i, m, delta):    
    if len(delta) != m * d:
        raise ValueError("Length of delta must be m*d")
    sqrt_d, sqrt_m = int(d**0.5), int(np.sqrt(m))
    
    lambda_i = float(0)
    r_offset = i%sqrt_m + sqrt_d * (sqrt_m) * (i // sqrt_m)
    for out_d in range(sqrt_d):
      offset = m * sqrt_d * out_d
      lambda_i += torch.sum(delta[offset + r_offset :offset + r_offset + sqrt_d * sqrt_m:sqrt_m])
      
    lambda_i /= d
    
    return lambda_i

def calculate_gamma(d, i, m, delta, log = False):
    if len(delta) != m * d:
        raise ValueError("Length of delta must be m*d")
    sqrt_d, sqrt_m = int(d**0.5), int(np.sqrt(m))
    
    lambda_i = float(0)
    for out_d in range(sqrt_d):
      offset = m * sqrt_d * out_d
      for in_d in range(sqrt_d):
          lambda_i += delta[offset + sqrt_m * in_d + i%sqrt_m + sqrt_d * (sqrt_m) * (i // sqrt_m)]
    lambda_i /= d
    
    return lambda_i
N = 4 
M = 2
d = N//M
delta = torch.arange(N**2).float()
gamma = torch.arange(M**2)

for i in range(M**2):
  gamma[i] = calculate_gamma(d**2,i,M**2, delta)
  '''
    Expected result
    0 tensor(5.)
    1 tensor(6.)
    2 tensor(9.)
    3 tensor(10.)
  '''

gamma_v1 = torch.tensor([ calculate_gamma_v1(d**2,i,M**2, delta) for i in range (M**2)])
gamma_v2_all = calculate_gamma_v2_all(d**2, M**2, delta)
gamma_v3_all = calculate_gamma_v3_all(d**2, M**2, delta)

# gamma_v2 = torch.tensor([ calculate_gamma_v2(d**2,i,M**2, delta) for i in range (M**2)])

# for i in range(M**2):
#   gamma_v1[i] = 

print("Gamma ", gamma)
print("gamma_v1 ", gamma_v1)
print("gamma_v2_all ", gamma_v2_all)
print("gamma_v3_all ", gamma_v3_all)

N = 64 
M = 2
d = N//M
delta = torch.arange(N**2)


# Start measuring time for the first function
start_time = time.time()
gamma = torch.arange(M**2).float()
for i in range(M**2):
    gamma[i] = calculate_gamma(d**2, i, M**2, delta)
end_time = time.time()
gamma_time = end_time - start_time

# Start measuring time for the second function
start_time = time.time()
gamma_v1 = torch.tensor([calculate_gamma_v1(d**2, i, M**2, delta) for i in range(M**2)])
end_time = time.time()
gamma_v1_time = end_time - start_time

# Start measuring time for the third function
start_time = time.time()
gamma_v2_all = calculate_gamma_v2_all(d**2, M**2, delta)
end_time = time.time()
gamma_v2_all_time = end_time - start_time

# Start measuring time for the third function
start_time = time.time()
gamma_v3_all = calculate_gamma_v3_all(d**2, M**2, delta)
end_time = time.time()
gamma_v3_all_time = end_time - start_time

# Print the outputs and time taken by each function
print("Gamma:", gamma)
print("Gamma_v1:", gamma_v1)
print("Gamma_v2_all:", gamma_v2_all)
print("Gamma_v3_all:", gamma_v3_all)

print("Time taken by gamma:", gamma_time)
print("Time taken by gamma_v1:", gamma_v1_time)
print("Time taken by gamma_v2_all:", gamma_v2_all_time)
print("Time taken by gamma_v3_all:", gamma_v3_all_time)

# Assert all versions produce the same output
assert torch.allclose(gamma, gamma_v1)
assert torch.allclose(gamma, gamma_v2_all)
assert torch.allclose(gamma, gamma_v3_all)