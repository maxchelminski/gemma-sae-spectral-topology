# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 15:08:59 2025

@author: Max
"""

import torch
from sae_lens import SAE
import matplotlib.pyplot as plt
import pickle

def write_pickle(obj_in, path_in, name_in):
    pickle.dump(obj_in, open(
        path_in + name_in + '.pk', 'wb'))

data_path = './data/'
device = 'cuda'

#Load SAE
sae = SAE.from_pretrained(
    release='gemma-scope-2b-pt-res-canonical', 
    sae_id='layer_20/width_16k/canonical', #Layer 20
    device=device
)

#Extract Feature Matrix
features = sae.W_dec.data

#Compute Adjacency Matrix (Cosine Similarity)
features = torch.nn.functional.normalize(features, p=2, dim=0)
adj_matrix = torch.mm(features.T, features)

#Clean diagonal
adj_matrix.fill_diagonal_(0)

#Threshold
adj_matrix[adj_matrix < 0.1] = 0
print(f'Graph constructed with {adj_matrix.count_nonzero()} edges.')

#Calculate Degree Matrix
degrees = adj_matrix.sum(dim=1)

#Compute Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees + 1e-8)) #avoid divide by 0
L_sym = torch.eye(features.shape[1], device=device) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

#Eigen Decomposition
eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)
fiedler_vector = eigenvectors[:, 1]

#Extracting Poles of Meaning
top_neg_indices = torch.topk(fiedler_vector, 10, largest=False).indices
write_pickle(top_neg_indices, data_path, 'top_neg_indices')
top_pos_indices = torch.topk(fiedler_vector, 10, largest=True).indices
write_pickle(top_pos_indices, data_path, 'top_pos_indices')
print(f'Cluster A (Negative Pole) Indices: {top_neg_indices.tolist()}')
print(f'Cluster B (Positive Pole) Indices: {top_pos_indices.tolist()}')

'''
NEXT STEP: Go to https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k
And search for the Feature IDs
'''


#Plotting the dimensions of gemma's SAEs
x_coords = eigenvectors[:, 1].cpu().numpy() #Fiedler
y_coords = eigenvectors[:, 2].cpu().numpy() #Next

plt.figure()
plt.scatter(x_coords, y_coords, alpha=0.5, s=1, c='blue')

plt.scatter(x_coords[top_neg_indices.cpu()], y_coords[top_neg_indices.cpu()], c='red', s=50, label='Cluster A (Formal System)')
plt.scatter(x_coords[top_pos_indices.cpu()], y_coords[top_pos_indices.cpu()], c='green', s=50, label='Cluster B (Narrative Reality)')

plt.title(f'Spectral Map of Gemma-2B Concepts (Layer 20)\nVisualizing {features.shape[1]} Features')
plt.xlabel('Dimension 1: Formal System <--> Narrative Reality')
plt.ylabel('Dimension 2')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{data_path}spectral_map.png')
plt.show()



#Recompute adjacency matrix to observe better shapes and connections


# adj_matrix = torch.mm(features.T, features)

# adj_matrix.fill_diagonal_(0)

# adj_matrix[adj_matrix < 0.01] = 0 

# degrees = adj_matrix.sum(dim=1)
# D_inv_sqrt = torch.diag(1.0 / torch.sqrt(degrees + 1e-8))
# L_sym = torch.eye(features.shape[1], device=device) - D_inv_sqrt @ adj_matrix @ D_inv_sqrt

# eigenvalues, eigenvectors = torch.linalg.eigh(L_sym)

# #Plotting the dimensions of gemma's SAEs again
# x_coords = eigenvectors[:, 1].cpu().numpy() #Fiedler
# y_coords = eigenvectors[:, 2].cpu().numpy() #Next

# plt.figure()
# plt.scatter(x_coords, y_coords, alpha=0.5, s=1, c='blue')

# plt.scatter(x_coords[top_neg_indices.cpu()], y_coords[top_neg_indices.cpu()], c='red', s=50, label='Cluster A (Formal System)')
# plt.scatter(x_coords[top_pos_indices.cpu()], y_coords[top_pos_indices.cpu()], c='green', s=50, label='Cluster B (Narrative Reality)')

# plt.title(f"Spectral Map of Gemma-2B Concepts (Layer 20)\nVisualizing {features.shape[1]} Features")
# plt.xlabel("Dimension 1: Formal System <--> Narrative Reality")
# plt.ylabel("Dimension 2")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()



