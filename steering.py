# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 14:58:26 2026

@author: Max
"""

#Steering
#Loading Model
from transformer_lens import HookedTransformer
from huggingface_hub import login
from sae_lens import SAE
import os
import pickle


def read_pickle(path_in, name_in):
    the_data_t = pickle.load(
        open(path_in + name_in + ".pk", "rb"))
    return the_data_t

data_path = './data/'
top_neg_indices = read_pickle(data_path, 'top_neg_indices')
top_post_indices = read_pickle(data_path, 'top_pos_indices')

login(token=os.getenv('TOKEN'))

device = 'cuda'
#Load SAE
sae = SAE.from_pretrained(
    release='gemma-scope-2b-pt-res-canonical', 
    sae_id='layer_20/width_16k/canonical',
    device=device
)

model = HookedTransformer.from_pretrained('gemma-2-2b', device=device)

#Defining the Steering Vector
#Take the mean of Cluster A features

cluster_a_indices = top_neg_indices
steering_vector = sae.W_dec[cluster_a_indices].mean(dim=0)

#Steering Hook to run inside the model at Layer 20
coefficients = [-150.0, 0.0, 150.0]
prompt = 'The chemistry of love is'

print('DETERMINISTIC SWEEP (Chemistry of Love)')
for coeff in coefficients:
    #Define hook with current coefficient
    def current_steering_hook(resid_pre, hook):
        #We add the steering vector * coeff
        resid_pre += steering_vector * coeff
        return resid_pre

    model.reset_hooks()
    
    #When coeff is 0, run normal generation
    if coeff == 0:
        output = model.generate(prompt, max_new_tokens=40, temperature=0.0, verbose=False)
        print(f'\n[Coeff {coeff} (Control)]: \n{output}')
    else:
        # Attach hook for this specific run
        with model.hooks(fwd_hooks=[('blocks.20.hook_resid_pre', current_steering_hook)]):
            output = model.generate(prompt, max_new_tokens=40, temperature=0.0, verbose=False)
            print(f'\n[Coeff {coeff}]: \n{output}')
            
            
prompt = 'The morning sun rises above the hills.'

print('DETERMINISTIC SWEEP (Beginning of a poem)')
for coeff in coefficients:
    #Define hook with current coefficient
    def current_steering_hook(resid_pre, hook):
        #We add the steering vector * coeff
        resid_pre += steering_vector * coeff
        return resid_pre

    model.reset_hooks()
    
    #When coeff is 0, run normal generation
    if coeff == 0:
        output = model.generate(prompt, max_new_tokens=40, temperature=0.0, verbose=False)
        print(f'\n[Coeff {coeff} (Control)]: \n{output}')
    else:
        # Attach hook for this specific run
        with model.hooks(fwd_hooks=[('blocks.20.hook_resid_pre', current_steering_hook)]):
            output = model.generate(prompt, max_new_tokens=40, temperature=0.0, verbose=False)
            print(f'\n[Coeff {coeff}]: \n{output}')
            

prompt = 'The chemistry of love is'

print('VARIABLE SWEEP (Chemistry of Love)')
for coeff in coefficients:
    #Define hook with current coefficient
    def current_steering_hook(resid_pre, hook):
        #We add the steering vector * coeff
        resid_pre += steering_vector * coeff
        return resid_pre

    model.reset_hooks()
    
    #When coeff is 0, run normal generation
    if coeff == 0:
        output = model.generate(prompt, max_new_tokens=40, temperature=1.0, verbose=False)
        print(f'\n[Coeff {coeff} (Control)]: \n{output}')
    else:
        # Attach hook for this specific run
        with model.hooks(fwd_hooks=[('blocks.20.hook_resid_pre', current_steering_hook)]):
            output = model.generate(prompt, max_new_tokens=40, temperature=1.0, verbose=False)
            print(f'\n[Coeff {coeff}]: \n{output}')