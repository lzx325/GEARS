#!/usr/bin/env python
# coding: utf-8



import sys
sys.path.append('../')

import pickle as pkl

import numpy as np
import anndata as ad
import pandas as pd
from gears import PertData, GEARS
from gears.utils import parse_single_pert,parse_combo_pert



# Load data. We use norman as an example.




data_path = './'
data_name = 'norman_umi_go'
model_name = 'gears_misc_umi_no_test'

pert_data = PertData(data_path)




pert_data.load(data_path = data_path + data_name)





n_combo=int(sys.argv[1])
mode=sys.argv[2]
print("n_combo =",n_combo)
pert_data.prepare_split(split = 'combo_seen2', seed = 1, split_path = "norman_umi_go/splits/custom_split_%d.pkl"%(n_combo))
pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)

# Create a model object; if you use [wandb](https://wandb.ai), you can easily track model training and evaluation by setting `weight_bias_track` to true, and specify the `proj_name` and `exp_name` that you like.

gears_model = GEARS(pert_data, device = 'cuda:0', 
                        weight_bias_track = False, 
                        proj_name = 'pertnet', 
                        exp_name = 'pertnet')
gears_model.model_initialize(hidden_size = 64, no_GO = True)
print(gears_model.tunable_parameters())

if mode=='train':
    gears_model.train(epochs = 10 , lr = 1e-3)
    gears_model.save_model('best_model-%d-no_GO'%(n_combo))

gears_model.load_pretrained('best_model-%d-no_GO'%(n_combo))

from gears.inference import evaluate,compute_metrics
test_res = evaluate(gears_model.dataloader['test_loader'], gears_model.best_model,
                    gears_model.config['uncertainty'], gears_model.device)
test_metrics, test_pert_res = compute_metrics(test_res)  
print(n_combo,test_metrics)


# Make prediction for new perturbation:

# Gene list can be found here:
