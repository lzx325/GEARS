import sys
sys.path.append('../')
import pickle as pkl

import numpy as np
import anndata as ad
import pandas as pd
import torch
import torch.optim

from gears import PertData, GEARS
from gears.utils import parse_single_pert,parse_combo_pert

from flow_model import GEARS_Model
from flow_train import train_basic, eval_basic
def main():
    # preparing data
    data_path = './'
    data_name = 'norman_umi_go'

    pert_data = PertData(data_path)
    pert_data.load(data_path = data_path + data_name)
    n_combo=66
    pert_data.prepare_split(split = 'combo_seen2', seed = 1, split_path = "norman_umi_go/splits/custom_split_%d.pkl"%(n_combo))
    pert_data.get_dataloader(batch_size = 32, test_batch_size = 128)
    # preparing model
    with open("gears_config.pkl",'rb') as f:
        config_dict=pkl.load(f)
        config=config_dict["config"]
        config["gene_names"]=config_dict["gene_names"]
        config["pert_names"]=config_dict["pert_names"]
        config["device"]="cuda"

    model = GEARS_Model(config).to(config["device"])
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = torch.nn.MSELoss()
    for epoch in range(20):
        avg_loss_train=train_basic(
            model, 
            pert_data.dataloader['train_loader'], 
            optimizer, 
            lr_scheduler, 
            criterion, 
            config["device"]
        )
        

        avg_loss_test=eval_basic(
            model, 
            pert_data.dataloader["test_loader"], 
            criterion, 
            config["device"]
        )

        print("Epoch %d, train loss %.4f, test loss %.4f"%(epoch,avg_loss_train,avg_loss_test))

if __name__ == '__main__':
    main()