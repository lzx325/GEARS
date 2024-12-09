import torch
from torch.nn.utils import clip_grad_norm_
import numpy as np
from tqdm import tqdm
CFG = False

def train_basic(model, dataloader, optimizer, lr_scheduler, criterion, device):
    
    model.train()  # Set the model to training mode
    total_loss = 0.0

    for i, batch in enumerate(tqdm(dataloader)):
        # Get the inputs and move them to the specified device
        n_genes = dataloader.dataset[0].x.shape[0]
        batch_size= batch.y.shape[0]
        control_x = batch.x.reshape((batch_size, n_genes)).to(device)
        perturb_x = batch.y.to(device)
        perturb_cond = batch.pert_idx
        if CFG:
            mask_indices = np.where(np.random.rand(len(batch))<0.1)[0]
            for i in mask_indices:
                perturb_cond[i] = [-1]
        t = torch.rand(batch_size).view(batch_size, 1).to(device)
        x_t =  t * perturb_x + (1.-t) * control_x
        # target = perturb_x - control_x
        target = perturb_x
        # noise = torch.randn_like(x_t) * t * (1-t)
        # x_t = x_t + noise
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        _, output = model(x_t, t, perturb_cond)
        
        # Compute the loss
        # import pdb; pdb.set_trace()
        loss = criterion(output, target)

        # Backward pass and optimization
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        lr_scheduler.step()

        # Add the batch's loss to the total loss for this epoch
        total_loss += loss.item()

    # Compute the average loss for this epoch
    avg_loss = total_loss / len(dataloader)
    return avg_loss

def eval_basic(model, dataloader, criterion, device):
    model.eval()  # Set the model to eval mode
    total_loss = 0.0

    for i, batch in enumerate(tqdm(dataloader)):
        n_genes = dataloader.dataset[0].x.shape[0]
        batch_size= batch.y.shape[0]
        # Get the inputs and move them to the specified device
        control_x = batch.x.reshape((batch_size, n_genes)).to(device)
        perturb_x = batch.y.to(device)
        perturb_cond = batch.pert_idx

        bsz = control_x.shape[0]

        t = torch.rand(bsz).view(bsz, 1).to(device)
        x_t =  t * perturb_x + (1.-t) * control_x
        # target = perturb_x - control_x
        target = perturb_x
        
        with torch.no_grad():
            _, output = model(x_t, t, perturb_cond)
        
            # Compute the loss
            # import pdb; pdb.set_trace()
            loss = criterion(output, target)


        # Add the batch's loss to the total loss for this epoch
        total_loss += loss.item()

    # Compute the average loss for this epoch
    avg_loss = total_loss / len(dataloader)
    return avg_loss
