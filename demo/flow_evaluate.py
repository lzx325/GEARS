import torch
from utils import repeat_n
import umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import wandb
from scipy.stats import pearsonr


@torch.no_grad()
def sample_ode(model, z0=None, N=None, l1=None):
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
        N = 1000   
    dt = 1./N
    traj = [] # to store the trajectory
    z = z0.detach().clone()
    batchsize = z.shape[0]

    # traj.append(z.detach().clone().cpu().numpy())
    for i in range(N):
        t = torch.ones(batchsize) * i / N
        _, pred = model(z, t.cuda(), l1)
        # import pdb; pdb.set_trace()
        z = z.detach().clone() + pred * dt
        
        # traj.append(z.detach().clone().cpu().numpy())

    return z.detach().clone().cpu().numpy()

@torch.no_grad()
def sample_ode_with_cfg(model, z0=None, N=None, l1=None, guidance=3.0):
    ### NOTE: Use Euler method to sample from the learned flow
    if N is None:
        N = 1000   
    dt = 1./N
    traj = [] # to store the trajectory
    z = z0.detach().clone()
    batchsize = z.shape[0]

    # traj.append(z.detach().clone().cpu().numpy())
    for i in range(N):
        t = torch.ones(batchsize) * i / N
        _, pred_cond = model(z, t.cuda(), l1)
        l2 = torch.zeros_like(l1)
        _, pred_uncond = model(z, t.cuda(), l2)
        # import pdb; pdb.set_trace()
        pred = pred_uncond + guidance * (pred_cond - pred_uncond)
        z = z.detach().clone() + pred * dt
        
        # traj.append(z.detach().clone().cpu().numpy())

    return z.detach().clone().cpu().numpy()



def umap_plot(control_data, perturb_data, synthetic_data, epoch=0):
    # Concatenate real and synthetic data
    combined_data = np.vstack((control_data, perturb_data, synthetic_data))

    # Perform PCA
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(combined_data)

    # Perform UMAP
    reducer = umap.UMAP(random_state=42)
    embedding = reducer.fit_transform(pca_result)
    fig, ax = plt.subplots()
                
    # Plotting
    ax.scatter(embedding[:len(control_data), 0], embedding[:len(control_data), 1], label='control data', alpha=0.5)
    ax.scatter(embedding[len(control_data):len(control_data)+len(perturb_data), 0], embedding[len(control_data):len(control_data)+len(perturb_data), 1], label='perturbed data', alpha=0.5)
    ax.scatter(embedding[len(control_data)+len(perturb_data):, 0], embedding[len(control_data)+len(perturb_data):, 1], label='synthetic data', alpha=0.5)
    plt.legend()
    plt.savefig(f"figures_3e-4/umap_{epoch}.png")
    # wandb.log({"umap embedding": fig})


def umap_plot(control_data, perturb_data_dict, synthetic_data_dict, epoch):
    # Step 1: Dimensionality reduction
    all_data = [control_data]
    labels = ["control"] * len(control_data)
    
    for perturb_name, perturb_data in perturb_data_dict.items():
        all_data.append(perturb_data)
        labels.extend([f"real-{perturb_name}"] * len(perturb_data))
        
    for perturb_name, synthetic_data in synthetic_data_dict.items():
        all_data.append(synthetic_data)
        labels.extend([f"synthetic-{perturb_name}"] * len(synthetic_data))
        
    all_data_np = np.vstack(all_data)
    # Perform PCA
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(all_data_np)

    reducer = umap.UMAP()
    embedding = reducer.fit_transform(pca_result)

    # Step 2: Visualization with matplotlib
    plt.figure(figsize=(10, 8))
    
    labels = np.array(labels)
    # Control data
    plt.scatter(embedding[labels == "control", 0], embedding[labels == "control", 1], c='black', label='Control')
    
    unique_perturbations = set(name.split('-')[-1] for name in labels if name != "control")
    color_map = plt.cm.get_cmap('viridis', len(unique_perturbations))
    
    for idx, perturb_name in enumerate(unique_perturbations):
        plt.scatter(embedding[labels == f"real-{perturb_name}", 0], embedding[labels == f"real-{perturb_name}", 1], c=[color_map(idx)], label=f"Real {perturb_name}")
        plt.scatter(embedding[labels == f"synthetic-{perturb_name}", 0], embedding[labels == f"synthetic-{perturb_name}", 1], c=[color_map(idx)], alpha=0.5, label=f"Synthetic {perturb_name}")
    
    plt.title(f"UMAP Projection - Epoch {epoch}")
    plt.legend()
    plt.savefig(f"figures/3e-4/umap_{epoch}.png")
    # import pdb; pdb.set_trace()







def sample_from_conditional_model(model, dataset, device, sample_size=1000, epoch=0, guidance=2.0):
    """
    Sample from the model 

    Z0 = dataset.control_X
    Z1 = dataset.perturb_X
    l1 = dataset.perturbations # l1 are the labels of Z1
    t = torch.ones(batchsize) * i / N
    pred = model(z0, t.cuda(), l1)
    
    Then check the distribution of pred compared with that of Z1
    """
    # import pdb; pdb.set_trace()
    control_x = dataset.control_X
    random_indices = np.random.randint(0, control_x.shape[0], size=sample_size)
    control_x = control_x[random_indices].to(device)
    perturb_x = dataset.perturb_X
    perturb_names = np.array(dataset.perturb_names)

    unique_perturbations = torch.unique(dataset.perturbations, dim=0)
    pearsonr_values = []
    pearsonr_values_cfg = []
    deg_peasonr_values = []
    deg_peasonr_values_cfg = []
    MSE_deg_values = []
    MSE_deg_values_cfg = []
    table_data = []
    perturb_data_dict = {}
    synthetic_data_dict = {}

    for perturb_cond_val in unique_perturbations:
        mask = (dataset.perturbations == perturb_cond_val).squeeze()
        mask = mask.all(dim=1)
        current_perturb_x = perturb_x[mask]
        current_perturb_names = perturb_names[mask][0]

        perturb_cond = perturb_cond_val.unsqueeze(0).repeat(control_x.shape[0], 1).to(device)
        # perturb_cond = torch.repeat_interleave(perturb_cond_val, control_x.shape[0]).to(device)
        synthetic_data = sample_ode(model, z0=control_x, N=1000, l1=perturb_cond)

        synthetic_data_cfg = sample_ode_with_cfg(model, z0=control_x, N=1000, l1=perturb_cond, guidance=guidance)

        # synthetic_data = traj[-1]
        # synthetic_data_cfg = traj_cfg[-1]
        
        mean_perturb = np.mean(current_perturb_x.numpy(), axis=0)
        mean_synthetic = np.mean(synthetic_data, axis=0)
        mean_synthetic_cfg = np.mean(synthetic_data_cfg, axis=0)

        pearsonr_perturb = pearsonr(mean_perturb, mean_synthetic)[0]
        pearsonr_perturb_cfg = pearsonr(mean_perturb, mean_synthetic_cfg)[0]
        
        pearsonr_values.append(pearsonr_perturb)
        pearsonr_values_cfg.append(pearsonr_perturb_cfg)
        
        # Get the pearson again with the top20 DE genes
        current_indice = dataset.top_gene_indices[current_perturb_names]
        pearsonr_perturb_deg = pearsonr(mean_perturb[current_indice], mean_synthetic[current_indice])[0]
        pearsonr_perturb_deg_cfg = pearsonr(mean_perturb[current_indice], mean_synthetic_cfg[current_indice])[0]
        MSE_deg = np.mean((mean_perturb[current_indice] - mean_synthetic[current_indice])**2)
        MSE_deg_cfg = np.mean((mean_perturb[current_indice] - mean_synthetic_cfg[current_indice])**2)


        deg_peasonr_values.append(pearsonr_perturb_deg)
        deg_peasonr_values_cfg.append(pearsonr_perturb_deg_cfg)
        MSE_deg_values.append(MSE_deg)
        MSE_deg_values_cfg.append(MSE_deg_cfg)


        table_data.append([current_perturb_names, pearsonr_perturb, pearsonr_perturb_cfg, pearsonr_perturb_deg, pearsonr_perturb_deg_cfg, MSE_deg, MSE_deg_cfg])

        perturb_data_dict[current_perturb_names] = current_perturb_x.numpy()
        synthetic_data_dict[current_perturb_names] = synthetic_data
        # If you have a umap_plot function that supports labels
        # You can potentially use current_perturb_names[0] as a representative label for the plot
    # umap_plot(control_x.cpu().numpy(), perturb_data_dict, synthetic_data_dict, epoch)

    mean_pearsonr = np.mean(pearsonr_values)
    mean_pearsonr_cfg = np.mean(pearsonr_values_cfg)

    wandb.log({
        "mean_pearsonr_perturb": mean_pearsonr, 
        "mean_pearsonr_perturb_cfg": mean_pearsonr_cfg,
        "mean_MSE_deg": np.mean(MSE_deg_values),
        "mean_MSE_deg_cfg": np.mean(MSE_deg_values_cfg),
        "mean_pearsonr_perturb_deg": np.mean(deg_peasonr_values),
        "mean_pearsonr_perturb_deg_cfg": np.mean(deg_peasonr_values_cfg),
        "epoch": epoch, 
        "perturbations_table": wandb.Table(data=table_data, columns=["perturbation", "pearsonr_perturb", "pearsonr_perturb_cfg", "pearsonr_perturb_deg", "pearsonr_perturb_deg_cfg", "MSE_deg", "MSE_deg_cfg"])
    })

    print(f"Mean pearsonr_perturb: {mean_pearsonr}, Mean pearsonr_perturb_cfg: {mean_pearsonr_cfg}")

    # perturb_cond = torch.repeat_interleave(dataset.perturbations[0], control_x.shape[0], dim=0).to(device).unsqueeze(1)

    # traj = sample_ode(model, z0=control_x, N=500, l1=perturb_cond)

    # control_data = control_x.cpu().numpy()
    # perturb_data = perturb_x.numpy()
    # synthetic_data = traj[-1]
    # mean_perturb = np.mean(perturb_data, axis=0)
    # mean_synthetic = np.mean(synthetic_data, axis=0)
    # mean_control = np.mean(control_data, axis=0)
    
    # pearsonr_perturb = pearsonr(mean_perturb, mean_synthetic)[0]
    # pearsonr_control = pearsonr(mean_control, mean_synthetic)[0]
    # wandb.log({"pearsonr_perturb": pearsonr_perturb, "pearsonr_control": pearsonr_control, "epoch": epoch})
    # print(f"pearsonr_perturb: {pearsonr_perturb}, pearsonr_control: {pearsonr_control}")
    # umap_plot(control_data, perturb_data, synthetic_data, epoch)

