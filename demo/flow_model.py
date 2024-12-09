import torch.nn as nn
from diffusers.models.embeddings import Timesteps, TimestepEmbedding, GaussianFourierProjection

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
import math

import torch
from torch import nn
from torch_geometric.nn import SGConv
# from utils import exists
import numpy as np


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2)
                              * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class MLP(torch.nn.Module):

    def __init__(self, sizes, batch_norm=True, last_layer_act="linear"):
        """
        Multi-layer perceptron
        :param sizes: list of sizes of the layers
        :param batch_norm: whether to use batch normalization
        :param last_layer_act: activation function of the last layer

        """
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers = layers + [
                torch.nn.Linear(sizes[s], sizes[s + 1]),
                torch.nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 1 else None,
                torch.nn.ReLU()
            ]

        layers = [l for l in layers if l is not None][:-1]
        self.activation = last_layer_act
        self.network = torch.nn.Sequential(*layers)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        return self.network(x)

class Autoencoder(nn.Module):
    def __init__(self, gene_dim, hidden_dim, pert_dim):
        super(Autoencoder, self).__init__()
        
        # Encoding layers
        input_dim = gene_dim
        encoding_layers = []
        prev_dim = gene_dim
        for h_dim in hidden_dim:
            encoding_layers.append(nn.Linear(prev_dim, h_dim))
            encoding_layers.append(nn.BatchNorm1d(h_dim))
            encoding_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoding_layers)

        # Decoding layers (excluding the last activation for reconstruction)
        decoding_layers = []
        for h_dim in reversed(hidden_dim[:-1]):
            decoding_layers.append(nn.Linear(prev_dim, h_dim))
            decoding_layers.append(nn.BatchNorm1d(h_dim))
            decoding_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoding_layers.append(nn.Linear(prev_dim, input_dim))  # Last layer for reconstruction
        self.decoder = nn.Sequential(*decoding_layers)
        self.time_embedding = nn.Sequential(GaussianFourierProjection(embed_dim=hidden_dim[-1]),
                                   nn.Linear(hidden_dim[-1], hidden_dim[-1]))
        self.pert_embedding = nn.Sequential(
            nn.Linear(pert_dim, hidden_dim[-1]),
            nn.SiLU(),
            nn.Linear(hidden_dim[-1], hidden_dim[-1]*2)
        )
        
    def forward(self, x, t, perturbations):
        t = self.time_embedding(t.squeeze())
        internal = self.encoder(x)
        scale, shift = self.pert_embedding(perturbations).chunk(2, dim=-1)
        x = t + internal
        x = x * (scale+1) + shift
        x = self.decoder(x)
        return internal, x


class block(nn.Module):
    r"""
    The block of the network. Employ the scale and shift trick for the conditional information
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.bn = nn.BatchNorm1d(output_dim)
        self.act = nn.SiLU()
        

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.bn(x)

        if exists(scale_shift):
            if not isinstance(scale_shift, tuple):
                scale_shift = scale_shift.chunk(2, dim=-1)
            scale, shift = scale_shift
            x = x * (scale+1) + shift

        x = self.act(x)
        return x

class GEARS_Model(torch.nn.Module):
    """
    GEARS model

    """

    def __init__(self, args):
        """
        :param args: arguments dictionary
        """

        super(GEARS_Model, self).__init__()
        self.args = args       
        self.num_genes = args['num_genes']
        self.num_perts = args['num_perts']
        hidden_size = args['hidden_size']
        self.uncertainty = args['uncertainty']
        self.num_layers = args['num_go_gnn_layers']
        self.indv_out_hidden_size = args['decoder_hidden_size']
        self.num_layers_gene_pos = args['num_gene_gnn_layers']
        self.no_perturb = args['no_perturb']
        self.no_GO = args['no_GO']
        self.pert_emb_lambda = 0.2
        
        # perturbation positional embedding added only to the perturbed genes
        self.pert_w = nn.Linear(1, hidden_size)
           
        # gene/globel perturbation embedding dictionary lookup            
        self.gene_emb = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.pert_emb = nn.Embedding(self.num_perts, hidden_size, max_norm=True)
        
        # time embedding
        self.time_embedding = nn.Sequential(
            GaussianFourierProjection(embed_dim=hidden_size),
            nn.Linear(hidden_size, hidden_size)
        )
        # transformation layer
        self.emb_trans = nn.ReLU()
        self.pert_base_trans = nn.ReLU()
        self.transform = nn.ReLU()
        self.emb_trans_v2 = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        self.pert_fuse = MLP([hidden_size, hidden_size, hidden_size], last_layer_act='ReLU')
        
        # gene co-expression GNN
        self.G_coexpress = args['G_coexpress'].to(args['device'])
        self.G_coexpress_weight = args['G_coexpress_weight'].to(args['device'])

        self.emb_pos = nn.Embedding(self.num_genes, hidden_size, max_norm=True)
        self.layers_emb_pos = torch.nn.ModuleList()
        for i in range(1, self.num_layers_gene_pos + 1):
            self.layers_emb_pos.append(SGConv(hidden_size, hidden_size, 1))
        
        ### perturbation gene ontology GNN
        self.G_sim = args['G_go'].to(args['device'])
        self.G_sim_weight = args['G_go_weight'].to(args['device'])

        self.sim_layers = torch.nn.ModuleList()
        for i in range(1, self.num_layers + 1):
            self.sim_layers.append(SGConv(hidden_size, hidden_size, 1))
        
        # decoder shared MLP
        self.recovery_w = MLP([hidden_size, hidden_size*2, hidden_size], last_layer_act='linear')
        
        # gene specific decoder
        self.indv_w1 = nn.Parameter(torch.rand(self.num_genes,
                                               hidden_size, 1))
        self.indv_b1 = nn.Parameter(torch.rand(self.num_genes, 1))
        self.act = nn.ReLU()
        nn.init.xavier_normal_(self.indv_w1)
        nn.init.xavier_normal_(self.indv_b1)
        
        # Cross gene MLP
        self.cross_gene_state = MLP([self.num_genes, hidden_size,
                                     hidden_size])
        # final gene specific decoder
        self.indv_w2 = nn.Parameter(torch.rand(1, self.num_genes,
                                           hidden_size+1))
        self.indv_b2 = nn.Parameter(torch.rand(1, self.num_genes))
        nn.init.xavier_normal_(self.indv_w2)
        nn.init.xavier_normal_(self.indv_b2)
        
        # batchnorms
        self.bn_emb = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base = nn.BatchNorm1d(hidden_size)
        self.bn_pert_base_trans = nn.BatchNorm1d(hidden_size)
        
        # uncertainty mode
        if self.uncertainty:
            self.uncertainty_w = MLP([hidden_size, hidden_size*2, hidden_size, 1], last_layer_act='linear')
        
        self.gene_names= args['gene_names']
        self.pert_names = args['pert_names']

    def get_pert_idx(self, pert_category):
        """
        Get perturbation index for a given perturbation category

        Parameters
        ----------
        pert_category: str
            Perturbation category

        Returns
        -------
        list
            List of perturbation indices

        """
        try:
            pert_idx = [np.where(p == self.pert_names)[0][0]
                    for p in pert_category.split('+')
                    if p != 'ctrl']
        except:
            print(pert_category)
            pert_idx = None
            
        return pert_idx 
    
    def get_gene_idx(self, gene_symbol):
        try:
            gene_idx = [np.where(gene_symbol == self.gene_names)[0][0]]
                    
        except:
            print(gene_symbol)
            gene_idx = None
            
        return gene_idx 
        
    def forward(self, x, t, pert_idx):
        """
        Forward pass of the model
        """
        assert x.shape[0]==t.shape[0]==len(pert_idx)
        if self.no_perturb:
            out = x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)           
            return torch.stack(out)
        else:
            num_graphs = len(pert_idx)

            # get base gene embeddings (empty gene embeddings)
            emb = self.gene_emb(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))        
            emb = self.bn_emb(emb)
            base_emb = self.emb_trans(emb)        

            # add co-expression GNN to base gene embeddings
            pos_emb = self.emb_pos(torch.LongTensor(list(range(self.num_genes))).repeat(num_graphs, ).to(self.args['device']))
            for idx, layer in enumerate(self.layers_emb_pos):
                pos_emb = layer(pos_emb, self.G_coexpress, self.G_coexpress_weight)
                if idx < len(self.layers_emb_pos) - 1:
                    pos_emb = pos_emb.relu()

            base_emb = base_emb + 0.2 * pos_emb
            base_emb = self.emb_trans_v2(base_emb)

            ## get perturbation index and embeddings

            pert_index = []
            for idx, i in enumerate(pert_idx):
                for j in i:
                    if j != -1:
                        pert_index.append([idx, j])
            pert_index = torch.tensor(pert_index).T

            # compute perturbation embeddings for all perturbations
            pert_global_emb = self.pert_emb(torch.LongTensor(list(range(self.num_perts))).to(self.args['device']))        
            ## augment global perturbation embedding with GNN
            if not self.no_GO:
                for idx, layer in enumerate(self.sim_layers):
                    pert_global_emb = layer(pert_global_emb, self.G_sim, self.G_sim_weight)
                    if idx < self.num_layers - 1:
                        pert_global_emb = pert_global_emb.relu()


            ## add global perturbation embedding to each gene in each cell in the batch
            base_emb = base_emb.reshape(num_graphs, self.num_genes, -1)
            # add perturbation embeddings (only those acutually applied to the cell) to base gene embeddings
            if pert_index.shape[0] != 0:
                ### in case all samples in the batch are controls, then there is no indexing for pert_index.
                pert_track = {}
                for i, j in enumerate(pert_index[0]):
                    if j.item() in pert_track:
                        pert_track[j.item()] = pert_track[j.item()] + pert_global_emb[pert_index[1][i]]
                    else:
                        pert_track[j.item()] = pert_global_emb[pert_index[1][i]]

                if len(list(pert_track.values())) > 0:
                    if len(list(pert_track.values())) == 1:
                        # circumvent when batch size = 1 with single perturbation and cannot feed into MLP
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values()) * 2))
                    else:
                        emb_total = self.pert_fuse(torch.stack(list(pert_track.values())))

                    for idx, j in enumerate(pert_track.keys()):
                        base_emb[j] = base_emb[j] + emb_total[idx]

            # add time embedding
            time_embed = self.time_embedding(t.squeeze())
            base_emb = base_emb + time_embed.view(num_graphs, 1, -1)

            base_emb = base_emb.reshape(num_graphs * self.num_genes, -1)
            base_emb = self.bn_pert_base(base_emb)

            ## apply the first MLP
            base_emb = self.transform(base_emb)        
            out = self.recovery_w(base_emb)
            out = out.reshape(num_graphs, self.num_genes, -1)
            out = out.unsqueeze(-1) * self.indv_w1
            w = torch.sum(out, axis = 2)
            out = w + self.indv_b1

            # Cross gene
            cross_gene_embed = self.cross_gene_state(out.reshape(num_graphs, self.num_genes, -1).squeeze(2))
            cross_gene_embed = cross_gene_embed.repeat(1, self.num_genes)

            cross_gene_embed = cross_gene_embed.reshape([num_graphs,self.num_genes, -1])
            cross_gene_out = torch.cat([out, cross_gene_embed], 2)

            cross_gene_out = cross_gene_out * self.indv_w2
            cross_gene_out = torch.sum(cross_gene_out, axis=2)
            out = cross_gene_out + self.indv_b2        
            out = out.reshape(num_graphs * self.num_genes, -1) + x.reshape(-1,1)
            out = torch.split(torch.flatten(out), self.num_genes)

            ## uncertainty head
            if self.uncertainty:
                out_logvar = self.uncertainty_w(base_emb)
                out_logvar = torch.split(torch.flatten(out_logvar), self.num_genes)
                return torch.stack(out), torch.stack(out_logvar)
            return None, torch.stack(out)
        

if __name__=="__main__":
    import pickle as pkl
    with open("gears_config.pkl",'rb') as f:
        config_dict=pkl.load(f)
        config=config_dict["config"]
        config["gene_names"]=config_dict["gene_names"]
        config["pert_names"]=config_dict["pert_names"]
        config["device"]="cuda"

    model = GEARS_Model(config).to(config["device"])

    batch_size=32
    x = torch.randn(batch_size, len(config["gene_names"])).to(config["device"])
    t = torch.rand(batch_size, 1).to(config["device"])
    pert_idx = [[i] for i in range(batch_size)]

    out=model(x, t, pert_idx)
    print(out.shape)
    
# class ResnetBlock(nn.Module):
#     r"""
#     Type 1. Resnet block. 
#     See https://arxiv.org/pdf/1512.03385.pdf
#     """
#     def __init__(self, input_dim, output_dim, hidden_dim, condition_dim):
#         super().__init__()
#         self.block1 = block(input_dim, hidden_dim)
#         self.block2 = block(hidden_dim, output_dim)
#         self.proj = nn.Linear(input_dim, output_dim)

#         self.time_proj = nn.Sequential(
#             nn.SiLU(), nn.Linear(condition_dim, output_dim*2)
#         )

#     def forward(self, x, time_embed = None):
#         scale_shift = None
#         if exists(time_embed):
#             time_embed = self.time_proj(time_embed)
#             scale_shift = time_embed.chunk(2, dim=-1)
        
#         h = self.block1(x, scale_shift)
#         h = self.block2(h)

#         return h + self.proj(x)



# class UNetEncoder(nn.Module):
#     r"""
#     This version does not use the skip connection. 
#     """
#     def __init__(self, gene_dim, hidden_dim, num_blocks):
#         super().__init__()
#         # Assumes hidden_dim is a list of the same length as num_blocks.
#         self.blocks = nn.ModuleList([block(gene_dim if i == 0 else hidden_dim[i-1], hidden_dim[i]) for i in range(num_blocks)])
        
#     def forward(self, x, condition):
#         outputs = []
#         for block in self.blocks:
#             x = block(x, condition)
#             outputs.append(x)
#         return outputs


# class UNetDecoder(nn.Module):
#     def __init__(self, gene_dim, hidden_dim, num_blocks):
#         super().__init__()
#         # Assumes hidden_dim is a list of the same length as num_blocks.
#         self.blocks = nn.ModuleList([block(hidden_dim[i] * 2 if i != num_blocks-1 else hidden_dim[i-1] * 2, hidden_dim[i] if i != num_blocks-1 else gene_dim) for i in range(num_blocks)])

#     def forward(self, x, condition, skip_connections):
#         for block, skip_connection in zip(self.blocks, reversed(skip_connections)):
#             x = torch.cat([x, skip_connection], dim=-1)
#             x = block(x, condition)
#         return x


# class UNetDiffModel(ModelMixin, ConfigMixin):
#     def __init__(self, gene_dim=977, 
#                  hidden_dim=[1024, 512, 256], 
#                  pert_dim=1
#                  ):
#         super().__init__()
        
#         num_blocks = len(hidden_dim)
#         self.encoder = UNetEncoder(gene_dim, hidden_dim, num_blocks)
#         self.decoder = UNetDecoder(gene_dim, list(reversed(hidden_dim)), num_blocks)

#         # self.time_embedding = nn.Sequential(
#         #         Timesteps(num_channels=hidden_dim[-1], flip_sin_to_cos=True, downscale_freq_shift=0), 
#         #         TimestepEmbedding(in_channels=hidden_dim[-1], time_embed_dim=hidden_dim[-1]) 
#         #         )
#         self.time_embedding = nn.Sequential(GaussianFourierProjection(embed_dim=hidden_dim[-1]),
#                                    nn.Linear(hidden_dim[-1], hidden_dim[-1]))
#         self.pert_embedding = nn.Sequential(
#             nn.Linear(pert_dim, hidden_dim[-1]),
#             nn.SiLU(),
#             nn.Linear(hidden_dim[-1], hidden_dim[-1])
#         )
        

#     def forward(self, x, t, pert):
#         time_embed = self.time_embedding(t.squeeze())
#         pert_embed = self.pert_embedding(pert)
#         time_embed = time_embed + pert_embed
#         import pdb; pdb.set_trace()
#         encoder_outputs = self.encoder(x, time_embed)
#         latent = encoder_outputs[-1]  # The last output is the latent representation
#         reconstructed = self.decoder(latent, time_embed, encoder_outputs[:-1])
#         return latent, reconstructed
    

# if __name__ == "__main__":
#     model = UNetDiffModel()
#     x = torch.randn(10, 977)
#     t = torch.randint(0, 100, (10,))
    
#     latent, reconstructed = model(x, t)