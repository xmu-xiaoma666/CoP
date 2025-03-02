import numpy as np
import sparseconvnet as scn
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from util import *
from config import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import os
from torch import nn
import torch
from einops.einops import rearrange, repeat
import numpy as np
import math 

m = 32
residual_blocks= True
block_reps = 2

dimension = 3
full_scale = 4096


class ParamDecoder(nn.Module):
    def __init__(self, mu_dim, need_in_dim,need_out_dim,k=32):
        super(ParamDecoder, self).__init__()
        self.need_in_dim=need_in_dim
        self.need_out_dim=need_out_dim
        self.k=k
        self.decoder = nn.Linear(mu_dim, need_in_dim*k) 
        self.V = nn.parameter.Parameter(torch.zeros(k,need_out_dim))
      
    def forward(self, t_feat):
        B=t_feat.shape[0]
        U = self.decoder(t_feat).reshape(B,self.need_in_dim,self.k)  # B x need_in_dim x k
        param=torch.einsum('bik,kj->bij',U,self.V).reshape(B,-1)
        return param

class DynamicLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, mu_dim: int, bias=True):
        super(DynamicLinear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mu_dim = mu_dim
        self.bias=bias
        self.decoder = ParamDecoder(mu_dim, in_dim + 1, out_dim)
    def forward(self, x, mu):

        param=rearrange(self.decoder(mu),'B (dim_A dim_B) -> B dim_A dim_B',dim_A=self.in_dim+1,dim_B=self.out_dim)
        weight=param[:,:-1,:]
        bias=param[:, -1, :]
        x=torch.einsum('b...d,bde->b...e',x,weight)
        if self.bias:
            bias=bias.view(((bias.shape[0],)+(1,)*(len(x.size())-2)+(bias.shape[-1],)))
            x=x+bias
        return x

class MuModuleList(nn.ModuleList):
    def forward(self,x,mu):
        for layer in self:
            if type(layer) == DynamicLinear:
                x=layer(x,mu)
            else:
                x=layer(x)
        return x


class ChannelGate(nn.Module):
    def __init__(self, gate_channels,text_dim, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = MuModuleList([
            DynamicLinear(gate_channels, gate_channels // reduction_ratio,text_dim),
            nn.ReLU(),
            DynamicLinear(gate_channels // reduction_ratio, gate_channels,text_dim)
        ])
    def forward(self, x ,mu):
        dynamic_x = self.mlp(x ,mu).mean(dim=1,keepdim=True) #1,D
        scale = torch.sigmoid( dynamic_x )
        res = x * scale
        return res


class SpatialGate(nn.Module):
    def __init__(self,gate_channels,mu_dim, reduction_ratio=16):
        super(SpatialGate, self).__init__()
        self.mlp =  MuModuleList([
            DynamicLinear(gate_channels, gate_channels // reduction_ratio,mu_dim),
            nn.ReLU(),
            DynamicLinear(gate_channels // reduction_ratio, gate_channels,mu_dim)
        ])
    def forward(self, x, mu):
        dynamic_x = self.mlp(x ,mu).mean(dim=-1,keepdim=True) #P,1
        scale = torch.sigmoid(dynamic_x) 
        res=x*scale # broadcasting (P,D)
        return res

class QueryDynamicAttention(nn.Module):
    def __init__(self,gate_channels=32,mu_dim=256, reduction_ratio=8,use_spatial=True,use_channel=True):
        super(QueryDynamicAttention,self).__init__()
        self.ChannelGate = ChannelGate(gate_channels,mu_dim, reduction_ratio)
        self.SpatialGate = SpatialGate(gate_channels,mu_dim, reduction_ratio)
        self.use_spatial=use_spatial
        self.use_channel=use_channel
    def forward(self, x,mu):
        if self.use_channel:
            x_new = self.ChannelGate(x,mu) 
        if self.use_spatial:
            x_new = self.SpatialGate(x_new,mu) 
        return x + x_new

class SCN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension, full_scale, mode=4))  # mode=4 allows scn to automatically detect the input data format
        .add(scn.SubmanifoldConvolution(dimension, 3, m, 3, False))
        .add(scn.UNet(dimension, block_reps, [m, 2*m, 3*m, 4*m, 5*m, 6*m, 7*m], residual_blocks))
        .add(scn.BatchNormReLU(m))
        .add(scn.OutputLayer(dimension))
        self.linear = nn.Linear(m, 20)
        self.linear1 = nn.Linear(m, m) 
        self.cen_pred = nn.Sequential(nn.Linear(m, m), nn.ReLU(), nn.Linear(m, 3))

    def forward(self, x):
        fv = self.sparseModel(x)
        y = self.linear(fv)
        fv = self.linear1(fv)
        offset = self.cen_pred(fv)  # Output offset prediction
        return y, fv, offset  # pc_sem, pc_feat, pc_offset


class SentenceEncoderGRU(nn.Module):
    def __init__(self, embedding_size=300, input_size=256, hidden_size=256, output_size=64, bidirectional=True):
        nn.Module.__init__(self)
        self.mlp = nn.Sequential(nn.Linear(embedding_size, input_size), nn.ReLU())
        self.biLSTM = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.lang_mlp = nn.Linear(hidden_size, output_size)

    def forward(self, lang_feat, lang_len):
        lang_feat = self.mlp(lang_feat)
        total_length = lang_feat.shape[1]
        # lang_feat is a tensor of language features with shape (batch_size, seq_len, hidden_size). The second parameter lang_len is a list of lengths for each sample in the batch.
        lang_feat = pack_padded_sequence(lang_feat, lang_len, batch_first=True, enforce_sorted=False)  # pack_padded_sequence packs a batch of variable-length sequences into a PackedSequence object and records each sequence's true length for use during training.
        output, hidden = self.biLSTM(lang_feat)
        # batch, sen_len, vec_size
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=total_length)  # pad_packed_sequence unpacks the packed Tensor back into its original form and converts it to batch_first mode.
        output = self.lang_mlp(output)
        return output, lang_len


class TARelationConv(nn.Module):  # Text-Guided Graph Neural Network specific module
    def __init__(self, lang_id, lang_od, pc_id, pc_od, k):
        nn.Module.__init__(self)
        self.k = k
        self.rel_encoder = nn.Sequential(nn.Linear(13, lang_od), nn.ReLU(), nn.Linear(lang_od, lang_od))
        self.lang_encoder = nn.Sequential(nn.Linear(lang_id, lang_od), nn.ReLU(), nn.Linear(lang_od, lang_od))
        self.feat_encoder = nn.Sequential(nn.Linear(pc_id, pc_od), nn.ReLU(), nn.Linear(pc_od, pc_od))

    def forward(self, feat, coord, lang_feat, lang_len):
        num_sen, num_obj, _ = feat.shape
        k = min(self.k, num_obj-1)
        d = ((coord.unsqueeze(1) - coord.unsqueeze(2))**2).sum(-1)  # Compute distance matrix between all objects, which is a series of diagonal matrices.
        indice0 = torch.arange(coord.shape[0]).view(coord.shape[0], 1, 1).repeat(1, num_obj, k+1)  # Repeat the index of this object as indice0.
        _, indice1 = torch.topk(d, k+1, dim=-1, largest=False)  # For each object, select the k closest objects, denoted as indice1.

        coord_expand = coord[indice0, indice1]  # Select coordinates of each object with its k closest objects based on indice0 and indice1.
        coord_expand1 = coord.unsqueeze(2).expand(coord.shape[0], coord.shape[1], k+1, coord.shape[-1])  # For each object, replicate its coordinates k+1 times, denoted as coord_expand1.
        rel_coord = coord_expand - coord_expand1  # Calculate the difference between coord_expand and coord_expand1, i.e., relative coordinates.
        d = torch.norm(rel_coord, p=2, dim=-1).unsqueeze(-1)  # Add an extra dimension representing the distance between two objects.
        dx = torch.abs(coord_expand[:,:,:,0]-coord_expand1[:,:,:,0]).unsqueeze(-1)
        dy = torch.abs(coord_expand[:,:,:,1]-coord_expand1[:,:,:,1]).unsqueeze(-1)
        dz = torch.abs(coord_expand[:,:,:,2]-coord_expand1[:,:,:,2]).unsqueeze(-1)
        rel = torch.cat([coord_expand, coord_expand1, rel_coord, d, dx, dy, dz], -1)  # Concatenate various components into one feature vector.
        rel = self.rel_encoder(rel)  # Relation Encoding

        rel = rel.view(rel.shape[0], -1, rel.shape[-1])
        num_sen, max_len, _ = lang_feat.shape
        mask = torch.arange(max_len).expand(num_sen, max_len).to(feat.device)
        mask = (mask < lang_len.unsqueeze(-1)).float().unsqueeze(1)  # Generate mask to mask out positions beyond the actual sentence length.
        lang_feat = self.lang_encoder(lang_feat)  # Encode text features.
        feat = self.feat_encoder(feat)  # Pass instance features through MLP.

        attn = torch.bmm(feat[indice0, indice1].view(feat.shape[0], -1, feat.shape[-1]), lang_feat.permute(0, 2, 1))  # Compute attention weights between each object and language features.
        attn = F.softmax(attn, -1) * mask  # Convert to attention weights using softmax and mask the sequence to avoid applying attention weights outside the valid range.
        attn = attn / (attn.sum(-1).unsqueeze(-1) + 1e-7)
        ins_attn_lang_feat = torch.bmm(attn, lang_feat)  # Instance-dependent text features as per the paper.

        dim = rel.shape[-1]
        rel = rel.view(num_sen, num_obj, k+1, dim)

        ins_attn_lang_feat = ins_attn_lang_feat.view(num_sen, num_obj, k+1, dim)
        feat = ((feat[indice0, indice1] * ins_attn_lang_feat) * rel).sum(2) + feat  # Multiply instance-dependent text features, local instance features, and relation encoding, then add the original feat.

        mask_lang_feat = lang_feat * mask.permute(0, 2, 1)
        attn2 = torch.bmm(mask_lang_feat, feat.view(feat.shape[0], -1, feat.shape[-1]).permute(0, 2, 1))  # Attention from text features to instance features.
        attn2 = F.softmax(attn2, -1)  # Normalize attention scores.
        attn2 = attn2 / (attn2.sum(-1).unsqueeze(-1) + 1e-7)
        new_lang_feat = torch.bmm(attn2, feat.view(feat.shape[0], -1, feat.shape[-1])) + lang_feat  # Instance-dependent text features as per the paper.

        score = feat.sum(-1)
        return new_lang_feat, feat, score

class TARelationConvBlock(nn.Module):
    def __init__(self, k):
        nn.Module.__init__(self)
        self.conv = TARelationConv(256, 128, 32, 128, k)
        self.conv1 = TARelationConv(128, 128, 128, 128, k)
        self.conv2 = TARelationConv(128, 128, 128, 128, k)
        # self.ffn = nn.Linear(128, 1)
        self.vis2share = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )
        self.lang2share = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )
        self.norm_vis = nn.LayerNorm(128)
        self.norm_lang = nn.LayerNorm(128)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    def forward(self, feat, coord, lang_feat, lang_len):
        lang_feat, feat, _ = self.conv(feat, coord, lang_feat, lang_len)
        feat = F.relu(feat)
        lang_feat, feat, _ = self.conv1(feat, coord, lang_feat, lang_len)
        feat = F.relu(feat)
        lang_feat, feat, _ = self.conv2(feat, coord, lang_feat, lang_len)
        feat = F.relu(feat)
        # score = self.ffn(feat).squeeze(-1)
        
        lang_feat = self.norm_lang(self.lang2share(lang_feat.max(dim=1,keepdim=True)[0])) #num_sen,1,128
        vis_feat = self.norm_vis(self.vis2share(feat)) #num_sen,num_obj,128
        score2 = torch.einsum('nsd,nod->nso',vis_feat,lang_feat).squeeze(dim=-1) #num_sen,num_obj
        logit_scale = self.logit_scale.exp()
        score2 = logit_scale*score2
        
        return feat, score2

class MLP(nn.Module):
	def __init__(self, in_d, out_d):
		nn.Module.__init__(self)
		self.fc = nn.Sequential(nn.Linear(in_d,out_d),nn.ReLU(),nn.Linear(out_d,out_d))
	def forward(self, x):
		return self.fc(x)


class ReasoningModule(nn.Module):
    def __init__(self, dim=32, expand=8):
        nn.Module.__init__(self)
        self.graph_weight1 = nn.Linear(dim,dim,bias=False)
        self.graph_weight2 = nn.Linear(dim,dim,bias=False)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn =nn.Sequential(
            nn.Linear(dim,dim*expand),
            nn.ReLU(),
            nn.Linear(dim*expand,dim)
        )
        self.ln3 = nn.LayerNorm(dim)
    def forward(self, edge1,edge2,vertex):
        out1 = self.graph_weight1(torch.einsum('bmn,bnc->bmc',edge1,vertex))
        out1 = self.ln1(out1 + vertex)
        out2 = self.graph_weight2(torch.einsum('bmn,bnc->bmc',edge2,out1))
        out2 = self.ln2(out2 + out1)
        ff = self.ffn(out2)
        res = self.ln3(out2 + ff)
        return res

class RefNetGRU(nn.Module):
    def __init__(self, k, N=6):
        nn.Module.__init__(self)
        # Language feature encoder using GRU
        self.lang_encoder = SentenceEncoderGRU(300,512,512,256)
        # Relation-aware convolution block
        self.relconv = TARelationConvBlock(k)  # Likely Text-Guided GNN
        # Sequential layers to parse language features into categories
        self.lang_parse = nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,3)
        )
        # Query dynamic attention module
        self.qdatt = QueryDynamicAttention()
        
        # Linear embedding layers for visual to language graph and within modalities
        self.vis2lang_graph_embed = nn.Linear(32,256,bias=False)
        self.vis2vis_graph_embed = nn.Linear(32,32,bias=False)
        self.lang2lang_graph_embed = nn.Linear(256,256,bias=False)
        
        # Sequential layers for linear embedding and relation position embedding
        self.linear_embed = nn.Sequential(
            nn.Linear(256+64,256),
            nn.ReLU(),
            nn.Linear(256,32)
        )
        self.relpos_embed = nn.Sequential(
            nn.Linear(4,32),
            nn.ReLU(),
            nn.Linear(32,1)
        )
        self.relu = nn.ReLU()
        
        self.N = N
        # Reasoning modules for iterative graph reasoning
        self.reasoning_modules = nn.ModuleList([
            ReasoningModule(32,8) for _ in range(self.N)
        ])

    def word_parsing(self, lang_feat):
        # Parsing language features into specific categories
        word_parse = self.lang_parse(lang_feat)
        word_parse = torch.softmax(word_parse,dim=-1) # num_sentence, num_word, 3 (categories: phrase, relation, unnecessary)
        return word_parse
    
    def entity_perception(self, obj_feat, lang_feat, word_parse, lang_len):
        # Dynamic attention mechanism to fuse object features with query from parsed language features
        query = word_parse[:,:,0].unsqueeze(-1)*lang_feat # num_sentence, 1, lang_dim
        query = torch.sum(query,dim=1,keepdim=True)
        fused_feat = self.qdatt(obj_feat, query)  # num_sentence, num_object, vis_dim
        return fused_feat
    
    def rel_pos_embed(self, obj_coord):
        # Embedding relative positions between objects
        rel_xyz = obj_coord.unsqueeze(1)-obj_coord.unsqueeze(2) # num_sentence,num_object,num_object,3
        rel_dis = torch.sqrt(torch.sum(rel_xyz**2,dim=-1,keepdim=True)) # num_sentence,num_object,num_object,1
        rel_matrix = torch.cat([rel_xyz,rel_dis],dim=-1)
        rel_matrix = self.relpos_embed(rel_matrix).squeeze(-1)
        return rel_matrix
    
    def build_graph(self, obj_feat, lang_feat, word_parse, obj_coord):
        # Building a graph structure with vertices and edges based on visual and language features
        vertex = self.vis2vis_graph_embed(obj_feat) # num_sentence,num_object,vis_dim
        rel_lang_feat = lang_feat*word_parse[:,:,1].unsqueeze(-1) # num_sentence,num_word,lang_dim
        embed_vis_feat = self.vis2lang_graph_embed(obj_feat) # num_sentence,num_object,lang_dim
        embed_lang_feat = self.lang2lang_graph_embed(rel_lang_feat) # num_sentence,num_word,lang_dim
        edge_vis = self.rel_pos_embed(obj_coord)  # num_sentence,num_object,num_object
        rel_matrix = torch.einsum('bnc,btc->bnt',embed_vis_feat,embed_lang_feat)  # num_sentence,num_object,num_word
        edge =  torch.einsum('bnt,bmt->bnm',rel_matrix,rel_matrix) # num_sentence,num_object,num_object
        edge = torch.softmax(edge,dim=-1)
        return vertex,edge,edge_vis
        
    def forward(self, scene):
        # Processing input scene with object features, coordinates, and language features
        num_obj, d = scene['obj_feat'].shape  # d is the feature dimension of each object, set to 32
        num_sen, max_len, _ = scene['lang_feat'].shape

        obj_feat = scene['obj_feat'].unsqueeze(0).expand(num_sen,num_obj,d)  # Expand object features across sentences
        obj_coord = scene['obj_coord'].unsqueeze(0).expand(num_sen,num_obj,3)  # Expand object coordinates across sentences
   
        # Encode language features
        lang_feat, lang_len = self.lang_encoder(scene['lang_feat'], scene['lang_len'])
        
        # Parse words into categories
        word_parse = self.word_parsing(lang_feat)
        # Perceive entities by fusing object and language features
        obj_feat_entity = self.entity_perception(obj_feat,lang_feat,word_parse,lang_len)
        # Build graph structures for reasoning
        vertex, edge, edge_vis = self.build_graph(obj_feat_entity,lang_feat,word_parse,obj_coord)
        # Iterative graph reasoning
        for i in range(self.N):
            vertex = self.reasoning_modules[i](edge,edge_vis,vertex)
            
        # Enhance features and predict scores
        query_lang_feat = (word_parse[:,:,0].unsqueeze(-1)+word_parse[:,:,1].unsqueeze(-1))*lang_feat # num_sentence, num_word, lang_dim
        query_lang_feat = torch.sum(query_lang_feat,dim=1,keepdim=True)  # num_sentence, 1, lang_dim
        query_lang_feat = query_lang_feat.repeat(1,num_obj,1)  # num_sentence, num_obj, lang_dim
        obj_feat = self.linear_embed(torch.cat([query_lang_feat,obj_feat_entity,vertex],dim=-1))
        feat, score = self.relconv(obj_feat, obj_coord, lang_feat, lang_len)  # Predict likelihood for each object type
        return score, word_parse

def gather(feat, lbl):
    # Gather features based on labels
    uniq_lbl = torch.unique(lbl)
    gather_func = scn.InputLayer(1, uniq_lbl.shape[0], mode=4)
    grp_f = gather_func([lbl.long().unsqueeze(-1), feat])
    grp_idx = grp_f.get_spatial_locations()[:,0]
    grp_idx, sorted_indice = grp_idx.sort()
    grp_f = grp_f.features[sorted_indice]
    return grp_f, grp_idx

def gather_1hot(feat, mask):
    # Aggregate features based on one-hot encoded masks
    obj_size = mask.sum(0)
    mean_f = torch.bmm(mask.unsqueeze(-1).float(), feat.unsqueeze(1))
    mean_f = mean_f.sum(0) / obj_size.float().unsqueeze(-1)
    idx = torch.arange(mask.shape[1]).cuda()
    return mean_f, idx