import time
import math
import json
import argparse
import itertools
import numpy as np
from math import pi
import os, sys, glob
print(os.getcwd())
from tqdm import tqdm
import sparseconvnet as scn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from util import *
from model_m1fix_m2fix_m3fix import *
from cluster import *
import data_val as data
# os.environ['CUDA_VISIBLE_DEVICES']='2'
import pickle
import torch

with open('/home/mayiwei/Project/3DRIS/TGNN/val_feat.pkl', 'rb') as f:
    feat_data = pickle.load(f)
    
    
parser = argparse.ArgumentParser()
parser.add_argument('--restore_epoch', type=int, default=-1, metavar='N', help='Epoch of model to restore')
parser.add_argument('--exp_name', type=str, default='cliploss1_model_m1fix_m2fix_m3ifx_aug4_seed1024', metavar='N', help='Name of the experiment')
args= parser.parse_args()

DIR = None
def _init_():
	DIR = './validation/' + args.exp_name + '/scenes/'
	print ('Save directory: ', DIR)

	if not os.path.exists('./validation'):
		os.mkdir('./validation')
	if not os.path.exists('./validation/' + args.exp_name):
		os.mkdir('./validation/' + args.exp_name)
	if not os.path.exists(DIR):
		os.mkdir(DIR)
_init_()
DIR = './validation/' + args.exp_name + '/scenes/'
io = IOStream('validation/' + args.exp_name + '/run.log')
    
def batch_val(batch, model, batch_size):
    ref_model = model['refer']


    IOUs = []
    idx = 0
    for i, num_p in enumerate(batch['num_points']):
        name = batch['names'][i]
        # io.cprint(name)

        scene_pcfeat = feat_data[name]['scene_pcfeat'].cuda()
        scene_sem = feat_data[name]['scene_sem'].cuda()
        scene_offset = feat_data[name]['scene_offset'].cuda()
        scene_coords = feat_data[name]['scene_coords'].cuda()
        grps = feat_data[name]['grps'].cuda()
        grp_feat = feat_data[name]['grp_feat'].cuda()
        grp_cen = feat_data[name]['grp_cen'].cuda()
        grps1 = feat_data[name]['grps1'].cuda()
        grp_feat1 = feat_data[name]['grp_feat1'].cuda()
        grp_cen1 = feat_data[name]['grp_cen1'].cuda()

        m = (scene_sem.argmax(-1) > 1)  # Perform argmax on the last dimension to get the most likely class for each point, and compare it with 1. Since semantic class numbering starts from 0, classes greater than 1 represent non-background classes.
        pred_sem = scene_sem.argmax(-1)  # Predict the instance class for each point.
        pred_cen = scene_coords + scene_offset  # Predicted center coordinates.

        scene_data = {}
        scene_data['lang_feat'] = batch['lang_feat'][i].float().cuda()  # Move language features to GPU.
        scene_data['lang_len'] = batch['lang_len'][i]  # Length of language features.

        ins_lbl = batch['y_ins'][idx:idx+num_p].cuda()  # Instance labels for the current batch.

        # Initialize an instance mask tensor [N, C], where N is the number of points and C is the number of predicted instance categories.
        ins_mask = torch.zeros(num_p, grps.shape[-1]).to(grps.device).long()
        ins_mask[m] = grps  # Assign values to non-background points, indicating the predicted category for each point.

        # Concatenate instance features and background features along dimension 0.
        obj_feat = torch.cat([grp_feat, grp_feat1], 0)
        # Concatenate instance center coordinates and background center coordinates along dimension 0.
        obj_coord = torch.cat([grp_cen, grp_cen1], 0)

        obj_num, dim = obj_feat.shape
        scene_data['obj_feat'] = obj_feat
        scene_data['obj_coord'] = obj_coord

        possible_obj_num = grp_feat.shape[0]  # Number of instances.
        total_score, _ = ref_model(scene_data)  # Get score predictions for all examples (including instances and background).
        total_score = total_score[:, 0:possible_obj_num]  # Only take score predictions for instance classes.
        total_score = F.softmax(total_score, -1)  # Apply softmax to scores. total_score[59, 29] represents prediction scores for 59 scenes, each with 29 possible instances.

        scores = [total_score.cpu().numpy()]
        # Obtain a [N, C'] matrix, representing which scenes each point appears in as a mask, i.e., predicted mask.
        pred = ins_mask[:, total_score.argmax(-1)]  # total_score.argmax(-1) gives the index of the highest-scoring instance for each scene.
        gt = batch['ref_lbl'][i].cuda()
        iou = (pred * gt).sum(0).float() / ((pred | gt).sum(0).float() + 1e-5)  # Calculate IoU: (pred * gt).sum(0) represents intersection, (pred | gt).sum(0) represents union. .sum(0) calculates the sum across all dimensions column-wise.
        IOUs.append(iou.cpu().numpy())

        precision_half = (iou > 0.5).sum().float() / iou.shape[0]
        precision_quarter = (iou > 0.25).sum().float() / iou.shape[0]
        outstr = 'mean IOU {:.4f} | P@0.5 {:.4f} | P@0.25 {:.4f}'.format(iou.mean().item(), precision_half, precision_quarter)
        # io.cprint(outstr)

        pc = [scene_coords, batch['x'][1][idx:idx+num_p], pred_cen, ins_lbl.unsqueeze(-1).float()]  # Combine various point cloud data.
        pc = torch.cat(pc, -1)  # Concatenate along the last dimension.
        sen = batch['sentences'][i]
        sen = np.array(sen)
        token = batch['tokens'][i]
        ref_objname = batch['lang_objname'][i]
        idx += num_p  # Increment the index by the number of points processed.
    IOUs = np.concatenate(IOUs, 0)  # Concatenate all IoU results.
    return IOUs


use_cuda = torch.cuda.is_available()

backbone_model = SCN().cuda()
backbone_model = nn.DataParallel(backbone_model)
ref_model=RefNetGRU(k=16).cuda()
ref_model.relconv = nn.DataParallel(ref_model.relconv)
models = {'backbone': backbone_model, 'refer': ref_model}

# training_epoch = checkpoint_restore(models, './checkpoints/'+args.exp_name+'/'+'models'+'/'+args.exp_name, io, use_cuda, args.restore_epoch)#加载模型参数
training_epoch = checkpoint_restore(models, './checkpoints/'+args.exp_name+'/'+'models'+'/'+args.exp_name, io, use_cuda, -2)#加载模型参数
for m in models:
    models[m].eval()

total_ious = []

for i,batch in enumerate(tqdm(data.val_data_loader)):
    if use_cuda:
        batch['x'][1]=batch['x'][1].cuda()
    with torch.no_grad():
        IOUs = batch_val(batch, models, data.batch_size)
        total_ious.append(IOUs)
    print ('({}/{}) Mean IOU so far {:.4f}'.format((i+1)*data.batch_size, len(data.loader_list), np.concatenate(total_ious, 0).mean()))
total_ious = np.concatenate(total_ious, 0)
IOU = total_ious.mean()

outstr = 'Mean IOU: {}'.format(IOU)
io.cprint(outstr)
Precision = (total_ious > 0.5).sum().astype(float)/total_ious.shape[0]
outstr = 'P@0.5: {}'.format(Precision)
io.cprint(outstr)
Precision = (total_ious > 0.25).sum().astype(float)/total_ious.shape[0]
outstr = 'P@0.25: {}'.format(Precision)
io.cprint(outstr)
