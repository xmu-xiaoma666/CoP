"""

Modified from: https://github.com/facebookresearch/SparseConvNet/blob/master/examples/ScanNet/unet.py

Training for referring model with gru encoder.

"""

import math
import argparse
import itertools
import numpy as np
import os, sys, glob
import os 
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from tqdm import tqdm
import sparseconvnet as scn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.parallel import DistributedDataParallel as DDP

import data as data
import data_val as data_val

from util import *
from model_CoP import *
import torch.nn.functional as F
import torch.nn as nn
import sng_parser
from pprint import pprint
import pickle
from torch.utils.tensorboard import SummaryWriter
import shutil
import random
import numpy as np
import torch

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--seed', type=int, default=1234, help='Initial learning rate')
parser.add_argument('--epochs', type=int, default=46, help='Number of epochs')
parser.add_argument('--exp_name', type=str, default='cliploss1_model_m1fix_m2fix_m3ifx', metavar='N', help='Name of the experiment')
args= parser.parse_args()

def _init_():
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not os.path.exists('./checkpoints/'+args.exp_name):
        os.makedirs('./checkpoints/'+args.exp_name)
    if not os.path.exists('./checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('./checkpoints/'+args.exp_name+'/'+'models')

    os.system('cp util.py ./checkpoints' + '/' + args.exp_name + '/' + 'util.py')
    os.system('cp data.py ./checkpoints' + '/' + args.exp_name + '/' + 'data.py')
    os.system('cp model.py ./checkpoints' + '/' + args.exp_name + '/' + 'model.py')
    os.system('cp config.py ./checkpoints' + '/' + args.exp_name + '/' + 'config.py')
    os.system('cp unet_gru.py ./checkpoints' + '/' + args.exp_name + '/' + 'unet_gru.py')

_init_()
io = IOStream('./checkpoints/' + args.exp_name + '/run.log')#IOStream 是一个辅助类，它将文本写入到日志文件，并同时将其输出到标准输出（console）中


def kll_loss(prediction, label):
    lang_obj_label = torch.zeros_like(prediction).to(prediction.device) #num_sen,num_obj
    num_sen,num_obj = lang_obj_label.shape
    for i in range(num_sen):
        lang_obj_label[i][label[i]] = 1
    probs1 = F.log_softmax(prediction, 1)
    probs2 = F.softmax(lang_obj_label, 1)
    loss = F.kl_div(probs1, probs2, reduction='mean')
    return loss


def batch_train(batch, model, optimizer, batch_size):
    ref_model = model['refer']
    backbone_model = model['backbone']
    
    with torch.no_grad():#backbone_model 是不进行训练的，而是用于提取特征的。
        pc_sem, pc_feat, pc_offset = backbone_model(batch['x'])#pc_feat [N,32]

    idx = 0
    loss = 0
    train_loss = 0
    total_pred = 0
    total_ttl_correct = 0
    loss4print = {'ttl':0}
    for i, num_p in enumerate(batch['num_points']):
        scene_pcfeat = pc_feat[idx:idx+num_p]#场景的点云特征表示 多个场景的点云存放在一起
        pc_ins = batch['y_ins'][idx:idx+num_p]#每一个点云所属类别表示
        pc_sem = batch['y'][idx:idx+num_p]
        m = pc_ins > -1 # Remove unlabeled points 除去没有标注类别的点云 [Ture,True,...]

        # Get instance mean features & coordinate using groundtruth labels
        #obj_feat：形状为 (n, C) 的二维张量，其中 n 是点云中实例的数量。该张量的每一行是一个实例的特征向量
        #obj_id：形状为 (n,) 的一维张量，表示每个实例的 ID。
        obj_feat, obj_id = gather(scene_pcfeat[m], pc_ins[m])  #聚合标注同一类别的点云特征，得到它们的类别信息 
        #具体来说，函数 gather 的作用是对输入的特征 feat 和对应的标签 lbl 进行聚合操作
        coord = batch['x'][0][idx:idx+num_p, 0:3].float().cuda()#得到点云的坐标信息
        coord/=data.scale#data.scale=50
        #shift
        coord = coord - (coord.max(0)[0]*0.5+coord.min(0)[0]*0.5)#这行代码是对 coord 中的所有坐标点进行平移，使它们的中心点位于原点。
        #coord[m]是一个形状为(N, 3)的tensor，其中N表示点的数量，3表示每个点的坐标（x、y、z）。pc_ins[m]是一个形状为(N,)的整数tensor，它表示每个点所属的物体实例编号。
        #通过将pc_ins[m]作为index传递给coord[m]，gather函数可以将coord[m]中属于pc_ins[m]中每个物体实例的点的坐标按照维度1（即列）进行聚合。
        obj_coord, _ = gather(coord[m], pc_ins[m])#得到平移后的物体的坐标

        # Referring---------------------------------------------
        lang_len = batch['lang_len'][i]
        lang_feat = batch['lang_feat'][i].float().cuda()#[lang_len,80,300]

        if lang_feat.shape[0] < 256:
            rand_idx = np.arange(lang_feat.shape[0])
        else:
            rand_idx = np.random.choice(lang_feat.shape[0], 256, replace=False)
            
        scene_data = {}
        scene_data['obj_feat'] = obj_feat
        scene_data['obj_coord'] = obj_coord
        scene_data['lang_len'] = lang_len[rand_idx]
        scene_data['lang_feat'] = lang_feat[rand_idx]
        
        ttl_score, word_parse = ref_model(scene_data)
        #首先从batch['lang_objID']中选取了rand_idx个物体的ID作为obj_gt，然后将这些ID与场景中物体的ID进行比较，生成一个和obj_id形状相同的二元张量，表示哪些物体在语句中被提到。
        obj_gt = batch['lang_objID'][i][rand_idx].unsqueeze(-1).cuda() == obj_id.cuda()
        obj_gt = obj_gt.float().argmax(-1)#将这个二元张量进行float()操作，将其中的True转换为1.0，False转换为0.0，然后在最后一维上使用argmax函数，得到每个句子中提到的物体的索引

        total_pred += ttl_score.shape[0]
        total_ttl_correct += (ttl_score.argmax(-1) == obj_gt).sum()#预测正确的数量

        if torch.isnan(ttl_score).any():#torch.isnan(ttl_score).any()将检查ttl_score中是否有任何一个NaN值，并返回相应的布尔值。
            print (ttl_score)
        ref_ttl_loss = F.cross_entropy(ttl_score, obj_gt)#计算预测类别和gt类别的交叉熵
        multi_kll_loss = kll_loss(ttl_score,obj_gt)
        loss += (ref_ttl_loss+multi_kll_loss)

        loss4print['ttl'] += ref_ttl_loss.item()
        idx += num_p
    train_loss += loss/batch_size 
    train_loss.backward()
    optimizer.step()
    for t in loss4print:
        loss4print[t]/=batch_size
    return loss4print['ttl'], total_pred, total_ttl_correct


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


        m = scene_sem.argmax(-1) > 1#对最后一维进行argmax操作，获取每个像素点最可能的类别，并与1进行比较。由于语义类别编号从0开始，因此大于1的类别编号表示的是非背景类别 
        pred_sem = scene_sem.argmax(-1)#每个点都预测了实例类别
        pred_cen = scene_coords + scene_offset#预测的中心

        scene_data = {}
        scene_data['lang_feat'] = batch['lang_feat'][i].float().cuda()
        scene_data['lang_len'] = batch['lang_len'][i]

        ins_lbl = batch['y_ins'][idx:idx+num_p].cuda()#y instance labels
        
        ins_mask = torch.zeros(num_p, grps.shape[-1]).to(grps.device).long()# [N,C] N点的数量 C预测实例类别的数量
        ins_mask[m] = grps#非背景类的点赋值，值是指每个点的类别预测 

        obj_feat = torch.cat([grp_feat, grp_feat1], 0)#实例特征和背景特征在0维度上拼接
        obj_coord = torch.cat([grp_cen, grp_cen1], 0)#实例中心坐标和背景中心坐标在0维度上拼接

        obj_num, dim = obj_feat.shape
        scene_data['obj_feat'] = obj_feat
        scene_data['obj_coord'] = obj_coord

        possible_obj_num = grp_feat.shape[0]#实例的数量
        total_score, _ = ref_model(scene_data)#所有样例(包括实例和背景)的得分预测
        total_score = total_score[:, 0:possible_obj_num]#仅取实例类的得分预测
        total_score = F.softmax(total_score, -1)#得分做softmax处理 total_score[59,29]，59个场景，每个场景中29个样例的预测得分
        
        scores = [total_score.cpu().numpy()]
        #得到的是一个[N,C']矩阵，表示每个点在C'个场景中的某几个场景这出现作为mask,即预测掩码
        pred = ins_mask[:, total_score.argmax(-1)]#total_score.argmax(-1)得到的是每个场景预测的样例类别,即得分最高的实例的索引
        gt = batch['ref_lbl'][i].cuda()
        iou = (pred*gt).sum(0).float()/((pred|gt).sum(0).float()+1e-5)#计算iou (pred*gt).sum(0)表示交，(pred|gt).sum(0)表示并 .sum(0)表示对所有维度按列求和
        IOUs.append(iou.cpu().numpy())

        precision_half = (iou > 0.5).sum().float()/iou.shape[0]
        precision_quarter = (iou > 0.25).sum().float()/iou.shape[0]
        outstr = 'mean IOU {:.4f} | P@0.5 {:.4f} | P@0.25 {:.4f}'.format(iou.mean().item(), precision_half, precision_quarter)
        # io.cprint(outstr)

        pc = [scene_coords, batch['x'][1][idx:idx+num_p], pred_cen, ins_lbl.unsqueeze(-1).float()]
        
        pc = torch.cat(pc, -1)
        sen = batch['sentences'][i]
        sen = np.array(sen)
        token = batch['tokens'][i]
        ref_objname = batch['lang_objname'][i]
        idx += num_p
    IOUs = np.concatenate(IOUs, 0)
    return IOUs


def eval_one_epoch(models):
    total_ious = []
    for m in models:
        models[m].eval()
    for i,batch in enumerate(tqdm(data_val.val_data_loader)):
        if use_cuda:
            batch['x'][1]=batch['x'][1].cuda()#将颜色数据放入cuda
        with torch.no_grad():
            IOUs = batch_val(batch, models, data_val.batch_size)
            total_ious.append(IOUs)
        # print ('({}/{}) Mean IOU so far {:.4f}'.format((i+1)*data_val.batch_size, len(data_val.loader_list), np.concatenate(total_ious, 0).mean()))
    total_ious = np.concatenate(total_ious, 0)
    IOU = total_ious.mean()
    IOU05 = (total_ious > 0.5).sum().astype(float)/total_ious.shape[0]
    IOU025 = (total_ious > 0.25).sum().astype(float)/total_ious.shape[0]
    for m in models:
        models[m].train()
    return IOU, IOU05, IOU025


use_cuda = torch.cuda.is_available()
io.cprint(args.exp_name)#io.cprint(args.exp_name) 会将 args.exp_name 字符串输出到控制台和文件中。

# Initialize backbone Sparse 3D-Unet and Text-Guided GNN
backbone_model = SCN().cuda()
backbone_model = nn.DataParallel(backbone_model)#还使用 nn.DataParallel 对这个模型进行包装，以便可以在多个 GPU 上并行训练和预测
ref_model = RefNetGRU(k=10).cuda()
ref_model.relconv = nn.DataParallel(ref_model.relconv)


# 从.pkl文件中加载Python字典或列表
with open('/home/mayiwei/Project/3DRIS/TGNN/val_feat_fix.pkl', 'rb') as f:
    feat_data = pickle.load(f)

# Load pretrained instance segmentation model for backbone
models = {}
models['backbone'] = backbone_model
training_epoch = checkpoint_restore(models, './checkpoints/model_insseg', io, use_cuda)#backbone加载预训练参数 
models['refer'] = ref_model

training_epoch = 1
training_epochs = args.epochs
io.cprint('Starting with epoch: ' + str(training_epoch))


params = ref_model.parameters()
optimizer = optim.Adam(params, lr=1)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30], gamma=0.2)
writer = SummaryWriter('./checkpoints/'+args.exp_name+'/'+'logs')
best_IOU = 0
seed_everything(args.seed)
# 定义学习率调整函数
def lr_lambda(epoch):
    if epoch < 5:
        return args.lr / 5 * (epoch + 1)
    if epoch < 20+5:
        return args.lr
    elif epoch < 30+5:
        return args.lr*0.2
    else:
        return args.lr*0.2*0.2

# 定义学习率调度器
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


for m in models:
    models[m].train()

for epoch in range(training_epoch, training_epochs+1):
    print("Epoch {}: lr = {}".format(epoch, optimizer.param_groups[0]['lr']))

    total_loss = {}
    pbar = tqdm(data.train_data_loader)

    total = 0
    ttl_correct = 0
    obj_correct = 0
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()
        if use_cuda:
            batch['x'][1] = batch['x'][1].cuda()  # [N,3] All point color information

        with torch.autograd.set_detect_anomaly(True):  # This function from PyTorch enables anomaly detection in the automatic differentiation process to detect issues like NaN or infinite values during gradient computation.
            loss, t, tc = batch_train(batch, models, optimizer, data.batch_size)
        total += t
        ttl_correct += tc
    scheduler.step()

    outstr = 'Epoch: {}, Loss: {:.4f}, '.format(epoch, loss) + 'Correct Objects: {:.4f}'.format(float(ttl_correct)/float(total))
    
    print('eval-',str(epoch))
    IOU, IOU05, IOU025 = eval_one_epoch(models)
    print("Epoch {}: lr = {}".format(epoch, optimizer.param_groups[0]['lr']))
    print("mean IOU= {}; P@0.5 = {}; P@0.25 = {}".format(IOU, IOU05, IOU025))

    # print(IOU, IOU05, IOU025)
    writer.add_scalar('learning rate', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('mean IOU', IOU, epoch)
    writer.add_scalar('P@0.5', IOU05, epoch)
    writer.add_scalar('P@0.25', IOU025, epoch)
    writer.add_scalar('loss', loss, epoch)
    writer.add_scalar('Correct Objects', float(ttl_correct)/float(total), epoch)
    
    
    io.cprint(outstr)
    checkpoint_save(models, './checkpoints/'+args.exp_name+'/'+'models'+'/'+args.exp_name, epoch, io, use_cuda)
    
    if best_IOU < IOU:
        best_IOU = IOU
        save_path = './checkpoints/'+args.exp_name+'/'+'models'+'/'+args.exp_name+'-%09d'%epoch+'.pth'
        best_path = './checkpoints/'+args.exp_name+'/'+'models'+'/'+args.exp_name+'-best.pth'
        shutil.copy(save_path, best_path)
