import numpy as np
import pandas as pd
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import os
import argparse
import time
import itertools
import random
import shutil
import torch
import torch.distributed as dist
from config import config as cfg
from dataset import DouBanDataset, CROSSNETCollator, NeighborSampler
from net import Model 
from utils.misc import mkdir
from utils.logger import setup_logger
from utils.collect_env import collect_env_info
from utils import comm
import pickle
import math
import heapq
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch dgl 暂不支持分布式训练')
parser.add_argument('--resume', '-r', type=str, required=False,
                    help='resume from checkpoint')
parser.add_argument("--local_rank", type=int, default=0, help="local rank")
parser.add_argument("--seed", type=int, default=1234, help="random seed")
parser.add_argument('-p', '--print-freq', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
args = parser.parse_args()
best_hr_a = float('-inf')
best_hr_b = float('-inf')
best_NDCG_a = float('-inf')
best_NDCG_b = float('-inf')
best_hr_a_epoch = float('inf')
best_hr_b_epoch = float('inf')
best_NDCG_a_epoch = float('inf')
best_NDCG_b_epoch = float('inf')

def main():
    global best_hr_a
    global best_hr_b
    global best_NDCG_a
    global best_NDCG_b
    global best_NDCG_a_epoch
    global best_NDCG_b_epoch
    global best_hr_a_epoch
    global best_hr_b_epoch
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    num_gpus = int(
        os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if comm.is_main_process() and cfg.output_dir:
        mkdir(cfg.output_dir)
        mkdir(cfg.checkpoint_dir)

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')

    logger = setup_logger("cross-net", cfg.output_dir, comm.get_rank(),
                          filename='train_log.txt')

    logger.info("Rank of current process: {}. World size: {}".format(
                comm.get_rank(), comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())
    logger.info("Command line arguments: " + str(args))

    with open(cfg.DATA.DATA_PATH, 'rb') as f:
        dataset = pickle.load(f)
    g = dataset['train_graph']
    test_user = dataset['test_user'] # shape:[num_user, 100]
    test_item_a = dataset['test_item_a']
    test_item_b = dataset['test_item_b']
    test_data = {'user':test_user,
                'da_tails':test_item_a,
                'db_tails':test_item_b}

    #import pdb;pdb.set_trace()
    dataset = DouBanDataset(g, cfg)
    neighbor_sampler = NeighborSampler(g, cfg)
    collator = CROSSNETCollator(neighbor_sampler, g, cfg)
    train_dataloader = torch.utils.data.DataLoader(dataset,drop_last=True, batch_size=cfg.DATALOADER.BATCH_SIZE, collate_fn=collator.collate_train, num_workers=0)

    model = Model(cfg)
    #logger.info(model.parameters())
    print(model)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            find_unused_parameters=True)

    optimizer = torch.optim.Adam(model.parameters(), 
                                    lr=cfg.SOLVER.LR, betas=(0.5, 0.999))

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
        # Map model to be loaded to specified single gpu.
        checkpoint = torch.load(args.resume, map_location=torch.device('cuda'))
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        logger.info("==> testing")
        #hr_a, NDCG_a, hr_b, NDCG_b = validate(logger, test_data, g, model, cfg)
        # evaluate on validation set
        #
        #
        #
        #

    for epoch in range(args.start_epoch, cfg.SOLVER.MAX_EPOCH):
        #if args.distributed:
         #   train_sampler.set_epoch(epoch)

        #adjust_learning_rate(optimizer, epoch, cfg)

        # train for one epoch
        train(logger, train_dataloader, model, optimizer, epoch, args)

        if (epoch+1) % 20 ==  0 and epoch != 0:
        # evaluate on validation set
            hr_a, NDCG_a, hr_b, NDCG_b = validate(logger, test_data, g, model, cfg)

            if comm.is_main_process():
            # remember best mape and save checkpoint
                if hr_a > best_hr_a:
                    best_hr_a_epoch = epoch
                if hr_b > best_hr_b:
                    best_hr_b_epoch = epoch
                if NDCG_a > best_NDCG_a:
                    best_NDCG_a_epoch = epoch
                if NDCG_b > best_NDCG_b:
                    best_NDCG_b_epoch = epoch
                is_best = hr_a > best_hr_a
                best_hr_a = max(hr_a, best_hr_a)
                best_hr_b = max(hr_b, best_hr_b)
                best_NDCG_a = max(NDCG_a, best_NDCG_a)
                best_NDCG_b = max(NDCG_b, best_NDCG_b)
            
            logger.info(f"best_hr_a {best_hr_a:.3f} at epoch {best_hr_a_epoch}")
            logger.info(f"best_hr_b {best_hr_b:.3f} at epoch {best_hr_b_epoch}")
            logger.info(f"best_NDCG_a {best_NDCG_a:.3f} at epoch {best_NDCG_a_epoch}")
            logger.info(f"best_NDCG_b {best_NDCG_b:.3f} at epoch {best_NDCG_b_epoch}")
            if comm.is_main_process():
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_hr_a': best_hr_a,
                'best_hr_b': best_hr_b,
                'best_NDCG_a':best_NDCG_a,
                'best_NDCG_b':best_NDCG_b,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename='logs/checkpoint/checkpoint.pth.tar')


def train(logger, train_loader, model, optimizer, epoch, args):
    len_loader = len(train_loader)
    # switch to train mode
    model.train()

    end = time.time()
    for i, (batch_data) in enumerate(train_loader):
        # compute loss
        loss = model( batch_data)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time = time.time() - end
        end = time.time()
        lr = optimizer.param_groups[0]['lr']
        if i % args.print_freq == 0:
            i = "%03d" % i
            logger.info(f"Train:[{epoch}][{i}/{len_loader}] batch_time: {batch_time:.4f} loss: {loss:.4f} lr: {lr:.6f}")


def validate(logger, test_data, g, model, cfg):
    # switch to evaluate mode
    heads = test_data['user']
    da_tails = test_data['da_tails']
    db_tails = test_data['db_tails']
    num_user_test = len(heads)
    hr_a = []
    hr_b = []
    NDCG_a = []
    NDCG_b = []
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i in tqdm(range(num_user_test)):
            
            da_tail = da_tails[i]
            db_tail = db_tails[i]
            user = heads[i]
            head = [user]*100
            data = sample_test(g, head, da_tail, db_tail)
            da_score, db_score = model(data)
            item_score_dict_a = {}
            da_gt = da_tail[0]
            da_sub_graph = data['da_sub_graph_test']
            tmp_hr_a = []
            tmp_NDCG_a = []
            #import pdb;pdb.set_trace()
            for item in da_tail:
                idx = da_sub_graph.edge_ids(user,item,etype=(cfg.DATA.USER_NTYPE, cfg.DATA.DA_ETYPE, cfg.DATA.DA_NTYPE))
                pred = da_score[cfg.DATA.USER_NTYPE,cfg.DATA.DA_ETYPE,cfg.DATA.DA_NTYPE][idx]
                #idx = da_sub_graph.edge_ids(item,user,etype=(cfg.DATA.DA_NTYPE, cfg.DATA.DA_ETYPE_INV, cfg.DATA.USER_NTYPE))
                #pred = da_score[cfg.DATA.DA_NTYPE, cfg.DATA.DA_ETYPE_INV, cfg.DATA.USER_NTYPE][idx]
                item_score_dict_a[item] = pred
            ranklist = heapq.nlargest(cfg.MODEL.TOPK, item_score_dict_a, key=item_score_dict_a.get)
            tmp_hr_a = getHitRatio(ranklist, da_gt)
            tmp_NDCG_a = getNDCG(ranklist, da_gt)
            hr_a.append(tmp_hr_a)
            NDCG_a.append(tmp_NDCG_a)

            item_score_dict_b = {}
            db_gt = db_tail[0]
            db_sub_graph = data['db_sub_graph_test']
            tmp_hr_b = []
            tmp_NDCG_b = []
            for item in db_tail:
                idx = db_sub_graph.edge_id(user,item,etype=(cfg.DATA.USER_NTYPE, cfg.DATA.DB_ETYPE, cfg.DATA.DB_NTYPE))
                pred = db_score[cfg.DATA.USER_NTYPE, cfg.DATA.DB_ETYPE, cfg.DATA.DB_NTYPE][idx]
                #idx = db_sub_graph.edge_id(item,user,etype=(cfg.DATA.DB_NTYPE, cfg.DATA.DB_ETYPE_INV, cfg.DATA.USER_NTYPE))
                #pred = db_score[cfg.DATA.DB_NTYPE, cfg.DATA.DB_ETYPE_INV, cfg.DATA.USER_NTYPE][idx]
                item_score_dict_b[item] = pred
            ranklist = heapq.nlargest(cfg.MODEL.TOPK, item_score_dict_b, key=item_score_dict_b.get)
            tmp_hr_b = getHitRatio(ranklist, db_gt)
            tmp_NDCG_b = getNDCG(ranklist, db_gt)
            hr_b.append(tmp_hr_b)
            NDCG_b.append(tmp_NDCG_b)
    HR_A, NDCG_A, HR_B, NDCG_B = np.mean(hr_a), np.mean(NDCG_a), np.mean(hr_b), np.mean(NDCG_b)
    val_time = time.time() - end
    logger.info(f'Current HR_A {HR_A:.3f} NDCG_A {NDCG_A:.3f}')
    logger.info(f'HR_B {HR_B:.3f} NDCG_B {NDCG_B:.3f}')
    logger.info(f'validate use val_time {val_time:.4f}')
    return np.mean(hr_a), np.mean(NDCG_a), np.mean(hr_b), np.mean(NDCG_b)

def sample_test_blocks(g, seeds):
    blocks = []
    for _ in range(cfg.MODEL.NUM_LAYER):
        sg = dgl.in_subgraph(g, seeds)
        block = dgl.to_block(sg, seeds)
        seeds = block.srcdata[dgl.NID]
        blocks.insert(0, block)
    return blocks    


def sample_test(g, heads, da_tails, db_tails):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
    heads = torch.tensor(heads)
    da_tails = torch.tensor(da_tails)
    db_tails = torch.tensor(db_tails)
    num_nodes_pre_types = {cfg.DATA.USER_NTYPE : heads.shape[0], 
                            cfg.DATA.DA_NTYPE : da_tails.shape[0],
                             cfg.DATA.DB_NTYPE : db_tails.shape[0]}
  
    edges_per_relation = {(cfg.DATA.USER_NTYPE, cfg.DATA.DA_ETYPE, cfg.DATA.DA_NTYPE) : list(zip(heads.reshape(-1), da_tails.reshape(-1))),
                            (cfg.DATA.USER_NTYPE, cfg.DATA.DB_ETYPE, cfg.DATA.DB_NTYPE): list(zip(heads.reshape(-1), db_tails.reshape(-1))),
                            (cfg.DATA.DB_NTYPE, cfg.DATA.DB_ETYPE_INV, cfg.DATA.USER_NTYPE) : list(zip(db_tails.reshape(-1), heads.reshape(-1)))}

    da_edges_per_relation = {(cfg.DATA.USER_NTYPE, cfg.DATA.DA_ETYPE, cfg.DATA.DA_NTYPE) : list(zip(heads.reshape(-1), da_tails.reshape(-1))),
                            (cfg.DATA.DA_NTYPE, cfg.DATA.DA_ETYPE_INV, cfg.DATA.USER_NTYPE) : list(zip(da_tails.reshape(-1), heads.reshape(-1)))}
                            
    sub_graph = dgl.heterograph(edges_per_relation, 
                                num_nodes_dict={ntype :g.num_nodes(ntype) for ntype in g.ntypes})

    da_sub_graph = dgl.heterograph(da_edges_per_relation, 
                                    num_nodes_dict={ntype : g.num_nodes(ntype) for ntype in g.ntypes})

    db_edges_per_relation = {(cfg.DATA.USER_NTYPE, cfg.DATA.DB_ETYPE, cfg.DATA.DB_NTYPE) : list(zip(heads.reshape(-1), db_tails.reshape(-1))),
                            (cfg.DATA.DB_NTYPE, cfg.DATA.DB_ETYPE_INV, cfg.DATA.USER_NTYPE) : list(zip(db_tails.reshape(-1), heads.reshape(-1)))}

    db_sub_graph = dgl.heterograph(db_edges_per_relation, 
                                    num_nodes_dict={ntype : g.num_nodes(ntype) for ntype in g.ntypes})
    #import pdb;pdb.set_trace()
    da_sub_graph_cal, db_sub_graph_cal,sub_graph_cal = dgl.compact_graphs([da_sub_graph,db_sub_graph, sub_graph])
    seeds = sub_graph_cal.ndata[dgl.NID]
    blocks = sample_test_blocks(g, seeds)
    return {'da_sub_graph':da_sub_graph_cal,'db_sub_graph':db_sub_graph_cal, 'blocks':blocks,
            'da_sub_graph_test':da_sub_graph, 'db_sub_graph_test':db_sub_graph}
    

def getHitRatio(ranklist, targetItem):
    for item in ranklist:
        if item == targetItem:
            return 1
    return 0
                 
def getNDCG(ranklist, targetItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == targetItem:
            return math.log(2) / math.log(i + 2)
    return 0

def save_checkpoint(state, is_best, filename='logs/checkpoint/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'logs/checkpoint/model_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, cfg):
    """Sets the learning rate to the initial LR decayed by 10 every 80 epochs"""
    lr = cfg.SOLVER.LR * (0.1 ** (epoch // 80))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
