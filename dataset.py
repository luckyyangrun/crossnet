import dgl
import torch
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd
from utils.build_graph import PandasGraphBuilder, rat2data, gen_testdata


class DouBanDataset(Dataset):
    def __init__(self, g, cfg):
        self.user_list = range((g.number_of_nodes(cfg.DATA.USER_NTYPE)))

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, index):
        heads = self.user_list[index]
        return heads



class NeighborSampler(object):
    def __init__(self, g, cfg):
        self.g = g
        self.user_ntype = cfg.DATA.USER_NTYPE
        self.da_ntype = cfg.DATA.DA_NTYPE
        self.db_ntype = cfg.DATA.DB_NTYPE 
        self.da_etype = cfg.DATA.DA_ETYPE
        self.db_etype = cfg.DATA.DB_ETYPE 
        self.neg_num = cfg.DATALOADER.NEG_NUM
        self.pos_num = cfg.DATALOADER.POS_NUM

        self.num_layers = cfg.MODEL.NUM_LAYER
        self.da_etype_inv = cfg.DATA.DA_ETYPE_INV
        self.db_etype_inv = cfg.DATA.DB_ETYPE_INV
        self.neighbor_rate = cfg.DATA.NEIGHBOR_RATE
        self.batch_size = cfg.DATALOADER.BATCH_SIZE
    def sample_mix_blocks(self, seeds):
        blocks = []
        for i in range(self.num_layers):
            #import pdb;pdb.set_trace()
            sg = dgl.in_subgraph(self.g, seeds)
            #new_edges_masks = {}
        # 遍历所有边的类型
            #for etype in sg.canonical_etypes:
            #    edge_mask = torch.zeros(sg.number_of_edges(etype))
            #    edge_mask.bernoulli_(self.neighbor_rate)
            #    new_edges_masks[etype] = edge_mask.bool()
        # 返回一个与初始图有相同节点的图作为边界
            #import pdb;pdb.set_trace()
            #frontier = dgl.edge_subgraph(sg, new_edges_masks, preserve_nodes=True)
            #block = dgl.to_block(frontier, seeds)
            block = dgl.to_block(sg, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks    
   
    def sample_mix_pairs(self, heads, da_tails, da_neg_tails, db_tails, db_neg_tails):
        # Create a graph with positive connections only and another graph with negative
        # connections only.
         
        da_tails = da_tails.squeeze(0)
        db_tails = db_tails.squeeze(0)
        da_neg_tails = da_neg_tails.squeeze(0)
        db_neg_tails = db_neg_tails.squeeze(0)
        pos_heads = heads.repeat(self.neg_num)
        #import pdb;pdb.set_trace()
        edges_per_relation = {(self.user_ntype, self.da_etype, self.da_ntype) : (pos_heads, da_tails),
                            (self.da_ntype, self.da_etype_inv, self.user_ntype) : (da_tails, pos_heads),
                            (self.user_ntype, self.db_etype, self.db_ntype): (pos_heads, db_tails),
                            (self.db_ntype, self.db_etype_inv, self.user_ntype) : (db_tails, pos_heads)}
        #import pdb;pdb.set_trace()
        num_nodes_dict={ntype : self.g.num_nodes(ntype) for ntype in self.g.ntypes}
        pos_graph = dgl.heterograph(edges_per_relation, 
                                    num_nodes_dict={ntype : self.g.num_nodes(ntype) for ntype in self.g.ntypes})
        da_edges_per_relation = {(self.user_ntype, self.da_etype, self.da_ntype) : (pos_heads, da_tails),
                            (self.da_ntype, self.da_etype_inv, self.user_ntype) : (da_tails, pos_heads)}

        da_pos_graph = dgl.heterograph(da_edges_per_relation, 
                                    num_nodes_dict={ntype : self.g.num_nodes(ntype) for ntype in self.g.ntypes})

        db_edges_per_relation = {(self.user_ntype, self.db_etype, self.db_ntype) : (pos_heads, db_tails),
                            (self.db_ntype, self.db_etype_inv, self.user_ntype) : (db_tails, pos_heads)}

        db_pos_graph = dgl.heterograph(db_edges_per_relation, 
                                    num_nodes_dict={ntype : self.g.num_nodes(ntype) for ntype in self.g.ntypes})
    
        da_neg_heads = heads.repeat(self.neg_num)
        db_neg_heads = heads.repeat(self.neg_num)
        #import pdb;pdb.set_trace()
        #edges_per_relation = {(self.user_ntype, self.da_etype, self.da_ntype) : (da_neg_heads, da_neg_tails),
        #                    (self.da_ntype, self.da_etype_inv, self.user_ntype) : (da_neg_tails, da_neg_heads),
        #                    (self.user_ntype, self.db_etype, self.db_ntype): (db_neg_heads, db_neg_tails),
        #                    (self.db_ntype, self.db_etype_inv, self.user_ntype) : (db_neg_tails, db_neg_heads)}
        #neg_graph = dgl.heterograph(edges_per_relation,
        #                            num_nodes_dict={ntype : self.g.num_nodes(ntype) for ntype in self.g.ntypes})
        da_neg_edges_per_relation = {(self.user_ntype, self.da_etype, self.da_ntype) : (da_neg_heads, da_neg_tails),
                            (self.da_ntype, self.da_etype_inv, self.user_ntype) : (da_neg_tails, da_neg_heads)}
        db_neg_edges_per_relation = {(self.user_ntype, self.db_etype, self.db_ntype) : (db_neg_heads, db_neg_tails),
                            (self.db_ntype, self.db_etype_inv, self.user_ntype) : (db_neg_tails, db_neg_heads)}
        da_neg_graph = dgl.heterograph(da_neg_edges_per_relation,
                                    num_nodes_dict={ntype : self.g.num_nodes(ntype) for ntype in self.g.ntypes})
        db_neg_graph = dgl.heterograph(db_neg_edges_per_relation,
                                    num_nodes_dict={ntype : self.g.num_nodes(ntype) for ntype in self.g.ntypes})
        
        da_pos_graph, db_pos_graph, da_neg_graph, db_neg_graph, pos_graph = dgl.compact_graphs([da_pos_graph,db_pos_graph, 
                                                                                    da_neg_graph, db_neg_graph, pos_graph])
        #seeds = {k:v[0] for k, v in edges_per_relation.items()}
        #pos_graph = dgl.compact_graphs([pos_graph])
        seeds = pos_graph.ndata[dgl.NID]
        #import pdb;pdb.set_trace()

        blocks = self.sample_mix_blocks(seeds)
        return da_pos_graph, db_pos_graph, da_neg_graph, db_neg_graph, blocks
    
    
    
    
    def sample_db_ssl_blocks(self, seeds):        
        blocks = []
        for _ in range(self.num_layers):
            sg = dgl.in_subgraph(self.g, seeds)
            

            new_edges_masks = {}
        # 遍历所有边的类型
            #for etype in sg.canonical_etypes:
                #edge_mask = torch.zeros(sg.number_of_edges(etype))
                #edge_mask.bernoulli_(self.neighbor_rate)
                #new_edges_masks[etype] = edge_mask.bool()
            #new_edges_masks[(self.user_ntype, self.da_etype, self.da_ntype)] = torch.zeros(sg.number_of_edges((self.user_ntype, self.da_etype, self.da_ntype))).bool()
            #new_edges_masks[(self.da_ntype, self.da_etype_inv, self.user_ntype)] = torch.zeros(sg.number_of_edges((self.da_ntype, self.da_etype_inv, self.user_ntype))).bool()
        # 返回一个与初始图有相同节点的图作为边界
            #frontier = dgl.edge_subgraph(sg, new_edges_masks, preserve_nodes=True)
            #block = dgl.to_block(frontier, seeds)
            frontier = dgl.edge_type_subgraph(sg, [(self.db_ntype, self.db_etype_inv, self.user_ntype),(self.user_ntype, self.db_etype, self.db_ntype)])
            block = dgl.to_block(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            #import pdb;pdb.set_trace()
            blocks.insert(0, block)
        return blocks

    def sample_da_ssl_blocks(self, seeds):        
        blocks = []
        for _ in range(self.num_layers):
            sg = dgl.in_subgraph(self.g, seeds)
            new_edges_masks = {}
        # 遍历所有边的类型
            #for etype in sg.canonical_etypes:
                #edge_mask = torch.zeros(sg.number_of_edges(etype))
                #edge_mask.bernoulli_(self.neighbor_rate)
                #new_edges_masks[etype] = edge_mask.bool()
            #new_edges_masks[(self.user_ntype, self.db_etype, self.db_ntype)] = torch.zeros(sg.number_of_edges((self.user_ntype, self.db_etype, self.db_ntype))).bool()
            #new_edges_masks[(self.db_ntype, self.db_etype_inv, self.user_ntype)] = torch.zeros(sg.number_of_edges((self.db_ntype, self.db_etype_inv, self.user_ntype))).bool()
        # 返回一个与初始图有相同节点的图作为边界
            #frontier = dgl.edge_subgraph(sg, new_edges_masks, preserve_nodes=True)
            #block = dgl.to_block(frontier, seeds)
            #block = dgl.to_block(sg, seeds)
            frontier = dgl.edge_type_subgraph(sg, [(self.da_ntype, self.da_etype_inv, self.user_ntype),(self.user_ntype, self.da_etype, self.da_ntype)])
            block = dgl.to_block(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks
        
 
class CROSSNETCollator(object):
    def __init__(self, sampler, g, cfg):
        self.sampler = sampler
        self.da_ntype = cfg.DATA.DA_NTYPE
        self.db_ntype = cfg.DATA.DB_NTYPE
        self.user_ntype = cfg.DATA.USER_NTYPE
        self.g = g
        self.da_etype = cfg.DATA.DA_ETYPE
        self.db_etype = cfg.DATA.DB_ETYPE
        self.batch_size = cfg.DATALOADER.BATCH_SIZE
        self.neg_num = cfg.DATALOADER.NEG_NUM
        self.pos_num = cfg.DATALOADER.POS_NUM
        

    def collate_mix_graph_train(self, batches):
        #import pdb;pdb.set_trace()
        heads, da_tails, da_neg_tails = batches[self.da_ntype]
        heads, db_tails, db_neg_tails = batches[self.db_ntype]
        da_pos_graph, db_pos_graph, da_neg_graph, db_neg_graph, blocks = self.sampler.sample_mix_pairs(heads, da_tails, da_neg_tails, db_tails, db_neg_tails)
        return da_pos_graph, db_pos_graph, da_neg_graph, db_neg_graph, blocks

    
    def da_collate_ssl_train(self, batches):
        heads, _, _ = batches[self.da_ntype]
        # Construct multilayer neighborhood via PinSAGE...
        blocks  = self.sampler.sample_da_ssl_blocks({self.user_ntype :heads})
        return blocks
    
    def db_collate_ssl_train(self, batches):
        heads, _, _ = batches[self.db_ntype]
        # Construct multilayer neighborhood via PinSAGE...
        blocks  = self.sampler.sample_db_ssl_blocks({self.user_ntype :heads})
        return blocks
    
    def collate_train(self, heads):
        da_tails, db_tails, da_neg_tails, db_neg_tails =[], [], [], []
        
        for i in range(len(heads)):
            for _ in range(self.neg_num):
                tmp_da_tails = dgl.sampling.random_walk(self.g, heads[i], metapath=[(self.user_ntype, self.da_etype, self.da_ntype)])[0][:,1]
                if tmp_da_tails == -1:tmp_da_tails = torch.tensor([1])
                tmp_db_tails = dgl.sampling.random_walk(self.g, heads[i], metapath=[(self.user_ntype, self.db_etype, self.db_ntype)])[0][:,1]
                if tmp_db_tails == -1:tmp_db_tails = torch.tensor([1])
                da_tails.append(tmp_da_tails)
                db_tails.append(tmp_db_tails)

        da_neg_tails = torch.randint(0, self.g.number_of_nodes(self.da_ntype), (self.batch_size * self.neg_num, ))
        db_neg_tails = torch.randint(0, self.g.number_of_nodes(self.db_ntype), (self.batch_size * self.neg_num, ))
        da_tails = torch.cat(da_tails).squeeze(0)
        db_tails = torch.cat(db_tails).squeeze(0)
        heads = torch.tensor(heads)
        batches =  {self.da_ntype : [heads, da_tails, da_neg_tails], 
                   self.db_ntype : [heads, db_tails, db_neg_tails]} 

        da_pos_graph, db_pos_graph, da_neg_graph, db_neg_graph, blocks = self.collate_mix_graph_train(batches)
        da_ssl_blocks= self.da_collate_ssl_train(batches)
        db_ssl_blocks= self.db_collate_ssl_train(batches)
        return {'da_pos_graph':da_pos_graph, 
                'db_pos_graph':db_pos_graph, 
                'da_neg_graph':da_neg_graph, 
                'db_neg_graph':db_neg_graph, 
                'blocks':blocks, 
                'da_ssl_blocks':da_ssl_blocks, 
                'db_ssl_blocks':db_ssl_blocks}
        
