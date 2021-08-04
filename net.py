from pickle import REDUCE
from functools import partial
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn.pytorch as dglnn
import dgl.function as fn 


class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.DEVICE
        self.in_dim = cfg.MODEL.EMBEDDING_IN_DIM
        self.hidden_dim = cfg.MODEL.GRAPH_HIDDEN_DIM
        self.out_dim = cfg.MODEL.EMBEDDING_OUT_DIM
        self.cross_hidden_dim = cfg.MODEL.CROSS_HEDDIEN_DIM
        self.proj_dim = cfg.MODEL.PROJECT_DIM
        self.n_views = cfg.MODEL.N_VIEWS
        self.da_ntype = cfg.DATA.DA_NTYPE
        self.db_ntype = cfg.DATA.DB_NTYPE

        self.da_etype = cfg.DATA.DA_ETYPE
        self.db_etype = cfg.DATA.DB_ETYPE
        self.da_etype_inv = cfg.DATA.DA_ETYPE_INV
        self.db_etype_inv = cfg.DATA.DB_ETYPE_INV
        self.user_ntype = cfg.DATA.USER_NTYPE
        self.etypes = cfg.DATA.ETYPES
        self.num_da_ntype = cfg.DATA.NUM_DA_NTYPE
        self.num_db_ntype = cfg.DATA.NUM_DB_NTYPE
        self.num_user_ntype = cfg.DATA.NUM_USER_NTYPE
        self.dnn_hidden_dim = cfg.MODEL.DNN_HIDDEN_DIM
        self.init_input  = nn.ModuleDict({self.user_ntype:nn.Embedding(self.num_user_ntype, self.in_dim), 
                                        self.da_ntype:nn.Embedding(self.num_da_ntype, self.in_dim),
                                        self.db_ntype:nn.Embedding(self.num_db_ntype, self.in_dim)})
        #self.init_input2  = nn.ModuleDict({self.user_ntype:nn.Embedding(self.num_user_ntype, self.in_dim), 
        #                                self.da_ntype:nn.Embedding(self.num_da_ntype, self.in_dim),
        #                                self.db_ntype:nn.Embedding(self.num_db_ntype, self.in_dim)})


        self.pre_conv = Feature_Pre_Conv()
        #self.rec_pre_conv = Feature_Pre_Conv()

        #self.da_conv_dict1 = nn.ModuleDict({self.da_etype: dglnn.GraphConv(self.in_dim, self.hidden_dim, norm='right'),
        #                                   self.da_etype_inv: dglnn.GraphConv(self.in_dim, self.hidden_dim, norm='right')})
        #self.da_conv_dict2 = nn.ModuleDict({self.da_etype: dglnn.GraphConv(self.hidden_dim, self.out_dim, norm='right'),
        #                                   self.da_etype_inv: dglnn.GraphConv(self.hidden_dim, self.out_dim, norm='right')})
        self.g_conv_dict1 = nn.ModuleDict({etype: dglnn.GraphConv(self.in_dim, self.hidden_dim, norm='right') for etype in self.etypes})
        self.g_conv_dict2 = nn.ModuleDict({etype: dglnn.GraphConv(self.hidden_dim, self.out_dim, norm='right') for etype in self.etypes})
        #self.da_rgcn = StochasticTwoLayerRGCN(self.da_conv_dict1, self.da_conv_dict2)
        self.da_rgcn = StochasticTwoLayerRGCN(self.g_conv_dict1, self.g_conv_dict2)

        #self.db_conv_dict1 = nn.ModuleDict({self.db_etype: dglnn.GraphConv(self.in_dim, self.hidden_dim, norm='right'),
        #                                   self.db_etype_inv: dglnn.GraphConv(self.in_dim, self.hidden_dim, norm='right')})
        #self.db_conv_dict2 = nn.ModuleDict({self.db_etype: dglnn.GraphConv(self.hidden_dim, self.out_dim, norm='right'),
        #                                   self.db_etype_inv: dglnn.GraphConv(self.hidden_dim, self.out_dim, norm='right')})
        #self.db_rgcn = StochasticTwoLayerRGCN(self.db_conv_dict1, self.db_conv_dict2)
        self.db_rgcn = StochasticTwoLayerRGCN(self.g_conv_dict1, self.g_conv_dict2)


        #self.mix_conv_dict1 = nn.ModuleDict({**self.da_conv_dict1, **self.db_conv_dict1})
        #self.mix_conv_dict2 = nn.ModuleDict({**self.da_conv_dict2, **self.db_conv_dict2})
        self.mix_conv_dict1 = nn.ModuleDict({**self.g_conv_dict1, **self.g_conv_dict1})
        self.mix_conv_dict2 = nn.ModuleDict({**self.g_conv_dict2, **self.g_conv_dict2})
        self.mix_rgcn = StochasticTwoLayerRGCN(self.mix_conv_dict1, self.mix_conv_dict2)


        self.pred = ScorePredictor()
        self.projector = nn.Sequential(nn.Linear(self.out_dim, self.proj_dim), nn.ReLU(), nn.Linear(self.proj_dim, self.out_dim))
        self.da_item_dnn = DNN(self.in_dim, self.dnn_hidden_dim, self.out_dim)
        self.db_item_dnn = DNN(self.in_dim, self.dnn_hidden_dim, self.out_dim)
        self.da_item_cross_dnn = CrossDNN(self.out_dim*2, self.cross_hidden_dim, self.out_dim)
        self.db_item_cross_dnn = CrossDNN(self.out_dim*2, self.cross_hidden_dim, self.out_dim)
        self.da_user_dnn = DNN(self.in_dim, self.dnn_hidden_dim, self.out_dim)
        self.db_user_dnn = DNN(self.in_dim, self.dnn_hidden_dim, self.out_dim)
        self.da_user_cross_dnn = CrossDNN(self.out_dim*2, self.cross_hidden_dim, self.out_dim)
        self.db_user_cross_dnn = CrossDNN(self.out_dim*2, self.cross_hidden_dim, self.out_dim)
        self.to(self.device)

    def forward(self, data): 
        if self.training:
            
            ###ssl-part
            da_ssl_blocks = self.block_to_gpu(data['da_ssl_blocks'])
            db_ssl_blocks = self.block_to_gpu(data['db_ssl_blocks'])
            da_embed = {}
            for key, item in da_ssl_blocks[0].srcdata[dgl.NID].items():
                feature = th.stack([self.init_input[key](item), da_ssl_blocks[0].srcdata['s2v'][key]])
                feature = feature.permute(1, 0, 2)
                da_embed[key] = feature
            da_embed = self.pre_conv(da_embed)
            da_ssl_blocks[0].srcdata['feature'] = da_embed

            db_embed = {}
            for key, item in db_ssl_blocks[0].srcdata[dgl.NID].items():
                feature = th.stack([self.init_input[key](item), db_ssl_blocks[0].srcdata['s2v'][key]])
                feature = feature.permute(1, 0, 2)
                db_embed[key] = feature
            db_embed = self.pre_conv(db_embed)
            db_ssl_blocks[0].srcdata['feature'] = db_embed


            da_ssl_feature = da_ssl_blocks[0].srcdata['feature']
            db_ssl_feature = db_ssl_blocks[0].srcdata['feature']
            da_ssl_user = self.projector(self.da_rgcn(da_ssl_blocks, da_ssl_feature)['user'])
            db_ssl_user = self.projector(self.db_rgcn(db_ssl_blocks, db_ssl_feature)['user'])
            ssl_user_encode = th.cat([da_ssl_user, db_ssl_user], dim=0)
            ssl_loss = self.info_nce_loss(features = ssl_user_encode)
            
        ###rec-part
            blocks = self.block_to_gpu(data['blocks'])
            indomain_embed = {}
            for key, item in blocks[-1].dstdata[dgl.NID].items():
                feature = th.stack([self.init_input[key](item), blocks[-1].dstdata['s2v'][key]])
                feature = feature.permute(1, 0, 2)
                indomain_embed[key] = feature

            indomain_feature = self.pre_conv(indomain_embed)
                            
            da_item_feature = indomain_feature[self.da_ntype]
            da_user_feature = indomain_feature[self.user_ntype]
            da_item_feature = self.da_item_dnn(da_item_feature)
            da_user_feature = self.da_user_dnn(da_user_feature)
        
            db_item_feature = indomain_feature[self.db_ntype]
            db_user_feature = indomain_feature[self.user_ntype]
            db_item_feature = self.db_item_dnn(db_item_feature)
            db_user_feature = self.db_user_dnn(db_user_feature)

            cross_embed = {}
            for key, item in blocks[0].srcdata[dgl.NID].items():
                feature = th.stack([self.init_input[key](item), blocks[0].srcdata['s2v'][key]])
                feature = feature.permute(1, 0, 2)
                cross_embed[key] = feature

            cross_feature = self.pre_conv(cross_embed)
            cross_feature = self.mix_rgcn(blocks, cross_feature)
            da_encoded = {self.da_ntype: self.da_item_cross_dnn(da_item_feature, cross_feature[self.da_ntype]), 
                    self.user_ntype: self.da_user_cross_dnn(da_user_feature, cross_feature[self.user_ntype])}
            db_encoded = {self.db_ntype: self.db_item_cross_dnn(db_item_feature, cross_feature[self.db_ntype]), 
                    self.user_ntype: self.db_user_cross_dnn(db_user_feature, cross_feature[self.user_ntype])}
            da_pos_graph = data['da_pos_graph'].to(self.device)
            db_pos_graph = data['db_pos_graph'].to(self.device)
            da_neg_graph = data['da_neg_graph'].to(self.device)
            db_neg_graph = data['db_neg_graph'].to(self.device)
            da_pos_score = self.pred(da_pos_graph, da_encoded)
            da_neg_score = self.pred(da_neg_graph, da_encoded)
            db_pos_score = self.pred(db_pos_graph, db_encoded)
            db_neg_score = self.pred(db_neg_graph, db_encoded)
            da_res_loss = self.compute_loss(da_pos_score, da_neg_score)
            db_res_loss = self.compute_loss(db_pos_score, db_neg_score)
            loss = ssl_loss + da_res_loss + db_res_loss
            #loss = da_res_loss + db_res_loss
            return loss
        else:
        ###rec-part
            blocks = self.block_to_gpu(data['blocks'])
            indomain_embed = {}
            for key, item in blocks[-1].dstdata[dgl.NID].items():
                feature = th.stack([self.init_input[key](item), blocks[-1].dstdata['s2v'][key]])
                feature = feature.permute(1, 0, 2)
                indomain_embed[key] = feature
            indomain_feature = self.pre_conv(indomain_embed)
            da_item_feature = indomain_feature[self.da_ntype]
            da_user_feature = indomain_feature[self.user_ntype]
            da_item_feature = self.da_item_dnn(da_item_feature)
            da_user_feature = self.da_user_dnn(da_user_feature)
        
            db_item_feature = indomain_feature[self.db_ntype]
            db_user_feature = indomain_feature[self.user_ntype]
            db_item_feature = self.db_item_dnn(db_item_feature)
            db_user_feature = self.db_user_dnn(db_user_feature)

            cross_embed = {}
            for key, item in blocks[0].srcdata[dgl.NID].items():
                feature = th.stack([self.init_input[key](item), blocks[0].srcdata['s2v'][key]])
                feature = feature.permute(1, 0, 2)
                cross_embed[key] = feature
            cross_feature = self.pre_conv(cross_embed)
            cross_feature = self.mix_rgcn(blocks, cross_feature)
        
            da_encoded = {self.da_ntype: self.da_item_cross_dnn(da_item_feature, cross_feature[self.da_ntype]), 
                    self.user_ntype: self.da_user_cross_dnn(da_user_feature, cross_feature[self.user_ntype])}
            db_encoded = {self.db_ntype: self.db_item_cross_dnn(db_item_feature, cross_feature[self.db_ntype]), 
                    self.user_ntype: self.db_user_cross_dnn(db_user_feature, cross_feature[self.user_ntype])}
            da_pos_graph = data['da_sub_graph'].to(self.device)
            db_pos_graph = data['db_sub_graph'].to(self.device)
            da_pos_score = self.pred(da_pos_graph, da_encoded)
            db_pos_score = self.pred(db_pos_graph, db_encoded)
            return da_pos_score, db_pos_score
            
    def block_to_gpu(self, blocks):
        for i in range(len(blocks)):
            blocks[i] = blocks[i].to(self.device)
        return blocks



        
    
    def info_nce_loss(self, features):
        labels = th.cat([th.arange(features.shape[0] // 2) for i in range(self.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = th.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = th.eye(labels.shape[0], dtype=th.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = th.cat([positives, negatives], dim=1)
        labels = th.zeros(logits.shape[0], dtype=th.long).to(self.device)

        logits = logits / 0.2
        loss = F.cross_entropy(logits, labels)
        return loss

    
    def compute_loss(self, pos_score, neg_score):
    # 间隔损失
        loss = 0
        for key in pos_score.keys():
            part_pos = pos_score[key]
            part_neg = neg_score[key]
            n_edges = part_pos.shape[0]
            loss+=(1 - part_pos.unsqueeze(1) + part_neg.view(n_edges, -1)).clamp(min=0).mean()
        return loss


class StochasticTwoLayerRGCN(nn.Module):
    def __init__(self, conv_dict1, conv_dict2):
        super().__init__()
        
        self.conv1 = HGraphConv(conv_dict1)
        self.conv2 = HGraphConv(conv_dict2)

    def forward(self, blocks, x):
        #import pdb;pdb.set_trace()
        x1 = self.conv1(blocks[0], x)
        x2 = self.conv2(blocks[1], x1)
        return x2

    
class ScorePredictor(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x

            #import pdb;pdb.set_trace()
            for etype in edge_subgraph.canonical_etypes:
                edge_subgraph.apply_edges(
                    fn.u_dot_v('x', 'x', 'score'), etype=etype)
            return edge_subgraph.edata['score']


class Feature_Pre_Conv(nn.Module):
    def __init__(self):
        super().__init__()
        conv1 = nn.Conv1d(2, 8, kernel_size = 1)
        conv2 = nn.Conv1d(8, 16, kernel_size = 1)
        conv3 = nn.Conv1d(16, 16, kernel_size = 1)
        conv4 = nn.Conv1d(16, 8, kernel_size = 1)
        conv5 = nn.Conv1d(8, 1, kernel_size = 1)
        self.pre_conv = nn.ModuleList([conv1, conv2, conv3, conv4, conv5])
    def forward(self, x):
        feature = {}
        for k, v in x.items():
            for layer in self.pre_conv:
                v = layer(v)
            #import pdb;pdb.set_trace()
            feature[k] = v.squeeze(1)
        return feature


class BaseBlock(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.fc = nn.Linear(channel, channel)
        self.bn = nn.BatchNorm1d(channel)
        self.relu = nn.ReLU()
    def forward(self, x):

        #import pdb;pdb.set_trace()
        x_out = self.relu(self.bn(self.fc(x)))
        return x_out


class DNN(nn.Module):
    def __init__(self, c_in, c_hidden, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.c_out = c_out

        self.fc1 = nn.Linear(c_in, c_hidden)
        self.block1 = BaseBlock(c_hidden)
        self.block2 = BaseBlock(c_hidden)
        self.block3 = BaseBlock(c_hidden)
        self.block4 = BaseBlock(c_hidden)

        self.fc4 = nn.Linear(c_hidden, c_out)
        
    def forward(self, x):
        x_down = self.fc1(x)
        x1 = x_down + self.block1(x_down)
        x2 = x1+ self.block2(x1)
        x3 = x2 + self.block3(x2)
        x4 = x3 + self.block4(x3)

        x_out = self.fc4(x4)
        return x_out
        
class CrossDNN(nn.Module):
    def __init__(self, c_in, c_hidden, c_out):
        super().__init__()
        self.c_in = c_in
        self.c_hidden = c_hidden
        self.c_out = c_out
        self.fc1 = nn.Linear(c_in, c_hidden)

        self.fc2 = nn.Linear(c_hidden, c_hidden)
        self.bn1 = nn.BatchNorm1d(c_hidden)
        self.relu1 = nn.ReLU()

        self.fc3 = nn.Linear(c_hidden, c_hidden)
        self.bn2 = nn.BatchNorm1d(c_hidden)
        self.relu2 = nn.ReLU()

        self.fc4 = nn.Linear(c_hidden, c_out)
        self.drop_out = nn.Dropout(0.5)
        
    def forward(self, x1, x2):
        x = th.cat((x1, x2), dim=1)
        assert x.shape[1] == self.c_in, 'c_in应该设置为两个向量concat起来的维度' 

        x_down = self.fc1(x)
        out = self.relu1(self.bn1(self.fc2(x_down)))
        out = self.relu2(self.bn2(self.fc3(out)))
        out +=x_down
        x_out = self.fc4(out)
        return x_out

    

class HGraphConv(nn.Module):
    def __init__(self, mods, aggregate='sum'):
        super(HGraphConv, self).__init__()
        self.mods = mods
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)
        if isinstance(aggregate, str):
            self.agg_fn = get_aggregate_fn(aggregate)
        else:
            self.agg_fn = aggregat
    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        """Forward computation

        Invoke the forward function with each module and aggregate their results.

        Parameters
        ----------
        g : DGLHeteroGraph
            Graph data.
        inputs : dict[str, Tensor] or pair of dict[str, Tensor]
            Input node features.
        mod_args : dict[str, tuple[any]], optional
            Extra positional arguments for the sub-modules.
        mod_kwargs : dict[str, dict[str, any]], optional
            Extra key-word arguments for the sub-modules.

        Returns
        -------
        dict[str, Tensor]
            Output representations for every types of nodes.
        """
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty : [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)
        return rsts


def _max_reduce_func(inputs, dim):
    return th.max(inputs, dim=dim)[0]

def _min_reduce_func(inputs, dim):
    return th.min(inputs, dim=dim)[0]

def _sum_reduce_func(inputs, dim):
    return th.sum(inputs, dim=dim)

def _mean_reduce_func(inputs, dim):
    return th.mean(inputs, dim=dim)

def _stack_agg_func(inputs, dsttype): # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    return th.stack(inputs, dim=1)

def _agg_func(inputs, dsttype, fn): # pylint: disable=unused-argument
    if len(inputs) == 0:
        return None
    stacked = th.stack(inputs, dim=0)
    return fn(stacked, dim=0)

def get_aggregate_fn(agg):
    """Internal function to get the aggregation function for node data
    generated from different relations.

    Parameters
    ----------
    agg : str
        Method for aggregating node features generated by different relations.
        Allowed values are 'sum', 'max', 'min', 'mean', 'stack'.

    Returns
    -------
    callable
        Aggregator function that takes a list of tensors to aggregate
        and returns one aggregated tensor.
    """
    if agg == 'sum':
        fn = _sum_reduce_func
    elif agg == 'max':
        fn = _max_reduce_func
    elif agg == 'min':
        fn = _min_reduce_func
    elif agg == 'mean':
        fn = _mean_reduce_func
    elif agg == 'stack':
        fn = None  # will not be called
    #else:
       # raise DGLError('Invalid cross type aggregator. Must be one of '
       #               '"sum", "max", "min", "mean" or "stack". But got "%s"' % agg)
    if agg == 'stack':
        return _stack_agg_func
    else:
        return partial(_agg_func, fn=fn)

