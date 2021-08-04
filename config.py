import os.path as osp
from easydict import EasyDict as edict


class Config:
    root_dir = osp.abspath(osp.dirname(__file__))
    output_dir = osp.join(root_dir, 'logs')
    checkpoint_dir = osp.join(output_dir, "checkpoint")
    DEVICE = "cuda"
    # solver
    SOLVER = edict()
    SOLVER.LR = 0.003
    SOLVER.CHECKPOINT_PERIOD = 5
    SOLVER.MAX_EPOCH = 100
    # MODEL
    MODEL = edict()
    MODEL.EMBEDDING_IN_DIM = 16
    MODEL.EMBEDDING_OUT_DIM = 16
    MODEL.CROSS_HEDDIEN_DIM = 64
    MODEL.DNN_HIDDEN_DIM = 32
    MODEL.PROJECT_DIM = 32
    MODEL.N_VIEWS = 2
    MODEL.GRAPH_HIDDEN_DIM = 32 
    MODEL.NUM_LAYER = 2
    MODEL.TOPK = 10 
    # DATA
    DATA = edict()
    DATA.NUM_DA_NTYPE = 9555 # MOVIE
    DATA.NUM_DB_NTYPE = 6777 # BOOK
    DATA.NUM_USER_NTYPE = 2107
    DATA.USER_NTYPE = 'user'
    DATA.DA_NTYPE = 'movie'
    DATA.DB_NTYPE = 'book'
    #DATA.DA_ETYPE = 'view'
    #DATA.DA_ETYPE_INV = 'view-by'
    DATA.DA_ETYPE = 'rate'
    DATA.DA_ETYPE_INV = 'rated-by'
    DATA.DB_ETYPE = 'rate'
    DATA.DB_ETYPE_INV = 'rated-by'
    #DATA.ETYPES = ['rate', 'rated-by', 'view', 'view-by']
    DATA.ETYPES = ['rate', 'rated-by']
    DATA.DATA_PATH = './data/data_s2v.pkl' 
    DATA.NEIGHBOR_RATE = 0.5
    #DATALOADER
    DATALOADER = edict()
    DATALOADER.BATCH_SIZE = 32
    
    DATALOADER.NEG_NUM = 10
    DATALOADER.POS_NUM = 32
    #这个参数我没有用






config = Config()
