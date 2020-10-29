import uuid
import logging
import time
import os
import sys
import glob

class ArgsInit(object):
    def __init__(self):
        self.description = 'DeeperGCN'
        # dataset
        self.dataset = "ogbg-molhiv"
        self.num_workers = 0
        self.batch_size = 32
        self.feature = 'full'
        self.add_virtual_node= True #'store_true'
        # training & eval settings
        self.use_gpu= True#'store_true'
        self.device=0
        self.epochs=300
        self.lr=0.01
        self.dropout=0.5
        # model
        self.num_layers=3
        self.mlp_layers=1
        self.hidden_channels=256
        self.block='res+'
        self.conv='gen'
        self.gcn_aggr='max'
        self.norm='batch'
        self.num_tasks=1
        # learnable parameters
        self.t=1.0
        self.p=1.0
        self.learn_t=True #'store_true'
        self.learn_p=True #'store_true'
        # message norm
        self.msg_norm= True #'store_true'
        self.learn_msg_scale= True #'store_true'
        # encode edge in conv
        self.conv_encode_edge= True #'store_true'
        # graph pooling type
        self.graph_pooling='mean'
        # save model
        self.model_save_path='./model_ckpt'
        self.save='experiment 1'
        # load pre-trained model
        self.model_load_path='./ogbg_molhiv_pretrained_model.pth'

    def __init__(self, description='DeeperGCN', batch_size=32, epochs=10, learning_rate=0.01, dropout=0.5, num_layers=3, mlp_layers=1, hidden_channels=256, gcn_aggr='max', graph_pooling='mean', experiment_save="default"):
        self.description = description #'DeeperGCN'
        # dataset
        self.dataset = "ogbg-molhiv"
        self.num_workers = 0
        self.batch_size = batch_size #32
        self.feature = 'full'
        self.add_virtual_node= True #'store_true'
        # training & eval settings
        self.use_gpu= True#'store_true'
        self.device=0
        self.epochs= epochs#300
        self.lr= learning_rate #0.01
        self.dropout= dropout #0.5
        # model
        self.num_layers= num_layers#3
        self.mlp_layers= mlp_layers #1
        self.hidden_channels= hidden_channels #256
        self.block='res+'
        self.conv='gen'
        self.gcn_aggr= gcn_aggr #'max' #options: [max, mean, add, softmax_sg, softmax, power]
        self.norm='batch'
        self.num_tasks=1
        # learnable parameters
        self.t=1.0
        self.p=1.0
        self.learn_t=True #'store_true'
        self.learn_p=True #'store_true'
        # message norm
        self.msg_norm= True #'store_true'
        self.learn_msg_scale= True #'store_true'
        # encode edge in conv
        self.conv_encode_edge= True #'store_true'
        # graph pooling type
        self.graph_pooling='mean' #options: [mean, max, add]
        # save model
        self.model_save_path='./model_ckpt'
        self.save= experiment_save #'experiment 1'
        # load pre-trained model
        # self.model_load_path='./ogbg_molhiv_pretrained_model.pth'


    def save_exp(self):
        print("yep")
        self.args.save = '{}-B_{}-C_{}-L_{}-F_{}-DP_{}' \
                    '-GA_{}-T_{}-LT_{}-P_{}-LP_{}' \
                    '-MN_{}-LS_{}'.format(self.args.save, self.args.block, self.args.conv,
                                          self.args.num_layers, self.args.hidden_channels,
                                          self.args.dropout, self.args.gcn_aggr,
                                          self.args.t, self.args.learn_t, self.args.p, self.args.learn_p,
                                          self.args.msg_norm, self.args.learn_msg_scale)
        print("ok")
        self.args.save = 'log/{}-{}-{}'.format(self.args.save, time.strftime("%Y%m%d-%H%M%S"), str(uuid.uuid4()))
        self.args.model_save_path = os.path.join(self.args.save, self.args.model_save_path)
        create_exp_dir(self.args.save, scripts_to_save=glob.glob('*.py'))
        log_format = '%(asctime)s %(message)s'
        logging.basicConfig(stream=sys.stdout,
                            level=logging.INFO,
                            format=log_format,
                            datefmt='%m/%d %I:%M:%S %p')
        fh = logging.FileHandler(os.path.join(self.args.save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        return self.args