import torch
import time


class Config():
    def __init__(self):

        self.random_seed = 124
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_path = './data/label.csv'
        self.col_select1, self.select1_num, self.col_select2, self.select2_num = '分类', 6, '正负', 1
        self.label_list = ['产业兴旺', '生态宜居', '乡风文明', '治理有效', '生活富裕']
        self.content_name = '博文'

        self.bert_dir = './models/bert-base-chinese'
        self.max_length = 200
        self.batch_size = 16

        self.loss = torch.nn.BCELoss()
        self.lr = 0.00001
        self.multi_label = True
        self.multi_class = False
        self.hidden_dim = 128
        self.lstm_layers = 3
        self.lstm_dropout = 0.5
        self.num_labels = 5

        self.num_epochs = 50
        self.print_stats = 10

        self.log_dir = './result/logs/' + time.strftime('%Y-%m-%d_%H.%M', time.localtime())
        self.train_metrics_path = './result/model_save/' + time.strftime('%Y-%m-%d-%H-%M', time.localtime()) + '_BertBiLstm_multi_labels_train_metrics.csv'
        self.model_save_path = './result/model_save/' + time.strftime('%Y-%m-%d-%H-%M', time.localtime()) + '_BertBiLstm_multi_labels_models.pt'

        self.multilabel_matrix_png = './result/picture/' + time.strftime('%Y-%m-%d_%H.%M', time.localtime()) + '_multilabel_matrix.png'

        self.checkpoint_path = './result/model_save/' + '2023-09-18-12-28_BertBiLstm_multi_labels_models.pt'  # test, predict load model


