import torch
import time


class Config():
    """配置参数"""
    def __init__(self):

        self.random_seed = 123
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.data_path = r'.\data\merge_data_clean_label_tocsv.csv'
        self.col_select1, self.select1_num, self.col_select2, self.select2_num = '分类', 6, '正负', 1   # 选择列和条件，不等于6，不等于1
        self.label_list = ['产业兴旺', '生态宜居', '乡风文明', '治理有效', '生活富裕']
        self.content_name = '博文'

        self.bert_dir = r'.\models\bert-base-chinese'
        self.max_length = 200  # 文本截断的最大长度 356
        self.batch_size = 16

        self.loss = torch.nn.BCELoss()
        self.lr = 0.00001
        self.multi_label = True
        self.multi_class = False
        self.hidden_dim = 128  # 256 bert
        self.lstm_layers = 3
        self.lstm_dropout = 0.5
        self.num_labels = 5

        self.num_epochs = 50
        self.print_stats = 10

        # 改成绝对路径
        self.log_dir = r'E:\05_RuralPortrait\04_experiment\04_experiment\logs' + r'\\' + time.strftime('%Y-%m-%d_%H.%M', time.localtime())  # 日志文件的保存目录
        self.train_metrics_path = r'E:\05_RuralPortrait\04_experiment\04_experiment\model_save' + r'\\' + time.strftime('%Y-%m-%d-%H-%M', time.localtime()) + '_BertBiLstm_multi_labels_train_metrics.csv'
        self.model_save_path = r'E:\05_RuralPortrait\04_experiment\04_experiment\model_save' + r'\\' + time.strftime('%Y-%m-%d-%H-%M', time.localtime()) + '_BertBiLstm_multi_labels_models.pt'

        # test matrix.png save path
        self.multilabel_matrix_png = r'E:\05_RuralPortrait\04_experiment\04_experiment\picture' + r'\\' + time.strftime('%Y-%m-%d_%H.%M', time.localtime()) + '_multilabel_matrix.png'

        # test or predict add model
        self.checkpoint_path = r'E:\05_RuralPortrait\04_experiment\04_experiment\model_save\\' + '2023-02-19-20-31_Bert_models.pt'  # test, predict load model


