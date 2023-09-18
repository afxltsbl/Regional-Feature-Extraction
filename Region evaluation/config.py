import os
import time


class Config:
    def __init__(self):
        root_dir = os.getcwd()

        self.topk = 100  # 取词频前多少的词
        self.topk_gcgj = 20
        self.topk_time = 5
        self.topk_gc = 100
        self.topk_year = 10000
        self.topk_yearaspect = 1000
        self.aspects = ['产业兴旺', '生态宜居', '乡风文明', '治理有效', '生活富裕']

        self.data_dir = os.path.join(root_dir, 'data', 'texts')
        self.data_path = os.path.join(self.data_dir, '分类结果合并最终结果.xlsx')

        self.result_time = time.strftime('%Y-%m-%d_%H.%M', time.localtime())
        self.result_dir = os.path.join(root_dir, 'results', self.result_time)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # 用户自定义词典
        self.user_dict = os.path.join(root_dir, 'data', 'user_settings', 'custom.txt')

        # 分词停用词词表
        self.stopwords = os.path.join(root_dir, 'data', 'user_settings', 'stopword.txt')

        # 乡村振兴停用词
        self.rural_revitalization_stopword = os.path.join(root_dir, 'data', 'user_settings', '乡村振兴停用词.txt')

        # 常用省市名称——地名
        self.common_province_city = os.path.join(root_dir, 'data', 'user_settings',  '省市区县镇.txt')

        # 常用城市名、行政区名、地名
        self.common_place = os.path.join(root_dir, 'data', 'user_settings', '地名列表.txt')

        # 常用村名
        self.common_village_name = os.path.join(root_dir, 'data', 'user_settings', '地区村名.txt')  # 常用地区村名称——地名  干什么？特例：年画村，年画

