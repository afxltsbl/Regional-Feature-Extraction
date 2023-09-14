import os
import pandas as pd
import jieba
import jieba.analyse as analyse
from resources import function as Fc


class TermFrequency:
    def __init__(self, config):
        self.config = config

        self.user_dict = Fc.read_txt_to_list(self.config.user_dict)

        self.stopwords = Fc.read_txt_to_list(self.config.stopwords)

        self.rural_revitalization_stopword = Fc.read_txt_to_list(self.config.rural_revitalization_stopword)

        self.common_province_city = Fc.read_txt_to_list(self.config.common_province_city)

        self.common_village_name = Fc.read_txt_to_list(self.config.common_village_name)

        self.common_place = Fc.read_txt_to_list(self.config.common_place)

    def getTimeVillagesAspectsTF(self, data, topk, save_dir):
        villages = data['village'].unique().tolist()
        result = pd.DataFrame(columns=['year', 'village', 'aspect', 'feature', 'TF-IDW'])
        for village_name in villages:
            for year in data[data['village'] == village_name]['Year'].unique().tolist():
                village_time_TF = self.getAspectsTF(data[(data['village'] == village_name) & (data['Year'] == year)], topk)
                village_time_TF['village'] = village_name
                village_time_TF['year'] = year
                result = pd.concat([result, village_time_TF])
        result.to_csv(os.path.join(save_dir, '02_各年各村各级_词频.csv'), index=False, encoding='utf-8_sig')

    def getVillagesAspectsTF(self, data, topk, save_dir):
        villages = data['village'].unique().tolist()
        result = pd.DataFrame(columns=['village', 'aspect', 'feature', 'TF-IDW'])
        for village_name in villages:
            village_TF = self.getAspectsTF(data[data['village'] == village_name], topk)
            village_TF['village'] = village_name
            result = pd.concat([result, village_TF])
        result.to_csv(os.path.join(save_dir, '01_各村各级_词频.csv'), index=False, encoding='utf-8_sig')

    def getVillagesTF(self, data, topk, save_dir):
        villages = data['village'].unique().tolist()
        result = pd.DataFrame(columns=['village', 'feature', 'TF-IDW'])
        for village_name in villages:
            village_TF = self.getListsTF(data[data['village'] == village_name]['博文'].tolist(), topk)
            village_TF = pd.DataFrame(village_TF, columns=['feature', 'TF-IDW'])
            village_TF['village'] = village_name
            result = pd.concat([result, village_TF])
        result.to_csv(os.path.join(save_dir, '00_各村_词频.csv'), index=False, encoding='utf-8_sig')

    def getYearAspectTF(self, data, topk, save_dir):
        years = data['Year'].unique().tolist()
        result = pd.DataFrame(columns=['year', 'aspect', 'feature', 'TF-IDW'])
        for year in years:
            df1 = data[data['Year'] == year]
            year_aspect = self.getAspectsTF(df1, topk)
            year_aspect['year'] = year
            result = pd.concat([result, year_aspect])
        result.to_csv(os.path.join(save_dir, '03_各年各级_词频.csv'), index=False, encoding='utf-8_sig')

    def getAspectsTF(self, data, topk):
        result = pd.DataFrame(columns=['aspect', 'feature', 'TF-IDW'])
        for aspect in self.config.aspects:
            df = data[data[aspect] == 1]  # select aspect data
            aspect_TF = self.getListsTF(df['博文'].tolist(), topk)
            aspect_df = pd.DataFrame(aspect_TF, columns=['feature', 'TF-IDW'])
            aspect_df['aspect'] = aspect
            result = pd.concat([result, aspect_df])
        return result

    def getListsTF(self, Lists, topk):
        jieba_cuts = Fc.jiebaCut(Lists, self.user_dict)
        del_stopwords = self.delStopwords(jieba_cuts)
        tf = self.countTF(del_stopwords, topk)
        return tf

    def getYearTF(self, df, topk, save_dir):
        years = df['Year'].unique().tolist()
        result = pd.DataFrame(columns=['year', 'feature', 'TF-IDW'])
        for year in years:
            temp = df[df['Year'] == year]
            yearTF = self.getListsTF(temp['博文'].tolist(), topk)
            yearTF = pd.DataFrame(yearTF, columns=['feature', 'TF-IDW'])
            yearTF['year'] = year
            result = pd.concat([result, yearTF])
        result.to_csv(os.path.join(save_dir, '03_各年_词频.csv'), index=False, encoding='utf-8_sig')

    def delStopwords(self, contents):
        contentsLists = Fc.delList_StopWords(contents, self.stopwords)
        contentsLists = Fc.delList_StopWords(contentsLists, self.common_province_city)
        contentsLists = Fc.delList_StopWords(contentsLists, self.common_village_name)
        contentsLists = Fc.delList_StopWords(contentsLists, self.common_place)
        contentsLists = Fc.delList_StopWords(contentsLists, self.rural_revitalization_stopword)
        return contentsLists

    def countTF(self, contents, topk):
        join_contents = ''
        for line in contents:
            join_line = ' '.join(line)
            join_contents += join_line
        jieba.load_userdict(self.user_dict)

        keyWords = jieba.analyse.extract_tags(join_contents, topK=topk, withWeight=True, )
        return keyWords


