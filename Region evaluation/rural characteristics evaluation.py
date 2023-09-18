import time
import datetime
import pandas as pd
import config
from resources import termfrequency
from resources import regioncharacter
from resources import picture
from resources import function
import os


if __name__ == '__main__':

    print('start run...', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    starttime = datetime.datetime.now()

    print('01 load config path...')
    config = config.Config()

    print('02 word frequency count...')
    df = function.load_data(config.data_path)
    wordTF = termfrequency.TermFrequency(config)

    print('02.1 count villages TF-IDW...')
    village_TF = wordTF.getVillagesTF(df, config.topk_gc)  # 分村-词频
    village_TF.to_csv(os.path.join(config.result_dir, '00_各村_词频.csv'), index=False, encoding='utf-8_sig')

    print('02.2 count villages, aspects, TF-IDW...')
    village_aspect_TF = wordTF.getVillagesAspectsTF(df, config.topk_gcgj)  # 分村分主题-维度
    village_aspect_TF.to_csv(os.path.join(config.result_dir, '01_各村各级_词频.csv'), index=False, encoding='utf-8_sig')

    print('02.3 count time, villages, aspects, TF-IDW...')
    year_village_tf = wordTF.getTimeVillagesAspectsTF(df, config.topk_time)  # 分村分主题分时间-维度
    year_village_tf.to_csv(os.path.join(config.result_dir, '02_各年各村各级_词频.csv'), index=False, encoding='utf-8_sig')

    print('02.4 count year TF-IDW...')
    year_villages = wordTF.getYearTF(df, config.topk_year)
    year_villages.to_csv(os.path.join(config.result_dir, '03_各年_词频.csv'), index=False, encoding='utf-8_sig')

    print('02.5 count year aspect TF-IDW')
    year_villages = wordTF.getYearAspectTF(df, config.topk_yearaspect)
    year_villages.to_csv(os.path.join(config.result_dir, '03_各年各级_词频.csv'), index=False, encoding='utf-8_sig')

    print('03 feature advantage count...')
    data = pd.read_csv(os.path.join(config.result_dir, '01_各村各级_词频.csv'))
    blog = pd.read_excel(config.data_path)
    relative_advantage = regioncharacter.getRelativeAdvantage(data, blog)
    relative_advantage.to_csv(os.path.join(config.result_dir, '03_relative_advantage.csv'), index=False, encoding='utf_8_sig')

    # print('04 radar picture...')
    # df = pd.read_csv(os.path.join(config.result_dir, '01_各村各级_词频.csv'))
    # query_df = pd.read_csv(os.path.join(config.result_dir, '00_各村_词频.csv'))
    # df_pre = regioncharacter.get_theme_advantage(df, query_df)
    # picture.plot_leidatu(df_pre, os.path.join(config.result_dir, '雷达图'))

    print('05 timing evoluation...')
    df = pd.read_csv(os.path.join(config.result_dir, '02_各年各村各级_词频.csv'))
    query_df = pd.read_csv(os.path.join(config.result_dir, '03_各年_词频.csv'))

    df_pre = regioncharacter.get_year_advantage(df, query_df)
    time_evaluation, alltimeseries_evaluation = regioncharacter.getTimingEvaluation(df_pre)
    time_evaluation.to_csv(os.path.join(config.result_dir, '06_villagetimeseries_evaluation.csv'), index=False, encoding='utf_8_sig')
    alltimeseries_evaluation.to_csv(os.path.join(config.result_dir, '07_alltimeseries_evaluation.csv'), index=False, encoding='utf_8_sig')

    print('end run...', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '程序运行时间：', (datetime.datetime.now() - starttime).seconds, 's')
