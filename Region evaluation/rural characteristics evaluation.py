import time
import datetime
import pandas as pd
import config
from resources import termfrequency
from resources import regioncharacter
from resources import picture
from resources import function


if __name__ == '__main__':

    print('start run...', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    starttime = datetime.datetime.now()

    print('01 load config path...')
    config = config.Config()

    print('02 word frequency count...')
    df = function.load_data(config.data_path)
    wordTF = termfrequency.TermFrequency(config)
    print('02.0 count villages TF-IDW...')
    wordTF.getVillagesTF(df, config.topk_gc, config.result_dir)  # 分村-词频
    print('02.1 count villages, aspects, TF-IDW...')
    wordTF.getVillagesAspectsTF(df, config.topk_gcgj, config.result_dir)  # 分村分主题-维度
    print('02.2 count time, villages, aspects, TF-IDW...')
    wordTF.getTimeVillagesAspectsTF(df, config.topk_time, config.result_dir)  # 分村分主题分时间-维度
    print('02.3 count year TF-IDW...')
    wordTF.getYearTF(df, config.topk_year, config.result_dir)
    print('02.4 count year aspect TF-IDW')
    wordTF.getYearAspectTF(df, config.topk_yearaspect, config.result_dir)

    print('03 feature advantage count...')
    data = pd.read_csv(r'E:\05_RuralPortrait\04_experiment\05_rural_characteristics\results\2023-02-21_09.41\01_各村各级_词频.csv')
    blog = pd.read_excel(config.data_path)
    result_dir = r'E:\05_RuralPortrait\04_experiment\05_rural_characteristics\results\2023-02-21_09.41'
    regioncharacter.getRelativeAdvantage(data, blog, result_dir)

    print('04 radar picture...')
    df = pd.read_csv(r'E:\05_RuralPortrait\04_experiment\05_rural_characteristics\results\2023-02-21_09.41\01_各村各级_词频.csv')
    query_df = pd.read_csv(r'E:\05_RuralPortrait\04_experiment\05_rural_characteristics\results\2023-02-21_09.41\00_各村_词频.csv')
    result_dir = r'E:\05_RuralPortrait\04_experiment\05_rural_characteristics\results\2023-02-21_09.41\雷达图'
    df_pre = pd.read_csv(r'E:\05_RuralPortrait\04_experiment\05_rural_characteristics\results\2023-02-21_09.41\04_乡村主题排名.csv')
    picture.plot_leidatu(df_pre, result_dir)

    print('05 timing evoluation...')
    df = pd.read_csv(r'E:\05_RuralPortrait\04_experiment\05_rural_characteristics\results\2023-02-21_09.41\02_各年各村各级_词频.csv')
    query_df = pd.read_csv(r'E:\05_RuralPortrait\04_experiment\05_rural_characteristics\results\2023-02-21_09.41\03_各年_词频.csv')
    result_dir = r'E:\05_RuralPortrait\04_experiment\05_rural_characteristics\results\2023-02-21_09.41'
    df_pre = regioncharacter.get_year_advantage(df, query_df)
    regioncharacter.getTimingEvaluation(df_pre, result_dir)

    print('end run...', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '程序运行时间：', (datetime.datetime.now() - starttime).seconds, 's')
