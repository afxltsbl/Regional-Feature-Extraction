import pandas as pd
import os


def getRelativeAdvantage(df, blog, result_dir):
    print('count blog_num、 feature_num、 village_num、 village_rank、 relative_advantage...')

    temp = df[['village', 'aspect']]
    temp = temp.drop_duplicates(subset=['village', 'aspect'], keep='first')
    for village in df['village'].unique().tolist():
        temp_data = blog[blog['village'] == village]
        for aspect in df[df['village'] == village]['aspect'].unique().tolist():
            temp.loc[(temp['village'] == village) & (temp['aspect'] == aspect), 'blog_num'] = temp_data[aspect].sum()
    df = pd.merge(df, temp, on=['village', 'aspect'], how='left')

    for village in df['village'].unique().tolist():
        for feature in df[df['village'] == village]['feature']:
            count = 0
            for blog_txt in blog[blog['village'] == village]['博文']:
                count += blog_txt.count(feature)
            df.loc[(df['village'] == village) & (df['feature'] == feature), 'feature_num'] = count

    feature_num = df[['village', 'feature', 'feature_num']]
    feature_num = feature_num.drop_duplicates(subset=['village', 'feature'], keep='first')
    feature_num[feature_num['feature'] == '非遗传承人']
    temp = feature_num.groupby(by=['feature'])['village'].count().reset_index(name='village_num')  # 每个特征出现在多少个村里，去重
    df = pd.merge(df, temp, on=['feature'], how='left')

    village_rank = feature_num.groupby(by=['feature']).apply(lambda x: x.sort_values(by='feature_num', ascending=False)).reset_index(drop=True)
    village_rank['village_rank'] = village_rank.groupby(by=['feature'])['feature_num'].rank(ascending=False, method='first')
    village_rank = village_rank[['village', 'feature', 'village_rank']]
    df = pd.merge(df, village_rank, on=['village', 'feature'], how='left')

    df['relative_advantage'] = 100 - 100 * (df['village_rank'] - 1) / df['village_num']

    df.to_csv(os.path.join(result_dir, '03_relative_advantage.csv'), index=False, encoding='utf_8_sig')


def getTimingEvaluation(df, result_dir):
    """计算村庄，历年各个方面特色的相对优势"""
    time_evaluation = df.groupby(by=['village', 'aspect', 'year']).sum().reset_index()
    time_evaluation.to_csv(os.path.join(result_dir, '06_villagetimeseries_evaluation.csv'), index=False, encoding='utf_8_sig')

    alltimeseries_evaluation = time_evaluation.groupby(by=['aspect', 'year']).sum().reset_index()
    alltimeseries_evaluation.to_csv(os.path.join(result_dir, '07_alltimeseries_evaluation.csv'), index=False, encoding='utf_8_sig')


def getRegionalCharacterRank(df, blog, result_dir):
    """v1.0 暂时不用了，后面优化删除"""
    print('开始计算村庄特色...')
    getVillageCharacter(df, blog, result_dir)

    print('开始计算区域特色排名...')
    getRegionalCharacterRank(df, blog, result_dir)


def getVillageCharacter(gcgj, blog, result_dir):
    """计算每个村，内部，各个方面的特色"""
    sum_gcgj = gcgj.groupby(by=['village', 'aspect']).sum().reset_index()
    mer = pd.merge(gcgj, sum_gcgj, on=['village', 'aspect'], how='left')
    if 'TF-IDW' not in mer.columns:
        mer.columns = ['village', 'aspect', 'feature', 'TF-IDW', 'TF-IDW_y']
    if mer['TF-IDW_y'].isnull().any():
        print('data merger have null!!!')
    mer['score'] = mer['TF-IDW'] / mer['TF-IDW_y']
    mer = mer.drop(['TF-IDW_y'], axis=1)
    vi_as = gcgj[['village', 'aspect']]
    vi_as.drop_duplicates(subset=['village', 'aspect'], keep='first', inplace=True)
    for village in blog['village'].unique().tolist():
        df = blog[blog['village'] == village]
        vi_as.loc[(vi_as['village'] == village) & (vi_as['aspect'] == '产业兴旺'), 'word_num'] = df['产业兴旺'].sum()
        vi_as.loc[(vi_as['village'] == village) & (vi_as['aspect'] == '生态宜居'), 'word_num'] = df['生态宜居'].sum()
        vi_as.loc[(vi_as['village'] == village) & (vi_as['aspect'] == '乡风文明'), 'word_num'] = df['乡风文明'].sum()
        vi_as.loc[(vi_as['village'] == village) & (vi_as['aspect'] == '治理有效'), 'word_num'] = df['治理有效'].sum()
        vi_as.loc[(vi_as['village'] == village) & (vi_as['aspect'] == '生活富裕'), 'word_num'] = df['生活富裕'].sum()
    aspect_num = pd.merge(mer, vi_as, on=['village', 'aspect'], how='left')

    aspect_num.to_csv(os.path.join(result_dir, '各村各级_村庄特色.csv'), index=False, encoding='utf-8_sig')


def getRegionalCharacterRank(gcgj, blog, result_dir):

    villages = gcgj['village'].unique().tolist()
    for village in villages:
        aspects = gcgj[gcgj['village'] == village]['aspect'].unique().tolist()
        for aspect in aspects:
            for feature in gcgj[(gcgj['village'] == village) & (gcgj['aspect'] == aspect)]['feature']:
                count = 0
                for content in blog[(blog['village'] == village) & (blog[aspect] == 1)]['博文']:
                    if feature in content:
                        count += 1
                gcgj.loc[(gcgj['village'] == village) & (gcgj['aspect'] == aspect) & (
                        gcgj['feature'] == feature), 'count'] = count
    sum_count = gcgj.groupby(by=['feature']).sum().reset_index()
    mer = pd.merge(gcgj, sum_count, on=['feature'], how='left')
    mer.columns = ['village', 'aspect', 'feature', 'TF-IDW', 'count', 'IDW_count', 'TF_count']
    mer['TF_rank'] = mer['count'] / mer['TF_count']
    col = mer[['village', 'feature', 'TF_rank']]
    col.drop_duplicates(subset=['village', 'feature'], keep='first', inplace=True)
    temp = col.groupby(by=['feature'])['village'].count()
    sum_feature_village = temp.to_frame().reset_index()
    col_count = pd.merge(col, sum_feature_village, on=['feature'], how='left')
    ranking = col_count.groupby(by=['feature']).apply(lambda x: x.sort_values(by='TF_rank', ascending=False)).reset_index(
        drop=True)
    ranking['village_rank'] = ranking.groupby(by=['feature']).cumcount() + 1
    ranking.columns = ['village', 'feature', 'TF_rank', 'village_num', 'village_rank']
    result = pd.merge(mer, ranking, on=['feature', 'village'], how='left')
    result = result.drop('TF_rank_y', axis=1)
    result.columns = ['village', 'aspect', 'feature', 'TF-IDW', 'count', 'IDW_count', 'TF_count', 'TF_rank',
                      'village_num', 'village_rank']

    result.to_csv(os.path.join(result_dir, '各村各级__区域特色排名.csv'), index=False, encoding='utf-8_sig')


def get_theme_advantage(data, query_data):
    """获取乡村内部主题优势"""
    villages = data['village'].unique().tolist()
    for village in villages:
        village_data = data[data['village'] == village]
        q_data = query_data[query_data['village'] == village]
        for tf in village_data['feature'].tolist():
            if tf in q_data['feature'].tolist():
                data.loc[(data['village'] == village) & (data['feature'] == tf), 'Q_TF_IDW'] = q_data[q_data['feature'] == tf]['TF-IDW'].values[0]
            else:
                data.loc[(data['village'] == village) & (data['feature'] == tf), 'Q_TF_IDW'] = 0
    return data


def get_year_advantage(data, query_data):
    """获取每年主题特征的优势"""
    years = data['year'].unique().tolist()
    for year in years:
        y_data = data[data['year'] == year]
        q_data = query_data[query_data['year'] == year]
        for tf in q_data['feature'].tolist():
            if tf in y_data['feature'].tolist():
                data.loc[(data['year'] == year) & (data['feature'] == tf), 'Q_TF_IDW'] = q_data[q_data['feature'] == tf]['TF-IDW'].values[0]
            else:
                data.loc[(data['year'] == year) & (data['feature'] == tf), 'Q_TF_IDW'] = 0
    return data








