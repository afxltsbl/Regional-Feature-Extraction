import pandas as pd
import os


def getRelativeAdvantage(df, blog):
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

    return df


def getTimingEvaluation(df):
    time_evaluation = df.groupby(by=['village', 'aspect', 'year']).sum().reset_index()
    alltimeseries_evaluation = time_evaluation.groupby(by=['aspect', 'year']).sum().reset_index()

    return time_evaluation, alltimeseries_evaluation


def get_theme_advantage(data, query_data):
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








