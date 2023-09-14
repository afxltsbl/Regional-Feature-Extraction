import re
import jieba
import pandas as pd
from datetime import datetime


def load_data(path):
    """读取数据"""
    data = pd.read_excel(path)
    data['发布时间'] = data['发布时间'].apply(lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S +0800 %Y"))
    data['Year'] = data['发布时间'].dt.year
    data['Month'] = data['发布时间'].dt.month
    data = data[['序号', 'village', '产业兴旺', '生态宜居', '乡风文明', '治理有效', '生活富裕', '博文', '发布时间', 'Year', 'Month']]
    return data


def read_txt_to_list(path):
    """读取txt文件，返回list"""
    wordlist = []
    try:
        file = open(path, 'r', encoding='utf8')
        for wordinline in file.readlines():
            wordlist.append(wordinline.strip())
    except Exception as e:
        print(e)

    return wordlist


def jiebaCut(contentsList, userDictPath):
    """使用结巴分词,逐行分词"""
    contents = []
    jieba.load_userdict(userDictPath)
    for content in contentsList:
        cutWord = jieba.lcut(content)
        if len(cutWord) > 1 and cutWord != '\r\n':
            contents.append(cutWord)
    f = open(r"E:\05_RuralPortrait\04_experiment\05_rural_characteristics\results\temp\k.txt", "a", encoding='utf-8')
    for line in contents:
        str = ' '.join(line)
        f.write(str + '\n')
    f.close()
    return contents


def delList_StopWords(reList1, stopWords):
    contentClean = []
    for line in reList1:
        lineClear = []
        for word in line:
            if word in stopWords:
                continue
            elif word == ' ' or bool(re.search(r'd', word)):
                continue
            else:
                lineClear.append(word)
        contentClean.append(lineClear)
    return contentClean


def delete_district(word):
    temp_return_word = ''
    tword = word.replace(' ', '')  # 去空格
    result = re.findall(
        r'[\u4e00-\u9fa5]{1,4}省|[\u4e00-\u9fa5]{1,4}市|[\u4e00-\u9fa5]{1,4}区|[\u4e00-\u9fa5]{1,4}县|[\u4e00-\u9fa5]{1,4}乡|[\u4e00-\u9fa5]{1,4}镇|[\u4e00-\u9fa5]{1,9}州|[\u4e00-\u9fa5]{0,4}街道|[\u4e00-\u9fa5]{1,4}村|[\u4e00-\u9fa5]{1,4}旗|[\u4e00-\u9fa5]{1,4}屯|[\u4e00-\u9fa5]{1,4}庄',tword)  # 分别匹配模式

    if len(result) > 0:
        pass
    else:
        temp_return_word = word

    return temp_return_word


def delete_gov(word):
    temp_return_word = ''
    tword = word.replace(' ', '')  # 去空格
    result = re.findall(r'[\u4e00-\u9fa5]{1,4}部|[\u4e00-\u9fa5]{1,6}局|[\u4e00-\u9fa5]{1,4}工会|[\u4e00-\u9fa5]{0,5}学校|[\u4e00-\u9fa5]{1,6}路|[\u4e00-\u9fa5]{0,4}机关|[\u4e00-\u9fa5]{0,4}公署|[\u4e00-\u9fa5]{0,4}政府|[\u4e00-\u9fa5]{0,4}支队|[\u4e00-\u9fa5]{1,3}电|[\u4e00-\u9fa5]{0,4}处|[\u4e00-\u9fa5]{0,4}部门|[\u4e00-\u9fa5]{0,6}集团|[\u4e00-\u9fa5]{1,4}厅|[\u4e00-\u9fa5]{0,4}产品|[\u4e00-\u9fa5]{0,4}设施|[\u4e00-\u9fa5]{0,4}功能|[\u4e00-\u9fa5]{0,6}公司|[\u4e00-\u9fa5]{1,6}会|[\u4e00-\u9fa5]{1,6}院|[\u4e00-\u9fa5]{1,6}银行|[\u4e00-\u9fa5]{0,4}委|[\u4e00-\u9fa5]{0,4}室|[\u4e00-\u9fa5]{0,4}队|[\u4e00-\u9fa5]{0,5}业|[\u4e00-\u9fa5]{1,5}所|[\u4e00-\u9fa5]{1,5}方|[\u4e00-\u9fa5]{1,5}社|[\u4e00-\u9fa5]{0,4}合同|[\u4e00-\u9fa5]{0,4}保险|[\u4e00-\u9fa5]{0,4}中心|[\u4e00-\u9fa5]{0,4}管理',tword)  # 分别匹配模式

    if len(result) > 0:
        pass
    else:
        temp_return_word = word

    return temp_return_word


def delete_usual(word):
    temp_return_word = ''
    tword = word.replace(' ', '')
    result = re.findall(r'[\u4e00-\u9fa5]{0,4}资源|[\u4e00-\u9fa5]{0,6}中学',tword)  # 分别匹配模式

    if len(result) > 0:
        pass
    else:
        temp_return_word = word

    return temp_return_word


def delete_common_placename(word, placenamelist):
    temp_return_word = ''
    if placenamelist:
        if word in placenamelist:
            pass
        else:
            temp_return_word = word
    else:
        print('需要载入常用省市名称词表！')