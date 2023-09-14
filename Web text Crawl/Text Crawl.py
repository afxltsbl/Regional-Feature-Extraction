import random
import pandas as pd
import requests
from urllib import parse
import time
import xlwt
import re


findquanwen = re.compile(r'<a.*?href="/status/.*?">(?P<qw>.*?)</a>', re.S)


def getJsonDict(url):
    time.sleep(random.randint(1, 2))
    cookielist = ["SCF=AhqBrxtYYN_AVb1YzoFVHdYK95cYolvkPzrU2RmmppyYLMxFIIctBtW5MTQIDl3nZr7RQet9Jt3uAZmcd5EL_GM.; SSOLoginState=1677154218; ALF=1679746218; _T_WM=70881870295; XSRF-TOKEN=4621d2; WEIBOCN_FROM=1110006030; MLOGIN=0; mweibo_short_token=ee7674c95b; M_WEIBOCN_PARAMS=lfid=102803&luicode=20000174&uicode=20000174",]
    user_Agentlist = ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36 Edg/94.0.992.50","Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.102 Safari/537.36 Edg/98.0.1108.56"]
    head = {
        "Cookie": cookielist[0],
        "User-Agent":user_Agentlist[random.randint(0,1)]
    }

    response = requests.get(url, headers=head)
    responsejson = response.json()
    response.close()
    if len(response.text) == 79:
        return 79
    else:
        return responsejson


def getOneBlogText(dict):
    text = dict["data"]["text"]
    # print(text)
    return text


def getBlogInfo(json_dict):
    datalist = []
    cards = json_dict['data']['cards']
    for card in cards:
        have_content = True
        try:
            if card['desc'] == "抱歉，未找到相关结果。":
                print("找到原因了")
                have_content = False
        except:
            have_content = True

        if have_content:
            try:
                if isinstance(card["card_group"], list):
                    if len(card['card_group']) == 1:
                        card_group = card['card_group'][0]
                        if card_group['card_type'] == 9:
                            mblog = card_group.get('mblog')
                            userid = mblog.get('user').get('id')
                            screen_name = mblog.get('user').get('screen_name')
                            source = mblog.get('source')
                            scheme = card_group.get('scheme')
                            text = mblog.get('text')
                            a = findquanwen.search(text)
                            if a:
                                findurl = re.compile(r'mblogid=(?P<mblogid>.*?)&')
                                id = findurl.search(scheme).group("mblogid")
                                url = "https://m.weibo.cn/statuses/show?id=" + str(id) +"&display=0&retcode=6102"
                                result = getJsonDict(url)
                                text = getOneBlogText(result)
                            else:
                                pass
                            # 处理文本中的标签
                            htmllable = re.compile('<.*?>', re.S)
                            text = htmllable.sub(" ", text)
                            created_at = mblog.get('created_at')
                            attitudes_count = mblog.get('attitudes_count')
                            reposts_count = mblog.get('reposts_count')
                            comments_count = mblog.get('comments_count')

                            data = [screen_name, userid, text, scheme, created_at, source, attitudes_count, reposts_count,
                                    comments_count]
                            datalist.append(data)
                            print([screen_name, userid, text, scheme, created_at, source, attitudes_count, reposts_count,
                                   comments_count])
                        else:
                            pass
                    elif len(card['card_group']) == 2:
                        pass
                    else:
                        pass
            except:
                pass
        else:
            print("此村庄无搜索结果")
            break
    return datalist


# 保存数据至excel
def saveData(savepath, data_list):
    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    sheet = book.add_sheet('微博爬虫', cell_overwrite_ok=True)
    col = ('博主', '博主id', '博文', '博文独立网址', '发布时间', '发布终端', '转发数', '评论数', '点赞')
    for i in range(0, 9):
        sheet.write(0, i, col[i])
    for i in range(0, len(data_list)):
        print('第%d条' % (i + 1))
        data = data_list[i]
        for j in range(0, 9):
            sheet.write(i + 1, j, data[j])
    book.save(savepath)
    print(savepath, '保存完毕')


def makeurl(villagenames):
    villages_url_list = []
    for name in villagenames:
        keyword = parse.quote(name)
        url = f"https://s.weibo.com/weibo?q={keyword}&page=2"
        villages_url_list.append(url)
    return villages_url_list


# 调用爬取链接,获得微博信息,并保存至excel
def villageSpider(village_url,name):
    content_list = []
    for page in range(1, 100):
        url = village_url + str(page)
        print("此时的链接是：", url)
        html_json = getJsonDict(url)
        if html_json == 79:
            break
        if 'data' in html_json:
            datalist = getBlogInfo(html_json)
            content_list = content_list + datalist

    savepath = r'content/'+str(len(content_list))+'_'+str(name)+'.xls'
    saveData(savepath, content_list)
    print('over!!！')


if __name__ == '__main__':
    print("此时爬虫开始的时间是：" + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    df = pd.read_excel(r'E:\05_RuralPortrait\04_experiment\03_Select122\100村名列表.xlsx')
    villagenames = df['100村名'].tolist()

    villages_url = makeurl(villagenames)
    num = 0
    for village_url in villages_url:
        print("此时是："+village_url)
        # villages_url.index(village_url)
        villagename = villagenames[villages_url.index(village_url)]
        print(f"此时开始爬取：    {villagename}")
        villageSpider(village_url, villagename)

