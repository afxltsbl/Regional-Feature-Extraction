import random
import pandas as pd
import requests
from urllib import parse
import time
import xlwt
import re
import json


findquanwen = re.compile(r'<a.*?href="/status/.*?">(?P<qw>.*?)</a>', re.S)


def getJsonDict(url):
    time.sleep(random.randint(1, 2))
    cookielist = ["XXXXX",]
    user_Agentlist = ["XXXXX",
                      "XXXXX"]
    head = {
        "Cookie": cookielist[0],
        "User-Agent": user_Agentlist[random.randint(0, 1)]
    }

    response = requests.get(url, headers=head)
    print(response.text)
    if response.status_code == 200:
        if response.text:
            try:
                responsejson = response.json()
            except json.JSONDecodeError as e:
                print("JSON 解析错误:", str(e))
        else:
            print("Response content is empty")
    else:
        print("Request failed with status code:", response.status_code)

    response.close()
    if len(response.text) == 79:
        return 79
    else:
        return responsejson


def getOneBlogText(dict):
    text = dict["data"]["text"]
    return text


def getBlogInfo(json_dict):
    datalist = []
    cards = json_dict['data']['cards']
    for card in cards:
        have_content = True
        try:
            if card['desc'] == "Sorry, no results found.":
                print("We found the cause.")
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
            print("No search results for this village.")
            break
    return datalist


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
    print(savepath, 'save done!!!')


def villageSpider(village_url, name):
    content_list = []
    for page in range(1, 100):
        url = village_url + str(page)
        print("the link is：", url)
        html_json = getJsonDict(url)
        if html_json == 79:
            break
        if 'data' in html_json:
            datalist = getBlogInfo(html_json)
            content_list = content_list + datalist

    savepath = r'craw result/'+str(len(content_list))+'_'+str(name)+'.xls'
    saveData(savepath, content_list)
    print('over!!！')


def makeurl(villagenames):
    villages_url_list = []
    for name in villagenames:
        keyword = parse.quote(name)
        url = f"https://s.weibo.com/weibo?q={keyword}&page="
        villages_url_list.append(url)
    return villages_url_list


if __name__ == '__main__':
    """
    Setting the User-Agent in the request header to simulate a browser request
    The User-Agent can be obtained from the browser developer tools, or use an existing User-Agent string.
    
    
    Add a cookie to simulate a login state
    Cookie information can be obtained from the browser developer tools or extracted from the request after login.
    """

    print("starts time：" + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))

    df = pd.read_csv(r'villages name.csv')
    villagenames = df['villages  name'].tolist()

    villages_url = makeurl(villagenames)
    num = 0
    for village_url in villages_url:
        print("link："+village_url)
        villagename = villagenames[villages_url.index(village_url)]
        print(f"start crawling：    {villagename}")
        villageSpider(village_url, villagename)

