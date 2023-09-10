
# https://medium.com/@keigi1203/python-%E9%80%B2%E9%9A%8E%E7%88%AC%E8%9F%B2%E6%8A%80%E5%B7%A7-selenium-chrome-d4ae4979a874
from typing import Dict, List, Union
import re
from easydict import EasyDict as edict
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import sys
sys.path.append('./')
from crawler.XDA.XDA.XDA.models import ThreadAttr, PostAttr


webpath = 'https://forum.xda-developers.com/t/warning-read-this-before-you-upgrade-to-android-13-stable.4482089/'
options = Options()
options.add_argument("start-maximized")
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
driver.get(webpath)
driver.implicitly_wait(3) # seconds

'''
1. to Scrapy
2. record reactions (simple count / count with usr info?)
3. refactor
4. clean text in 'bbWrapper'
5. page-changer

'''
cssD: Dict[str, Dict] = {
        'main': {
            'all_posts': 'article[id*="js-post"]',
        },
        'thread': {
            ThreadAttr.title: 'h1[class="p-title-value"]::text',
            ThreadAttr.date_time: 'time[class="u-dt"]::attr(datetime)',
        },
        'post': {
            PostAttr.date_time: 'time[class="u-dt"]::attr(datetime)',
            PostAttr.author: '::attr(data-author)',
            PostAttr.user_title: 'h5[class^="user"]::text',
            PostAttr.post_id: '::attr(id)',
            PostAttr.text: 'div[class="bbWrapper"]::text',
            PostAttr.reaction: 'a[class*="--reaction0"]::text',
        },
        'section': {
            'main': 'div.message-cell.message-cell--main',
            'user': 'div.message-cell.message-cell--user',
        }
    }
XpathD: Dict[str, Dict] = {
    'forum': {
        'name':
        '//*[@id="top"]/div[3]/div[2]/div/div[2]/div[1]/div[2]/div[1]/div/ul/li[4]/a/span',
        'page_num':
        '/html/body/div[2]/div/div[3]/div[2]/div/div[2]/div[1]/div[2]/div[2]/div[4]/div[1]/nav/div[1]/ul/li[5]/a/text()',
    },
    'thread': {
        'category':
        '//*[@id="top"]/div[3]/div[2]/div/div[2]/div[1]/div[1]/div/div/div[1]/h1/span[1]/text()',
        'curr_page':
        '/html/body/div[2]/div/div[3]/div[2]/div/div[2]/div[1]/div[2]/div[2]/div[4]/div[1]/nav/div[1]/ul/li[2]/a/text()',
        'page_num':
        '/html/body/div[2]/div/div[3]/div[2]/div/div[2]/div[1]/div[2]/div[3]/div[1]/div[1]/nav/div[1]/ul/li[6]/a/text()'
    },
    'post':{
        'floor': '//*[@id="js-post-{post_id}"]/div/div[2]/div/header/ul[2]/li[2]/a',
        'react_url': '//*[@id="js-post-{post_id}"]/div/div[2]/div/header/ul[2]/li[2]/a'
    }
}

XpathD = edict(XpathD)
CssD = edict(cssD)
def get_integer(text: Union[str, List]) -> int:
    if isinstance(text, list):
        try:
            text = text[0]
            number = int(re.search(r'\d+', text).group())
        except:
            number = 1
    return number


comments = driver.find_elements(By.CSS_SELECTOR, "article[id*='js-post']")
for idx, comment in enumerate(comments[:2]):
    print(f'{idx}:')
    author = comment.get_attribute('data-author')
    content = comment.get_attribute('data-content')
    id_string = comment.get_attribute('id')
    id = re.search(r'\d+', id_string).group()
    print(f'author: {author}')
    print(f'content: {content}')
    print(f'id: {id}')


    # .//ancestor::tr
    textElem = comment.find_element(By.CLASS_NAME, "bbWrapper")
    # print(textElem.text)
    # get user info
    usrElem = comment.find_element(By.CSS_SELECTOR, "div.message-cell.message-cell--user")
    usrTitle = usrElem.find_element(By.CSS_SELECTOR, "h5[class^='user']").text
    print(f'User title: {usrTitle}')
    # get time
    mainElem = comment.find_element(By.CSS_SELECTOR, "div.message-cell.message-cell--main")
    time = mainElem.find_element(By.CSS_SELECTOR, "time[class='u-dt']")
    datetime_str = time.get_attribute('datetime')
    # get floor
    floor_path = XpathD.post.floor.format(post_id=id)
    floor = driver.find_element(By.XPATH, floor_path)
    print(f'floor: {floor.text}')
    # floor = mainElem.find_element(By.CSS_SELECTOR, "a[class='u-concealed']")
    print(f'date time: {datetime_str}')
    print()

    # https://forum.xda-developers.com/t/warning-read-this-before-you-upgrade-to-android-13-stable.4482089/page-2-

driver.close()


