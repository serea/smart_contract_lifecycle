# coding:utf-8

import logging
from bs4 import BeautifulSoup
import ssl
import os, sys, re
import requests
import time
import pandas as pd
from progressbar import *
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException

class StateofthedappsCrawler:
    def __init__(self):
        context = ssl._create_unverified_context()
        requests.adapters.DEFAULT_RETRIES = 500
        self.headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'}
        self.proxy = {'http': 'socks5://127.0.0.1:1086','https':'socks5://127.0.0.1:1086'}
        self.allTransList = [] # 存储这个user的所有交易信息，dict格式
        self.errorList = []

        # 使用headless chrome来加载异步js的页面
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        self.driver = webdriver.Chrome(chrome_options=chrome_options)


    def IndexCrawler(self):
        """crawl dapp collection website by index
        """

        logging.debug('getTransactionAddressByUserCrawler - BEGIN')
        index_url = 'https://www.stateofthedapps.com/rankings/platform/ethereum' # 目录页
        base_url = "https://www.stateofthedapps.com"  # dapp详情页
        
        beginPageNum = 1  # 页面数字
        lastPageNum = self.getPageNum(index_url)  # 获取页数（便于print进度）
        print("共有 %d 页需要进行爬取..." % lastPageNum)
        for pageNum in range(beginPageNum, lastPageNum+1):
            print(">>> 开始爬取第%d页：" % pageNum)
            # 对所有的index进行循环操作
            res = None
            s = requests.session()
            s.keep_alive = False

            # 如果没有get到，就sleep 10，然后再爬，共尝试30次
            for countdown in range(1,30):
                try:
                    res = s.get(index_url+'?page='+str(pageNum), verify=False, timeout=100)
                    break
                except:
                    time.sleep(10)
                    continue
            # 对爬到的这个目录页面进行处理
            if res is not None:
                soup = BeautifulSoup(res.content,'lxml')
                table_body = soup.find('div', class_="table-body")
                progress = ProgressBar() # 每个循环前都要初始化进度条
                for child in progress(table_body.find_all('a', class_="icon-link")):
                    dapp_url = base_url + child['href']
                    self.DappCrawler(dapp_url)
                
            else:
                print("Request Failed: "+index_url+'?page='+str(pageNum))
                self.errorList.append(index_url+'?page='+str(pageNum))
            
            # 当爬完一页后，将关于这个user的所有info都存储到pandas里
            c = pd.DataFrame.from_dict(self.allTransList)
            c = c[["DappName", "DappHomePage", "SmartContractAddress", "categories", "tags", "platform",
                        "DappDescribe", "Author", "SoftwareLicense", "Status", "ActiveUsers", "Tx", "Volume"]] # 修改列顺序
            # 以防万一先保存一把
            try:
                c.to_csv('./stateofthedapps2/' + str(pageNum) + ".csv", index=False)
            except Exception as e:
                print("存储" + index_url + "?page=" + str(pageNum) + " 的爬取结果失败。")
                print(e)
                self.save_errorlist()
            
            pageNum += 1 # 爬取下一页
        logging.debug('getTransactionAddressByUserCrawler - END')


    def DappCrawler(self, dapp_url):
        # 有异步js，需要用selenium来爬取
        # 处理dapp对应的url主页
        # 如果没有get到，就sleep 10，然后再爬，共尝试30次
        for countdown in range(1,30):
            try:
                self.driver.get(dapp_url)
                self.expand_contract() # mimic click
                break
            except:
                time.sleep(10)
                continue
        # 对爬到的这个目录页面进行处理
        if self.driver is not None:
            # 将driver的形式转为用soup处理
            soup = BeautifulSoup(self.driver.page_source,'lxml')
            self.allTransList.append(self.this_dapp_info(soup, dapp_url))
        else:
             print("Request Failed: " + dapp_url)
             self.errorList.append(dapp_url)

        return

    def getPageNum(self, index_url):
        """获取整个页面的页数
        
        Args:
            index_url ([type]): [description]
        
        Returns:
            [type]: [description]
        """

        try:
            s = requests.session()
            s.keep_alive = False
            res = s.get(index_url, verify=False, timeout=100)
            soup = BeautifulSoup(res.content,'lxml')
            lastPage = soup.find('button', class_="button number last")
            return int(lastPage.string)
        except Exception as e:
            print("获取页数出错")
            print(e)
            return 100
        
    def this_dapp_info(self, soup, dapp_url):
        """处理得到dapp的信息
        
        Args:
            soup        ([type]): [该dapp_url网页的soup]
            dapp_url    ([string]): [dapp在这个review网站上的值]
        
        Returns:
            [dict]: [dapp处理好的信息]
        """
        dapp_name = None
        description = None
        homepage = None
        platform = None
        categories = None
        tags = None
        sm_address = None
        status = None
        author = None
        mLicense = None
        activeusers = None
        tx = None
        volume = None
        info_from = None

        try:
            # 获取dapp的name
            dapp_name = dapp_url.split("/")[-1]
            # 信息来源
            info_from = dapp_url.split("/")[2]

            # 获取dapp的描述
            if soup.find('div', class_="description") is not None:
                description = soup.find('div', class_="description").text

            # 获取homepage的地址
            if soup.find('div', class_="DappDetailBodyContentCtas").find("a") is not None:
                homepage = soup.find('div', class_="DappDetailBodyContentCtas").find("a")['href']

            # 获取platform
            if soup.find('div', class_="component-DappDetailBodyContentPlatform").a is not None:
                platform = soup.find('div', class_="component-DappDetailBodyContentPlatform").a.string
            
            # 获取categories
            if soup.find("li", class_="category-item") is not None:
                categories = soup.find("li", class_="category-item").a.string

            # 获取tags
            if soup.find("div", class_="component-DappDetailBodyContentBadges") is not None:
                tags_position = soup.find("div", class_="component-DappDetailBodyContentBadges").find_all("li")
                tags = [tag.a.string for tag in tags_position]
                tags = ",".join(tags)

            # 获取和这个dapp相关的智能合约的地址
            # 获取到 Mainnet contracts 里的合约，其他的都不要
            addr_component = soup.find('div', class_="component-DappDetailBodyContentModulesContracts")
            if addr_component is not None:
                for h4 in addr_component.find_all("h4", class_="contract-name"):
                    if "Mainnet" in h4.text:
                        addr_list = h4.parent.find_all('li', class_="DappDetailBodyContentModulesContractsAddress")
                        sm_address = [li.a.span.string for li in addr_list]
                        sm_address = ",".join(sm_address)
                    else:
                        continue
            else:
                sm_address = ""

            # 获取status, author, mLicense
            if soup.find('div', class_="component-DappDetailBodyContentModulesStatus") is not None:
                status = soup.find('div', class_="component-DappDetailBodyContentModulesStatus").strong.string 

            if soup.find('div', class_="component-DappDetailBodyContentModulesAuthors").p.span is not None:
                author = soup.find('div', class_="component-DappDetailBodyContentModulesAuthors").p.span.text
            else:
                author = soup.find('div', class_="component-DappDetailBodyContentModulesAuthors").p.text

            if soup.find('div', class_="component-DappDetailBodyContentModulesLicense").p is not None:
                mLicense = soup.find('div', class_="component-DappDetailBodyContentModulesLicense").p.text
            
            # 活跃用户数、交易量、钱财数量
            stats = soup.find_all('div', class_="component-DappDetailBodyContentModulesStats")
            if len(stats) != 0:
                for stat in stats:
                    # activeusers
                    if 'Active' in stat.find('h4', class_="subtitle").text:
                        activeusers_list = [self.FormatStrNum(li.find('span', class_="stat-value").string) for li in stat.find_all('li', class_="stat-item")]
                        activeusers = "-".join(activeusers_list)
                    # tx
                    if 'Transactions' in stat.find('h4', class_="subtitle").text:
                        tx_list = [self.FormatStrNum(li.find('span', class_="stat-value").string) for li in stat.find_all('li', class_="stat-item")]
                        tx = "-".join(tx_list)
                    # volume
                    if 'Volume' in stat.find('h4', class_="subtitle").text:
                        volume_list = [self.FormatStrNum(li.find('span', class_="stat-value").string) for li in stat.find_all('li', class_="stat-item")]
                        volume = "-".join(volume_list)

        except Exception as e:
            print(e)
            self.errorList.append(dapp_url)

        # 整合信息
        info = {
                'DappName': dapp_name,
                'DappDescribe': description,
                'DappHomePage':homepage,
                'SmartContractAddress': sm_address,
                'categories': categories,
                'tags': tags,
                'Status': status,
                'Author':author,
                'platform': platform,
                'SoftwareLicense': mLicense,
                'InfoFrom': info_from,
                'ActiveUsers':activeusers,
                'Tx':tx,
                'Volume': volume
            }
        return info

    def save_errorlist(self):
        with open('error_urls2.txt', 'w') as f:
            for item in self.errorList:
                f.write("%s\n" % item)

    def expand_contract(self):
        try:
            hide = self.driver.find_element_by_xpath("//span[@class='show-hide']")
            if hide.size != 0:
                hide.click()
        except NoSuchElementException:
            return 
        except Exception as e:
            print(e)
            print(">> 点击扩展的时候出错了...")

    def FormatStrNum(self, s):
        # 格式化str类型的num，eg. '1,657' -> '1657'
        return "".join(s.split(","))


if __name__=="__main__":

    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    context = ssl._create_unverified_context()
    requests.adapters.DEFAULT_RETRIES = 500

    dapps = StateofthedappsCrawler() # init crawler
    dapps.IndexCrawler()
    dapps.driver.quit() # 写入析构函数会报错，全网无解法，暂时先这样吧
    
    # crawl a single dapp
    # dapp_url = "https://www.stateofthedapps.com/dapps/storj"
    # dapps.DappCrawler(dapp_url)

    
    
