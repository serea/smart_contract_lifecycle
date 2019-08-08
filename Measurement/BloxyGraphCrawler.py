# coding:utf-8
import logging
from bs4 import BeautifulSoup
import ssl
import os, sys, re
import requests
import time
from tqdm import tqdm
import pymysql
import datetime
import optparse

def parse_option():
    parser = optparse.OptionParser()

    parser.add_option(
        "-b",
        "--begin",
        dest="begin",
        type="string",
        action="store",
        default=0,
        help="name tag for result.txt")
    parser.add_option(
        "-e",
        "--end",
        dest="end",
        type="int",
        action="store",
        default=1000,
        help="soft time range(units : days)")
    return parser


class BloxyGraphCrawler:
    """directly get bloxy-generated graph of transactions, then insert into Database.
    """
    def __init__(self):
        context = ssl._create_unverified_context()
        requests.adapters.DEFAULT_RETRIES = 500
        self.headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36'}
        self.proxy = {'http': 'socks5://127.0.0.1:1086','https':'socks5://127.0.0.1:1086'}
        self.errorList = []


    def getAllTxhashSequence(self, begin, end):
        new_db = pymysql.connect("localhost","root","hello","dapp_analysis_rearrange")
        cursor = new_db.cursor()
        sql = "SELECT td.TxHash FROM (SELECT td.TxHash FROM TransactionDescription AS td) AS td LEFT JOIN TRANSACTION AS t ON td.TxHash=t.hashkey WHERE t.hashkey IS NULL;"
        cursor.execute(sql)
        repetition = cursor.fetchall()
        print("\n >>> 需要爬的总共有" +str(len(repetition)) + "个")
        r = [ i[0] for i in repetition][int(begin):int(end)]
        txhash_set = set(r)
        new_db.close()
        return txhash_set

    def grpahCrawler(self, begin=1, end=1000):
        # 爬取txgraph的数据
        s = requests.session()
        s.keep_alive = False
        graph_sql = "INSERT INTO `TransactionGraph_extendtx` (`hashkey`, `jsonContent`) VALUES (%s,%s) ON DUPLICATE KEY UPDATE jsonContent = %s"

        new_db = pymysql.connect("localhost","root","hello","dapp_analysis_rearrange")
        cursor = new_db.cursor()

        # 需要爬取的txhash set
        txhash_set = set(self.read_list("lack.txt"))
        # txhash_set = self.getAllTxhashSequence(begin, end)

        # 存储爬取结果
        txgraph_list = []
        for txhash in tqdm(txhash_set):
            g = self.TxDetailGraphCrawler(txhash, s)

            try:
                cursor.execute(graph_sql, g)
                new_db.commit()
            except Exception as e:
                print(e)
                print("%s 导入数据库出错" % txhash)
                self.errorList.append(txhash)
                new_db.rollback() # 回滚到导入之前的状态
        new_db.close()

        
    def TxDetailGraphCrawler(self, txhash, s):
        # 获取这个tx的图json数据
        base_url = "https://bloxy.info/tx_graph_expanded_results/"
        while 1:
            try:
                res = s.get(base_url + txhash, verify=False, timeout=30)
                if res.status_code == 429:
                    print(base_url + txhash + "  429 Too Many Requests ing ...")
                    time.sleep(5)
                else:
                    break
            except '<html>\r\n<head>' in res.content.decode('utf-8'):
                time.sleep(10)
                print(res.status_code)
                print(res.content.decode('utf-8'))
                print(base_url + txhash)
                continue
            except Exception as e:
                time.sleep(10)
                print(e)
                continue
        j = res.content.decode('utf-8')
        return (txhash, j, j)

    def save_errorlist(self):
        with open('./grapherror_urls.txt', 'w') as f:
            for item in self.errorList:
                f.write("%s\n" % item)

    def read_list(self, file_name):
        places = []
        with open(file_name, 'r') as filehandle:  
            for line in filehandle:
                places.append(line.strip("\"\n"))
        return places

if __name__=="__main__":

    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    context = ssl._create_unverified_context()
    requests.adapters.DEFAULT_RETRIES = 500

    # 爬取graph数据
    bloxy = BloxyGraphCrawler()

    # get args
    global options, args
    parser = parse_option()
    options, args = parser.parse_args()
    print(options.begin)
    bloxy.grpahCrawler(begin=options.begin, end=options.end)
    
    
    
    
