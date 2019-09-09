# coding:utf-8

import logging
from bs4 import BeautifulSoup
import ssl
import os, sys, re
import requests
import time
import pandas as pd
from tqdm import tqdm
import pymysql
import datetime
import json, csv
from web3.auto.infura import w3

if w3.isConnected() is not True:
    print("connect to node error, please check your setting....")
    print(">>> https://web3py.readthedocs.io/en/stable/providers.html")
    
class ABICrawler:
    def __init__(self):
        context = ssl._create_unverified_context()
        requests.adapters.DEFAULT_RETRIES = 500

        self.headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3112.90 Safari/537.36'}
        self.proxy = {'http': 'socks5://127.0.0.1:1086','https':'socks5://127.0.0.1:1086'}


    def ABIListCrawler(self):
        base_url = "https://www.4byte.directory/api/v1/signatures/?page=%d"

        # 存储tx的列表
        # 获取数据库里全部待爬的tx值
        db = pymysql.connect(   host='localhost',
                                user='root',
                                password='hello',
                                db='dapp_analysis_rearrange'
                            )
        cursor = db.cursor()
        updateSql = "INSERT INTO MethodABI( id, created_at, text_signature, hex_signature, bytes_signature) VALUES  (%s, %s, %s, %s, %s);"

        pageNum = 1
        while True:
            print(">>> Page #%d" % pageNum)
            res = requests.get(base_url % pageNum, verify=False, timeout=50)
            text = json.loads(res.text)
            if res.status_code == 404:
                break
            result = text['results']
            for r in result:
                ABI = (r['id'], r['created_at'], r['text_signature'], r['hex_signature'], w3.toHex(bytes(r['bytes_signature'], 'utf-8')))
                try:
                    cursor.execute(updateSql, ABI)
                    db.commit()
                except Exception as e:
                    print(e)
                    print( " %d dump error..." % r['id'])
                    db.rollback()
            pageNum +=1
        db.close()

        return



if __name__=="__main__":

    from requests.packages.urllib3.exceptions import InsecureRequestWarning
    requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
    context = ssl._create_unverified_context()
    requests.adapters.DEFAULT_RETRIES = 500

    # 爬取hex对应的method名字
    abi = ABICrawler()
    abi.ABIListCrawler()

