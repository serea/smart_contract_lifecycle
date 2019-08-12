# -*- coding: utf-8 -*-
import os, sys
import requests, ssl
import pymysql
import logging
from bs4 import BeautifulSoup
from sqlalchemy import create_engine
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from web3.auto.infura import w3

if w3.isConnected() is not True:
    print("connect to node error, please check your setting....")
    print(">>> https://web3py.readthedocs.io/en/stable/providers.html")
    

class Type:
    def __init__(self, __load__ = "full"):
        context = ssl._create_unverified_context()
        requests.adapters.DEFAULT_RETRIES = 500

        self.headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3112.90 Safari/537.36'}
        self.proxy = {'http': 'socks5://127.0.0.1:1086','https':'socks5://127.0.0.1:1086'}

        if __load__ == "empty":
            self.address2type = dict()
            self.address2killed = dict()
        elif __load__ == "full": # load all key-value from DB
            self.address2type, self.address2killed = self.getTypeDict()
        else:
            raise("No this load options, please check...")
        self.address2code = dict()

    def getTypeDict(self):
        # 获取contract和type的对照字典
        db = pymysql.connect(   host='localhost', user='root', password='hello', db='dapp_analysis_rearrange')
        cursor = db.cursor()
        sql = "select address, type, killed from SmartContract_transfer;"
        cursor.execute(sql)
        repetition = cursor.fetchall()
        db.close()

        address2type = dict()
        for r in repetition:
            address2type[r[0]] = r[1]

        address2killed = dict()
        for r in repetition:
            address2killed[r[0]] = r[2]

        return address2type, address2killed


    def typeIdentify(self, addr):
        """identify the addr type, would be split to [exchange, user, contract]
        
        Args:
            addr ([type]): [the address of contract or wallet]
        
        Returns:
            [type]: return type in [exchange, user, contract]
        """
        # 验证是否是地址
        if addr is None or len(addr) != 42 or addr[:2] != "0x":
            return None
        # 验证地址是user还是contract
        if addr not in self.address2type.keys():
            code = w3.eth.getCode(w3.toChecksumAddress(addr))
            if len(code)<1 and self.isExchange(addr) is True:
                self.address2type[addr] = "exchange"
            elif len(code) < 1 and self.isKilledAddress(addr) is True:
                self.address2code[addr] = code
                self.address2type[addr] = 'contract'
            elif len(code) < 1:
                self.address2type[addr] = "user"
            elif len(code) > 1:
                self.address2code[addr] = code
                self.address2type[addr] = 'contract'
            self.saveAddressToDB(addr, self.address2type[addr])
        return self.address2type[addr]

    def isKilledAddress(self, addr):
        """use etherscan to identify if addr is killed contract.
        """
        url = "https://etherscan.io/address/" + addr
        res = requests.get(url, verify=False, timeout=50)
        soup = BeautifulSoup(res.content,'lxml')

        alist = soup.find('li',id="ContentPlaceHolder1_li_code")
        if alist is not None:
            return True
        return False

    def isExchange(self, addr):
        """user etherscan to identify if addr is an exchange.
        """
        url = "https://etherscan.io/address/" + addr
        res = requests.get(url, verify=False, timeout=50)
        soup = BeautifulSoup(res.content,'lxml')

        alist = soup.find_all('a',class_="u-label--secondary")
        for a in alist:
            if a.text == "Exchange":
                return True
        return False

    def saveAddressToDB(self, addr, addrType):
        db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
        cursor = db.cursor()
        updateSql = "INSERT INTO SmartContract_transfer(address, type, byteCode, sourceCode, `level`, `detail`, killed) VALUES (%s, %s, %s, %s, %s, %s, %s);"
        r = (addr, addrType, None, None, None, None, None)
        try:
            cursor.execute(updateSql, r)
            db.commit()
        except Exception as e:
            db.rollback()
            print(e)
            print(" address %s save error" % (addr))
            print(r)

        db.close()

    def isKilledAddressByCode(self, addr):
        """identify whether a contract is killed by the code field. Only suit for confirmed contract address.
        """
        code = w3.eth.getCode(w3.toChecksumAddress(addr))
        if len(code) == 0:
            return True
        return False

if __name__ == "__main__":
    t = Type(__load__="empty")  # load with no DB pre-store type information. Good choice for light tasks.
    t = Type(__load__="full")   # load with Million-level DB pre-store type information. Save time for 10k-level tasks.

    