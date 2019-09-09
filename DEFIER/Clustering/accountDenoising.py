# -*- coding: utf-8 -*-
import ssl
import os, sys, re
import pymysql
import numpy as np
import pandas as pd
import networkx as nx
import gmatch4py as gm
import traceback
import datetime
import optparse
import logging
from bs4 import BeautifulSoup
import requests
import time
from tqdm import tqdm
import datetime
import json, csv
from web3.auto.infura import w3
from baseFunction import getDappAddrInTest


class Denoising:
    def __init__(self, gameName):
        context = ssl._create_unverified_context()
        requests.adapters.DEFAULT_RETRIES = 500

        self.headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3112.90 Safari/537.36'}
        self.proxy = {'http': 'socks5://127.0.0.1:1086','https':'socks5://127.0.0.1:1086'}
        self.gameName = gameName
        self.gameAddress = set()
        self.gameDict = dict()


    def getTXbyGameName(self):
        # 利用游戏名获取交易中的所有控制用户地址userAddress
        db = pymysql.connect(   host='localhost', user='root', password='hello', db='dapp_analysis_rearrange')
        cursor = db.cursor()
        userAddressList = []
        sql = "SELECT td.*,g.jsonContent FROM TransactionGraph_unknown AS g LEFT JOIN TransactionDescription_testset_enlarger AS td ON g.hashKey=td.hashKey WHERE td.gameName=\"%s\" limit 10000;"% self.gameName
        cursor.execute(sql)
        repetition = cursor.fetchall()
        db.close()

        return repetition

    def getOwnerOfGame(self):
        owner = set()
        for g in self.gameAddress:
            url = "https://etherscan.io/address/" + g
            res = requests.get(url, verify=False, timeout=50)
            soup = BeautifulSoup(res.content,'lxml')

            alist = soup.find('div',id="ContentPlaceHolder1_trContract")
            if alist is not None:
                for a in alist.find_all('a'):
                    if "address" in a['href']:
                        owner.add(a['href'].split("/address/")[1])
        for addr in owner:
            self.gameDict[addr] = "owner"        

        return

    def getAddressByTx(self, repetition):
        gameAddress = set([i[3] for i in repetition]) # for td
        # gameAddress = set([i[-4] for i in repetition]) # for extendtx
        self.gameAddress = self.gameAddress.union(gameAddress)
        for g in gameAddress:
            self.gameDict[g] = "gameAddress"
        return

    def getAddressByGameDB(self):
        # 通过之前爬的数据库来获取游戏地址
        db = pymysql.connect(   host='localhost', user='root', password='hello', db='dapp_analysis_rearrange')
        cursor = db.cursor()
        userAddressList = []
        sql = "SELECT gameAddress FROM EthereumGame WHERE gameName=\"%s\";"% self.gameName
        cursor.execute(sql)
        repetition = cursor.fetchall()
        db.close()
        for r in repetition:
            self.gameDict[r[0]] = "gameAddress"

        return

    def getAddressByTrace(self, repetition):
        # 通过trace来获取address
        for r in tqdm(repetition):
            hashKey = r[0]
            sender = r[1]
            traces = [t for t in json.loads(r[-1]) if t[2] == "call"]
            for trace in traces:
                if trace[0]['id'] in self.gameAddress and trace[1]['id'] != sender:
                    self.gameDict[trace[1]['id']] = trace[1]['label']

        return

    def getOwnerCreatedContract(self, owner):
        # mode = time，表示只获取时间周期内的tx
        beginBlock = "0"
        flag = True
        created_contract_set = set()
        while flag:
            for i in range(5):
                url = "http://api.etherscan.io/api?module=account&action=txlist&address=%s&startblock=%s&endblock=99999999&sort=asc&apikey=JR1M2DJC1R71TRYGPHRIUYMD1K7YGZ8N56" % (owner, beginBlock)
                res = requests.get(url, verify=False, timeout=50)
                text = json.loads(res.text)
                if text['message'] == 'OK':
                    break
            txlist = text['result']
            if len(txlist) != 10000: # 说明这个就是最后的endblock了
                flag = False
            else:
                beginBlock = txlist[-1]['blockNumber']
            created_contracts = [tx['contractAddress'] for tx in txlist if tx['contractAddress'] != ""]
            created_contract_set =  created_contract_set.union(set(created_contracts))

        # 收集到owner创建的全部地址后, 去掉已经获取过的addr
        created_contract_set = [i for i in created_contract_set if i not in self.gameDict]
        # 存入数据库
        self.saveCreatedContractIntoDB(created_contract_set)
        return

    def saveCreatedContractIntoDB(self, created_contract_set):
        # save dict into db
        db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
        cursor = db.cursor()
        updateSql = "INSERT INTO GameContract( `gameName`, smartContractAddress, identity, `origin`) VALUES  (%s, %s, %s, %s);"
        for addr in tqdm(created_contract_set):
            try:
                r = (self.gameName, addr, "ownerCreatedContract", 2)
                cursor.execute(updateSql, r)
                db.commit()
            except Exception as e:
                db.rollback()
                print(e)
                continue

        db.close()

    def saveIntoDB(self):
        # save dict into db
        db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
        cursor = db.cursor()
        updateSql = "INSERT INTO GameContract( `gameName`, smartContractAddress, identity, `origin`) VALUES  (%s, %s, %s, %s);"
        for addr in tqdm(self.gameDict.keys()):

            try:
                r = (self.gameName, addr, self.gameDict[addr], 1 if addr in self.gameAddress else 0)
                cursor.execute(updateSql, r)
                db.commit()
            except Exception as e:
                db.rollback()
                print(e)
                continue

        db.close()

    def accountDenoising(self):
        repetition = self.getTXbyGameName()
        
        # 获取tx里的gameaddress字段
        self.getAddressByTx(repetition)

        # 获取创建这个游戏的人的信息
        self.getOwnerOfGame()
        # 获取这个游戏在网上的地址
        self.getAddressByGameDB()
        # 获取trace里 游戏地址 主动访问过的地址
        self.getAddressByTrace(repetition)

        self.saveIntoDB()
        return

if __name__ == "__main__":
    context = ssl._create_unverified_context()
    requests.adapters.DEFAULT_RETRIES = 500

    gameList = getDappAddrInTest("enlarger")
    for i in range(0, len(gameList)):
        dappName = gameList[i][0]
        dappAddr = gameList[i][1]
        print("# %d -- %s" % (i, dappName))
        denoising = Denoising(dappName)
        denoising.gameAddress.add(dappAddr)
        denoising.gameDict[dappAddr] = "gameAddress"
        denoising.accountDenoising()

        # denoising owner created contracts
        owners = [i for i in denoising.gameDict if denoising.gameDict[i] == "owner"]
        for owner in owners:
            denoising.getOwnerCreatedContract(owner)
