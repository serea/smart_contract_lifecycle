# -*- coding: utf-8 -*-
import zipfile, os, sys
import requests, ssl, json
import pymysql
import re
import numpy as np
import pandas as pd
import json, csv
import traceback
import logging
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from sqlalchemy import create_engine
import base64
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
import random
from web3.auto.infura import w3
from copy import copy
from bisect import bisect_left, bisect_right
from baseFunction import read_list, save_list, graphConstructor, calcGraphSimilarityByGED
import calendar

if w3.isConnected() is not True:
    print("connect to node error, please check your setting....")
    print(">>> https://web3py.readthedocs.io/en/stable/providers.html")
    
def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))
    
    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator
def getAllABIInDB():
    # 获取数据表中全部的hashkey
    db = pymysql.connect(   host='localhost',
                            user='root',
                            password='hello',
                            db='dapp_analysis_rearrange'
                        )
    cursor = db.cursor()
    sql = "SELECT text_signature, hex_signature FROM MethodABI;"
    cursor.execute(sql)
    repetitioon = cursor.fetchall()
    
    db.close()
    return repetitioon

def getAllTxInDB():
    db = pymysql.connect(   host='localhost', user='root', password='hello', db='dapp_analysis_rearrange')
    cursor = db.cursor()
    sql = "select distinct `hashKey` from TransactionDescription_trainset_extend_test;"
    cursor.execute(sql)
    existedTxSet = set([r[0] for r in cursor.fetchall()])
    db.close()

    return existedTxSet

def getTypeDict():
    # 获取contract和type的对照字典
    db = pymysql.connect(   host='localhost', user='root', password='hello', db='dapp_analysis_rearrange')
    cursor = db.cursor()
    sql = "select address, type from SmartContract_transfer;"
    cursor.execute(sql)
    repetition = cursor.fetchall()
    db.close()

    address2type = dict()
    for r in repetition:
        address2type[r[0]] = r[1]

    return address2type

abiTuple = getAllABIInDB()
# existedTxSet = getAllTxInDB()
address2type = getTypeDict()

class Preprocessing(object):

    def __init__(self, gameName, gameAddr):
        context = ssl._create_unverified_context()
        requests.adapters.DEFAULT_RETRIES = 500

        self.headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3112.90 Safari/537.36'}
        self.proxy = {'http': 'socks5://127.0.0.1:1086','https':'socks5://127.0.0.1:1086'}

        self.gameName = gameName
        self.gameAddr = gameAddr
        self.denoisingAddr = self.getDenoisingAddrByGameDB()
        self.gameAddress = set()
        self.gameDict = dict()

        self.address2type = address2type
        self.address2code = dict()

        self.mode = "test"

    def getTXbyGameNameInET(self):
        # 利用游戏名获取交易中的所有控制用户地址userAddress
        db = pymysql.connect(   host='localhost', user='root', password='hello', db='dapp_analysis_rearrange')
        cursor = db.cursor()
        sql = "SELECT td.*,g.jsonContent FROM TransactionGraph AS g LEFT JOIN ExtendTx AS td ON g.hashKey=td.hashKey WHERE td.gameName=\"%s\";"% self.gameName
        cursor.execute(sql)
        repetition = cursor.fetchall()

        handled_tx = "select distinct `seed` from TransactionDescription_trainset_extend_test;"
        cursor.execute(handled_tx)
        handled_rep = set([r[0] for r in cursor.fetchall()])
        db.close()

        repetition = [r for r in repetition if r[0] not in handled_rep]

        return repetition


    def saveTXbyGameNameInET(self):
        # 利用游戏名获取交易中的所有控制用户地址userAddress
        db = pymysql.connect(   host='localhost', user='root', password='hello', db='dapp_analysis_rearrange')
        cursor = db.cursor()
        sql = "SELECT td.*,g.jsonContent FROM TransactionGraph AS g LEFT JOIN ExtendTx AS td ON g.hashKey=td.hashKey WHERE td.gameName=\"%s\";"% self.gameName
        cursor.execute(sql)
        repetition = cursor.fetchall()

        handled_tx = "select distinct `seed` from TransactionDescription_trainset_extend_test;"
        cursor.execute(handled_tx)
        handled_rep = set([r[0] for r in cursor.fetchall()])
        db.close()

        repetition = [r for r in repetition if r[0] not in handled_rep]

        return repetition


    def getTXbyGameNameInTD(self):
        # 利用游戏名获取交易中的所有控制用户地址userAddress
        db = pymysql.connect(   host='localhost', user='root', password='hello', db='dapp_analysis_rearrange')
        cursor = db.cursor()
        sql = "SELECT td.*,g.jsonContent FROM TransactionGraph AS g LEFT JOIN TransactionDescription AS td ON g.hashKey=td.hashKey WHERE td.gameName=\"%s\";"% self.gameName
        cursor.execute(sql)
        repetition = cursor.fetchall()

        handled_tx = "select distinct `seed` from TransactionDescription_trainset_extend_test;"
        cursor.execute(handled_tx)
        handled_rep = set([r[0] for r in cursor.fetchall()])
        db.close()

        repetition = [r for r in repetition if r[0] not in handled_rep]

        return repetition

    def getDenoisingAddrByGameDB(self):
        # 获取需要denoising的地址
        db = pymysql.connect(   host='localhost', user='root', password='hello', db='dapp_analysis_rearrange')
        cursor = db.cursor()
        sql = "SELECT smartContractAddress FROM GameContract WHERE gameName=\"%s\";"% self.gameName
        cursor.execute(sql)
        repetition = cursor.fetchall()
        db.close()

        denoisingAddr = [i[0] for i in repetition]
        return denoisingAddr
    
    def getEOAsOfGame(self, repetition):
        # 获取全部和这个游戏交互的地址
        addrSet = set([ r[1] for r in repetition]).union(set([ r[6] for r in repetition]))
        addrSet = set([addr for addr in addrSet if addr not in self.denoisingAddr and addr is not None])
        # 拆分user和contract
        userSet = set()
        contractSet = set()
        for addr in addrSet:
            addrType = self.typeIdentify(addr)
            if addrType == "user":
                userSet.add(addr)
            elif addrType == "contract":
                contractSet.add(addr)
            else:
                print("%s 的属性为 %s, 不属于user或contract" % (addr, self.address2type[addr]))
                continue
        return userSet, contractSet

    def getEOAsOfGameInET(self, repetition):
        # 获取全部和这个游戏交互的地址
        userSet = set([ r[8] for r in repetition if r[9] == "user"])
        contractSet = set([r[11] for r in repetition if r[11] not in self.denoisingAddr and r[12] == "contract"])
        return userSet, contractSet

    def pickTxbyETInWeb(self, wholeTxList, period):
        pickTxList = [tx for tx in wholeTxList if int(tx['timeStamp']) >= period[0] and int(tx['timeStamp']) <= period[1]]
        timestampArray = np.array([tx['timeStamp'] for tx in wholeTxList], dtype=int)
        startIdx = np.searchsorted(timestampArray, period[0])
        endIdx   = np.searchsorted(timestampArray, period[1], side='right')
        return wholeTxList[startIdx:endIdx]

    def preprocessing(self):
        repetition = self.getTXbyGameName()
        userSet, contractSet = self.getEOAsOfGame(repetition)
        
        for user in tqdm(userSet):
            focusTxlist = [r for r in repetition if r[1] == user or r[6] == user]
            periodList = self.periodListCalcu(user, focusTxlist)
            wholeTxList = self.getTxlistByAddress(user) # 获取到全部的tx列表
            pickedTxList = self.pickTx(wholeTxList, periodList)
            self.saveTxlistToDB(pickedTxList)

        return

    def pickTx(self, wholeTxList, periodList):
        timestampArray = np.array([int(tx['timeStamp']) for tx in wholeTxList], dtype=int)
        pickTxList = []
        for period in periodList:
            startIdx = np.searchsorted(timestampArray, period[0])
            endIdx   = np.searchsorted(timestampArray, period[1], side='right')
            for tx in wholeTxList[startIdx : endIdx]:
                tx['seed'] = period[2]
                pickTxList.append(tx)
        dupSeedHash = [item for item, count in Counter([tx['hash'] for tx in pickTxList]).items() if count > 1]
        
        dupSeedTxlist = []
        for hashKey in dupSeedHash:
            dupSeedTxs = [tx for tx in pickTxList if tx['hash'] == hashKey]
            tx = dupSeedTxs[0]
            tx['seed'] = ", ".join(set([tx['seed'] for tx in dupSeedTxs]))
            pickTxList = [tx for tx in pickTxList if tx['hash'] != hashKey]
            pickTxList.append(tx)
        
        return pickTxList

    def distance(self, r1, r2):
        g1 = graphConstructor(json.loads(r1[-1]))
        g2 = graphConstructor(json.loads(r2[-1]))
        # 计算相邻两张图的编辑距离
        graphDistance = calcGraphSimilarityByGED(g1, g2) # 返回的是距离，值越小表示越相似
        date1 = r1[4]
        date2 = r2[4]
        dateDistance = abs((date1-date2).total_seconds()/60) # 按分钟来算相差的时间

        alpha = 1
        beta = 1
        distance = alpha*graphDistance + beta*dateDistance
        return distance

    def getCodeByAddress(self, address):
        if address in self.address2code.keys():
            return self.address2code[address]
        else:
            code = w3.eth.getCode(w3.toChecksumAddress(address))
            self.address2code[address] = code
            return code

    def CodeSimilarity(self, address):
        simList = []
        for i in self.documentedContractSet:
            simList.append(jaccard_similarity(w3.toHex(self.getCodeByAddress(address)), w3.toHex(self.getCodeByAddress(i))))

        return max(simList)


    # --------------------------------------------------------------
    def typeIdentify(self, addr):
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
        url = "https://etherscan.io/address/" + addr
        res = requests.get(url, verify=False, timeout=50)
        soup = BeautifulSoup(res.content,'lxml')

        alist = soup.find('li',id="ContentPlaceHolder1_li_code")
        if alist is not None:
            return True
        return False

    def isExchange(self, addr):
        url = "https://etherscan.io/address/" + addr
        res = requests.get(url, verify=False, timeout=50)
        soup = BeautifulSoup(res.content,'lxml')

        alist = soup.find_all('a',class_="u-label--secondary")
        for a in alist:
            if a.text == "Exchange":
                return True
        return False

    def fillTimeInfo(self):
        timestampArray = [ calendar.timegm(i.timetuple()) for i in self.groupDate]
        self.timestampMean = np.mean(timestampArray)

        # self.timestampStd = np.std(timestampArray) if len(timestampArray) > 1 else 100000
        self.timestampStd = 20000
        self.timeMin = min(timestampArray) - self.timestampStd
        self.timeMax = max(timestampArray) + self.timestampStd
        print(">>> time mean = " +  datetime.utcfromtimestamp(self.timestampMean).strftime("%Y-%m-%d %H:%M:%S") + " var = " + str(self.timestampStd))
        print(">>> timeMin = " + datetime.utcfromtimestamp(self.timeMin).strftime("%Y-%m-%d %H:%M:%S") + "; timeMax = " + datetime.utcfromtimestamp(self.timeMax).strftime("%Y-%m-%d %H:%M:%S"))
        pass

    def getTxlistByAddress(self, address, startblock=0, endblock=99999999):
        # mode = time，表示只获取时间周期内的tx

        for i in range(5):
            url = "http://api.etherscan.io/api?module=account&action=txlist&address=" + address +"&startblock=" + str(startblock) + "&endblock=" + str(endblock) + "&sort=desc&apikey=WQ5Y216EK6SP2E9SJIBVDJNI1BI7KAIR42"
            res = requests.get(url, verify=False, timeout=50)
            text = json.loads(res.text)
            if text['message'] == 'OK':
                break
            elif "Please select a smaller result dataset" in text['message'] and (endblock - startblock) > 1:
                print("# 数据过大，startblock %d, endblock %d" % (startblock, endblock))
                endblock = (endblock - startblock)/10 + startblock

        
        txlist = text['result'][::-1] # etherscan返回的结果里有tx是多余的

        if len(txlist) > 2000:
            print("%s 地址的tx长度为%d" % (address, len(txlist)))
        for i in range(0, len(txlist)):
            txlist[i]['from'] = txlist[i]['from'].lower()
            txlist[i]['to'] = txlist[i]['to'].lower()
        return txlist

    def periodListCalcu(self, address, focusTxlist):
        periodList = []
        for tx in focusTxlist:
            time = datetime.utcfromtimestamp(int(tx['timeStamp']))
            timeMin = time - timedelta(days=1)
            # timeMax = time + timedelta(days=1)
            timeMax = time + timedelta(seconds=1)
            periodList.append( (calendar.timegm(timeMin.timetuple()), calendar.timegm(timeMax.timetuple()), tx['hash']) )

        return periodList

    def periodListCalcuForEnlarger(self, address, focusTxlist):
        periodList = []
        for tx in focusTxlist:
            time = tx[1]
            timeMin = time - timedelta(days=1)
            # timeMax = time + timedelta(days=1)
            timeMax = time + timedelta(days=1)
            periodList.append( (calendar.timegm(timeMin.timetuple()), calendar.timegm(timeMax.timetuple()), tx[0]) )

        return periodList

    def periodListCalcuForTest(self, address, focusTxlist):
        periodList = []
        for tx in focusTxlist:
            time = tx[4]
            timeMin = time - timedelta(days=1)
            timeMax = time + timedelta(days=1)
            # timeMax = time + timedelta(seconds=1)
            periodList.append( (calendar.timegm(timeMin.timetuple()), calendar.timegm(timeMax.timetuple()), tx[0]) )

        return periodList

    def getTXbyGameAddressInWeb(self, dapp, startblock=0, endblock=99999999):
        txlist = self.getTxlistByAddress(dapp, startblock=startblock, endblock=endblock)
        return txlist

    def decodeMethod(self, tx):
        if len(tx['to']) == 0 and len(tx["input"])>2:
            return "contract creation"
        elif tx['input'] == '0x':
            return "transfer"
        elif len(tx["input"]) > 10:
            return self.Hex2Text(tx["input"][:10], tx["input"])
        else:
            return tx["input"]
    def Hex2Text(self, hex, input):
        candidateSign = [i[0] for i in abiTuple if i[1] == hex]
        if len(candidateSign) == 1:
            return candidateSign[0]
        elif len(candidateSign) == 0:
            return hex
        else:
            s = len(input[10:])/64
            for c in candidateSign:
                paraCnt = len(c.split(","))
                if paraCnt == s:
                    return c
            return hex

        return hex
    def tupleBuild(self, tx, user):
        tx['txDate'] = datetime.utcfromtimestamp(int(tx['timeStamp'])).strftime("%Y-%m-%d %H:%M:%S")
        tx["txMethod"]  = self.decodeMethod(tx)
        tx['senderType']    = self.typeIdentify(tx['from'])
        tx['receiverType']  = self.typeIdentify(tx['to'])
        tx["value"] = str(w3.fromWei(int(tx["value"]), 'ether'))

        r = (   tx['hash'], tx['txDate'], self.gameName, self.gameAddr, tx['seed'], None,
                tx['from'], tx['senderType'], None, tx['txMethod'], tx['to'], tx['receiverType'], None, 
                None, None, None, None, None, None, None, 
                tx['value'], tx['input'], tx['isError'], tx['contractAddress'], None, self.gameName, None, None, 5, user
        )
        return r

    def saveTxlistToDB(self, pickedTxList, user):
        
        db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
        cursor = db.cursor()

        handled_tx = "select distinct `hashKey` from TransactionDescription_goodset_extend where gameName = \"%s\";" % self.gameName
        cursor.execute(handled_tx)
        handled_rep = set([r[0] for r in cursor.fetchall()])

        # updateSql = "INSERT INTO TransactionDescription_testset_extend(hashKey, txDate, gameName, gameAddress, seed, similarity, sender, senderType, senderName, txMethod, receiver, receiverType, receiverName, txLabel, txLabelNew, labelReason, suicide, profit, traceCnt, prepare, value, input, isError, contractAddress, ori, seedGame, seedEOA) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        # updateSql = "INSERT INTO TransactionDescription_trainset_extend_test(hashKey, txDate, gameName, gameAddress, seed, similarity, sender, senderType, senderName, txMethod, receiver, receiverType, receiverName, txLabel, txLabelNew, labelReason, suicide, profit, traceCnt, prepare, value, input, isError, contractAddress, ori, seedGame, graphDistance, dateDistance, stage, seedEOA) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        updateSql = "INSERT INTO TransactionDescription_goodset_extend(hashKey, txDate, gameName, gameAddress, seed, similarity, sender, senderType, senderName, txMethod, receiver, receiverType, receiverName, txLabel, txLabelNew, labelReason, suicide, profit, traceCnt, prepare, value, input, isError, contractAddress, ori, seedGame, graphDistance, dateDistance, stage, seedEOA) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"

        for tx in pickedTxList:
            try:
                if tx['hash'] not in handled_rep:
                    r = self.tupleBuild(tx, user)
                    cursor.execute(updateSql, r)
                    db.commit()
            except Exception as e:
                db.rollback()
                print(e)
                print(" address is %s and hashkey is %s" % (tx['seed'], tx['hash']))
                # print(r)
                continue

        db.close()


    def saveAddressToDB(self, addr, addrType):
        db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
        cursor = db.cursor()
        updateSql = "INSERT INTO SmartContract_transfer(address, type, byteCode, sourceCode, `level`, `detail`, killed) VALUES (%s, %s, %s, %s, %s, %s, %s);"
        r = (addr, addrType, None, None, None, None, None)
        try:
            cursor.execute(updateSql, r)
            # print(r)
            db.commit()
        except Exception as e:
            db.rollback()
            print(e)
            print(" address %s 保存出错！" % (addr))
            print(r)

        db.close()


    def getTxByEOAInTest(self, game, eoa, mode="goodset"):
        # 利用游戏名获取交易中的所有控制用户地址userAddress
        db = pymysql.connect(   host='localhost', user='root', password='hello', db='dapp_analysis_rearrange')
        cursor = db.cursor()
        if self.mode == "test":
            sql = "SELECT td.*,g.jsonContent FROM TransactionDescription_testset AS td LEFT JOIN TransactionGraph_unknown AS g ON g.hashKey=td.hashKey WHERE td.gameName=\"" + game + "\" and td.controlUserAddress=\"" + eoa + "\" order by txDate;"
        elif self.mode == "enlarger":
            sql = "SELECT td.*,g.jsonContent FROM TransactionDescription_testset_enlarger AS td LEFT JOIN TransactionGraph_unknown AS g ON g.hashKey=td.hashKey WHERE td.gameName=\"" + game + "\" and td.sender=\"" + eoa + "\" order by txDate;"
        elif self.mode == "goodset":
            sql = "SELECT td.*,g.jsonContent FROM TransactionDescription_goodset_enlarger AS td LEFT JOIN TransactionGraph_goodset AS g ON g.hashKey=td.hashKey WHERE td.gameName=\"" + game + "\" and td.sender=\"" + eoa + "\" order by txDate;"
        else:
            raise("不存在该mode，请检查")

        cursor.execute(sql)
        repetition = cursor.fetchall()

        handled_tx = "select distinct `seed` from TransactionDescription_goodset_extend where gameName=\"" + game + "\";"
        cursor.execute(handled_tx)
        handled_rep = set([r[0] for r in cursor.fetchall()])
        handled_rep = set()
        db.close()

        repetition = [r for r in repetition if r[0] not in handled_rep]
        return repetition


    def getTxByEOAInTestET(self, game, eoa):
        # 利用游戏名获取交易中的所有控制用户地址userAddress
        db = pymysql.connect(   host='localhost', user='root', password='hello', db='dapp_analysis_rearrange')
        cursor = db.cursor()
        sql = "SELECT td.*,g.jsonContent FROM TransactionDescription_testset_extend AS td LEFT JOIN TransactionGraph AS g ON g.hashKey=td.hashKey WHERE td.gameName=\"" + game + "\" and td.sender=\"" + eoa + "\" order by txDate;"
        cursor.execute(sql)
        repetition = cursor.fetchall()
        return repetition

    def tupleBuildInTest(self, r):
        try:
            tx = dict()
            tx['hashKey'] = r[0]
            tx['txDate'] = r[4]
            tx['gameName'] = r[2]
            tx['gameAddress'] = r[3]
            tx["txMethod"]  = r[5]
            tx['sender'] = r[1]
            tx['senderType']    = self.typeIdentify(r[1])
            tx['senderName']    = None
            tx['receiver']  = r[6]
            tx['receiverType']  = self.typeIdentify(r[6])
            tx['receiverName'] = r[7]
            tx['txLabel'] = None
            tx['txLabelNew'] = None
            tx['labelReason'] = "ExtendTx"
            tx['profit'] = None
            tx['traceCnt'] = None
            seedGame = r[2]
        except Exception as e:
            print(e)
            print(r[6])
        r = (   tx['hashKey'], tx['txDate'], tx['gameName'], tx['gameAddress'], tx['hashKey'], 0,
                tx['sender'], tx['senderType'], tx['senderName'], tx['txMethod'], tx['receiver'], tx['receiverType'], tx['receiverName'], 
                tx['txLabel'], tx['txLabelNew'], tx['labelReason'], None, None, None, None,
                None, None, None, None, "1", seedGame, None
        )
        return r

    def tupleBuildInEnlarger(self, r):
        try:
            tx = dict()
            tx['hashKey'] = r[0]
            tx['txDate'] = r[4]
            tx['gameName'] = r[2]
            tx['gameAddress'] = r[3]
            tx["txMethod"]  = r[5]
            tx['sender'] = r[1]
            tx['senderType']    = self.typeIdentify(r[1])
            tx['senderName']    = None
            tx['receiver']  = r[6]
            tx['receiverType']  = self.typeIdentify(r[6])
            tx['receiverName'] = r[7]
            tx['txLabel'] = None
            tx['txLabelNew'] = None
            tx['labelReason'] = "ExtendTx"
            tx['profit'] = None
            tx['traceCnt'] = None
            seedGame = r[2]
        except Exception as e:
            print(e)
            print(r[6])
        r = (   tx['hashKey'], tx['txDate'], tx['gameName'], tx['gameAddress'], tx['hashKey'], 0,
                tx['sender'], tx['senderType'], tx['senderName'], tx['txMethod'], tx['receiver'], tx['receiverType'], tx['receiverName'], 
                tx['txLabel'], tx['txLabelNew'], tx['labelReason'], None, None, None, None,
                None, None, None, None, "1", seedGame, None
        )
        return r

    def saveTestsetToDB(self, pickedTxList):
        
        db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
        cursor = db.cursor()
        updateSql = "INSERT INTO TransactionDescription_testset_extend(hashKey, txDate, gameName, gameAddress, seed, similarity, sender, senderType, senderName, txMethod, receiver, receiverType, receiverName, txLabel, txLabelNew, labelReason, suicide, profit, traceCnt, prepare, value, input, isError, contractAddress, ori, seedGame, stage) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        rep = []
        for tx in pickedTxList:
            try:
                r = self.tupleBuildInTest(tx)
                rep.append(r)
                cursor.execute(updateSql, r)
                db.commit()
            except Exception as e:
                db.rollback()
                print(e)
                # print("hashkey is %s")
                print(r)
                continue

        db.close()

    def saveEnlargerToDB(self, pickedTxList):
        db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
        cursor = db.cursor()

        handled_tx = "select distinct `hashKey` from TransactionDescription_goodset_extend where gameName = \"%s\";" % self.gameName
        cursor.execute(handled_tx)
        handled_rep = set([r[0] for r in cursor.fetchall()])

        updateSql = "INSERT INTO TransactionDescription_goodset_extend(hashKey, txDate, gameName, gameAddress, seed, similarity, sender, senderType, senderName, txMethod, receiver, receiverType, receiverName, txLabel, txLabelNew, labelReason, suicide, profit, traceCnt, prepare, value, input, isError, contractAddress, ori, seedGame, graphDistance, dateDistance, stage, seedEOA, txGameName, txGameAddress) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        for r in pickedTxList:
            if r[0] not in handled_rep:
                try:
                    cursor.execute(updateSql, r[:-1])
                    db.commit()
                except Exception as e:
                    db.rollback()
                    print(e)
                    print("hashkey is %s" % r[0])
                    # print(r)
                    continue

        db.close()

if __name__ == "__main__":
    context = ssl._create_unverified_context()
    requests.adapters.DEFAULT_RETRIES = 500
    game = "godgame"

    pre = Preprocessing(game)
    pre.preprocessing()



