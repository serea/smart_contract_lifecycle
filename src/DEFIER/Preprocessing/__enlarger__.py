# -*- coding: utf-8 -*- 
import math
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from collections import Counter
import networkx as nx
import matplotlib.pyplot as plt
import zipfile, os, sys
import json
import pymysql
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from tqdm import tqdm
from sqlalchemy import create_engine
from baseFunction import read_list, save_list, graphConstructor, calcGraphSimilarityByGED, getSingleGraphByTx, getDappAddrInTest, getHandledDappNameInTestET
from baseFunction import getTopEOAByDappInTest, getHandledEOAs, getAllEOAByDappInEnlarger
from Preprocessing import Preprocessing
import calendar
from web3.auto.infura import w3
import random

if w3.isConnected() is not True:
    print("connect to node error, please check your setting....")
    print(">>> https://web3py.readthedocs.io/en/stable/providers.html")

if 'DEFIER' not in os.getcwd():
    os.chdir("./DEFIER")

def extendTxBucket(tempFocusTxs, bucketIdx, pre):
    # pre.saveEnlargerToDB(tempFocusTxs) # 将这个用户的所有数据存到extend里，然后获取这100条的扩展tx
    periodList = pre.periodListCalcuForEnlarger(user, tempFocusTxs) # 获取需要爬的时间段
    # 通过w3获取tx的信息
    try:
        c = w3.eth.getTransaction(tempFocusTxs[0][0])['blockNumber']
        sblock = c-5500 if c > 5500 else c # 分区域获取tx数量，避免短期内tx > 1w的情况
        eblock = w3.eth.getTransaction(tempFocusTxs[-1][0])['blockNumber']+1
    except Exception as e:
        print(e)
        sblock = 0
        eblock = 99999999

    wholeTxList = pre.getTxlistByAddress(user, startblock=sblock, endblock=eblock) # 获取到这个user的全部tx列表
    
    pickedTxList = pre.pickTx(wholeTxList, periodList)
    print(" - bucket #(%d, %d) 扩展tx  %d 条" % (bucketIdx[0], bucketIdx[1], len(pickedTxList)))
    # pre.saveTxlistToDB(pickedTxList, user)


def getBucketRange(focusTxlist):
    startIdx = 0
    startDate = focusTxlist[startIdx][1]
    endDate = startDate + timedelta(days=1)
    bucketRange = []
    for i in range(len(focusTxlist)):
        date = focusTxlist[i][1]
        if date > endDate:
            if startIdx < i-1:
                bucketRange.append((startIdx, i-1))
                startIdx = i
                startDate = date
                endDate =  startDate + timedelta(days=1)
            elif startIdx == i-1:
                endDate += timedelta(days=1)
    if len(bucketRange) == 0:
        bucketRange.append((0, len(focusTxlist)))
    else:
        bucketRange.append((bucketRange[-1][1]+1, len(focusTxlist)))
    return bucketRange

def cleanTestset(gameName, gameAddr, pre, denoisingAddrs):
    # 将所有的testset里没有放入enlarger的数据都填充进enlarger里
    db = pymysql.connect(   host='localhost', user='root', password='hello', db='dapp_analysis_rearrange')
    cursor = db.cursor()
    sql = "SELECT t.*,g.jsonContent FROM TransactionDescription_testset_enlarger AS t LEFT JOIN TransactionDescription_testset_extend AS e ON t.hashKey=e.hashKey LEFT JOIN TransactionGraph_unknown AS g ON t.hashKey=g.hashKey WHERE e.hashKey IS NULL AND t.gameName=\"%s\" AND t.gameAddress=\"%s\" AND t.sender IS NOT NULL order by t.txDate;" % (gameName, gameAddr)

    cursor.execute(sql)
    repetition = cursor.fetchall()
    db.close()

    tempFocusTxs = [i for i in repetition if i[6] not in denoisingAddrs]
    pre.saveEnlargerToDB(tempFocusTxs)
    return

if __name__ == "__main__":
    # Firstly, to avoid confusing transactions, you need run accountDenoising to store denoising address belonging to this dapp.

    # 1. 获取游戏的地址
    DappsRep = getDappAddrInTest(mode="enlarger")

    # 从unknown数据里提取出需要爬取的eoa和tx，并存入extend表
    clock = []
    for dapp in DappsRep:
        # 2. 获取这个游戏在我们数据集里的extendtx列表和时间范围，extendtx作为带标签的tx, 去重
        dappName = dapp[0]
        dappAddr = dapp[1]
        
        print("\n# Handle %s addr %s" % (dappName, dappAddr))
        pre = Preprocessing(dappName, dappAddr) #初始化
        pre.mode = "enlarger" # set 
        denoisingAddrs = pre.getDenoisingAddrByGameDB() # 排除掉不需要的addr
        userList = getAllEOAByDappInEnlarger(dappName)
        handledUsers = getHandledEOAs(dappName, dappAddr)
        
        # userList = [i for i in userList if i not in handledUsers and i not in denoisingAddrs]
        userList = [i for i in userList if i not in denoisingAddrs]
        userCnt = len(userList)
        print("共有%d个user需要extend"%len(userList))

        if len(userList) > 100:
            userList = random.sample(userList, 100)
        for i in range(len(userList)):
            user = userList[i]
            a = time.time()
            print("\n# %s 游戏的 第%d个user: %s" %(dappName, i, user))
            focusTxlist = pre.getTxByEOAInTest(dappName, user, mode="enlarger") # testset里的数据
            if len(focusTxlist) == 0 or len(focusTxlist) > 5000:
                continue
            bucketRange = getBucketRange(focusTxlist)
            if len(bucketRange) > 200:
                continue
            for i in bucketRange:
                tempFocusTxs = focusTxlist[i[0]: i[1]]
                extendTxBucket(tempFocusTxs, i, pre)
            b = time.time()
            print("time is %d" % int(b-a))
            clock.append((dappName, dappAddr, userCnt, user, b-a))
        clockDf = pd.DataFrame(clock, columns=("dappName", "dappAddr", "userCnt", "user", "extendTime"))
        clockDf.to_csv("extend_clock.csv", index=False)

        
