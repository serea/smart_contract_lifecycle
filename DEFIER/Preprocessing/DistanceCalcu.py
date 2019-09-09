#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 12:52:37 2019

@author: vera
"""
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
import gmatch4py as gm
from sqlalchemy import create_engine

from baseFunction import read_list, save_list, graphConstructor, calcGraphSimilarityByGED, getSingleGraphByTx
alpha = 0.9
beta = 0.1

if 'DEFIER' not in os.getcwd():
    os.chdir("./DEFIER")

def distance( r1, r2):
    graphDistance   = getGraphDistance(r1, r2)
    dateDistance    = getDateDistance(r1, r2)
    distance = alpha*graphDistance + beta*dateDistance
    return distance

def getGraphDistance(r1, r2):
    try:
        g1 = graphConstructor(json.loads(r1[-1]))
        g2 = graphConstructor(json.loads(r2[-1]))
        s = calcGraphSimilarityByGED(g1, g2)
        if np.isinf(s):
            s = 0
            if len(r1[-1])==2 or len(r2[-1])==2:
                s = abs(len(json.loads(r1[-1])) - len(json.loads(r2[-1])))
    except Exception as e:
        print(e)
        return 0

    # 计算相邻两张图的编辑距离
    return s # 返回的是距离，值越小表示越相似

def getDateDistance(r1, r2):
    date1 = r1[1]
    date2 = r2[1]
    return abs((date1-date2).total_seconds()/3600) # 除以小时的总秒数，值越小表示时间越相近


def addGameNameandAddress():
    db = pymysql.connect("localhost","root","hello","dapp_analysis_rearrange")
    cursor = db.cursor()

    dapp_df = pd.DataFrame(columns=['gameName','gameAddress'])
    sql = "SELECT gameName,gameAddress FROM EthereumGame"
    cursor.execute(sql)
    temp = cursor.fetchall()
    for item in temp:
        dapp_df = dapp_df.append([{'gameName': item[0], 'gameAddress':item[1]}],ignore_index=True)
    print("数据库里共有gameName-gameAddress %d 个" % len(dapp_df))

    sql_all = "SELECT t.hashKey,g.jsonContent FROM TransactionDescription_goodset_extend AS t LEFT JOIN TransactionGraph_goodset AS g ON t.hashKey=g.hashKey WHERE g.hashKey IS NOT NULL and t.suicide is null ;"

    cursor.execute(sql_all)
    repetition = cursor.fetchall()

    for item in tqdm(repetition):
        try:
            hashKey = item[0]
            jsonContent = json.loads(item[1].decode('utf-8'))

            traceNum = 0
            gameAddress=None
            gameName=None
            traceNum = len(jsonContent)
            gameName, gameAddress = getGraphDapp(dapp_df,jsonContent)
            suicide = getSuicideFlag(item[1].decode('utf-8'))

            update_sql = "UPDATE TransactionDescription_goodset_extend SET traceCnt=%s, txGameName=%s, txGameAddress=%s, suicide = %s WHERE hashKey=%s"
            sqlarg = [traceNum, gameName, gameAddress, suicide,  hashKey]
            cursor.execute(update_sql, sqlarg)
            db.commit()
        except Exception as e:
            print(e)
            print(hashKey)
            db.rollback() # 回滚到导入之前的状态
    db.close()

def getSuicideFlag(graph):
    if 'suicide'.lower() in graph.lower():
        return 1
    else:
        return 0

def getGraphDapp(dapp_df,graph):

    gameName = []
    gameAddress=[]
    graphObj = graph
    # 0x8d12a197cb00d4747a1fe03395095ce2a5cc6819 forkdelta
    for item in graphObj:
        if item[2] in ['call','transfer_to']  and item[3] in dapp_df['gameAddress'].tolist():
            game = dapp_df.ix[dapp_df['gameAddress']==item[3]]['gameName'].values[0]
            if game not in gameName:
                if game=='forkdelta' and len(gameName)>0:
                    continue
                elif 'forkdelta' in gameName:
                    gameName.append(game)
                    gameAddress.append(item[3])
                else:
                    gameName.append(game)
                    gameAddress.append(item[3])
        if item[2] in ['call','transfer_to'] and item[4] in dapp_df['gameAddress'].tolist():
            game = dapp_df.ix[dapp_df['gameAddress']==item[4]]['gameName'].values[0]
            if game not in gameName:
                if game=='forkdelta' and len(gameName)>0:
                    continue
                elif 'forkdelta' in gameName:
                    gameName.append(game)
                    gameAddress.append(item[4])
                else:
                    gameName.append(game)
                    gameAddress.append(item[4])
    gameName_str = ','.join(gameName)
    gameAddress_str = ','.join(gameAddress)
    return gameName_str,gameAddress_str


def getAllRepInDB():
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    sql = "SELECT  td.*, g.jsonContent FROM TransactionDescription_goodset_extend AS td LEFT JOIN TransactionGraph_goodset AS g ON g.hashKey=td.hashKey where g.hashKey is not null;"
    cursor.execute(sql)
    repetition = cursor.fetchall()
    repDict = dict()
    for r in repetition:
        repDict[r[0]] = r
    db.close()
    return repDict

def addDistance():
    repDict = getAllRepInDB() # 获取全部需要处理的tx

    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    waitHandle = [hashKey for hashKey in repDict.keys()]
    for hashKey in tqdm(waitHandle):
        r = repDict[hashKey]
        
        if r[4] in repDict:
            focusTx = repDict[r[4]] # 通过seed找到这个tx
        else:
            print("seed tx %s not in DB" % r[4])
            continue
        
        try:
            graphDistance = getGraphDistance(r, focusTx)
            dateDistance = getDateDistance(r, focusTx)
            distance = alpha * graphDistance + beta * dateDistance
        except Exception as e:
            print(e)
            print(" Distance calculate Error ! hashkey is %s" % r[0])
            continue

        updateSql = "UPDATE TransactionDescription_goodset_extend SET graphDistance=%f, dateDistance=%f, similarity=%f where hashKey = \"%s\""
        try:
            t = (graphDistance, dateDistance, distance, hashKey)
            cursor.execute(updateSql % t)
            db.commit()
        except Exception as e:
            db.rollback()
            print(e)
            print(" ERROR ! hashkey is %s" % r[0])
            continue

    db.close()

if __name__ == "__main__":

    # 补充距离
    addDistance()
    # 补充图
    addGameNameandAddress()
