# -*- coding: utf-8 -*-
import os, sys
import pymysql
import numpy as np
import pandas as pd
import networkx as nx
import gmatch4py as gm
import traceback
import argparse
from sqlalchemy import create_engine
import datetime
import optparse
import json
import operator

def getDappAddrInDB():
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    sql = "SELECT gameName,gameAddress,s.type,cnt FROM SmartContract_transfer AS s LEFT JOIN (SELECT gameAddress ,gameName ,count(*) AS cnt FROM ExtendTx AS e GROUP BY gameName,gameAddress ORDER BY cnt) AS e ON e.gameAddress=s.address WHERE e.gameAddress IS NOT NULL ORDER BY cnt;"
    cursor.execute(sql)
    repetition = [i for i in cursor.fetchall()]
    db.close()
    return repetition

def getDappAddrInTest(mode="test"):
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    if mode == "test":
        sql = "SELECT gameName,gameAddress,count(*) AS cnt FROM TransactionDescription_testset GROUP BY gameName,gameAddress ORDER BY gameName, cnt DESC;"
    elif mode == "enlarger":
        sql = "SELECT gameName,gameAddress,count(*) AS cnt FROM TransactionDescription_testset_enlarger GROUP BY gameName,gameAddress ORDER BY cnt;"
    elif mode == "goodset":
        sql = "SELECT gameName,gameAddress,count(*) AS cnt FROM TransactionDescription_goodset_enlarger GROUP BY gameName,gameAddress ORDER BY cnt;"
    else:
        raise("不存在该mode，请检查")
    cursor.execute(sql)
    repetition = [i for i in cursor.fetchall()]
    db.close()
    return repetition

def getDappNameInTest():
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    sql = "SELECT gameName, count(*) AS cnt FROM TransactionDescription_testset GROUP BY gameName ORDER BY cnt DESC;"
    cursor.execute(sql)
    repetition = [i for i in cursor.fetchall()]
    db.close()
    return repetition

def getHandledDappNameInTDET():
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    sql = "SELECT `gameName`,count(*) FROM TransactionDescription_trainset_extend_test GROUP BY gameName;"
    cursor.execute(sql)
    repetition = [i[0] for i in cursor.fetchall()]
    db.close()
    return set(repetition)

def getHandledDappNameInTestET():
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    sql = "SELECT `gameName`,count(*) FROM TransactionDescription_testset_extend where ori is null GROUP BY gameName;"
    cursor.execute(sql)
    repetition = [i[0] for i in cursor.fetchall()]
    db.close()
    return set(repetition)

def getHandledEOAs(game, gameAddr, mode = "enlarger"):
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    if mode == "enlarger":
        sql = "SELECT seedEOA FROM TransactionDescription_testset_extend WHERE gameName='%s' AND gameAddress='%s';" % (game, gameAddr)
    elif mode == "goodset":
        sql = "SELECT distinct seedEOA FROM TransactionDescription_goodset_extend WHERE gameName='%s' AND gameAddress='%s';" % (game, gameAddr)
    cursor.execute(sql)
    repetition = [i[0] for i in cursor.fetchall()]
    db.close()
    return set(repetition)

def intIdentify(offset):
    try:
        if type(offset) is int or offset is None:
            pass
        elif offset.isdigit():
            offset = int(offset)
        else:
            raise("Error! 'Offset' Input is not int! ")
    except Exception as e:
        print(e)
    return offset

def getTopEOAByDappInTest(game, number, offset=3):
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()

    offset = intIdentify(offset)
    number = intIdentify(number)
    
    if number is None and offset != 0: # 获取全部(offset默认为3)
        sql = "SELECT controlUserAddress,count(*) AS cnt FROM TransactionDescription_testset WHERE gameName=\"%s\" and controlUserAddress is not null group by controlUserAddress order by cnt desc OFFSET %d;" % (game, offset)
    elif number is None and offset == 0:
        sql = "SELECT controlUserAddress,count(*) AS cnt FROM TransactionDescription_testset WHERE gameName=\"%s\" and controlUserAddress is not null group by controlUserAddress order by cnt desc;" % (game)
    elif number is not None and offset != 0: # 获取指定的number数
        sql = "SELECT controlUserAddress,count(*) AS cnt FROM TransactionDescription_testset WHERE gameName=\"%s\" and controlUserAddress is not null group by controlUserAddress order by cnt desc LIMIT %d OFFSET %d;" % (game, number, offset)
    elif number is not None and offset == 0: # 获取指定的number数
        sql = "SELECT controlUserAddress,count(*) AS cnt FROM TransactionDescription_testset WHERE gameName=\"%s\" and controlUserAddress is not null group by controlUserAddress order by cnt desc LIMIT %d ;" % (game, number)

    cursor.execute(sql)
    repetition = [i[0] for i in cursor.fetchall()]
    db.close()
    return repetition

def getTOP100EOAByDappInTest(game):
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    
    cursor.execute(sql)
    repetition = [i[0] for i in cursor.fetchall()]
    db.close()
    return repetition



def getAllEOAByDappInEnlarger(game, mode="enlarger"):
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    if mode == "enlarger":
        sql = "SELECT sender, count(*) AS cnt FROM TransactionDescription_testset_enlarger WHERE gameName=\"%s\" and sender is not null group by sender order by cnt;" % game
    elif mode == "goodset":
        sql = "SELECT eoa FROM goodset_eoa WHERE gameName=\"%s\";" % game
    else:
        raise("不存在该mode，请检查")
    cursor.execute(sql)
    repetition = [i[0] for i in cursor.fetchall()]
    db.close()
    return repetition

def getUserAddressList(gameName):
    # 利用游戏名获取交易中的所有控制用户地址userAddress
    db = pymysql.connect(   host='localhost',
                            user='root',
                            password='',
                            db='dapp_analysis_rearrange'
                        )
    cursor = db.cursor()
    userAddressList = []
    sql = "SELECT DISTINCT TransactionDescription.controlUserAddress from TransactionDescription WHERE TransactionDescription.controlUserAddress in (select smartContractAddress from SmartContract_Game where SmartContract_Game.smartContractType='0') and TransactionDescription.gameName=\"%s\" and TransactionDescription.txDate is not NULL;"% gameName
    cursor.execute(sql)
    repetition = cursor.fetchall()
    userAddressList = [i[0] for i in repetition]
    db.close()
    userAddressList = list(set(userAddressList))#- set(getGameAddressList(gameName))
    if len(userAddressList) == 0:
            print("There is no smartcontract with dapp_name \"%s\" , Please check your gamename." % gameName)
    return userAddressList

def GameAddressDetector(gameName):
    # 检测地址账号是否是game related的账号
    db = pymysql.connect(   host='localhost',
                            user='root',
                            password='hello',
                            db='eth_smart_contract'
                        )
    cursor = db.cursor()
    userAddressList = []
    cntList = []

    sql = "SELECT s.GameName,s.smart_contract,count(DISTINCT s.controler) AS controlerCnt FROM ( SELECT DISTINCT a.smart_contract,game.GameName,a.controler FROM dapp_contract_hashkey_index AS a LEFT JOIN FomoTypeGame AS game ON game.SmartContractAddress=a.dapp_address WHERE game.GameName=\"%s\") s GROUP BY s.GameName,s.smart_contract ORDER BY controlerCnt DESC;" % gameName
    cursor.execute(sql)
    repetition = cursor.fetchall()
    userAddressList = [i[1] for i in repetition]
    cntList = [i[2] for i in repetition]
    
    result = pd.value_counts(cntList).sort_index()
    result.plot(figsize=(20,5))
    threshold=3
    mean_1 = np.mean(cntList)
    std_1 =np.std(cntList)
    
    outliers = []
    for index, y in enumerate(cntList):
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(index)
        
    gameAddressList = [userAddressList[i] for i in outliers]
    sql = "SELECT DISTINCT SmartContractAddress FROM FomoTypeGame WHERE GameName=\"%s\";" % gameName
    cursor.execute(sql)
    repetition = cursor.fetchall()
    for i in repetition:
        gameAddressList.append(i[0])
    db.close()
    return list(set(gameAddressList))

def getGameAddressList(gameName):
    # 返回需要处理的用户地址列表
    db = pymysql.connect(   host='localhost',
                            user='root',
                            password='hello',
                            db='eth_smart_contract'
                        )
    cursor = db.cursor()
    gameAddressList = []
    sql = "SELECT DISTINCT SmartContractAddress FROM FomoTypeGame WHERE GameName=\"%s\";" % gameName
    cursor.execute(sql)
    repetition = cursor.fetchall()
    gameAddressList = [i[0] for i in repetition]

    sql2 = "SELECT DISTINCT smart_contract FROM gameContract WHERE GameName=\"%s\";" % gameName
    cursor.execute(sql2)
    repetition2 = cursor.fetchall()
    for i in repetition2:
        gameAddressList.append(i[0])
    db.close()

    if len(gameAddressList) == 0:
            print("There is no smartcontract with dapp_name \"%s\" , Please check your gamename." % gameName)
    return list(set(gameAddressList))

def calcGraphSimilarity(graph1, graph2):
    # 计算图的相似度

    def select_k(spectrum, minimum_energy = 0.9):
        running_total = 0.0
        total = sum(spectrum)
        if total == 0.0:
            return len(spectrum)
        for i in range(len(spectrum)):
            running_total += spectrum[i]
            if running_total / total >= minimum_energy:
                return i + 1
        return len(spectrum)
    laplacian1 = nx.spectrum.laplacian_spectrum(graph1)
    laplacian2 = nx.spectrum.laplacian_spectrum(graph2)

    k1 = select_k(laplacian1)
    k2 = select_k(laplacian2)
    k = min(k1, k2)

    similarity = sum((laplacian1[:k] - laplacian2[:k])**2)
    return similarity

class_ = gm.VertexEdgeOverlap
comparator=class_()
comparator.set_attr_graph_used("nodetype", "edgetype")

def calcGraphSimilarityByGED(graph1, graph2):
    # 计算两个图的相似度
    matrix = comparator.compare([graph1,graph2],None)
    distance = comparator.distance(matrix)
    return (distance[0][1]+distance[1][0])/2.0 # 手动计算两个图的相似度

def graphConstructor(graphObj):
    graphObj = sorted(graphObj, key=operator.itemgetter(2, 5, 6)) # 排序
    # 根据图的json构建图结构
    node_dict = {}
    G = nx.DiGraph() # Graph DiGraph MultiDiGraph 有向图的基础类
    for links in graphObj: # graphObj 里每一行都是links
        if links[5] == "reference":
            continue
        if not links[3] in node_dict:
            node_dict[links[3]] = len(node_dict)
        if not links[4] in node_dict:
            node_dict[links[4]] = len(node_dict)

        G.add_node(node_dict[links[3]], nodetype = links[0]['group'])
        G.add_node(node_dict[links[4]], nodetype = links[1]['group'])
        G.add_edge(node_dict[links[3]], node_dict[links[4]], edgetype=(links[5] + str(links[-1])))    
    return G

def read_list(file_name):
    places = []
    with open(file_name, 'r') as filehandle:  
        for line in filehandle:
            places.append(line.strip("\"\n"))
    return places

def mkdir(path):
    try:
        folder = os.path.exists(path)
        if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
    except Exception as e:
        print(e)

def save_list(alist, filename="001.txt"):
    with open(filename, 'w') as f:
        for item in list(alist):
            f.write("%s\n" % item)


def getSingleGraphByTx(hashKey):
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    sql = "SELECT jsonContent FROM TransactionGraph_extendtx WHERE hashKey=\"%s\";"% hashKey
    cursor.execute(sql)
    repetition = cursor.fetchone()[0]
    db.close()
    return repetition

def getTxByEOA():
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    sql = "SELECT * FROM TransactionGraph_unknown;"
    cursor.execute(sql)
    repetition = cursor.fetchall()
    return repetition

if __name__ == "__main__":
    jsonContent = getSingleGraphByTx("0x86a0504cc8b20a5d00eeb75d6ab4c6bbcb51b18f4483dfade96aa978206f1e3c")
    j = json.loads(jsonContent)
    graph = graphConstructor(json.loads(jsonContent))

    repetition = getTxByEOA()

    for i in tqdm(repetition):
        jsonContent = str(i[1]).lower()
        if "PVP" in jsonContent or "pvp" in jsonContent or "0x5b1fef120" in jsonContent:
            print(i[0])