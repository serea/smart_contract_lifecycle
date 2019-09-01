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
from ..baseFunction import read_list, save_list, graphConstructor, calcGraphSimilarityByGED, getSingleGraphByTx, mkdir

import KMean

if 'Kmeans' not in os.getcwd():
    os.chdir("./DEFIER/kmeans-clustering-master/Kmeans")


def getDiffResultNameInDB(note):
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    sql = "SELECT distinct dataset_type FROM KMeans where note = '%s';" % note
    cursor.execute(sql)
    repetition = cursor.fetchall()
    db.close()
    return [i[0] for i in repetition]

def getGroupName(mode = "testset"):
    """[get the list of handle-required game]
    
    Args:
        mode (str, optional): [pick testset or goodset dataset]. Defaults to "testset".
    
    Returns:
        [list]: [handle-required game list under this mode]
    """
    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    if mode == "testset":
        sql = "SELECT DISTINCT gameName FROM TransactionDescription_testset_extend ORDER BY gameName;"
    elif mode == "goodset":
        sql = "SELECT DISTINCT gameName FROM TransactionDescription_goodset_extend ORDER BY gameName;"
    else:
        raise("don't have this mode, please check...")
    cursor.execute(sql)
    repetition = cursor.fetchall()
    return [i[0] for i in repetition]

def ignoreExchanges(games):
    exchanges = ['IDEX', 'Kyber-Network', 'Local-Ethereum', 'ForkDelta', 'wibson','Token-Store', 'Bancor', 'radar-relay', 'FunFair', 'SONM', "foam-map", "ethfinex-trustless"]
    games = [i for i in games if i not in exchanges]

    return games


def PackageUnknownSet(note):
    """In cluster module, such as `testset()`, we can manually set the name of the cluster result, e.g. \"unknownset-threshold3\".
        And here, you can use this module to package sets belonging to your cluster name `note`
    
    Args:
        note ([type]): [The cluster's name, such as "unknown", "unknownset-threshold3"]
    """
    games = ignoreExchanges(getGroupName())

    db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
    cursor = db.cursor()
    for game in tqdm(games):
        try:
            params = (game, note)
            sql_1 = "SET group_concat_max_len=99999999999999999;"
            sql = "SELECT GROUP_CONCAT(sender ORDER BY txDate ASC SEPARATOR ', ') AS controler,GROUP_CONCAT(gameName ORDER BY txDate ASC SEPARATOR ', ') AS gameName,GROUP_CONCAT(gameAddress ORDER BY txDate ASC SEPARATOR ', ') AS gameAddress,GROUP_CONCAT(hashKey ORDER BY txDate ASC SEPARATOR ', ') AS txlist,count(*) AS cnt FROM (SELECT k.hashKey,k.`group`,t.txDate,t.sender,t.gameName,t.gameAddress FROM KMeans AS k LEFT JOIN TransactionDescription_testset_extend AS t ON k.hashKey=t.hashKey WHERE k.note=\"%s\" AND k.dataset_type=\"%s\") AS a GROUP BY a.`group`;" % params
            cursor.execute(sql_1)
            cursor.execute(sql)
            repetition = cursor.fetchall()
        except Exception as e:
            print(e)
            print(game)
            continue
        
        try:
            df = pd.DataFrame(list(repetition), columns=["controler", "gameName", "gameAddress", "txlist", "cnt"])
            mkdir("./result/" + note + "/")
            df.to_csv("./result/" + note + "/" + game + ".csv", index=False)
            print(" # 保存游戏 %s 的结果成功" % game)
        except Exception as e:
            print(e)
            print(" ERROR! game %s " % game)

    db.close()
    return 


def goodset():
    groupList = getGroupName()
    
    kmean = KMean.KMean()
    for g in groupList:
        params = (g, kmean.alpha, kmean.beta, kmean.maxIterTimes, kmean.maxLenLimit)
        print("\n\n# handle campaign %s : alpha = %f, beta = %f, maxIterSeeds = %d, maxLenLimit = %d" % params)
        kmean.points = kmean.getAllRepInDB(g, mode = "goodset")
        groups, iterSeeds = kmean.run()
        kmean.saveTxlistToDB(groups, g, iterSeeds, "goodset-threshold3")


def trainset():
    groupList = getGroupName()
    # alphaList = [0.1, 0.5, 0.9, 0.01, 0.3, 0.7, 0.99]
    # betaList = [0.9, 0.5, 0.1, 0.99, 0.7, 0.3, 0.01]
    alphaList = [0.9]
    betaList = [0.1]

    kmean = KMean.KMean()
    for g in groupList:
        for idx in range(len(alphaList)):
            kmean.alpha = alphaList[idx]
            kmean.beta = betaList[idx]
            kmean.maxIterTimes = 6
            kmean.maxLenLimit = 50
            params = (g, kmean.alpha, kmean.beta, kmean.maxIterTimes, kmean.maxLenLimit)
            print("\n\n# handle campaign %s : alpha = %f, beta = %f, maxIterSeeds = %d, maxLenLimit = %d" % params)
            kmean.points = kmean.getAllRepInDB(g, mode="trainset")
            groups, iterSeeds = kmean.run()
            kmean.saveTxlistToDB(groups, g, iterSeeds, "-".join([str(i) for i in params]))
    
    # evaluation result
    whole_result = pd.DataFrame(columns=['note', 'dataset_type', 'stage', 'accuracy', 'true_cnt', 'all_cnt'])
    for g in groupList:
        diff_result = getDiffResultNameInDB(g)
        for r in diff_result:
            if r == "trainset":
                continue
            params = tuple(r.split("-"))
            if len(params) < 4:
                continue
            if params[3] != "6":
                continue
            print("\n# %s Params Setting: alpha = %s, beta = %s, maxIterSeeds = %s, maxLenLimit = %s" % params)
            result = kmean.evaluate(g, r)
            whole_result = pd.concat([whole_result, result], ignore_index=True)
    
    # save evaluation result
    whole_result.to_excel("kmeans-result-iter1.xlsx", index=False)



def testset():
    groupList = getGroupName()
    groupList = ignoreExchanges(groupList)

    kmean = KMean.KMean()
    run_result = []
    col = ("game", "time", "clusterCnt", "hashCnt", "alpha", "beta", "maxIterTimes", "maxLenLimit", "lenLimit", "timePeriod")

    for g in groupList:
        params = (g, kmean.alpha, kmean.beta, kmean.maxIterTimes, kmean.maxLenLimit)
        print("\n\n# handle campaign %s : alpha = %f, beta = %f, maxIterSeeds = %d, maxLenLimit = %d" % params)
        t = time.time()
        kmean.points = kmean.getAllRepInDB(g, mode="testset")
        if len(kmean.points) > 300000:
            continue
        groups, iterSeeds = kmean.run()
        kmean.saveTxlistToDB(groups, g, iterSeeds, "unknownset-threshold3")
        t2 = time.time()
        print("\nDuration: "+str(t2-t))
        run_result.append((g, t2-t, kmean.k, len(kmean.points), kmean.alpha, kmean.beta, kmean.maxIterTimes, kmean.maxLenLimit, kmean.lenLimit, kmean.timePeriod))
        df = pd.DataFrame(run_result, columns=col)
        df.to_csv("run_result.csv", index=False)


if __name__ == "__main__":
    t = time.time()
    # trainset()
    # goodset()
    testset()
    PackageUnknownSet("unknownset-threshold3")
    t2 = time.time()
    print("\nDuration: "+str(t2-t))



