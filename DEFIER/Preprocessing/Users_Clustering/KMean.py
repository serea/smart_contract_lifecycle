#!/usr/bin/env python
# -*- coding: utf-8 -*- 
from tqdm import tqdm
from sqlalchemy import create_engine
from ..baseFunction import read_list, save_list, graphConstructor, calcGraphSimilarityByGED, getSingleGraphByTx
import math
import matplotlib.pyplot as plt
import random
from collections import Counter
import networkx as nx
import gmatch4py as gm
import matplotlib.pyplot as plt
import zipfile, os, sys
import json
import pymysql
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import normalize

class_ = gm.VertexEdgeOverlap
comparator=class_()
comparator.set_attr_graph_used("nodetype", "edgetype")


class KMean:
    def __init__(self):
        """Attributes
            delta -- 判定两点重合的最小距离
            distance -- 计算两点距离
            center --- 计算一组点的中心
        """
        self.delta = 0.001
        self.maxIterTimes = 3
        self.timePeriod = 10 # hour unit
        self.alpha = 0.9
        self.beta = 0.1
        self.points = None
        self.k = None
        self.lenLimit = None
        self.maxLenLimit = 100
        self.extendThreshold = 3

    def getAllRepInDB(self, game, mode="testset"):
        db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
        cursor = db.cursor()
        if mode == "testset":
            sql = "SELECT t.hashKey,t.txDate,a.jsonContent FROM TransactionDescription_testset_extend AS t LEFT JOIN TransactionGraph_unknown AS a ON t.hashKey=a.hashKey WHERE t.gameName=\"%s\" and t.similarity < %d ORDER BY txDate;" % (game, self.extendThreshold)
        elif mode == "goodset":
            sql = "SELECT t.hashKey,t.txDate,a.jsonContent FROM TransactionDescription_goodset_extend AS t LEFT JOIN TransactionGraph_goodset AS a ON t.hashKey=a.hashKey WHERE t.gameName=\"%s\" and t.similarity < %d ORDER BY txDate;" % (game, self.extendThreshold)
        elif mode == "trainset":
            sql = "select * from (SELECT e.hashKey,e.txDate,gt.jsonContent FROM TransactionGraph_trainset AS gt LEFT JOIN (SELECT hashKey,txDate FROM TransactionDescription_trainset_extend_test WHERE seedGame=\"%s\") AS e ON e.hashKey=gt.hashKey WHERE e.hashKey IS NOT NULL UNION ALL SELECT e.hashKey,e.txDate,gt.jsonContent FROM TransactionGraph_trainset AS gt LEFT JOIN (SELECT hashKey,txDate FROM ExtendTx WHERE seedGame=\"%s\") AS e ON e.hashKey=gt.hashKey WHERE e.hashKey IS NOT NULL ) as a order by txDate;" % (game, game)
            
        cursor.execute(sql)
        repetition = cursor.fetchall()
        repDict = dict()
        cnt = 0
        for r in repetition:
            try:
                repDict[r[0]] = (r[1].timestamp(), graphConstructor(json.loads(r[2])))
            except Exception as e:
                cnt += 1
                continue
        db.close()
        print("共有%d个无法获取图的tx"% cnt)
        return repDict

    def distance(self, r1, r2):
        graphDistance   = self.getGraphDistance(r1, r2)
        dateDistance    = self.getDateDistance(r1, r2)
        distance = self.alpha * graphDistance + self.beta * dateDistance
        return distance

    def getGraphDistance(self, r1, r2):
        return calcGraphSimilarityByGED(r1[1], r2[1]) # 返回的是距离，值越小表示越相似

    def getDateDistance(self, r1, r2):
        return abs((r1[0]-r2[0])/3600) # 除以小时的总秒数，值越小表示时间越相近

    def getDateCenter(self, dateList):
        return sum(map(datetime.timestamp, dateList)) / len(dateList)

    def get_date_dis_matrix(self, dates):
        matrix = np.zeros(shape=(len(dates),len(dates)))
        for idx1 in range(0, len(dates)):
            d1 = dates[idx1]
            for idx2 in range(idx1+1, len(dates)):
                d2 = dates[idx2]
                dis = abs((d1-d2)/3600)
                matrix[idx1, idx2] = dis
                matrix[idx2, idx1] = dis
        
        return normalize(matrix, axis=1, norm='l1') # 做l1归一化

    def _calcu_dis_matrix_of_groups(self, group):
        graphs = [self.points[hashKey][1] for hashKey in group]
        graph_dis_matrix = comparator.distance(comparator.compare(graphs, None))

        dates = [self.points[hashKey][0] for hashKey in group]
        date_dis_matrix  = self.get_date_dis_matrix(dates)

        matrix = self.alpha * graph_dis_matrix + self.beta * date_dis_matrix
        return matrix

    def center(self, group):
        """计算中心点
        
        Args:
            group ([type]): [需要寻找中心点的group]
        
        Returns:
            [type]: [中心点的hashkey]
        """
        matrix = self._calcu_dis_matrix_of_groups(group)
        avgList = np.average(matrix, axis=1)
        return group[np.argmin(avgList)]
        
    def run(self):
        """执行划分，并返回划分的信息，直接调用此方法返回最终结果
        Param:
            points -- 需要划分的点的数组[p1,p2,p3,...]
            k -- 参数k，分成k个区域
            seeds -- 种子点的数组[s1,s2,...]，seeds中的点和points中的点为统一数据类型。len(seeds) == k
        Return: 
            返回划分好的节点列表，类型如：[(p1,p2,..),(p3,p4,..),..]
        """   
        # 初始化获得K值、center信息和对group进行初次分类
        lenLimit = 4
        k, seeds, groups = self._getKandCenter(lenLimit)
        iter = 0
        while k < 2:
            print('while lenLimit = %d, timePeriod = %d K <2, recreate K' % (lenLimit, self.timePeriod))
            if int(lenLimit/2) == 0:
                lenLimit = lenLimit
            else:
                lenLimit = int(lenLimit/2)
            self.timePeriod = int(self.timePeriod/2)
            k, seeds, groups = self._getKandCenter(lenLimit)
            iter += 1
            if iter > 10:
                raise KMeanError("please check the data ... we can't handle it...")
        
        self.k = k
        self.lenLimit = lenLimit

        #迭代种子
        iterSeeds = seeds
        #是否继续迭代
        flag = True 
        iterTimes = 1
        while flag and iterTimes < self.maxIterTimes:
            print("\n # %d 次迭代" % iterTimes)
            print(" ## 重新归类每一个tx")
            groups = self._runOnce( k, iterSeeds, groups)
            tmpSeeds = []
            print(" ## 重新计算每一个cluster里的中心点")
            for group in tqdm(groups): # 计算每个group的中心点
                if len(group) != 0:
                    tmpSeeds.append(self.center(group))
                else:
                    tmpSeeds.append(iterSeeds[groups.index(group)])
            # 判断group的聚合程度是否已足够紧（上一次和这一次的中心点差距不大）
            flag = False
            for i in range(len(iterSeeds)):
                if self.distance(self.points[iterSeeds[i]], self.points[tmpSeeds[i]]) > self.delta:
                    flag = True
                    break
            iterSeeds = tmpSeeds
            iterTimes += 1
        return groups, iterSeeds

    def _runOnce(self, k, seeds, groups):
        """执行一次划分，并返回分类的信息
        Param:
            points -- 需要划分的点的数组[p1,p2,p3,...],数组的每个元素都要实现distance方法来确定和另外一个点的距离
                如：p1.distance(p2)返回p1 到 p2 的距离
            k -- 参数k，分成k个区域
            seeds -- 种子点的数组[s1,s2,...]，seeds中的点和points中的点为统一数据类型。len(seeds) == k
        Return: 
            返回划分好的节点列表，类型如：[[p1_index,p2_index,..],[p3_index,p4_index,..],..],每个划分里存储的是位置
        """
        if not k == len(seeds):
            raise KMeanError('k must equals len(seeds)')
        #存放划分结果的数组，长度为k，初始化都为空
        new_groups = [[] for x in range(k)]

        # 对每一个group中的每个hashkey，和前中后3个group的中心点做距离计算，判断pos是这三个中的谁
        for g_idx in tqdm(range(k)):
            for hashKey in groups[g_idx]:
                candidateSeeds = [ seeds[idx] for idx in self._get_g_range(g_idx, k)]
                g_hash = self._minDistancePoint( hashKey, candidateSeeds)
                pos = seeds.index(g_hash)
                new_groups[pos].append(hashKey)
        return new_groups
    
    def _get_g_range(self, g_idx, group_len):
        if g_idx == 0: # 第一个
            return range(g_idx, g_idx+2)
        elif g_idx == group_len-1: # 最后一个
            return range(g_idx-1, g_idx+1)
        else:
            return range(g_idx-1, g_idx+2)

    def _minDistancePoint(self, hashKey, candidateSeeds):
        """找出points中离point最近的点
        Return
            最近的点所在的位置.
        """    
        #minDistance: 目前最近的点的距离
        #minPos: 目前最近点在数组中的位置
        candidateSeeds.append(hashKey)
        matrix = self._calcu_dis_matrix_of_groups(candidateSeeds)
        line = matrix[-1][:-1]
        return candidateSeeds[np.argmin(line)]
         
    def _getKandCenter(self, lenLimit):
        """初始化，得到K值，center和group的初步分类
        
        Args:
            point_list ([type]): [description]
        
        Returns:
            [type]: [description]
        """
        point_list_argu = []
        # point_list虽然是dict，但还是按照txdate的顺序排的
        # 第一个hash
        beginHash = list(self.points.keys())[0]
        beginDate = self.points[beginHash][0]
        point_list_argu.append(beginHash)

        # 找到后面的center，并进行第一次group划分
        nowDate = beginDate
        nextExpectedDate = nowDate + self.timePeriod*3600
        groups = []
        nowGroup = []
        for hashKey in tqdm(self.points.keys()):
            
            if self.points[hashKey][0] >= nowDate and self.points[hashKey][0] < nextExpectedDate and len(nowGroup) < self.maxLenLimit:
                nowGroup.append(hashKey)
            elif self.points[hashKey][0] >= nowDate and self.points[hashKey][0] < nextExpectedDate and len(nowGroup) >= self.maxLenLimit: # 超过设置的最大边界了，也重新创一个新的group
                groups.append(nowGroup)
                nowGroup = []
                nowGroup.append(hashKey)
                point_list_argu.append(hashKey) # 下一个group的center
            elif self.points[hashKey][0] >= nextExpectedDate and len(nowGroup) < lenLimit: # 现在group里的数量太少，再继续扩充
                nowDate = nextExpectedDate
                nextExpectedDate = nowDate + self.timePeriod*3600
                nowGroup.append(hashKey)
            elif self.points[hashKey][0] >= nextExpectedDate: # 超出当前的时间范围了，创建一个新的nowGroup
                groups.append(nowGroup)
                nowGroup = []
                nowDate = nextExpectedDate
                nextExpectedDate = nowDate + self.timePeriod*3600
                nowGroup.append(hashKey)
                point_list_argu.append(hashKey) # 下一个group的center
        groups.append(nowGroup)            
        return len(groups), point_list_argu, groups

    def saveTxlistToDB(self, groups, groupName, seeds, dataset_type):
        pymysql.converters.encoders[np.float64] = pymysql.converters.escape_float
        pymysql.converters.conversions = pymysql.converters.encoders.copy()
        pymysql.converters.conversions.update(pymysql.converters.decoders)
        db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
        cursor = db.cursor()
        updateSql = "INSERT INTO KMeans(hashKey, `group`, note, similarity, graph_similarity, date_similarity, k, lenLimit, timePeriod, dataset_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);"
        print(" ## 存入数据库")
        for idx in tqdm(range(len(groups))):
            for g in groups[idx]:
                try:
                    r = (   g, idx, groupName, 
                            self.distance(self.points[g], self.points[seeds[idx]]),
                            self.getGraphDistance(self.points[g], self.points[seeds[idx]]),
                            self.getDateDistance(self.points[g], self.points[seeds[idx]]),
                            self.k, self.lenLimit, self.timePeriod, dataset_type
                        )
                    cursor.execute(updateSql, r)
                    db.commit()
                except Exception as e:
                    db.rollback()
                    print(e)
                    print(" hashkey is %s" % g)
                    continue

        db.close()  
        return

    def evaluate(self, note, dataset_type = "trainset"):
        df = self.getHashLabelBySeedGame(note, dataset_type)
        group_result, cluster_result = self._accuracy_in_clutser(df)
        
        result = self._evaluate_in_whole_situation(group_result, note, dataset_type)
        avg = np.average(result['accuracy'].tolist())
        temp = pd.DataFrame([(cluster_result, None, None, None, None, avg), (None, None, None, None, None, None)], columns=['note', 'dataset_type', 'stage', 'accuracy', 'true_cnt', 'all_cnt'])
        result = pd.concat([result, temp], ignore_index=True)

        return result

    def _accuracy_in_clutser(self, df):
        accuracyList = []
        group_result = []
        groupSet = set(df['predict_label'].tolist())
        for g in groupSet:
            stage = df[df['predict_label'] == g]['true_label'].tolist()
            c = Counter(stage).most_common(1)[0]
            focus_stage = c[0]
            single_accuracy = c[1]/len(stage)
            accuracyList.append(single_accuracy)
            group_result.append([g, focus_stage, True if single_accuracy >= 0.8 else False, single_accuracy])
        cluster_result = " accuracy in clutser : Mean(%f), Max(%f), Min(%f)" % (sum(accuracyList)/len(accuracyList), max(accuracyList), min(accuracyList))
        print(cluster_result + "\n")
        return pd.DataFrame(group_result, columns=["g", "focus_cluster", "label", "single_accuracy"]), cluster_result

    def _evaluate_in_whole_situation(self, df, note, dataset_type):
        l = []
        for s in sorted(list(set(df['focus_cluster'].tolist()))):
            true_cnt = len(df[(df['focus_cluster'] == s) & (df['label'] == True)])
            all_cnt = len(df[(df['focus_cluster'] == s)])
            accuracy = true_cnt / all_cnt
            print(" note \"%s\" stage %s accuracy = %f (%d / %d)" % (note, s, accuracy, true_cnt, all_cnt))
            l.append((note, dataset_type, s, accuracy, true_cnt, all_cnt))
        result = pd.DataFrame(l, columns=['note', 'dataset_type', 'stage', 'accuracy', 'true_cnt', 'all_cnt'])
        return result

    def getHashLabelBySeedGame(self, note, dataset_type):
        db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
        cursor = db.cursor()
        sql = "SELECT hashKey, true_label, predict_label FROM (SELECT y.hashKey,y.stage AS true_label,y.txDate,p.`group` AS predict_label FROM ExtendTx AS y LEFT JOIN KMeans AS p ON y.hashKey=p.hashKey WHERE seedGame=\"%s\" and p.dataset_type =\"%s\" UNION ALL SELECT y.hashKey,y.stage AS true_label,y.txDate,p.`group` AS predict_label FROM TransactionDescription_trainset_extend_test AS y LEFT JOIN KMeans AS p ON y.hashKey=p.hashKey WHERE seedGame=\"%s\"  and p.dataset_type =\"%s\" ) AS a ORDER BY true_label,txDate;" % (note, dataset_type, note, dataset_type)
        cursor.execute(sql)
        repetition = cursor.fetchall()
        db.close()

        df = pd.DataFrame(list(repetition), columns=["hashKey", "true_label", "predict_label"])
        return df

    def getAllResultInDB(self, note):
        db = pymysql.connect('localhost', 'root', 'hello', 'dapp_analysis_rearrange')
        cursor = db.cursor()
        sql = "SELECT e.hashKey,k1.`group`, stage,txDate FROM ExtendTx AS e LEFT JOIN KMeans AS k1 ON e.hashKey=k1.hashKey WHERE k1.hashKey IS NOT NULL AND note=\"" + note + "\" ORDER BY e.txDate;"
        cursor.execute(sql)
        repetition = cursor.fetchall()
        db.close()
        return [i for i in repetition]


class KMeanError(Exception):
    """
        Attributes:
            message -- 错误信息            
    """
    def __init__(self,message = ''):
        self.message = message