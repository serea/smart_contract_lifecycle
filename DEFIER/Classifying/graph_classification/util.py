from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os
import math
import networkx as nx
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import json, csv
from zipfile import ZipFile

import argparse, operator

from sqlalchemy import create_engine
import pymysql

cmd_opt = argparse.ArgumentParser(description='Argparser for graph_classification')
cmd_opt.add_argument('-mode', default='cpu', help='cpu/gpu')
cmd_opt.add_argument('-gm', default='mean_field', help='mean_field/loopy_bp')
cmd_opt.add_argument('-data', default=None, help='data folder name')
cmd_opt.add_argument('-batch_size', type=int, default=50, help='minibatch size')
cmd_opt.add_argument('-seed', type=int, default=1, help='seed')
cmd_opt.add_argument('-feat_dim', type=int, default=0, help='dimension of node feature')
cmd_opt.add_argument('-edge_feat_dim', type=int, default=0, help='dimension of edge feature')
cmd_opt.add_argument('-num_class', type=int, default=2, help='#classes')
cmd_opt.add_argument('-fold', type=int, default=1, help='fold (1..10)')
cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='number of epochs')
cmd_opt.add_argument('-latent_dim', type=int, default=64, help='dimension of latent layers')
cmd_opt.add_argument('-out_dim', type=int, default=1024, help='s2v output size')
cmd_opt.add_argument('-hidden', type=int, default=100, help='dimension of regression')
cmd_opt.add_argument('-max_lv', type=int, default=4, help='max rounds of message passing')
cmd_opt.add_argument('-learning_rate', type=float, default=0.0001, help='init learning_rate')
cmd_opt.add_argument('-trainset', type=str, default="", help='init training set')
cmd_opt.add_argument('-testset', type=str, default="", help='init test set')
cmd_opt.add_argument('-isvalidate', type=int, default=0, help='mode is validate')

cmd_args, _ = cmd_opt.parse_known_args()

print(cmd_args)

class S2VGraph(object):
    def __init__(self, g, gid, node_tags, edge_tags, label, user_node_id = None, dapp_node_id = None):
        self.num_nodes = len(node_tags)
        self.g = g
        self.gid = gid
        self.node_tags = node_tags
        self.edge_tags = edge_tags

        self.user_node_id = user_node_id
        self.dapp_node_id = dapp_node_id

        self.label = label

        x, y = zip(*g.edges())
        self.num_edges = len(x)        
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.int32)
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = self.edge_pairs.flatten()


def getTransactionJsonContent(db, transHash):
    cursor = db.cursor()
    try:
        sql = "SELECT jsonContent FROM `TransactionGraph_unknown` WHERE hashkey='%s';"%(transHash)
        cursor.execute(sql)
        repetition = cursor.fetchone()
        if repetition is None:
            sql = "SELECT jsonContent FROM `TransactionGraph_testset`  WHERE hashkey='%s';"%(transHash)
            cursor.execute(sql)
            repetition = cursor.fetchone()
            if repetition is None:
                sql = "SELECT jsonContent FROM `TransactionGraph_trainset`  WHERE hashkey='%s';"%(transHash)
                cursor.execute(sql)
                repetition = cursor.fetchone()
                if repetition is None:
                    sql = "SELECT jsonContent FROM `TransactionGraph`  WHERE hashkey='%s';"%(transHash)
                    cursor.execute(sql)
                    repetition = cursor.fetchone()
                    if repetition is None:
                        sql = "SELECT jsonContent FROM `TransactionGraph_selected` WHERE hashkey='%s';"%(transHash)
                        cursor.execute(sql)
                        repetition = cursor.fetchone()
        return repetition[0]
    except Exception as e:
        print('getTransactionJsonContent')
        print(transHash)
        print(e)
        db.rollback()
        return None

def getTransactionTag(db, transHash):
    cursor = db.cursor()
    try:
        sql = "SELECT GameAddress, GameName FROM TransactionTag AS tt WHERE hashkey='%s';"%(transHash)
        cursor.execute(sql)
        repetition = cursor.fetchone()
        if repetition is not None:
            return repetition
        else:
            return None
    except Exception as e:
        print(e)
        db.rollback()


def load_transaction_seq_data(seq_file, test_seq_file=[], isTemporal = False):
    seq_list = []
    db = pymysql.connect(host="localhost", user="root", password="", database="dapp_analysis_rearrange", port=3306, )
    dapp_df = pd.concat([pd.read_csv(f, header=0, sep=',', skipinitialspace=True) for f in seq_file],ignore_index=True)
    
    def typicalSampling(group, amount):
        frac = amount
        return group.sample(n=amount, frac=None, random_state=2, replace=True)
    dapp_df = dapp_df.groupby('label', group_keys=False).apply(typicalSampling, 2000)
    #dapp_df = dapp_df[(dapp_df['label']!=3)]
    #replacementstr = lambda x: 0 if x==4 else 1
    #dapp_df['label'] = dapp_df['label'].apply(replacementstr)
    dapp_df = dapp_df.drop_duplicates()
    print(dapp_df.describe())

    sql_nodetype="SELECT address, origin, killed from GameContract_all"
    cursor = db.cursor(pymysql.cursors.DictCursor)
    cursor.execute(sql_nodetype)
    repetition = cursor.fetchall()
    nodetype_df = pd.DataFrame(repetition)
    nodetype_df=nodetype_df.set_index(['address'])

    feat_dict = {}
    edge_feat_dict = {}
    for index, row in dapp_df.iterrows():
        g_list = []
        # print(row)
        for (gitem, g_user, g_game) in zip(json.loads(json.dumps(eval(row['txlist']))), json.loads(json.dumps(eval(row['controler']))), json.loads(json.dumps(eval(row['game']))),): #garphHashList[0:-1]
            node_dict = {}
            g_label = row['label'] # garphHashList[-1]
            G = nx.DiGraph() # Graph DiGraph MultiDiGraph
            G.label = g_label
            G.transID = gitem
            jsonContent = getTransactionJsonContent(db, gitem)
            #relatedGameInfo = getTransactionTag(db, gitem)
            
            if jsonContent is None:
                print(gitem+' not find graph in trainset')
                continue
            graphObj = json.loads(jsonContent.decode('utf-8'))
            if graphObj == []:
                continue
            graphObj = sorted(graphObj, key=operator.itemgetter(2, 5, 6)) 
            for links in graphObj:
                if not isTemporal:

                    if links[5] == "reference":
                        continue
                    if not links[3] in node_dict:
                        mapped = len(node_dict)
                        node_dict[links[3]] = mapped
                    
                    if links[3] in nodetype_df.index.values:
                        if nodetype_df.loc[links[3],'killed'] is not None:
                            node_type='killed'
                        elif nodetype_df.loc[links[3],'origin']=='0':
                            node_type='gameName'
                        elif nodetype_df.loc[links[3],'origin']=='1':
                            node_type='gameRelatedAddress'
                        elif nodetype_df.loc[links[3],'origin']=='2':
                            node_type='gameOwnerSmartContract'
                        else:
                            print("we could find origin of "+links[3])
                    else:
                        node_type = links[0][u'group'] if (links[0][u'label'].startswith('address') or links[0][u'label'].startswith('0x')) else links[0][u'label']
                    if not node_type in feat_dict:
                        mapped = len(feat_dict)
                        feat_dict[node_type] = mapped

                    G.add_node(node_dict[links[3]], nodetype = node_type)


                    if not links[4] in node_dict:
                        mapped = len(node_dict)
                        node_dict[links[4]] = mapped
                    if links[4] in nodetype_df.index.values:
                        if nodetype_df.loc[links[4],'killed'] is not None:
                            node_type='killed'
                        elif nodetype_df.loc[links[4],'origin']=='0':
                            node_type='gameName'
                        elif nodetype_df.loc[links[4],'origin']=='1':
                            node_type='gameRelatedAddress'
                        elif nodetype_df.loc[links[4],'origin']=='2':
                            node_type='gameOwnerSmartContract'
                        else:
                            print("we could find origin of "+links[4])
                    else:
                        node_type = links[0][u'group'] if (links[0][u'label'].startswith('address') or links[0][u'label'].startswith('0x')) else links[0][u'label']
                    if not node_type in feat_dict:
                        mapped = len(feat_dict)
                        feat_dict[node_type] = mapped

                    G.add_node(node_dict[links[4]], nodetype = node_type)

                    if links[2]=='transfer_to':
                        edge_type=len(str(links[6]))
                    else:
                        edge_type = links[5]
                    if not edge_type in edge_feat_dict:
                        mapped = len(edge_feat_dict)
                        edge_feat_dict[edge_type] = mapped

                    G.add_edge(node_dict[links[3]], node_dict[links[4]], edgetype = edge_type)
                elif isTemporal:
                    srcTuple = [str(links[3]), '0']
                    dstTuple = [str(links[4]), '0']
                    while '|'.join(srcTuple) in node_dict:
                        srcTuple[-1] = str(int(srcTuple[-1])+1)
                    while '|'.join(dstTuple) in node_dict:
                        dstTuple[-1] = str(int(dstTuple[-1])+1)

                    links[3] = '|'.join(srcTuple)
                    if not links[3] in node_dict:
                        mapped = len(node_dict)
                        node_dict[links[3]] = mapped
                        node_type = links[0][u'group']
                        if not node_type in feat_dict:
                            mapped = len(feat_dict)
                            feat_dict[node_type] = mapped
                    links[4] = '|'.join(dstTuple)
                    if not links[4] in node_dict:
                        mapped = len(node_dict)
                        node_dict[links[4]] = mapped
                        node_type = links[1][u'group']
                        if not node_type in feat_dict:
                            mapped = len(feat_dict)
                            feat_dict[node_type] = mapped
                    if links[3]=='transfer_to':
                            edge_type=len(str(links[6]))
                    else:
                        edge_type = links[5]
                    if not edge_type in edge_feat_dict:
                        mapped = len(edge_feat_dict)
                        edge_feat_dict[edge_type] = mapped

                    if int(srcTuple[-1]) > 0 and int(dstTuple[-1]) > 0 :
                        G.add_node(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1])-1)], nodetype = links[0][u'group'])
                        G.add_node(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1])-1)], nodetype = links[1][u'group'])
                        G.add_node(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], nodetype = links[0][u'group'])
                        G.add_node(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], nodetype = links[1][u'group'])
                        G.add_edge(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1])-1)], node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], edgetype = links[5])
                        G.add_edge(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1])-1)], node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], edgetype = links[5])
                        G.add_edge(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1])-1)], node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], edgetype = links[5])
                        G.add_edge(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1])-1)], node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], edgetype = links[5])
                    elif int(srcTuple[-1]) == 0 and int(dstTuple[-1]) > 0 :
                        G.add_node(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], nodetype = links[0][u'group'])
                        G.add_node(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], nodetype = links[1][u'group'])
                        G.add_node(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1])-1)], nodetype = links[1][u'group'])
                        G.add_edge(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1])-1)], node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], edgetype = links[5])
                        G.add_edge(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1])-1)], node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], edgetype = links[5])
                    elif int(srcTuple[-1]) > 0 and int(dstTuple[-1]) == 0 :
                        G.add_node(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1])-1)], nodetype = links[0][u'group'])
                        G.add_node(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], nodetype = links[0][u'group'])
                        G.add_node(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], nodetype = links[1][u'group'])
                        G.add_edge(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1])-1)], node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], edgetype = links[5])
                        G.add_edge(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1])-1)], node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], edgetype = links[5])
                    elif int(srcTuple[-1]) == 0 and int(dstTuple[-1]) == 0 :
                        G.add_node(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], nodetype = links[0][u'group'])
                        G.add_node(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], nodetype = links[1][u'group'])
                        G.add_edge(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], edgetype = links[5])
                      
            node_tags = []
            edge_tags = []      
            for (n, nt) in sorted(G.nodes(data=True)):
                node_tags.append(feat_dict[nt['nodetype']])

            sns, dns, attrs = zip(*G.edges(data=True))
            for attr in attrs:
                edge_tags.append(edge_feat_dict[attr['edgetype']])

            g_list.append(S2VGraph(G, G.transID, node_tags, edge_tags, g_label, node_dict[g_user] if g_user!='' and g_user is not None and g_user in node_dict.keys() else -1, node_dict[g_game] if g_game!='' and g_game is not None and g_game in node_dict.keys() else -1)) #edge_tags
        if len(g_list)>0:
            seq_list.append(g_list)

    test_seq_list = []
    if test_seq_file != []:
        dapp_df_test = pd.concat([pd.read_csv(f, header=0, sep=',', skipinitialspace=True) for f in test_seq_file],ignore_index=True)

        def typicalSampling(group, amount):
            frac = amount
            return group.sample(n=amount, frac=None, random_state=2, replace=True)
        #dapp_df_test = dapp_df_test.groupby('label', group_keys=False).apply(typicalSampling, 200)
        #dapp_df_test = dapp_df_test.drop_duplicates()
        print(dapp_df_test.describe())
        for index, row in dapp_df_test.iterrows():
            g_list = []
            for (gitem, g_user, g_game) in zip(json.loads(json.dumps(eval(row['txlist']))), json.loads(json.dumps(eval(row['controler']))), json.loads(json.dumps(eval(row['game']))),): #garphHashList[0:-1]
                node_dict = {}
                g_label = row['label'] # garphHashList[-1]
                G = nx.DiGraph() # Graph DiGraph MultiDiGraph
                G.label = g_label
                G.transID = gitem
                jsonContent = getTransactionJsonContent(db, gitem)
                
                #print(gitem)
                if jsonContent is None:
                    print(gitem +'not find graph in test set')
                    continue
                graphObj = json.loads(jsonContent.decode('utf-8'))
                if graphObj == []:
                    continue
                graphObj = sorted(graphObj, key=operator.itemgetter(2, 5, 6)) 
                for links in graphObj:
                    if not isTemporal:
                        if links[5] == "reference":
                            continue
                        if not links[3] in node_dict:
                            mapped = len(node_dict)
                            node_dict[links[3]] = mapped
                        if links[3] in nodetype_df.index.values:
                            if nodetype_df.loc[links[3],'killed'] is not None:
                                node_type='killed'
                            elif nodetype_df.loc[links[3],'origin']=='0':
                                node_type='gameName'
                            elif nodetype_df.loc[links[3],'origin']=='1':
                                node_type='gameRelatedAddress'
                            elif nodetype_df.loc[links[3],'origin']=='2':
                                node_type='gameOwnerSmartContract'
                            else:
                                print("we could find origin of "+links[3])
                        else:
                            node_type = links[0][u'group'] if (links[0][u'label'].startswith('address') or links[0][u'label'].startswith('0x')) else links[0][u'label']

                        if not node_type in feat_dict:
                            mapped = len(feat_dict)
                            feat_dict[node_type] = mapped

                        G.add_node(node_dict[links[3]], nodetype = node_type)

                        if not links[4] in node_dict:
                            mapped = len(node_dict)
                            node_dict[links[4]] = mapped
                        if links[4] in nodetype_df.index.values:
                            if nodetype_df.loc[links[4],'killed'] is not None:
                                node_type='killed'
                            elif nodetype_df.loc[links[4],'origin']=='0':
                                node_type='gameName'
                            elif nodetype_df.loc[links[4],'origin']=='1':
                                node_type='gameRelatedAddress'
                            elif nodetype_df.loc[links[4],'origin']=='2':
                                node_type='gameOwnerSmartContract'
                            else:
                                print("we could find origin of "+links[4])
                        else:
                            node_type = links[0][u'group'] if (links[0][u'label'].startswith('address') or links[0][u'label'].startswith('0x')) else links[0][u'label']

                        if not node_type in feat_dict:
                            mapped = len(feat_dict)
                            feat_dict[node_type] = mapped

                        G.add_node(node_dict[links[4]], nodetype = node_type)


                        if links[2]=='transfer_to':
                            edge_type=len(str(links[6]))
                        else:
                            edge_type = links[5]
                        if not edge_type in edge_feat_dict:
                            mapped = len(edge_feat_dict)
                            edge_feat_dict[edge_type] = mapped

                        G.add_edge(node_dict[links[3]], node_dict[links[4]], edgetype = edge_type)
                    
                    elif isTemporal:
                        srcTuple = [str(links[3]), '0']
                        dstTuple = [str(links[4]), '0']
                        while '|'.join(srcTuple) in node_dict:
                            srcTuple[-1] = str(int(srcTuple[-1])+1)
                        while '|'.join(dstTuple) in node_dict:
                            dstTuple[-1] = str(int(dstTuple[-1])+1)

                        links[3] = '|'.join(srcTuple)
                        if not links[3] in node_dict:
                            mapped = len(node_dict)
                            node_dict[links[3]] = mapped
                            node_type = links[0][u'group']
                            if not node_type in feat_dict:
                                mapped = len(feat_dict)
                                feat_dict[node_type] = mapped
                        links[4] = '|'.join(dstTuple)
                        if not links[4] in node_dict:
                            mapped = len(node_dict)
                            node_dict[links[4]] = mapped
                            node_type = links[1][u'group']
                            if not node_type in feat_dict:
                                mapped = len(feat_dict)
                                feat_dict[node_type] = mapped
                        edge_type = links[5]
                        if not edge_type in edge_feat_dict:
                            mapped = len(edge_feat_dict)
                            edge_feat_dict[edge_type] = mapped

                        if int(srcTuple[-1]) > 0 and int(dstTuple[-1]) > 0 :
                            G.add_node(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1])-1)], nodetype = links[0][u'group'])
                            G.add_node(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1])-1)], nodetype = links[1][u'group'])
                            G.add_node(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], nodetype = links[0][u'group'])
                            G.add_node(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], nodetype = links[1][u'group'])
                            G.add_edge(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1])-1)], node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], edgetype = links[5])
                            G.add_edge(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1])-1)], node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], edgetype = links[5])
                            G.add_edge(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1])-1)], node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], edgetype = links[5])
                            G.add_edge(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1])-1)], node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], edgetype = links[5])
                        elif int(srcTuple[-1]) == 0 and int(dstTuple[-1]) > 0 :
                            G.add_node(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], nodetype = links[0][u'group'])
                            G.add_node(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], nodetype = links[1][u'group'])
                            G.add_node(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1])-1)], nodetype = links[1][u'group'])
                            G.add_edge(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1])-1)], node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], edgetype = links[5])
                            G.add_edge(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1])-1)], node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], edgetype = links[5])
                        elif int(srcTuple[-1]) > 0 and int(dstTuple[-1]) == 0 :
                            G.add_node(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1])-1)], nodetype = links[0][u'group'])
                            G.add_node(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], nodetype = links[0][u'group'])
                            G.add_node(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], nodetype = links[1][u'group'])
                            G.add_edge(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1])-1)], node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], edgetype = links[5])
                            G.add_edge(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1])-1)], node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], edgetype = links[5])
                        elif int(srcTuple[-1]) == 0 and int(dstTuple[-1]) == 0 :
                            G.add_node(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], nodetype = links[0][u'group'])
                            G.add_node(node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], nodetype = links[1][u'group'])
                            G.add_edge(node_dict[srcTuple[0]+'|'+str(int(srcTuple[-1]))], node_dict[dstTuple[0]+'|'+str(int(dstTuple[-1]))], edgetype = links[5])
                        
                node_tags = []
                edge_tags = []         
                for (n, nt) in sorted(G.nodes(data=True)):
                    node_tags.append(feat_dict[nt['nodetype']])

                sns, dns, attrs = zip(*G.edges(data=True))
                for attr in attrs:
                    edge_tags.append(edge_feat_dict[attr['edgetype']])

                g_list.append(S2VGraph(G, G.transID, node_tags, edge_tags, g_label, node_dict[g_user] if g_user!='' and g_user is not None and g_user in node_dict.keys() else -1, node_dict[g_game] if g_game!='' and g_game is not None and g_game in node_dict.keys() else -1)) #edge_tags
            if len(g_list)>0:
                test_seq_list.append(g_list)

    cmd_args.feat_dim = len(feat_dict)
    cmd_args.edge_feat_dim = len(edge_feat_dict)
    print('# classes: %d' % cmd_args.num_class)
    print('# nodes: %d ' % len(node_dict))
    print('# node features: %d' % cmd_args.feat_dim)
    print('# edge features: %d' % cmd_args.edge_feat_dim)
    
    return seq_list, test_seq_list
    