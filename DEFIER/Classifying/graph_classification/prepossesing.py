# coding:utf-8
import time
import os, sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import pymysql
import datetime
import ast



def addLabel(dataset_ori,datasetlabel,outputdatasetset):
    db = pymysql.connect("localhost","root","","dapp_analysis_rearrange")
    cursor = db.cursor()
    datasetset_list=[]
    input_df = pd.concat([pd.read_csv(dataset_ori+f, header=0, sep='\n', skipinitialspace=True,delimiter=",") for f in os.listdir(dataset_ori)],ignore_index=True)

    for index,row in input_df.iterrows():

        if row['controler'] is np.nan or row['gameAddress'] is np.nan:
            controler_list=[]
            gameAddress_list=[]
            for item in row['txlist'].replace(' ','').split(','):
                sql = "SELECT sender,txGameAddress FROM TransactionDescription_testset_extend where hashKey='%s';"%item
                cursor.execute(sql)
                txObj = cursor.fetchall()
                if len(txObj)==0:
                    break
                txObj=txObj[0]
                controler = txObj[0]
                controler_list.append(controler)
                gameAddress = txObj[1].split(',')[0]
                gameAddress_list.append(gameAddress)
            tx_dict={"txlist":row['txlist'].replace(' ','').split(','),"controler":controler_list,"gameAddress":gameAddress_list,"label":datasetlabel}
        else:
            tx_dict={"txlist":row['txlist'].replace(' ','').split(','),"controler":row['controler'].replace(' ','').split(','),"gameAddress":row['gameAddress'].replace(' ','').split(','),"label":datasetlabel}
        
        datasetset_list.append(tx_dict)

    datasetset_df = pd.DataFrame(datasetset_list)
    print(datasetset_df.shape)
    datasetset_df.to_csv(outputdatasetset,index=None)
    return datasetset_df

def slideWin(input_df,win,outputgoodset):
    print(input_df.shape)
    txDes=input_df
    label0_df = txDes[txDes['label']==0]
    label1_df = txDes[txDes['label']==1]
    label2_df = txDes[txDes['label']==2]
    label3_df = txDes[txDes['label']==3]
    label4_df = txDes[txDes['label']==4]
    print("number of 0: "+str(len(label0_df)))
    print("number of 1: "+str(len(label1_df)))
    print("number of 2: "+str(len(label2_df)))
    print("number of 3: "+str(len(label3_df)))
    print("number of 4: "+str(len(label4_df)))
    new_txlist = []
    for index,row in txDes.iterrows():

        if len(row['txlist'])==1:
            continue
        if row["controler"] is np.nan:
            row["controler"] = []
        if row["gameAddress"] is np.nan:
            row["gameAddress"]=[]

        if len(row["txlist"])>win:
            for index in range(len(row["txlist"])-win+1):
                tx_label_dict = {"txlist":row["txlist"][index:index+win], "label":row["label"], "controler":row["controler"][index:index+win], "gameAddress":row["gameAddress"][index:index+win]}
                new_txlist.append(tx_label_dict)
        else:
            tx_label_dict = {"txlist":row["txlist"], "label":row["label"], "controler":row["controler"], "controler":row["controler"], "gameAddress":row["gameAddress"]}
            new_txlist.append(tx_label_dict)
    txDes_new = pd.DataFrame(new_txlist)        
    txDes_new.to_csv(outputgoodset[:-4]+"_"+str(win)+".csv",index=False)
    label0_df = txDes_new[txDes_new['label']==0]
    label1_df = txDes_new[txDes_new['label']==1]
    label2_df = txDes_new[txDes_new['label']==2]
    label3_df = txDes_new[txDes_new['label']==3]
    label4_df = txDes_new[txDes_new['label']==4]
    print("number of 0: "+str(len(label0_df)))
    print("number of 1: "+str(len(label1_df)))
    print("number of 2: "+str(len(label2_df)))
    print("number of 3: "+str(len(label3_df)))
    print("number of 4: "+str(len(label4_df)))

def removeTransfer(inputfile,outputbadset):
    db = pymysql.connect("localhost","root","","dapp_analysis_rearrange")
    cursor = db.cursor()
    badset_df = pd.read_csv(inputfile, header=0, skipinitialspace=True, low_memory=False, encoding='UTF-8')
    print(badset_df.head())
    for index,row in badset_df.iterrows():
        row['txlist']=row["txlist"].replace(' ','').split(',')
        newtx=[]
        for tx in row['txlist']:
            try:
                sql_stage = "SELECT stage FROM ExtendTx WHERE hashKey='%s'"%tx
                cursor.execute(sql_stage)
                stage = cursor.fetchone()
                stage = stage[0]
                if stage!=0:
                    newtx=row['txlist']
                    break

                sql_select = "SELECT jsonContent FROM TransactionGraph_trainset WHERE hashKey='%s'"%tx
                cursor.execute(sql_select)
                graph = cursor.fetchone()
                graph = json.loads(graph[0].decode('utf-8'))
                if not istransfer(graph):
                    newtx.append(tx)
            except Exception as e:
                print(e)
                db.rollback() # 如果发生错误则回滚
                db.close()
        if len(newtx)==0:
            print(row)
            badset_df.drop(labels=None, axis=0, index=index, inplace=True)
        else:
            if(len(row['txlist'])!=len(newtx)):
                print(len(row['txlist']))
                print(len(newtx))
            badset_df.loc[index,'txlist']=','.join(newtx)
    badset_df.to_csv(outputbadset)
    db.close()
    return badset_df

def labelUnknown(labelFile,resultTable):
    db = pymysql.connect("localhost","root","","dapp_analysis_rearrange")
    cursor = db.cursor()
    columns=['label','txlist']
    label_df = pd.read_csv(labelFile, header=None, names=columns, skipinitialspace=True, low_memory=False, encoding='UTF-8')
    print(label_df.head())
    for index,row in label_df.iterrows():
        row['txlist']=ast.literal_eval(row["txlist"])
        row['label'] = row['label'][8]
        # print(row['txlist'],row['label'])
        for tx in row['txlist']:
            try:
                sql_update = "UPDATE %s set `label`='%s' WHERE hashKey='%s'"%(resultTable,row['label'],tx)
                print(sql_update)
                cursor.execute(sql_update)
                db.commit()
            except Exception as e:
                print(e)
                db.rollback() # 如果发生错误则回滚
                db.close()
    db.close()

def statisticUnknownResult(resultTable,detailTabel):
    game_list=["0xUniverse","1000Guess","Ace Dice","Airswap","Auctionity","Augur","Axie-Infinity","bingo4beast long official (b4b)","BlackJack","Blockchain-Cuties","Bloom","Chibi-Fighters","CityMayor","CoinFlip","Compound","Crypt-Oink","cryptoflowers","CryptoKitties","CryptoRoulette","DAO","dice for slice","dice2.win","dicegame","diceroll.app","Edgeless","ETH-Town","ethcrystal fomo","ethdice","Ether-Kingdoms","EtherCartel","Etheremon","EtherLotto","Etheroll","EtherPot","Ethraffle_v4b","EthStick","FCK-2","FirePonzi","fomo lightning","fomo short","fomo super","fomo2d","fomo3d","fomo3d loop","fomo3dlong","fomo4d","fomo4d chess","fomo777","fomofast","fomogame","fomojp","fomolol","fomosports","fomowar","fomoﬁve","full fomo","Gods-Unchained","GoodFellas","GovernMental Ponzi Scheme","grand theft fomo","HyperDragons","ICO","imfomo","infinity","koreanfomo3d","LastIsMe","lastwinner","lucky dice","LuckyBlock","LuckyDoubler","MegaCryptoPolis","Minds","MLB-Crypto-Baseball","multiplayer dice online","My-Crypto-Heroes","notfomo3d","Numerai","OpenAddressLottery","origin-protocol","play0x","playtowin-io","powerofbubble","powh-3d","ratscam","Roulette","RussianRoulette","slotthereum","SmartBillions","smartdice","souha","spacewar","StackyGame","STOX","supercard","switcheo-2","TheRun","unnamed1","unnamed2","unnamed3","unnamed4","vdize","Wings","world fomo","xether-2"]
    game_detail_list=[]
    db = pymysql.connect("localhost","root","","dapp_analysis_rearrange")
    cursor = db.cursor(pymysql.cursors.DictCursor)
    print('get connect')
    sql_create_tabel="create table %s as SELECT k.*, td.txDate, td.seed, td.sender, td.senderType, td.txMethod, td.isError, td.txGameName, td.txGameAddress, td.txLabelNew FROM %s AS k LEFT JOIN TransactionDescription_testset_extend AS td ON k.hashKey = td.hashKey WHERE k.label is not NULL ORDER BY k.note, k.`group`, k.label" %(detailTabel,resultTable)
    
    # 获取每个游戏每个打标的hash个数
    sql_all="SELECT count(distinct(hashKey)),note,label FROM %s where label is not NULL group by note,label order by note,label"%detailTabel
    # 获取有真实标签的，每个游戏每个打标的hash个数
    sql_label="SELECT count(distinct(hashKey)),note,label FROM %s where txLabelNew is not NULL group by note,label order by note,label"%detailTabel
    # 获取每个游戏每个真实标签对应的hash个数
    sql_txLabelNew="SELECT count(distinct(hashKey)),note,txLabelNew FROM %s where txLabelNew is not NULL group by note,txLabelNew"%detailTabel
    
    # 获取每个游戏每个打标的group个数
    sql_all_group="SELECT count(DISTINCT(`group`)),note,label FROM %s where label is not NULL group by note,label order by note,label"%detailTabel
    # 获取有真实标签的，每个游戏每个打标的group个数
    sql_label_group="SELECT count(DISTINCT(`group`)),note,label FROM %s where txLabelNew is not NULL group by note,label order by note,label"%detailTabel
    # 获取每个游戏每个真实标签对应的group个数
    sql_txLabelNew_group="SELECT count(DISTINCT(`group`)),note,txLabelNew FROM %s where txLabelNew is not NULL group by note,txLabelNew order by note,label"%detailTabel

    try:
        print(sql_create_tabel)
        cursor.execute(sql_create_tabel)
        db.commit()
        print(sql_all)
        cursor.execute(sql_all)
        repetition_all = cursor.fetchall()
        repetition_all=pd.DataFrame(repetition_all)
        print(repetition_all.head())
        repetition_all.to_csv('./repetition_all.csv')
        # repetition_all = pd.read_csv('./repetition_all.csv')

        print(sql_label)
        cursor.execute(sql_label)
        repetition_label = cursor.fetchall()
        repetition_label=pd.DataFrame(repetition_label)
        print(repetition_label.head())
        repetition_label.to_csv('./repetition_label.csv')
        # repetition_label = pd.read_csv('./repetition_label.csv')

        print(sql_txLabelNew)
        cursor.execute(sql_txLabelNew)
        repetition_txLabelNew = cursor.fetchall()
        repetition_txLabelNew=pd.DataFrame(repetition_txLabelNew)
        print(repetition_txLabelNew.head())
        repetition_txLabelNew.to_csv('./repetition_txLabelNew.csv')
        # repetition_txLabelNew = pd.read_csv('./repetition_txLabelNew.csv')

        print(sql_all_group)
        cursor.execute(sql_all_group)
        repetition_all_group = cursor.fetchall()
        repetition_all_group=pd.DataFrame(repetition_all_group)
        print(repetition_all_group.head())
        repetition_all_group.to_csv('./repetition_all_group.csv')
        # repetition_all_group = pd.read_csv('./repetition_all_group.csv')

        print(sql_label_group)
        cursor.execute(sql_label_group)
        repetition_label_group = cursor.fetchall()
        repetition_label_group=pd.DataFrame(repetition_label_group)
        print(repetition_label_group.head())
        repetition_label_group.to_csv('./repetition_label_group.csv')
        # repetition_label_group = pd.read_csv('./repetition_label_group.csv')

        print(sql_txLabelNew_group)
        cursor.execute(sql_txLabelNew_group)
        repetition_txLabelNew_group = cursor.fetchall()
        repetition_txLabelNew_group=pd.DataFrame(repetition_txLabelNew_group)
        print(repetition_txLabelNew_group.head())
        repetition_txLabelNew_group.to_csv('./repetition_txLabelNew_group.csv')
        # repetition_txLabelNew_group = pd.read_csv('./repetition_txLabelNew_group.csv')

        # read from csv, the label type is int, read from sql, the label type is str.
        for item in tqdm(game_list):
            print(item)
            #------------compute the tx number with classification result label
            all_df=repetition_all[repetition_all['note']==item]
            label_0=all_df[all_df['label']=='0']['count(distinct(hashKey))'].values[0] if all_df[all_df['label']=='0']['count(distinct(hashKey))'].values else None
            label_1=all_df[all_df['label']=='1']['count(distinct(hashKey))'].values[0] if all_df[all_df['label']=='1']['count(distinct(hashKey))'].values else None
            label_2=all_df[all_df['label']=='2']['count(distinct(hashKey))'].values[0] if all_df[all_df['label']=='2']['count(distinct(hashKey))'].values else None
            label_3=all_df[all_df['label']=='3']['count(distinct(hashKey))'].values[0] if all_df[all_df['label']=='3']['count(distinct(hashKey))'].values else None
            label_4=all_df[all_df['label']=='4']['count(distinct(hashKey))'].values[0] if all_df[all_df['label']=='4']['count(distinct(hashKey))'].values else None

            #------------compute the classification result of tx with known label
            label_df=repetition_label[repetition_label['note']==item]
            label_true_0=label_df[label_df['label']=='0']['count(distinct(hashKey))'].values[0] if label_df[label_df['label']=='0']['count(distinct(hashKey))'].values else None
            label_true_1=label_df[label_df['label']=='1']['count(distinct(hashKey))'].values[0] if label_df[label_df['label']=='1']['count(distinct(hashKey))'].values else None
            label_true_2=label_df[label_df['label']=='2']['count(distinct(hashKey))'].values[0] if label_df[label_df['label']=='2']['count(distinct(hashKey))'].values else None
            label_true_3=label_df[label_df['label']=='3']['count(distinct(hashKey))'].values[0] if label_df[label_df['label']=='3']['count(distinct(hashKey))'].values else None
            label_true_4=label_df[label_df['label']=='4']['count(distinct(hashKey))'].values[0] if label_df[label_df['label']=='4']['count(distinct(hashKey))'].values else None

            #------------compute the number tx with the specific known label
            txLabelNew_df=repetition_txLabelNew[repetition_txLabelNew['note']==item]
            txLabelNew_0=txLabelNew_df[txLabelNew_df['txLabelNew']=='0']['count(distinct(hashKey))'].values[0] if txLabelNew_df[txLabelNew_df['txLabelNew']=='0']['count(distinct(hashKey))'].values else None
            txLabelNew_1=txLabelNew_df[txLabelNew_df['txLabelNew']=='1']['count(distinct(hashKey))'].values[0] if txLabelNew_df[txLabelNew_df['txLabelNew']=='1']['count(distinct(hashKey))'].values else None
            txLabelNew_2=txLabelNew_df[txLabelNew_df['txLabelNew']=='2']['count(distinct(hashKey))'].values[0] if txLabelNew_df[txLabelNew_df['txLabelNew']=='2']['count(distinct(hashKey))'].values else None
            txLabelNew_3=txLabelNew_df[txLabelNew_df['txLabelNew']=='3']['count(distinct(hashKey))'].values[0] if txLabelNew_df[txLabelNew_df['txLabelNew']=='3']['count(distinct(hashKey))'].values else None
            txLabelNew_4=txLabelNew_df[txLabelNew_df['txLabelNew']=='4']['count(distinct(hashKey))'].values[0] if txLabelNew_df[txLabelNew_df['txLabelNew']=='4']['count(distinct(hashKey))'].values else None


            #------------compute the group number with classification result label
            all_group_df=repetition_all_group[repetition_all_group['note']==item]
            label_group_0=all_group_df[all_group_df['label']=='0']['count(DISTINCT(`group`))'].values[0] if all_group_df[all_group_df['label']=='0']['count(DISTINCT(`group`))'].values else None
            label_group_1=all_group_df[all_group_df['label']=='1']['count(DISTINCT(`group`))'].values[0] if all_group_df[all_group_df['label']=='1']['count(DISTINCT(`group`))'].values else None
            label_group_2=all_group_df[all_group_df['label']=='2']['count(DISTINCT(`group`))'].values[0] if all_group_df[all_group_df['label']=='2']['count(DISTINCT(`group`))'].values else None
            label_group_3=all_group_df[all_group_df['label']=='3']['count(DISTINCT(`group`))'].values[0] if all_group_df[all_group_df['label']=='3']['count(DISTINCT(`group`))'].values else None
            label_group_4=all_group_df[all_group_df['label']=='4']['count(DISTINCT(`group`))'].values[0] if all_group_df[all_group_df['label']=='4']['count(DISTINCT(`group`))'].values else None

            #------------compute the classification result of group with known label
            label_group_df=repetition_label_group[repetition_label_group['note']==item]
            label_group_true_0=label_group_df[label_group_df['label']=='0']['count(DISTINCT(`group`))'].values[0] if label_group_df[label_group_df['label']=='0']['count(DISTINCT(`group`))'].values else None
            label_group_true_1=label_group_df[label_group_df['label']=='1']['count(DISTINCT(`group`))'].values[0] if label_group_df[label_group_df['label']=='1']['count(DISTINCT(`group`))'].values else None
            label_group_true_2=label_group_df[label_group_df['label']=='2']['count(DISTINCT(`group`))'].values[0] if label_group_df[label_group_df['label']=='2']['count(DISTINCT(`group`))'].values else None
            label_group_true_3=label_group_df[label_group_df['label']=='3']['count(DISTINCT(`group`))'].values[0] if label_group_df[label_group_df['label']=='3']['count(DISTINCT(`group`))'].values else None
            label_group_true_4=label_group_df[label_group_df['label']=='4']['count(DISTINCT(`group`))'].values[0] if label_group_df[label_group_df['label']=='4']['count(DISTINCT(`group`))'].values else None

            #------------compute the number of group with the specific known label
            txLabelNew_group_df=repetition_txLabelNew_group[repetition_txLabelNew_group['note']==item]
            txLabelNew_group_0=txLabelNew_group_df[txLabelNew_group_df['txLabelNew']=='0']['count(DISTINCT(`group`))'].values[0] if txLabelNew_group_df[txLabelNew_group_df['txLabelNew']=='0']['count(DISTINCT(`group`))'].values else None
            txLabelNew_group_1=txLabelNew_group_df[txLabelNew_group_df['txLabelNew']=='1']['count(DISTINCT(`group`))'].values[0] if txLabelNew_group_df[txLabelNew_group_df['txLabelNew']=='1']['count(DISTINCT(`group`))'].values else None
            txLabelNew_group_2=txLabelNew_group_df[txLabelNew_group_df['txLabelNew']=='2']['count(DISTINCT(`group`))'].values[0] if txLabelNew_group_df[txLabelNew_group_df['txLabelNew']=='2']['count(DISTINCT(`group`))'].values else None
            txLabelNew_group_3=txLabelNew_group_df[txLabelNew_group_df['txLabelNew']=='3']['count(DISTINCT(`group`))'].values[0] if txLabelNew_group_df[txLabelNew_group_df['txLabelNew']=='3']['count(DISTINCT(`group`))'].values else None
            txLabelNew_group_4=txLabelNew_group_df[txLabelNew_group_df['txLabelNew']=='4']['count(DISTINCT(`group`))'].values[0] if txLabelNew_group_df[txLabelNew_group_df['txLabelNew']=='4']['count(DISTINCT(`group`))'].values else None


            game_detail_dict={'gameName':item, 'label_0':label_0, 'label_1':label_1, 'label_2':label_2, 'label_3':label_3, 'label_4':label_4,
            'label_true_0':label_true_0,'label_true_1':label_true_1,'label_true_2':label_true_2,'label_true_3':label_true_3,'label_true_4':label_true_4,
            'txLabelNew_0':txLabelNew_0,'txLabelNew_1':txLabelNew_1,'txLabelNew_2':txLabelNew_2,'txLabelNew_3':txLabelNew_3,'txLabelNew_4':txLabelNew_4,
            'label_group_0':label_group_0,'label_group_1':label_group_1,'label_group_2':label_group_2,'label_group_3':label_group_3,'label_group_4':label_group_4,
            'label_group_true_0':label_group_true_0,'label_group_true_1':label_group_true_1,'label_group_true_2':label_group_true_2,'label_group_true_3':label_group_true_3,'label_group_true_4':label_group_true_4,
            'txLabelNew_group_0':txLabelNew_group_0,'txLabelNew_group_1':txLabelNew_group_1,'txLabelNew_group_2':txLabelNew_group_2,'txLabelNew_group_3':txLabelNew_group_3,'txLabelNew_group_4':txLabelNew_group_4}
            # print(game_detail_dict)
            game_detail_list.append(game_detail_dict)

        game_detail_df = pd.DataFrame(game_detail_list)
        game_detail_df.to_csv('./statisticUnknownResult.csv')
        game_detail_df.to_excel('./statisticUnknownResult.xlsx')

    except Exception as e:
        print(e)
        db.rollback() # 如果发生错误则回滚

    db.close()

if __name__ == "__main__":

    beforeClassify=1
    if beforeClassify==1:
        """In prepossesing module, we gather goodset_threshold3 and slide window of these clusters.
        
        Args:
            files: ./goodset_threshold3/
        """
        goodsetlabel=0
        goodset_ori="./data/goodset_threshold3/"
        outputgoodset="./data/goodset.csv"
        goodset=addLabel(goodset_ori,goodsetlabel,outputgoodset)
        wins=[5,8,10]
        for win in wins:
            slideWin(goodset,win,outputgoodset)

        """In prepossesing module, we remove the transfer transaction of badset_ori and slide window of these clusters.
        
        Args:
            files: ./badset_ori
        """
        badset_ori="./data/badset_ori.csv"
        outputbadset="./data/badset_notransfer.csv"
        badset = removeTransfer(badset_ori,outputbadset)
        wins=[5,8,10]
        for win in wins:
            slideWin(badset,win,outputbadset)

        """In prepossesing module, we remove the transfer transaction of badset_ori and slide window of these clusters.
        
        Args:
            files: ./badset_ori
        """
        unknownsetlabel=0
        unknownset_ori="./data/0724-allgame/"
        for dic in os.listdir(unknownset_ori):
            if dic[-4:]=='game':
                outputunknownset='./data/'+dic+".csv"
                unknwonset=addLabel(unknownset_ori+dic+'/',unknownsetlabel,outputunknownset)
    else:
        """After classification, we label the result into Database.
        
        Args:
            resultDictionary: ./result/
        """
        # resultDictionary='./result/'
        resultTable='KMeans_0908'
        for (root, dirs, files) in os.walk(os.path.dirname(resultDictionary)):
            for file in tqdm(files):
                if file[-4:]=='.csv' and os.path.getsize(root+'/'+file)!=0:  
                    labelUnknown(root+'/'+file,resultTable)

        """After classification, we generate result detail table on Database and statistic result.
        
        Args: resultTable
        """
        detailTabel='game_104_result_'+resultTable[7:]
        print(detailTabel)
        statisticUnknownResult(resultTable,detailTabel)

    