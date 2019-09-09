# -*- coding: utf-8 -*-
import os, sys
import pymysql
from sqlalchemy import create_engine
    
def getAllABIInDB():
    # 获取数据表中全部的hex2method
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

def getAbiDict():
    db = pymysql.connect(   host='localhost',
                            user='root',
                            password='hello',
                            db='dapp_analysis_rearrange'
                        )
    cursor = db.cursor()
    sql = "SELECT text_signature, hex_signature FROM MethodABI;"
    cursor.execute(sql)
    abiTuple = cursor.fetchall()
    db.close()

    abiDict = dict()
    for i in abiTuple:
        if i[1] not in abiDict:
            abiDict[i[1]] = [i[0]]
        else:
            abiDict[i[1]].append(i[0])

    return abiDict

if __name__ == "__main__":
    abiTuple = getAllABIInDB()
    abiDict = getAbiDict()