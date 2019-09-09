# smart_contract_lifecycle

## Introduction

This is the repository for paper "Evil Under the Sun: Discovering and Understanding Attacks on Ethereum Decentralized Applications".

## Run

Envs: python3.6

```
cd smart_contract_lifecycle
pip install -r requirements.txt
```
- `Measurement` contains code of getting transaction data and transaction description data of trainset(including meansurement part and goodset), testset.
- `DEFIER` includes code of our model DEFIER
 -`cluster` clustering the transactions based on the transaction distance
 -`classification` classifying the clusters
