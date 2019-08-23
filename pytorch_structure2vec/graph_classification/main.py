from __future__ import print_function
import sys, os
import csv
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
import random
import torch
import itertools

sys.path.append('%s/../s2v_lib' % os.path.dirname(os.path.realpath(__file__)))
from embedding import EmbedMeanField, EmbedLoopyBP
from mlp import MLPClassifier, LSTMTagger

from util import cmd_args, load_data, load_transaction_json_data, load_user_and_transaction_json_data, load_transaction_seq_data

from sklearn import metrics
from sklearn.metrics import (auc, classification_report, roc_curve, zero_one_loss)
from sklearn import preprocessing
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm

import matplotlib.pyplot as plt
from scipy import interp
from itertools import cycle

from node2vec import Node2Vec

from datetime import datetime

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        if cmd_args.gm == 'mean_field':
            model = EmbedMeanField
        elif cmd_args.gm == 'loopy_bp':
            model = EmbedLoopyBP
        else:
            print('unknown gm %s' % cmd_args.gm)
            sys.exit()

        out_dim = cmd_args.out_dim
        if out_dim == 0:
            out_dim = cmd_args.latent_dim

        self.s2v = model(latent_dim=cmd_args.latent_dim, 
                output_dim=cmd_args.out_dim,
                num_node_feats=cmd_args.feat_dim, 
                num_edge_feats=cmd_args.edge_feat_dim,
                max_lv=cmd_args.max_lv)

        self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class)
        #self.mlp = LSTMTagger(embedding_dim=out_dim, hidden_dim=cmd_args.hidden, vocab_size=9999, target_size=cmd_args.num_class)

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0
        n_edges = 0
        concat_feat = []
        concat_edge_feat = []
        for i in range(len(batch_graph)):
            labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            n_edges += batch_graph[i].num_edges
            if batch_graph[i].node_tags is not None:
                concat_feat += batch_graph[i].node_tags
            if batch_graph[i].edge_tags is not None:
                concat_edge_feat += batch_graph[i].edge_tags
                concat_edge_feat += batch_graph[i].edge_tags

        node_feat = None
        if len(concat_feat) != 0:
            concat_feat = torch.LongTensor(concat_feat).view(-1, 1)
            node_feat = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_feat.scatter_(1, concat_feat, 1)


        edge_feat = None
        if len(concat_edge_feat) != 0:
            concat_edge_feat = torch.LongTensor(concat_edge_feat).view(-1, 1)
            edge_feat = torch.zeros(n_edges * 2, cmd_args.edge_feat_dim)
            edge_feat.scatter_(1, concat_edge_feat, 1)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda() 
            edge_feat = edge_feat.cuda() 
            labels = labels.cuda()

        return node_feat, edge_feat, labels

    def forward(self, batch_graph): 
        node_feat, edge_feat, labels = self.PrepareFeatureLabel(batch_graph)
        embed = self.s2v(batch_graph, node_feat, edge_feat)

        return self.mlp(embed, labels)


class SeqClassifier(nn.Module):
    def __init__(self):
        super(SeqClassifier, self).__init__()
        out_dim = cmd_args.out_dim
        if out_dim == 0:
            out_dim = cmd_args.latent_dim

        self.s2v = EmbedMeanField(latent_dim=cmd_args.latent_dim, 
                output_dim=cmd_args.out_dim,
                num_node_feats=cmd_args.feat_dim, 
                num_edge_feats=cmd_args.edge_feat_dim,
                max_lv=cmd_args.max_lv)

        #self.mlp = MLPClassifier(input_size=out_dim, hidden_size=cmd_args.hidden, num_class=cmd_args.num_class)
        self.lstm = LSTMTagger(embedding_dim=out_dim, hidden_dim=cmd_args.hidden, vocab_size=128*20, target_size=cmd_args.num_class)
        

    def PrepareFeatureLabel(self, batch_graph):
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0
        n_edges = 0
        concat_feat = []
        concat_edge_feat = []
        for i in range(len(batch_graph)):
            #labels[i] = batch_graph[i].label
            n_nodes += batch_graph[i].num_nodes
            n_edges += batch_graph[i].num_edges
            if batch_graph[i].node_tags is not None:
                concat_feat += batch_graph[i].node_tags
            if batch_graph[i].edge_tags is not None:
                concat_edge_feat += batch_graph[i].edge_tags
                concat_edge_feat += batch_graph[i].edge_tags

        node_feat = None
        if len(concat_feat) != 0:
            concat_feat = torch.LongTensor(concat_feat).view(-1, 1)
            node_feat = torch.zeros(n_nodes, cmd_args.feat_dim)
            node_feat.scatter_(1, concat_feat, 1)

        edge_feat = None
        if len(concat_edge_feat) != 0:
            concat_edge_feat = torch.LongTensor(concat_edge_feat).view(-1, 1)
            edge_feat = torch.zeros(n_edges * 2, cmd_args.edge_feat_dim)
            edge_feat.scatter_(1, concat_edge_feat, 1)

        if cmd_args.mode == 'gpu':
            node_feat = node_feat.cuda() 
            edge_feat = edge_feat.cuda() 
            labels = labels.cuda()

        return node_feat, edge_feat, labels

    def forward(self, batch_graph):
        embed_list = []
        embed_node_eoa_list = []
        embed_node_d_list = []
        embed_list_temp_file = []
        labels = torch.LongTensor(len(batch_graph), 1) #.random_() % 2
        #labels_one_hot = torch.zeros(len(batch_graph), 2).scatter_(1, labels, 1)
        i = 0
        g_list = []
        
        batch_size = len(batch_graph)
        embed_length = []
        max_length = 0
        for graphseq in batch_graph:
            if len(graphseq)>max_length:
                max_length = len(graphseq)

        for graphseq in batch_graph:
            if len(graphseq)<=0:
                continue
            embed_temp = None
            embed_node_eoa_temp = None
            embed_node_d_temp = None
            # if graphseq[0].label < 4 and graphseq[0].label > 0:
            if len(graphseq)>0:
                labels[i] = graphseq[0].label
                # print(graphseq[0])
                # print(graphseq[0].label)
                # print(type(graphseq[0].label))
                # print(labels[i])
                i += 1
            #node_feat, edge_feat, _ = self.PrepareFeatureLabel(graphseq)
            #embed = self.s2v(graphseq, node_feat, edge_feat)
            #embed_list_temp_file.append([[embed.detach().numpy().tolist()],graphseq[0].label])

            #for e in embed:
            #    embed_temp = embed_temp+e if embed_temp is not None else e
            '''
            for g in graphseq:
                node_feat, edge_feat, _ = self.PrepareFeatureLabel([g])
                embed = self.s2v([g], node_feat, edge_feat)
                
                for e in embed:
                    embed_temp = torch.cat([embed_temp,e], 0) if embed_temp is not None else e
            
            embed_list.append(embed_temp.detach().numpy())

            for g in graphseq:
                node_feat, edge_feat, _ = self.PrepareFeatureLabel([g])
                embed = self.s2v([g], node_feat, edge_feat)
                
                for e in embed:
                    embed_temp = torch.cat([embed_temp,e], 0) if embed_temp is not None else e
            embed_list.append(embed_temp)
            '''

            # Node2Vec
            for g in graphseq:
                node2vec = Node2Vec(g.g, dimensions=64, walk_length=10, num_walks=10, workers=4)  # walk_length=10, num_walks=10
                node2vec_model = node2vec.fit(window=5, min_count=1, batch_words=4)  # Any keywords acceptable by gensim.Word2Vec can be passed, `diemnsions` and `workers` are automatically passed (from the Node2Vec constructor)
                
                ae_user = np.zeros(shape=(64))
                if g.user_node_id != -1:
                    ae_user = node2vec_model.wv[str(g.user_node_id)]

                ae_dapp = np.zeros(shape=(64))
                if g.dapp_node_id != -1:
                    ae_dapp = node2vec_model.wv[str(g.dapp_node_id)]

                embed_node_eoa = torch.FloatTensor(ae_user).unsqueeze(0)
                embed_node_d = torch.FloatTensor(ae_dapp).unsqueeze(0)

                embed_node_eoa_temp = torch.cat((embed_node_eoa_temp, embed_node_eoa), 0) if embed_node_eoa_temp is not None else embed_node_eoa
                embed_node_d_temp = torch.cat((embed_node_d_temp, embed_node_d), 0) if embed_node_d_temp is not None else embed_node_d
                
                node_feat, edge_feat, _ = self.PrepareFeatureLabel([g])
                embed_struc = self.s2v([g], node_feat, edge_feat)
                embed_temp = torch.cat((embed_temp, embed_struc), 0) if embed_temp is not None else embed_struc
            
            embed_node_eoa_temp = np.lib.pad(embed_node_eoa_temp.detach().numpy(), pad_width=((0,max_length-len(embed_temp)),(0,0)), mode='constant', constant_values=((0,-1),(-1,-1)))
            embed_node_eoa_list.append(embed_node_eoa_temp.tolist())

            embed_node_d_temp = np.lib.pad(embed_node_d_temp.detach().numpy(), pad_width=((0,max_length-len(embed_temp)),(0,0)), mode='constant', constant_values=((-1,-1),(-1,-1)))
            embed_node_d_list.append(embed_node_d_temp.tolist())
            
            embed_length.append(len(embed_temp))
            embed_temp = np.lib.pad(embed_temp.detach().numpy(), pad_width=((0,max_length-len(embed_temp)),(0,0)), mode='constant', constant_values=((-1,-1),(-1,-1)))
            embed_list.append(embed_temp.tolist())

            #embed_atten_temp = np.lib.pad(embed_atten_temp.detach().numpy(), pad_width=((0,max_length-len(embed_temp)),(0,0)), mode='constant', constant_values=((-1,-1),(-1,-1)))
            #embed_atten_list.append(embed_atten_temp.tolist())

        #embed_list = list(itertools.zip_longest(*embed_list, fillvalue = -1))
        embed_list = torch.FloatTensor(embed_list)
        embed_node_eoa_list = torch.FloatTensor(embed_node_eoa_list)
        embed_node_d_list = torch.FloatTensor(embed_node_d_list)

        #print(sorted_seq_lengths)
        # print(embed_list)
        # print(embed_list.size())
        # print(len(sorted_seq_lengths.cpu().numpy()))
        #embed_pack = torch.nn.utils.rnn.pack_padded_sequence(input=embed_list, lengths=np.array(embed_length), batch_first=False, enforce_sorted=False)

        #return self.mlp(torch.Tensor(embed_list), labels.squeeze())
        return self.lstm(np.array(embed_length), batch_size, embed_list, labels, embed_node_eoa_list, embed_node_d_list) #.long()


def loop_dataset(g_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]

        batch_graph = [g_list[idx] for idx in selected_idx]
        logits, loss, acc = classifier(batch_graph)
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        if optimizer is None:
            pred = logits.data.max(1, keepdim=True)[1]

        loss = loss.data.cpu().item()
        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )

        total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)
    total_loss = np.array(total_loss)
    avg_loss = np.sum(total_loss, 0) / n_samples
    return avg_loss

def load_model_and_test(test_graphs):
    classifier = Classifier()
    classifier.load_state_dict(torch.load('./epoch-best-mlp0119.model', map_location=lambda storage, loc: storage))

    logits, loss, acc = classifier(test_graphs)
    pred = logits.data.max(1, keepdim=True)[1]
    y_test = [test_graphs[idx].label for idx in range(len(test_graphs))]
    allTransList = [test_graphs[idx].gid for idx in range(len(test_graphs))]
    print(classification_report(y_test, pred, digits = 5))
    print(metrics.confusion_matrix(y_test, pred, labels=None, sample_weight=None))

    '''
    with open("./fp1128.loopy1.csv","a+") as fpcsvfile, open("./fn1128.loopy1.csv","a+") as fncsvfile: 
        for (yt,yp,transHash) in zip(y_test, pred, allTransList):
            if yp==1 and yt == 0:
                csv.writer(fpcsvfile).writerow([transHash])
            elif yp==0 and yt == 1:
                csv.writer(fncsvfile).writerow([transHash])
    '''

def main(train_graphs, validate):
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    #train_graphs, validate = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(validate)))

    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()
    
    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        avg_loss = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1]))
        
        test_loss = loop_dataset(validate, classifier, list(range(len(validate))))
        print('\033[93maverage test of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1]))

        if best_loss is None or test_loss[0] < best_loss:
            best_loss = test_loss[0]
            print('----saving to best model since this is the best valid loss so far.----')
            torch.save(classifier.state_dict(), './epoch-best-mlp0119.model')
            #save_args(cmd_args.save_dir + '/epoch-best-args.pkl', cmd_args)


def loop_seqdataset_for_lstm(seq_list, classifier, sample_idxes, optimizer=None, bsize=cmd_args.batch_size):
    total_loss = []
    total_iters = (len(sample_idxes) + (bsize - 1) * (optimizer is None)) // bsize
    pbar = tqdm(range(total_iters), unit='batch')

    n_samples = 0
    for pos in pbar:
        selected_idx = sample_idxes[pos * bsize : (pos + 1) * bsize]
        batch_graph = [seq_list[idx] for idx in selected_idx]

        logits, loss, acc = classifier(batch_graph)
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        if optimizer is None:
            pred = logits.data.max(1, keepdim=True)[1]

        pbar.set_description('loss: %0.5f acc: %0.5f' % (loss, acc) )

        total_loss.append(np.array([loss, acc]) * len(selected_idx))

        n_samples += len(selected_idx)
    if optimizer is None:
        assert n_samples == len(sample_idxes)

    avg_loss = np.sum(total_loss, 0) / n_samples
    return avg_loss

def load_lstm_model_and_test(test_graphs):
    classifier = SeqClassifier()
    classifier.load_state_dict(torch.load('./epoch-test-lstm.model', map_location=lambda storage, loc: storage))
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    logits, loss, acc = classifier(test_graphs)
    pred_prob = logits.data.max(1, keepdim=True)[0]
    pred = logits.data.max(1, keepdim=True)[1]

    if cmd_args.mode == 'gpu':
        pred = pred.cpu()
 
    y_test = [test_graphs[idx][0].label for idx in range(len(test_graphs))]

    print(classification_report(y_test, pred, digits = 5))
    print(metrics.confusion_matrix(y_test, pred, labels=None, sample_weight=None))

    #false_positive_rate, recall, thresholds = roc_curve(y_test, pred_prob)
    #roc_auc = auc(false_positive_rate, recall)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_one_hot = label_binarize(y_test, np.arange(cmd_args.num_class))
    for i in range(cmd_args.num_class):
        fpr[i], tpr[i], _ = roc_curve(y_one_hot[:, i], logits.data[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_one_hot.ravel(), logits.data.detach().numpy().ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(cmd_args.num_class)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(cmd_args.num_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= cmd_args.num_class
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    lw=2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','chocolate','olive'])
    for i, color in zip(range(cmd_args.num_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


    allTransList = []
    for idx in range(len(test_graphs)):
        allTransList.append([test_graphs[idx][idxx].gid for idxx in range(len(test_graphs[idx]))])

    with open("./result.csv","a+") as resfile:
        for (yt,yp,transHash) in zip(y_test, pred, allTransList):
            csv.writer(resfile).writerow([yp, transHash])
    # with open("./fp.seq.csv","a+") as fpcsvfile, open("./fn.seq.csv","a+") as fncsvfile, open("./label-0.seq.csv","a+") as label0file, open("./label-1.seq.csv","a+") as label1file: 
    #     for (yt,yp,transHash) in zip(y_test, pred, allTransList):
    #         if yp==1 and yt == 0:
    #             csv.writer(fpcsvfile).writerow([transHash])
    #         elif yp==0 and yt == 1:
    #             csv.writer(fncsvfile).writerow([transHash])
    #         elif yp==0 and yt == 0:
    #             csv.writer(label0file).writerow([transHash])
    #         elif yp==1 and yt == 1:
    #             csv.writer(label1file).writerow([transHash])


def main_lstm(train_graphs, validate):
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    #train_graphs, validate = load_data()
    print('# train: %d, # test: %d' % (len(train_graphs), len(validate)))

    classifier = SeqClassifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)
    #optimizer = optim.SGD(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    best_loss = None
    for epoch in range(cmd_args.num_epochs):
        random.shuffle(train_idxes)
        avg_loss = loop_seqdataset_for_lstm(train_graphs, classifier, train_idxes, optimizer=optimizer)
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, avg_loss[0], avg_loss[1]))
        
        test_loss = loop_seqdataset_for_lstm(validate, classifier, list(range(len(validate))))
        print('\033[93maverage test of epoch %d: loss %.5f acc %.5f\033[0m' % (epoch, test_loss[0], test_loss[1]))

        if best_loss is None or test_loss[0] < best_loss:
            best_loss = test_loss[0]
            print('----saving to best model since this is the best valid loss so far.----')
            torch.save(classifier.state_dict(), './epoch-test-lstm.model')

if __name__ == '__main__':
    #'./data/splited_2b_nodup.csv'
    
    '''
    # './data/splited_2b_nodup.csv',    './data/splited_23b_nodup.csv',
    train_graphs, test_graphs = load_transaction_json_data(['./data/splited_1125_report_nodup.csv','./data/splited_2b_nodup.csv', ], ['./data/splited_1124_confirm_nodup.csv','./data/splited_23b_nodup.csv',], '/Users/yaoyepeng/Project/contract_analysis/')
    train, validate = train_test_split(train_graphs, test_size=0.1, random_state=1)
    #main(train, validate)
    load_model_and_test(test_graphs)
    '''

    #test_graphs = load_user_and_transaction_json_data(['/Users/yaoyepeng/Project/contract_analysis/positiveUser_temp_212.csv',], '/Users/yaoyepeng/Project/contract_analysis/')
    #train, validate = train_test_split(test_graphs, test_size=0.1, random_state=1)
    #main(train, validate)
    #load_model_and_test(test_graphs)
    start_main = datetime.now()
    train_graphs, test_graphs = load_transaction_seq_data([
        #'./data/cross_bad_report.csv',
        #'./data/single_bad_report.csv',
        #'./data/cross_good_report.csv',

        #'./data/extend_control_game/DICE1-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/DICE2-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/DICE3-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/FOMO2-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/FOMO3-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/godgame-cluster_extend_control_game.csv'
        #'./data/extend_control_game/PARITY-PIE-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/PRIVATE1-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/godgame-cluster_extend_control_game.csv',
        
        #'./data/extend_control_game/badset_ori_extend_control_game_5.csv',
        #'./data/extend_control_game/badset_ori_extend_control_game_8.csv',
        #'./data/extend_control_game/badset_ori_extend_control_game_10.csv',
        #'./data/extend_control_game/goodset_ori_extend_control_game_5.csv',
        #'./data/extend_control_game/goodset_ori_extend_control_game_8.csv',

        # './data/badset_ori_group/badset_ori_badRan_extend_control_game_5.csv',
        # './data/badset_ori_group/badset_ori_badRan_extend_control_game_8.csv',
        # './data/badset_ori_group/badset_ori_overflow_extend_control_game_5.csv',
        # './data/badset_ori_group/badset_ori_overflow_extend_control_game_8.csv',
        # './data/badset_ori_group/badset_ori_reentran_extend_control_game_5.csv',
        # './data/badset_ori_group/badset_ori_reentran_extend_control_game_8.csv',
        # './data/badset_ori_group/badset_ori_dos_extend_control_game_5.csv',
        # './data/badset_ori_group/badset_ori_dos_extend_control_game_8.csv',

        # './data/extend_control_game_no_transfer/goodset_ori_0_extend_control_game_5.csv',
        # './data/extend_control_game_no_transfer/goodset_ori_0_extend_control_game_8.csv',
        # './data/extend_control_game_no_transfer/goodset_ori_0_extend_control_game_10.csv',
        './data/goodset_new.csv',
        './data/extend_control_game_no_transfer/badset_ori_notransfer_no3_extend_control_game_5.csv',
        './data/extend_control_game_no_transfer/badset_ori_notransfer_no3_extend_control_game_8.csv',
        './data/extend_control_game_no_transfer/badset_ori_notransfer_no3_extend_control_game_10.csv',
        
        ],
        [
        # './data/all_user_labeled_except_null.csv',
        # './data/cross_user_by_game_30day_labeled.csv',
        # './data/single_bad_d2w.csv',
        # './data/cross_bad_d2w.csv',

        #'./data/extend_control_game/DICE1-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/DICE2-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/DICE3-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/FOMO2-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/FOMO3-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/godgame-cluster_extend_control_game.csv'
        #'./data/extend_control_game/FOMO2-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/PARITY-PIE-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/PRIVATE1-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/RocketCoin3-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/RocketCoin4-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/Rubixi1-max150-cluster_extend_control_game.csv',
        #'./data/extend_control_game/THRONE1-max150-cluster_extend_control_game.csv',
        
        #'./data/extend_control_game/badset_ori_extend_control_game_5.csv',
        #'./data/extend_control_game/goodset_ori_extend_control_game_5.csv',

        # './data/badset_ori_group/badset_ori_imAuth_extend_control_game_5.csv',
        # './data/badset_ori_group/badset_ori_imAuth_extend_control_game_8.csv',
        # './data/badset_ori_group/badset_ori_imAuth_extend_control_game_10.csv',
        
        #'./data/unknownset_0713.csv',
        './data/unknown_class_newtype_rerun.csv'
        ])
    
    # train_set, test_set = train_test_split(train_graphs, test_size=0.1, random_state=1)
    # train, validate = train_test_split(train_set, test_size=0.1, random_state=1)#train_graphs[:]+test_graphs[600:]

    train, validate = train_test_split(train_graphs, test_size=0.1, random_state=1)
    main_lstm(train, validate)
    print('Train Sets #: %d'%(len(train)+len(validate)))
    
    train_main = datetime.now()
    print("\n* Total training time of: is " + str(train_main - start_main))

    test_set = test_graphs
    print('Test Sets #: %d'%(len(test_set)))
    random.shuffle(test_set)
    load_lstm_model_and_test(test_set)

    test_main = datetime.now()
    print("\n* Total testing time of: is " + str(test_main - train_main))
