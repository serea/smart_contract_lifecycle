from __future__ import print_function

import os
import sys
import numpy as np
import torch
import math
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import itertools
from torch.autograd import Function

from pytorch_util import weights_init

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        pred = self.h2_weights(h1)
        
        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            return pred, mae, mse
        else:
            return pred

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(MLPClassifier, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, hidden_size)
        self.h3_weights = nn.Linear(hidden_size, num_class)

        weights_init(self)
        
    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        h2 = self.h1_weights(h1)
        h2 = F.relu(h2)

        logits = self.h3_weights(h2)
        logits = F.log_softmax(logits, dim=1)

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum() / float(y.size()[0])
            return logits, loss, acc
        else:
            return logits



def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(weight, seq[i].t())
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0,1)
        #if(nonlinearity=='tanh'):
        #    _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if(s is None):
            s = _s_bias
        else:
            s = torch.cat((s,_s_bias),0)
    return s.squeeze()

def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i].unsqueeze(0), weight)
        #if(nonlinearity=='tanh'):
        #    _s = torch.tanh(_s)
        #_s = _s.unsqueeze(0)
        if(s is None):
            s = _s
        else:
            s = torch.cat((s,_s),0)
    return s.squeeze()

def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    print(rnn_outputs.size())
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        print(att_weights[i].unsqueeze(0).size())
        a_i = att_weights[i].unsqueeze(0).expand_as(h_i) #
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if(attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors,h_i),0)
    return torch.sum(attn_vectors, 0)

class LinearF(Function):
     def forward(self, input, weight, bias=None):
         self.save_for_backward(input, weight, bias)

         output = torch.mm(weight.t(), input)
         if bias is not None:
             output += bias.unsqueeze(0).expand_as(output)

         return output

     def backward(self, grad_output):
         input, weight, bias = self.saved_tensors

         grad_input = grad_weight = grad_bias = None
         if self.needs_input_grad[0]:
             grad_input = torch.mm(grad_output, weight)
         if self.needs_input_grad[1]:
             grad_weight = torch.mm(grad_output.t(), input)
         if bias is not None and self.needs_input_grad[2]:
             grad_bias = grad_output.sum(0).squeeze(0)

         if bias is not None:
             return grad_input, grad_weight, grad_bias
         else:
             return grad_input, grad_weight

class NodeAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NodeAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = Parameter(torch.Tensor(hidden_size, 2*hidden_size))
        self.bias_ih = Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x_in, x_node_eoa, x_node_d):
        softmax = nn.Softmax()
        print('1:')
        print(x_in.size())
        print(x_node_eoa.permute(0, 2, 1).size())
        print(x_node_d.size())
        x_node = torch.bmm(x_node_eoa.permute(0, 2, 1), x_node_d)

        print('2:')
        print(x_node.size())
        print(x_in.permute(1, 2, 0).size())
        embed_a = softmax(torch.bmm(x_node, x_in.permute(1, 2, 0)))

        print('3:')
        print(embed_a.size())
        print(x_in.permute(1, 0, 2).size())
        embed_e = torch.bmm(embed_a, x_in.permute(1, 0, 2))

        print('4:')
        print(torch.cat((x_node, embed_e), -1).size())
        print(self.weight_ih.size())
        print(self.bias_ih.size())
        #c = LinearF()(torch.cat((x_node, embed_e), -1), self.weight_ih, self.bias_ih)
        c = batch_matmul_bias(torch.cat((x_node, embed_e), -1), self.weight_ih, self.bias_ih)
        c = torch.sigmoid(c)

        print('5:')
        print(c.size())
        print(self.weight_ih.size())
        print(self.bias_ih.size())
        output = (torch.ones(c.size(0), self.hidden_size, self.hidden_size) - c).matmul(embed_e) + c.matmul(x_node)

        print('Fin:')
        print(output.size())

        return output

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, target_size):
        super(LSTMTagger,self).__init__()
        self.embedding_dim=embedding_dim
        self.hidden_dim=hidden_dim
        self.vocab_size=vocab_size
        self.target_size=target_size

        self.attention = NodeAttention(self.embedding_dim, self.hidden_dim)

        self.lstm = torch.nn.LSTM(input_size = self.embedding_dim, hidden_size = self.hidden_dim, batch_first=False, bidirectional=True, dropout=0) #
        self.log_softmax = torch.nn.LogSoftmax()
        self.graph_embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim)
        #self.hidden=(torch.autograd.Variable(torch.zeros(1,1,self.hidden_dim)),torch.autograd.Variable(torch.zeros(1,1,self.hidden_dim)))

        self.hidden2tag = torch.nn.Linear(in_features=self.hidden_dim*2, out_features=self.target_size)
        self.sigmoid = torch.nn.Sigmoid()
        
        #self.cirterion = torch.nn.BCELoss(size_average=True)
        self.cirterion = torch.nn.MSELoss()
    
    def zeroPadding(self,l, fillvalue=0):
        return list(itertools.zip_longest(*l, fillvalue=fillvalue))

    def binaryMatrix(self,l):
        m = []
        for i, seq in enumerate(l):
            m.append([])
            for token in seq:
                if token == 0:
                    m[i].append(0)
                else:
                    m[i].append(1)
        return m

    def forward(self, list_int, batch_size, x, y = None, x_node_eoa = None, x_node_d = None):
        self.PAD_VALUE  = 0
        input_emb = self.attention(x, x_node_eoa, x_node_d)
        input_emb = input_emb.permute(1, 0, 2)
        #input_emb = x

        #sorted_length_list, idx_sort = torch.sort(mask_list, dim=0, descending=True)
        #_, idx_unsort = torch.sort(idx_sort, dim=0, descending=False)
        #input_emb = input_emb[:, idx_sort]
        #input_emb_packed = torch.nn.utils.rnn.pack_padded_sequence(input_emb, sorted_length_list.cpu().numpy(), batch_first=False) #.view(-1,len(x),self.embedding_dim)
        
        #self.norm = torch.nn.BatchNorm1d(max(mask_list).float(), momentum=0.5)
        #input_emb = self.norm(input_emb)

        #targets_batch = torch.LongTensor(self.zeroPadding(y))
        #targets_mask = torch.ByteTensor(self.binaryMatrix(self.zeroPadding(y)))
        print(input_emb.size())
        #out, self.hidden = self.lstm(input_emb.view(-1,1,self.embedding_dim), self.hidden) # .view(-1,1,self.embedding_dim)
        #self.h = torch.autograd.Variable(torch.zeros(2, batch_size, self.hidden_dim))                                                                                                                                
        #self.c = torch.autograd.Variable(torch.zeros(2, batch_size, self.hidden_dim)) 

        #embed_pack = torch.nn.utils.rnn.pack_padded_sequence(input=input_emb, lengths=np.array(list_int), batch_first=False, enforce_sorted=False)

        out, _ = self.lstm(input_emb) #, (self.h, self.c)
        
        #out, length = nn.utils.rnn.pad_packed_sequence(out, batch_first=False)
        #out = out[0]#.index_select(0, idx_unsort)
        #out = out.contiguous()
        #out = out.view(-1, self.hidden_dim*2)

        out = self.hidden2tag(out)
        tags = self.log_softmax(out)
        print(tags)
        print(tags.size())
        
        '''
        num_element = 0
        #list_int = list(idx_unsort)
        cur_batch_max_len = max(list_int)
        mask_mat = Variable(torch.ones(len(list_int),cur_batch_max_len))
        print(mask_mat)
        for idx_sample in range(len(list_int)):
            num_element += list_int[idx_sample] * self.target_size
            if list_int[idx_sample] != cur_batch_max_len:
                mask_mat[idx_sample, list_int[idx_sample]:] = 0.0

        tags = tags * mask_mat
        '''

        #tags = self.log_softmax(self.linear1(out.view(-1,self.hidden_dim*2)))
        #return tags


        if y is not None:
            y_onehot = torch.FloatTensor(len(y),2)
            y_onehot.zero_()
            y_onehot.scatter_(1,y,1)

            loss = self.cirterion(tags, y_onehot)
            #loss = (loss * mask_mat).sum() / num_element
            
            pred = tags.data.max(1, keepdim=True)[1]

            acc = pred.eq(y.data.view_as(pred)).cpu().sum().float() / float(y.size()[0])
            return tags, loss, acc
        else:
            return tags

