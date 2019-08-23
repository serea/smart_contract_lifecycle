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
        _s = torch.mm(weight, seq[i])
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
        print('Function:')
        print(weight.unsqueeze(0).expand(input.size(0), -1, -1).size())
        output = torch.bmm(input, weight.unsqueeze(0).expand(input.size(0), -1, -1))
        print(output.size())
        if bias is not None:
            print(bias.unsqueeze(0).expand_as(output).size())
            output += bias.unsqueeze(0).expand_as(output)

        return output

    def backward(self, grad_output):
        print('backward')
        input, weight, bias = self.saved_tensors

        grad_input = grad_weight = grad_bias = None
        if self.needs_input_grad[0]:
            grad_input = torch.bmm(grad_output, weight)
        if self.needs_input_grad[1]:
            grad_weight = torch.bmm(grad_output, input) # orig: grad_output.t()
        if bias is not None and self.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        #weight_temp = None
        #for i in range(grad_weight.size(0)):
        #    weight_temp = weight_temp.mul(grad_weight[i]) if weight_temp is not None else grad_weight[i]

        grad_weight = grad_weight+weight
        print(grad_weight)
        if bias is not None:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight

class NodeAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NodeAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = Parameter(torch.Tensor(2*hidden_size, 1))
        self.bias_ih = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x_in, x_node_eoa, x_node_d):
        batch_size = x_in.size(0)
        seq_length = x_in.size(1)
        softmax = nn.Softmax()
        #print('1:')
        #print(x_in.size())
        #print(x_node_eoa.permute(0, 2, 1).size())
        #print(x_node_d.size())
        x_node = torch.cat((x_node_eoa, x_node_d), -1)

        #print('2:')
        #print(x_node.view(-1, x_node.size(2)).unsqueeze(1).size())
        #print(x_in.view(-1, x_in.size(2)).unsqueeze(1).permute(0, 2, 1).size())
        embed_a = softmax(torch.bmm(x_node.view(-1, x_node.size(2)).unsqueeze(1), x_in.view(-1, x_in.size(2)).unsqueeze(1).permute(0, 2, 1)))

        #print('3:')
        #print(embed_a.size())
        #print(x_in.view(-1, x_in.size(2)).unsqueeze(1).size())
        embed_e = torch.bmm(embed_a, x_in.view(-1, x_in.size(2)).unsqueeze(1))

        #print('4:')
        #print(embed_e.size())
        #print(torch.cat((x_node.view(-1, x_node.size(2)).unsqueeze(1), embed_e), -1).size())
        #print(self.weight_ih.size())
        #print(self.bias_ih.size())
        c = LinearF()(torch.cat((x_node.view(-1, x_node.size(2)).unsqueeze(1), embed_e), -1), self.weight_ih, self.bias_ih)
        #print(c.size())
        c = torch.sigmoid(c)

        #print('5:')
        #print(c.size())
        #print(torch.ones(batch_size*seq_length, 1).size())
        #print(embed_e.size())
        output_left = (torch.ones(batch_size*seq_length, 1,1) - c).bmm(embed_e)
        #print(output_left.size())
        #output_right = c.bmm(x_node.view(-1, x_node.size(2)).unsqueeze(1))
        output_right = c.bmm(x_in.view(-1, x_in.size(2)).unsqueeze(1))
        output =  output_left + output_right

        #print('Fin:')
        #print(output.size())
        output = output.view(batch_size, seq_length, -1)
        output = output.permute(1, 0, 2)
        #print(output.size())

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
        #self.hidden=(torch.autograd.Variable(torch.zeros(1,1,self.hidden_dim)),torch.autograd.Variable(torch.zeros(1,1,self.hidden_dim)))

        self.hidden2tag = torch.nn.Linear(in_features=self.hidden_dim*2, out_features=self.target_size)
        self.sigmoid = torch.nn.Sigmoid()
        
        #self.cirterion = torch.nn.BCELoss(size_average=True)
        self.cirterion = torch.nn.MSELoss()
        #self.cirterion = torch.nn.MultiLabelSoftMarginLoss()

    def forward(self, list_int, batch_size, x, y = None, x_node_eoa = None, x_node_d = None):
        self.PAD_VALUE  = 0
        
        input_emb = self.attention(x, x_node_eoa, x_node_d)
        input_emb = input_emb.permute(1, 0, 2)

        #input_emb = x
        input_emb = list(itertools.zip_longest(*(input_emb.detach().numpy().tolist()), fillvalue = -1))
        input_emb = torch.FloatTensor(input_emb)

        #self.h = torch.autograd.Variable(torch.zeros(2, batch_size, self.hidden_dim))                                                                                                                                
        #self.c = torch.autograd.Variable(torch.zeros(2, batch_size, self.hidden_dim)) 

        embed_pack = torch.nn.utils.rnn.pack_padded_sequence(input=input_emb, lengths=np.array(list_int), batch_first=False, enforce_sorted=False)

        out, _ = self.lstm(embed_pack) #, (self.h, self.c)
        
        out, length = nn.utils.rnn.pad_packed_sequence(out, batch_first=False)
        out = out[0]#.index_select(0, idx_unsort)
        out = out.contiguous()
        #out = out.view(-1, self.hidden_dim*2)

        #out = self.hidden2tag(out)
        #tags = self.softmax(out)
        tags = self.log_softmax(self.hidden2tag(out))
        #print(tags)
        print(tags.size())

        #tags = self.log_softmax(self.linear1(out.view(-1,self.hidden_dim*2)))
        #return tags

        if y is not None:
            y_onehot = torch.FloatTensor(len(y), self.target_size)
            y_onehot.zero_()
            print(y)
            y_onehot.scatter_(1,y,1)

            print(tags.size()) #([64, 181, 2])
            print(y_onehot.size()) #([64, 2])
            loss = self.cirterion(tags, y_onehot)
            #loss = (loss * mask_mat).sum() / num_element
            
            pred = tags.data.max(1, keepdim=True)[1]

            acc = pred.eq(y.data.view_as(pred)).cpu().sum().float() / float(y.size()[0])
            return tags, loss, acc
        else:
            return tags

