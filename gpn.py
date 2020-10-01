import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Attention(nn.Module):
    def __init__(self, n_hidden):
        super(Attention, self).__init__()
        self.size = 0
        self.batch_size = 0
        self.dim = n_hidden
        
        v  = torch.FloatTensor(n_hidden).cuda()
        self.v  = nn.Parameter(v)
        self.v.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        # parameters for pointer attention
        self.Wref = nn.Linear(n_hidden, n_hidden)
        self.Wq = nn.Linear(n_hidden, n_hidden)
    
    
    def forward(self, q, ref):       # query and reference, hidden state of current node, and the context
        self.batch_size = q.size(0)     #512
        self.size = int(ref.size(0) / self.batch_size)   #51
        ############embedding
        q = self.Wq(q)     # (B, dim)   affine layer
        ref = self.Wref(ref)    #affine layer
        ref = ref.view(self.batch_size, self.size, self.dim)  # (B, size, dim)
        
        
        q_ex = q.unsqueeze(1).repeat(1, self.size, 1) # (B, size, dim)
        # v_view: (B, dim, 1)
        v_view = self.v.unsqueeze(0).expand(self.batch_size, self.dim).unsqueeze(2)
        
        # (B, size, dim) * (B, dim, 1)
        u = torch.bmm(torch.tanh(q_ex + ref), v_view).squeeze(2)
        
        return u, ref


class LSTM(nn.Module):
    def __init__(self, n_hidden):
        super(LSTM, self).__init__()
        
        # parameters for input gate
        self.Wxi = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whi = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wci = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for forget gate
        self.Wxf = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whf = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wcf = nn.Linear(n_hidden, n_hidden)    # w(ct)
        
        # parameters for cell gate
        self.Wxc = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Whc = nn.Linear(n_hidden, n_hidden)    # W(ht)
        
        # parameters for forget gate
        self.Wxo = nn.Linear(n_hidden, n_hidden)    # W(xt)
        self.Who = nn.Linear(n_hidden, n_hidden)    # W(ht)
        self.wco = nn.Linear(n_hidden, n_hidden)    # w(ct)
    
    
    def forward(self, x, h, c):       # query and reference
        
        # input gate
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + self.wci(c))
        # forget gate
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + self.wcf(c))
        # cell gate
        c = f * c + i * torch.tanh(self.Wxc(x) + self.Whc(h))
        # output gate
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + self.wco(c))
        
        h = o * torch.tanh(c)
        
        return h, c


class GPN(torch.nn.Module):
    
    def __init__(self, args, n_feature, n_hidden):
        super(GPN, self).__init__()
        self.city_size = args['size']
        self.batch_size = args['batch_size']
        self.dim = n_hidden
        
        # lstm for first turn
        self.lstm0 = nn.LSTM(n_hidden, n_hidden)
        
        # pointer layer
        self.pointer = Attention(n_hidden)
        
        # lstm encoder
        self.encoder = LSTM(n_hidden)
        
        # trainable first hidden input
        h0 = torch.FloatTensor(n_hidden).cuda()
        c0 = torch.FloatTensor(n_hidden).cuda()
        
        # trainable latent variable coefficient
        alpha = torch.ones(1).cuda()
        
        self.h0 = nn.Parameter(h0)
        self.c0 = nn.Parameter(c0)
        
        self.alpha = nn.Parameter(alpha)
        self.h0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        self.c0.data.uniform_(-1/math.sqrt(n_hidden), 1/math.sqrt(n_hidden))
        
        r1 = torch.ones(1).cuda()
        r2 = torch.ones(1).cuda()
        r3 = torch.ones(1).cuda()
        self.r1 = nn.Parameter(r1)
        self.r2 = nn.Parameter(r2)
        self.r3 = nn.Parameter(r3)
        
        # embedding
        self.embedding_x = nn.Linear(n_feature, n_hidden)
        self.embedding_all = nn.Linear(n_feature, n_hidden)
        self.x_norm = nn.LayerNorm(n_hidden)
        self.context_norm = nn.LayerNorm([self.city_size+1, n_hidden])
        
        
        
        # weights for GNN
        self.W1 = nn.Linear(n_hidden, n_hidden)
        self.W2 = nn.Linear(n_hidden, n_hidden)
        self.W3 = nn.Linear(n_hidden, n_hidden)
        
        # aggregation function for GNN
        self.agg_1 = nn.Linear(n_hidden, n_hidden)
        self.agg_2 = nn.Linear(n_hidden, n_hidden)
        self.agg_3 = nn.Linear(n_hidden, n_hidden)
    
    
    def forward(self, x, X_all, mask, mask_fin, h=None, c=None, latent=None):
        '''
        Inputs:
        
        x: current city coordinate (B, 5)
        
        X_all: all cities' cooridnates (B, size, 5)
        
        mask: mask visited cities  ([B, size])
        
        mask_fin: used to check which problem in batch has finished decoding
        
        h: hidden variable (B, dim)
        
        c: cell gate (B, dim)
        
        latent: latent pointer vector from previous layer (B, size, dim)
        
        ###########
        
        Outputs:
        
        softmax: probability distribution of next city (B, size)
        
        h: hidden variable (B, dim)
        
        c: cell gate (B, dim)
        
        latent_u: latent pointer vector for next layer
        '''
        
        
        variable_size = True
        
        
        self.batch_size = X_all.size(0)
        self.city_size = X_all.size(1)

        if variable_size:
            #check which problems have masked all nodes
            unfinished = np.ones(X_all.size(0), dtype=bool)        
            for i in range(X_all.size(0)):
                if torch.isfinite(mask_fin[i]).sum().item() == 0:    #all node mask are set to -inf, the problem is finished
                    unfinished[i] = False
                    
            unfinished_idx = unfinished.nonzero()[0]  #get index of unfinished problem
            
            
            #define current batch(without finished problem)
            X_all = X_all[unfinished_idx]
            x = x[unfinished_idx]
            if h is not None and c is not None: 
                h = h[unfinished_idx]
                c = c[unfinished_idx]
            mask = mask[unfinished_idx]


        
        # =============================
        # vector context
        # =============================
        

        x = self.embedding_x(x)     #depot   
        x = self.x_norm(x)
        context = self.embedding_all(X_all)  #all nodes
        context = self.context_norm(context)
        
        
        # =============================
        # process hidden variable for lSTM
        # =============================
        
        first_turn = False
        if h is None or c is None:
            first_turn = True
        
        if first_turn:
            # init h0 and c0 for LSTM 
            
            h0 = self.h0.unsqueeze(0).expand(self.batch_size, self.dim)
            c0 = self.c0.unsqueeze(0).expand(self.batch_size, self.dim)

            h0 = h0.unsqueeze(0).contiguous()
            c0 = c0.unsqueeze(0).contiguous()
            
            input_context = context.permute(1,0,2).contiguous()
            _, (h_enc, c_enc) = self.lstm0(input_context, (h0, c0))
            
            # let h0, c0 be the hidden variable of first turn
            h = h_enc.squeeze(0)
            c = c_enc.squeeze(0)
        
        
        # =============================
        # graph neural network encoder
        # =============================
        
        # (B, size, dim)
        context = context.reshape(-1, self.dim)
        
        context = self.r1 * self.W1(context)\
            + (1-self.r1) * F.relu(self.agg_1(context/(self.city_size-1)))

        context = self.r2 * self.W2(context)\
            + (1-self.r2) * F.relu(self.agg_2(context/(self.city_size-1)))
        
        context = self.r3 * self.W3(context)\
            + (1-self.r3) * F.relu(self.agg_3(context/(self.city_size-1)))
        
        
        # LSTM encoder
        h, c = self.encoder(x, h, c)
        
        # query vector
        q = h
        
        # pointer
        u, _ = self.pointer(q, context)
        

        
        

            
            
        if variable_size:
            # for finished problems, h, c = 0(no more needed by next step)       
            h_out = torch.zeros(self.batch_size, self.dim).cuda()
            h_out[unfinished_idx] = h
            c_out = torch.zeros(self.batch_size, self.dim).cuda()
            c_out[unfinished_idx] = c
            
            # for finished problems, u_out is masking everything else but depot
            # so the vehicle stay in depot
            u_out = (torch.ones(self.batch_size, self.city_size)*-np.inf).cuda()
            u_out[:,0] = 1
            u_out[unfinished_idx] = u
    
            # if it is low level GPN, for finished problems, latent_u is uniform
            latent_u = torch.zeros_like(u_out).cuda()
            latent_u[unfinished_idx] = u 
            
            # mask visited or unsatisfiable nodes
            u_out[unfinished_idx] = 10 * torch.tanh(u_out[unfinished_idx]) + mask
            if latent is not None:
                u_out += self.alpha * latent
            
        else:
            latent_u = u.clone()
            # mask visited or unsatisfiable nodes
            u = 10 * torch.tanh(u) + mask
            if latent is not None:
                u += self.alpha * latent
                
            return F.softmax(u, dim=1), h, c, latent_u
        
    
        return F.softmax(u_out, dim=1), h_out, c_out, latent_u

