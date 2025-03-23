import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models


BN_MOMENTUM = 0.1



class ScaledDotProductAttention(nn.Module):
    def __init__(self, opt):
        super(ScaledDotProductAttention, self).__init__()
        self.opt = opt
        self.pos_k = nn.Embedding(opt.n_heads * opt.num_nodes, opt.d_k) if opt.use_pos_enc else None
        self.pos_v = nn.Embedding(opt.n_heads * opt.num_nodes, opt.d_v) if opt.use_pos_enc else None
        self.pos_ids = torch.LongTensor(list(range(opt.n_heads * opt.num_nodes))).view(1, opt.n_heads, opt.num_nodes) if opt.use_pos_enc else None

    def forward(self, Q, K, V):
        if self.opt.use_pos_enc:
            K_pos = self.pos_k(self.pos_ids.cuda())
            V_pos = self.pos_v(self.pos_ids.cuda())
            scores = torch.matmul(Q, (K + K_pos).transpose(-1, -2)) / np.sqrt(self.opt.d_k)
            attn = nn.Softmax(dim=-1)(scores)
            context = torch.matmul(attn, V + V_pos)
        else:
            scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.opt.d_k)
            attn = nn.Softmax(dim=-1)(scores)
            context = torch.matmul(attn, V)

        return context, attn




class MultiHeadAttention(nn.Module):
   def __init__(self, opt):
       super(MultiHeadAttention, self).__init__()
       self.opt = opt
       self.d_model = opt.gtn_output_dim + opt.spatial_dim

       # Projection layers for Q, K, V with different input dimensions
       self.W_Q = nn.Linear(self.d_model, opt.d_k * opt.n_heads)
       self.W_K = nn.Linear(self.d_model, opt.d_k * opt.n_heads)
       self.W_V = nn.Linear(self.d_model, opt.d_v * opt.n_heads)

       self.att = ScaledDotProductAttention(opt)

       # Output projection layer
       self.W_O = nn.Linear(opt.n_heads * opt.d_v, self.d_model)

       self.norm = nn.LayerNorm(self.d_model)

   def forward(self, Q, K, V, add_residual):
       residual, batch_size = Q, Q.size(0)
       q_s = self.W_Q(Q).view(batch_size, -1, self.opt.n_heads, self.opt.d_k).transpose(1,2)
       k_s = self.W_K(K).view(batch_size, -1, self.opt.n_heads, self.opt.d_k).transpose(1,2)
       v_s = self.W_V(V).view(batch_size, -1, self.opt.n_heads, self.opt.d_v).transpose(1,2)
       context, attn = self.att(q_s, k_s, v_s)
       context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.opt.n_heads * self.opt.d_v)

       output = self.W_O(context)

       if add_residual:
        return self.norm(output + residual), attn
       else:
        return self.norm(output), attn
    

class FgBgAttention(nn.Module):
   def __init__(self, opt):
       super(FgBgAttention, self).__init__()
       self.opt = opt
       self.att = MultiHeadAttention(opt)

   def forward(self, fg_feats, bg_feats):
       output, attn = self.att(fg_feats, bg_feats, bg_feats, self.opt.add_residual)
       return output, attn


############################################################################################################### Regression
class FgBgRegression(nn.Module):  
    def __init__(self, opt):
        super(FgBgRegression, self).__init__()
        self.regressor = nn.Sequential(
                                                
            nn.Linear(opt.gtn_output_dim + opt.spatial_dim + opt.d_noise  , 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 3),
        )

    def forward(self, x):
        out = self.regressor(x)
        return out
    







