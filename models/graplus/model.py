import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from torch_geometric.data import Batch
                                                                                    
from network import FgBgAttention, FgBgRegression

from SG_Networks import GraphEmbeddings, GTN

class Normalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.mean = mean
        self.std = std

    def __call__(self, img, is_cuda=True):
        assert (img.shape[1] == 3)
        proc_img = torch.zeros(img.shape)
        if is_cuda:
            proc_img = proc_img.cuda()
        proc_img[:, 0, :, :] = (img[:, 0, :, :] - self.mean[0]) / self.std[0]
        proc_img[:, 1, :, :] = (img[:, 1, :, :] - self.mean[1]) / self.std[1]
        proc_img[:, 2, :, :] = (img[:, 2, :, :] - self.mean[2]) / self.std[2]

        return proc_img
    
def weights_init_normal(m):
    classname = m.__class__.__name__
                                                                     
    if classname.find('Conv') != -1 :
        torch.nn.init.normal_(m.weight, 0.0, 0.02)

    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)


def get_params(model, key): 
    if key == "g10x":
        for m in model.named_modules():                            
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.Linear) or isinstance(m[1], nn.BatchNorm1d) or isinstance(m[1], nn.Embedding) or isinstance(m[1], nn.LayerNorm):
                for p in m[1].parameters():
                    yield p

    if key == "d1x":
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d) or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], nn.Linear) or isinstance(m[1], nn.BatchNorm1d) or isinstance(m[1], nn.Embedding) or isinstance(m[1], nn.LayerNorm):
                for p in m[1].parameters():
                    yield p


#################################################################################################
class SysNet(nn.Module): #####
    def __init__(self, opt, img_size, init_weight=True):
        super(SysNet, self).__init__()
        self.opt = opt
        self.img_size = img_size

        self.Embedding = GraphEmbeddings(opt)
        self.graph_operator = GTN(opt) 
        if opt.use_spatial_info : self.spatial_proj = nn.Linear(9, opt.spatial_dim)
        self.fg_cat_proj = nn.Linear(opt.embed_dim, opt.spatial_dim+opt.gtn_output_dim)
        self.att = FgBgAttention(opt)
        self.regress_net = FgBgRegression(opt)

        #--------------------------------------------#
        if init_weight:
            self.initialize_weight()
    
    def initialize_weight(self):    
        if self.opt.use_spatial_info :
            modules = [self.att, self.regress_net, self.fg_cat_proj, self.spatial_proj, self.graph_operator]   
        else:
            modules = [self.att, self.regress_net, self.fg_cat_proj, self.graph_operator]   
                               

        for m in modules:
            m.apply(weights_init_normal)

    def gen_blend(self, bg_img, fg_img, fg_msk, fg_bbox, trans):
        batch_size = len(trans)
        theta = torch.cat((
            1 / (trans[:,0] + 1e-6), torch.zeros(batch_size).cuda(), (1 - 2 * trans[:,1]) * (1 / (trans[:,0] + 1e-6) - fg_bbox[:,2] / self.img_size),
            torch.zeros(batch_size).cuda(), 1 / (trans[:,0] + 1e-6), (1 - 2 * trans[:,2]) * (1 / (trans[:,0] + 1e-6) - fg_bbox[:,3] / self.img_size)
        ), dim=0).view(2, 3, batch_size).permute(2, 0, 1).contiguous()
        grid = F.affine_grid(theta, fg_img.size(), align_corners=True)
        fg_img_out = F.grid_sample(fg_img, grid, align_corners=True)
        fg_msk_out = F.grid_sample(fg_msk, grid, align_corners=True)
        comp_out = fg_msk_out * fg_img_out + (1 - fg_msk_out) * bg_img

        return comp_out, fg_msk_out
    
    #################################################################################### #graph
    def process_scene_graph(self, batch_size, node_attr, edge_index, edge_attr, cats_fg):
        """
        Process the scene graph using the Graph Transformer layers.

        Args:
            graph_data: Data object containing x (node labels), edge_index (edge indices), edge_attr (edge labels).
                in train mode (batch of graphs):
                    type: torch_geometric.data.batch.DataBatch
                in not train mode(1 graph):
                    type:torch_geometric.data.data.Data (batch loaded from dataloader) 

            cats_fg: list strings for categories for foregrounds in the batch
        Returns:
            Pooled graph features.
        """

        node_attr  = node_attr.long()  # Node labels    Shape: (total_num_nodes,    )
        edge_attr  = edge_attr.long()  # Edge labels    Shape: (total_num_edges,    )
        # edge_index                   # Edge indices   Shape: (2  , total_num_edges)
        
        # print("#*20"+" in Sysnet ...")
        # print("node_attr.shape:", node_attr.shape)
        # print("edge_attr.shape:", edge_attr.shape)
        # print("#*20"+" in Sysnet ...")

        # print("edge_index.shape:", edge_index.shape)
        #=================================================================# foreground category embedding
        # Get the indices of each word in the list
        fg_cat_indices = [self.Embedding.node_labels.index(word) for word in cats_fg]

        # Convert the list of indices to a PyTorch tensor
        fg_cat_indices = torch.tensor(fg_cat_indices).cuda()

        fg_cat_features = self.Embedding.node_embedding(fg_cat_indices) # Shape: (batch_size, embedding_dim)

        #=================================================================# graph Embedding
        # Node embeddings
        node_features = self.Embedding.node_embedding(node_attr)          # Shape: (total_num_nodes, embedding_dim)

        # Edge embeddings
        edge_features = self.Embedding.edge_embedding(edge_attr)  # Shape: (total_num_edges, embedding_dim)

        #=================================================================# Graph Operation
        # Pass through Graph Operator Model
        # node_features Shape:  (total_num_nodes, gtn_output_dim)
        # edge_features Shape:  (total_num_edges, gtn_output_dim)

        node_features, edge_features = self.graph_operator(node_features, edge_index, edge_features)
    
        #=================================================================# reshaping node features 
        node_features = node_features.view(batch_size, self.opt.num_nodes, -1).float() # shape: (batch_size, num_nodes, gtn_output_dim)

        return node_features, fg_cat_features
    
    
    ####################################################################################

    # def forward(self,bg_size, catnms, bg_img, fg_img, fg_msk, fg_bbox, sg_bg, sg_bg_bbox_org,sg_bg_trans, is_train=True ):
    def forward(self, bg_size, catnms, bg_img, fg_img, fg_msk, fg_bbox, node_attr, edge_attr, edge_index, sg_bg_bbox_org,sg_bg_trans, is_train=True ):


        #=========================#
        bg_size = bg_size.cuda()
        node_attr = node_attr.cuda()
        edge_attr = edge_attr.cuda()
        edge_index = edge_index.cuda()
        sg_bg_bbox_org = sg_bg_bbox_org.cuda()
        sg_bg_trans = sg_bg_trans.cuda()

        batch_size = len(catnms)

        #=============================================================================# 
        # node_features shape  :(batch_size, num_nodes, gtn_output_dim)
        # fg_cat_features_ shape:(batch_size, embed_dim)
        node_features, fg_cat_features_ = self.process_scene_graph(batch_size, node_attr, edge_index, edge_attr, catnms) 

        # fg_cat_features  shape:  (batch_size, num_nodes, gtn_output_dim + spatial_dim)
        fg_cat_features = self.fg_cat_proj(fg_cat_features_)

        #=============================================================================#
        # tensor of W and H of the original size of bg image
        bg_size = bg_size.unsqueeze(1).repeat(1, self.opt.num_nodes, 1)  # Shape: (batch_size, num_nodes, 2)

        if self.opt.use_spatial_info :
            # sg_bg_bbox shape:  (batch_size, num_nodes, 2+4+3)
            spatial_info = torch.cat((bg_size, sg_bg_bbox_org, sg_bg_trans), dim=2)
            # bg_bbox_feat shape:  (batch_size, num_nodes, spatial_dim)
            spatioal_features = self.spatial_proj(spatial_info)
            
        #=============================================================================#
        if self.opt.use_spatial_info :
            # concated_node_features shape:  (batch_size, num_nodes,  gtn_output_dim + spatial_dim )
            concated_node_features = torch.cat((node_features, spatioal_features), dim=2)
        else:
            # concated_node_features shape:  (batch_size, num_nodes,  gtn_output_dim )
            concated_node_features = node_features

        # fg_cat_features shape:(batch_size, 1, gtn_output_dim + spatial_dim)
        fg_cat_features = fg_cat_features.unsqueeze(1)

        #=============================================================================#
        # fg_cat_features shape:        (batch_size, 1        ,  gtn_output_dim + spatial_dim )
        # concated_node_features shape: (batch_size, num_nodes,  gtn_output_dim + spatial_dim )
        # attn_feats_1 shape:           (batch_size, 1        ,  gtn_output_dim + spatial_dim )
        attn_feats, attn = self.att(fg_cat_features, concated_node_features)

        randomness = torch.randn((attn_feats.shape[0], self.opt.d_noise)).cuda()
                                                                                                                                    
        trans_ = self.regress_net(torch.cat((attn_feats.squeeze(1), randomness), dim=1))
        trans = torch.tanh(trans_) / 2.0 + 0.5


        if is_train :
            blend_img_1, blend_msk_1 = self.gen_blend(bg_img, fg_img, fg_msk, fg_bbox, trans)
            return blend_img_1, blend_msk_1, trans
        else:
            return trans, attn


class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=True, init_weight=True):
        super(Discriminator, self).__init__()
        self.normalize = Normalize()

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        sequence_stream = []
        for n in range(len(sequence)):
            sequence_stream += sequence[n]
        self.model = nn.Sequential(*sequence_stream)

        if init_weight:
            self.initialize_weight()

    def initialize_weight(self):
        self.model.apply(weights_init_normal)

    def forward(self, img, mask):
        img_norm = self.normalize(img)
        out = self.model(torch.cat((img_norm, mask), dim=1))
        out_avg = F.adaptive_avg_pool2d(out, (1, 1))
        return out_avg.view(out_avg.shape[0])


class GAN(object):
    def __init__(self, opt):
        self.Eiters = 0
        self.opt = opt
        self.generator = SysNet(opt, img_size=opt.img_size)
        self.discriminator = Discriminator(input_nc=4)
        self.to_cuda()
        self.optimizer_G = self.get_G_optimizer(opt)
        self.optimizer_D = self.get_D_optimizer(opt)
        self.discri_loss = torch.nn.BCELoss()
        self.recons_loss_no_reduction = torch.nn.MSELoss(reduction='none')

    def get_G_optimizer(self, opt):
        return torch.optim.Adam(
            params=[
                {
                    "params": get_params(self.generator, key="g10x"), 
                    "lr": opt.g_lr,         ###
                    "initial_lr": opt.g_lr, ###
                    "weight_decay": opt.g_weight_decay,
                },
            ], 
            betas=(opt.b1, opt.b2)
        )

    def get_D_optimizer(self, opt):
        return torch.optim.Adam(
            params=[
                {
                    "params": get_params(self.discriminator, key="d1x"),
                    "lr":  opt.d_lr,
                    "initial_lr": opt.d_lr,
                    "weight_decay": opt.d_weight_decay,
                }
            ], 
            betas=(opt.b1, opt.b2)
        )

    def start_train(self):
        self.generator.train()
        self.discriminator.train()

    def start_eval(self):
        self.generator.eval()
        self.discriminator.eval()

    def to_cuda(self):
        self.generator = self.generator.cuda()
        self.discriminator = self.discriminator.cuda()

    def state_dict(self):
        model_dict = dict()
        model_dict["generator"]     = self.generator.state_dict()
        model_dict["discriminator"] = self.discriminator.state_dict()
        return model_dict

    def optimizer_dict(self):
        optimizer_state_dict = dict()
        optimizer_state_dict["generator"] = self.optimizer_G.state_dict()
        optimizer_state_dict["discriminator"] = self.optimizer_D.state_dict()
        return optimizer_state_dict

    def load_state_dict(self, pretrained_dict, strict=False):
        for k in pretrained_dict:
            if k == "generator":
                self.generator.load_state_dict(pretrained_dict[k], strict=strict)
            elif k == "discriminator":
                self.discriminator.load_state_dict(pretrained_dict[k], strict=strict)

    def load_opt_state_dict(self, pretrained_dict):
        for k in pretrained_dict:
            if k == "generator":
                self.optimizer_G.load_state_dict(pretrained_dict[k])
            elif k == "discriminator":
                self.optimizer_D.load_state_dict(pretrained_dict[k])
                 
                 
                 
    
    
                                                                                         #####
    def train_disc_gen(self, bg_size, catnms, bg_img, fg_img, fg_msk, fg_bbox, comp_img, comp_msk, 
                   label, trans_label, node_attr, edge_attr, edge_index, sg_bg_bbox_org, sg_bg_trans):    
        
        self.Eiters += 1
        batch_size = len(label)
        
        #####
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
        # sg_bg = sg_bg.to(device)
        # sg_bg = sg_bg.cuda()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Data preprocessing
        node_attr = node_attr.to(device)
        edge_attr = edge_attr.to(device)
        edge_index = edge_index.to(device)

        bg_img_v = bg_img.to(device).requires_grad_(False)
        fg_img_v = fg_img.to(device).requires_grad_(False)
        fg_msk_v = fg_msk.to(device).requires_grad_(False)
        fg_bbox_v = fg_bbox.float().to(device).requires_grad_(False)
        comp_img_v = comp_img.to(device).requires_grad_(False)
        comp_msk_v = comp_msk.to(device).requires_grad_(False)
        label_v = label.float().to(device).requires_grad_(False)
        trans_label_v = trans_label.to(device).requires_grad_(False)
        valid = Variable(torch.ones(batch_size), requires_grad=False).cuda()
        fake = Variable(torch.zeros(batch_size), requires_grad=False).cuda()
        #-----------------------------------------------------------------------------#

        # Generate composites
        gen_comps, gen_msks, gen_trans = self.generator(bg_size, catnms, bg_img_v, fg_img_v, fg_msk_v, fg_bbox_v, 
                                                                          node_attr, edge_attr, edge_index, sg_bg_bbox_org, sg_bg_trans
                                                                          )
                                                                            
        real_indices = (label_v == 1).nonzero(as_tuple=True)[0]
        fake_indices = (label_v == 0).nonzero(as_tuple=True)[0]

        # ----------------------------------------------- ### Step 1: Train Discriminator on real samples (label_v == 1)
        real_comp_img_v = comp_img_v[real_indices]
        real_comp_msk_v = comp_msk_v[real_indices]
        pos_labels = valid[real_indices]
        
        
        for _ in range(self.opt.num_d_up_real):
            ### discrimination prediction on "positive" samples of the dataset
            d_pred_real_data = self.discriminator(real_comp_img_v, real_comp_msk_v) 
            d_loss_real_data = self.discri_loss(d_pred_real_data, pos_labels)

            self.optimizer_D.zero_grad()
            d_loss_real_data.backward()
            self.optimizer_D.step()
        
        #----------------------------------------------- ### Step 2: Train Discriminator on negative samples and generated fake samples together
        fake_comp_img_v = comp_img_v[fake_indices]
        fake_comp_msk_v = comp_msk_v[fake_indices]
        neg_labels = fake[fake_indices]

        ### discrimination prediction on "negative" samples of the dataset
        d_pred_fake_data = self.discriminator(fake_comp_img_v, fake_comp_msk_v) 
        d_loss_fake_data = self.discri_loss(d_pred_fake_data, neg_labels)

        ### discrimination prediction on generated samples fromm generator
        d_pred_gen_data_detach = self.discriminator(gen_comps.detach(), gen_msks.detach())
        d_loss_gen_data = self.discri_loss(d_pred_gen_data_detach, fake)

        d_loss_fake = d_loss_fake_data + d_loss_gen_data

        self.optimizer_D.zero_grad()
        d_loss_fake.backward()
        self.optimizer_D.step()

        # ----------------------------------------------- ###  Step 3: Train Generator
        d_pred_gen_data = self.discriminator(gen_comps, gen_msks)
        g_loss = self.discri_loss(d_pred_gen_data, valid)

        g_rec_loss = ((self.recons_loss_no_reduction(gen_trans, trans_label_v) * torch.cat((torch.sin(gen_trans[:,:1] * np.pi / 2), torch.cos(gen_trans[:,:1] * np.pi / 2), torch.cos(gen_trans[:,:1] * np.pi / 2)), dim=1).detach()).mean(dim=1) * label_v).sum() / label_v.sum()

        G_loss = g_loss + self.opt.rec_loss_weight*g_rec_loss

        self.optimizer_G.zero_grad()
        G_loss.backward()
        self.optimizer_G.step()
    
        #########################################################################################################    
        num_real = label_v.sum().item()         # Number of real samples in the batch (where label_v == 1)
        num_fake = (label_v == 0).sum().item()  # Number of fake samples in the batch (where label_v == 0)

        #===================================================================== Accuracy calculations for each case ###
        with torch.no_grad():

            #========================================================# generator
            gen_pred = (d_pred_gen_data > 0.5).float()
            # Calculate the number of correct predictions by comparing with the actual labels
            gen_correct = (gen_pred == fake).float().sum().item()     # Number of correct predictions for gen_comps_1

            #========================================================# dataset
            d_label_real_data = (d_pred_real_data > 0.5).float()
            real_correct = (d_label_real_data == label_v[real_indices]).float().sum().item()  # Correct predictions for real samples
            
            d_label_fake_data = (d_pred_fake_data > 0.5).float()
            fake_correct = (d_label_fake_data == label_v[fake_indices]).float().sum().item()  # Correct predictions for fake samples
            
            return G_loss, g_loss, g_rec_loss, d_loss_real_data, d_loss_fake, gen_correct, real_correct, fake_correct, num_real, num_fake



                                                              
    def test_genorator(self,bg_size, catnms, bg_img, fg_img, fg_msk, fg_bbox, node_attr, edge_index , edge_attr, sg_bg_bbox_org, sg_bg_trans):
        bg_size = bg_size.cuda()
        bg_img = bg_img.cuda()
        fg_img = fg_img.cuda()
        fg_msk = fg_msk.cuda()
        fg_bbox = fg_bbox.float().cuda()
        
        node_attr = node_attr.cuda()
        edge_index = edge_index.cuda()
        edge_attr = edge_attr.cuda()
        sg_bg_bbox_org = sg_bg_bbox_org.cuda()
        sg_bg_trans = sg_bg_trans.cuda()

        return self.generator(bg_size, catnms, bg_img, fg_img, fg_msk, fg_bbox, node_attr, edge_attr, edge_index, sg_bg_bbox_org, sg_bg_trans, is_train=False)      

