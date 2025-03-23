import os
import sys
from thop import profile, clever_format



# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Move two levels up
main_dir = os.path.abspath(os.path.join(current_dir, "../../"))

# Add this directory to sys.path
sys.path.append(main_dir)
##########################################################

import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch

from loader import get_sg_loader
from loader.utils import gen_composite_image


######
OPA_path    = os.path.abspath(os.path.join(current_dir, "../../../OPA_dataset"))
SG_OPA_path = os.path.abspath(os.path.join(current_dir, "../../../OPA_SG/"))


gpt2_path = os.path.abspath(os.path.join(current_dir, "../../../word_embedding_checkpoints/gpt2"))


########################################################## Sampling
def sample(sample_dataset, model, iter, gen_dir):
    model.start_eval()
    cat_list = ["dog", "bus", "bench" , "bicycle", "cake", "laptop", "fork", "car", "boat", "bottle", "motorcycle", "bear", "horse", "vase", "book", "truck", "potted_plant", "airplane", "person","remote","elephant","suitcase","pizza","orange","cup","knife","donut","mouse","fire_hydrant","sandwich"]

    id_list = [0, 3, 4, 5, 6, 37, 35, 49, 72, 77, 110, 120, 106, 149, 212, 269, 509, 854, 10422, 5029, 5190,8154,1838,5920, 6194,6831,82,5824,5858,3977]
    
    #======================================================================================#
    # Check if the file 'attentions.pth' exists, and load it or create a new dictionary
    attention_file = os.path.join(gen_dir, "attentions.pth")
    if os.path.exists(attention_file):
        # Load the existing attention dictionary
        attention_dict = torch.load(attention_file)
    else:
        # Create a new attention dictionary with keys from cat_list and empty dictionaries
        attention_dict = {cat: {} for cat in cat_list}
    #======================================================================================#

    for idx, id in enumerate(id_list):
        cat = cat_list[idx]

        sample_dir = os.path.join(gen_dir, cat)
        os.makedirs(sample_dir, exist_ok=True)
  
              
        indices , bg_size, annids, scids, bg_img_arrs, fg_img_arrs, fg_msk_arrs, bg_img_feats, fg_img_feats, fg_msk_feats, fg_bboxes, catnms, sg_bg_trans, sg_bg_bbox_org, node_attr, edge_index , edge_attr = sample_dataset[id]

        bg_img_arr, fg_img_arr, fg_msk_arr = bg_img_arrs, fg_img_arrs, fg_msk_arrs
        
        fg_bboxes = torch.tensor(fg_bboxes).unsqueeze(0)
        catnms = [catnms]
        sg_bg_bbox_org = sg_bg_bbox_org.unsqueeze(0)

        bg_img_feats = bg_img_feats.unsqueeze(0)
        fg_img_feats = fg_img_feats.unsqueeze(0)
        fg_msk_feats = fg_msk_feats.unsqueeze(0)
        bg_size = bg_size.unsqueeze(0)
        sg_bg_trans = sg_bg_trans.unsqueeze(0)

        pred_trans_, att = model.generator(bg_size, catnms, bg_img_feats, fg_img_feats, fg_msk_feats, fg_bboxes, 
            node_attr, edge_attr, edge_index, sg_bg_bbox_org, sg_bg_trans, is_train=False) 
        
        #================================================#
         # Flatten the attention map into a 2D tensor
        attention = att.squeeze(0,2)  # Assuming attention is a 2D tensor
 
        # Convert the attention tensor to a list of lists
        att_list = attention.tolist()

        # Add the attention list to the dictionary under the appropriate category and iteration
        attention_dict[cat][iter] = att_list

        # Save the updated attention dictionary using torch.save
        torch.save(attention_dict, attention_file)
        #================================================#

        gen_comp_img, _, _ = gen_composite_image(
            bg_img=Image.fromarray(bg_img_arr.astype(np.uint8)).convert('RGB'), 
            fg_img=Image.fromarray(fg_img_arr.astype(np.uint8)).convert('RGB'), 
            fg_msk=Image.fromarray(fg_msk_arr.astype(np.uint8)).convert('L'), 
            trans=(pred_trans_.cpu().numpy().astype(np.float32)[0]).tolist(),
            fg_bbox=None
        )            
        gen_comp_img.save(os.path.join(sample_dir, '{}.jpg'.format(iter)))
                    
    
        
##########################################################

def infer(eval_loader, opt, model=None, repeat=1):
    def csv_title():
        return 'annID,scID,bbox,catnm,label,img_path,msk_path'
    def csv_str(annid, scid, gen_comp_bbox, catnm, gen_file_name):
        return '{},{},"{}",{},-1,images/{}.jpg,masks/{}.png'.format(annid, scid, gen_comp_bbox, catnm, gen_file_name, gen_file_name)

    assert (repeat >= 1)
    save_dir = os.path.join(main_dir,'result', opt.expid)
    eval_dir = os.path.join(save_dir, opt.eval_type, str(opt.epoch))
    assert (not os.path.exists(eval_dir))
    img_sav_dir  = os.path.join(eval_dir, 'images')
    msk_sav_dir  = os.path.join(eval_dir, 'masks')
    csv_sav_file = os.path.join(eval_dir, '{}.csv'.format(opt.eval_type))
    os.makedirs(eval_dir)
    os.mkdir(img_sav_dir)
    os.mkdir(msk_sav_dir)

    attention_file = os.path.join(eval_dir, f"{opt.expid}_{opt.epoch}_{opt.eval_type}_attentions.pth") #Att
    assert (not os.path.isfile(attention_file)) #Att

    #====================================================# when module gets run by itself!
    if model is None:
        from model import GAN
        model_dir = os.path.join(save_dir, 'models')
        model_path = os.path.join(model_dir, str(opt.epoch) + '.pth')
        assert(os.path.exists(model_path))

        model = GAN(opt)
        loaded = torch.load(model_path)
        assert(opt.epoch == loaded['epoch'])
        try:
            model.load_state_dict(loaded['model'], strict=True)
        except RuntimeError as e:
            print(f"Error loading model state dict: {e}")
            return
    #====================================================#


    model.start_eval()

    gen_res = []
    
    attention_dict = {} #Att

    for i, batch in enumerate(tqdm(eval_loader)):
            
        # Skip if the loader returns None
        if batch is None:
            continue
        
        indices , bg_size, annids, scids, bg_img_arrs, fg_img_arrs, fg_msk_arrs, bg_img_feats, fg_img_feats, fg_msk_feats, fg_bboxes, catnms, sg_bg_trans, sg_bg_bbox_org, node_attr, edge_index , edge_attr = batch

        index, annid, scid, bg_img_arr, fg_img_arr, fg_msk_arr, catnm = indices[0], annids[0], scids[0], bg_img_arrs[0], fg_img_arrs[0], fg_msk_arrs[0], catnms[0]

        for repeat_id in range(repeat):                    

            pred_trans_, att = model.test_genorator(bg_size, catnms, bg_img_feats, fg_img_feats, fg_msk_feats, fg_bboxes,node_attr, edge_index , edge_attr, sg_bg_bbox_org, sg_bg_trans)

            gen_comp_img, gen_comp_msk, gen_comp_bbox = gen_composite_image(
                bg_img=Image.fromarray(bg_img_arr.numpy().astype(np.uint8)).convert('RGB'), 
                fg_img=Image.fromarray(fg_img_arr.numpy().astype(np.uint8)).convert('RGB'), 
                fg_msk=Image.fromarray(fg_msk_arr.numpy().astype(np.uint8)).convert('L'), 
                trans=(pred_trans_.cpu().numpy().astype(np.float32)[0]).tolist(),
                fg_bbox=None
            )
            if repeat == 1:
                gen_file_name = "{}_{}_{}_{}_{}_{}_{}".format(index, annid, scid, gen_comp_bbox[0], gen_comp_bbox[1], gen_comp_bbox[2], gen_comp_bbox[3])
            else:
                gen_file_name = "{}_{}_{}_{}_{}_{}_{}_{}".format(index, repeat_id, annid, scid, gen_comp_bbox[0], gen_comp_bbox[1], gen_comp_bbox[2], gen_comp_bbox[3])
            gen_comp_img.save(os.path.join(img_sav_dir, '{}.jpg'.format(gen_file_name)))
            gen_comp_msk.save(os.path.join(msk_sav_dir, '{}.png'.format(gen_file_name)))
            gen_res.append(csv_str(annid, scid, gen_comp_bbox, catnm, gen_file_name))

            #==================================================================# #Att
            # Flatten the attention map into a 2D tensor
            attention = att.squeeze(0,2)  # Assuming attention is a 2D tensor
    
            # Convert the attention tensor to a list of lists
            att_list = attention.tolist()

            attention_dict[gen_file_name]= att_list
            #==================================================================#

    torch.save(attention_dict, attention_file) #Att
            
    with open(csv_sav_file, "w") as f:
        f.write(csv_title() + '\n')
        for line in gen_res:
            f.write(line + '\n')


##########################################################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst", type=str, default="SG_OPADst", help="dataloder type")                                                                             
    parser.add_argument("--expid", type=str, required=True, help="experiment name")
    parser.add_argument("--epoch", type=int, required=True, help="epoch number to evaluate")
    parser.add_argument("--eval_type", type=str, choices=["train", "trainpos", "sample", "eval", "evaluni","real_eval", "real_evaluni", "eval_train" ], default="eval", help="evaluation type")
    parser.add_argument("--repeat", type=int, default=1, help="number of times to repeat inference")
    parser.add_argument("--add_residual", action='store_true', default=True, help="to add residual in attention block")
    parser.add_argument("--img_size", type=int, default=256, help="size of image")
    
    parser.add_argument("--num_nodes", type=int, default=20, help="number of nodes in each scene graph")
    parser.add_argument("--data_root", type=str, default=OPA_path, help="dataset root")
    parser.add_argument("--sg_root", type=str, default=SG_OPA_path, help="scene graph dataset root")
    parser.add_argument("--gpt2_path", type=str, default=gpt2_path, help="path for node/edge gpt2 embeddings ")
    parser.add_argument("--embed_dim", type=int, default=768, help="node/edge embeding dimension")
    parser.add_argument("--embed_freeze", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="to freeze the embedding layer or not")
    parser.add_argument("--gpt2_node_mode", type=str, choices=['category_embedding', 'description_embedding', 'cat_desc_embedding', 'placement_embedding', 'cat_place_embedding', 'cat_desc_place_embedding'], default='cat_desc_place_embedding', help="Mode for selecting node embedding type")

    parser.add_argument("--use_spatial_info", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="to concatenate spatial information on GTN output")
    parser.add_argument("--spatial_dim", type=int, default=256, help="output dimension of Linear layer for bg image objects spatial_info")

    parser.add_argument("--gtn_num_head", type=int, default=8, help="number of attention heads in GTN")
    parser.add_argument("--gtn_num_layer", type=int, default=5, help="number of gtn layers in GTN")
    parser.add_argument("--gtn_hidden_dim", type=int, default=768, help="output dimension of hidden layers in GTN")
    parser.add_argument("--gtn_output_dim", type=int, default=768, help="node/edge dimension of GNN output ")

    #---------------------------------------------------------------------------------------------------------------------# Attention properties
    parser.add_argument("--d_k", type=int, default=64, help="dimension of key in multi-head attention")
    parser.add_argument("--d_v", type=int, default=64, help="dimension of value in multi-head attention")
    parser.add_argument("--n_heads", type=int, default=8, help="number of heads in multi-head attention")

    parser.add_argument("--use_pos_enc", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="to use positional encoding in attention block")

    parser.add_argument("--d_noise", type=int, default=2048, help="dimension of random noise/vector")

    parser.add_argument("--g_lr", type=float, default=0.0001, help="adam: generator learning rate")
    parser.add_argument("--g_weight_decay", type=float, default=0, help="generator weight decay")

    parser.add_argument("--d_lr", type=float, default=0.00002, help="adam: discriminator learning rate")
    parser.add_argument("--d_weight_decay", type=float, default=0, help="discriminator weight decay")

    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    parser.add_argument("--rec_loss_weight", type=float, default=50.0, help="reconstruction loss weight")
    parser.add_argument("--num_d_up_real", type=int, default=2, help="size of bg image cropped objects")

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_args()
    
    eval_loader = get_sg_loader(opt.dst, batch_size=1, num_workers=1, image_size=opt.img_size, mode_type=opt.eval_type, data_root=opt.data_root, sg_root=opt.sg_root, num_nodes = opt.num_nodes)
    with torch.no_grad():        
        infer(eval_loader, opt, model=None, repeat=opt.repeat)
