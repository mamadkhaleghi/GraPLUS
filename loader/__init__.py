from torch.utils.data import DataLoader
from functools import partial
import torch 

if __name__ == 'loader':
    from .base import OPABasicDataset, SG_OPABasicDataset, BalancedSampler, TwoToOneSampler, PositiveSampler
    from .datasets import OPADst1, OPADst3, SG_OPADst
elif __name__ == '__init__':
    from base import OPABasicDataset, SG_OPABasicDataset, BalancedSampler, TwoToOneSampler, PositiveSampler
    from datasets import OPADst1, OPADst3, SG_OPADst
else:
    raise NotImplementedError

dataset_dict = {"OPABasicDataset": OPABasicDataset, "SG_OPABasicDataset": SG_OPABasicDataset, "OPADst1": OPADst1, "OPADst3": OPADst3, "SG_OPADst": SG_OPADst}


##########################################################################
def get_loader(name, batch_size, num_workers, image_size, shuffle, mode_type, data_root):
    dset = dataset_dict[name](size=image_size, mode_type=mode_type, data_root=data_root)
    loader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def get_dataset(name, image_size, mode_type, data_root):
    dset = dataset_dict[name](size=image_size, mode_type=mode_type, data_root=data_root)
    return dset



##########################################################################

def get_sg_loader(name, batch_size, num_workers, image_size, mode_type, data_root, sg_root, num_nodes,
               flip_augment_prob=0.5, 
               color_jitter_augment_prob=0.5,
               grayscale_augment_prob = 0.2,
               gaussian_blur_augment_prob = 0.1,
               augment_flag = False,
               sampler_type="default"):
    
    dset = dataset_dict[name](size=image_size, mode_type=mode_type, data_root=data_root, sg_root=sg_root, num_nodes=num_nodes,
                              flip_augment_prob=flip_augment_prob, 
                              color_jitter_augment_prob=color_jitter_augment_prob,
                              grayscale_augment_prob = grayscale_augment_prob,
                              gaussian_blur_augment_prob =gaussian_blur_augment_prob,
                              augment_flag = augment_flag)
    
    my_sampler = None # default sampler (no custom sampler)
   #=======================================================================# sampler_selection
    # Select the appropriate sampler only if "train" is in mode_type
    if "train" in mode_type:
        if sampler_type == "balance_sampler":
            my_sampler = BalancedSampler(dset.pos_indices, dset.neg_indices, batch_size)
        elif sampler_type == "TwoToOne_sampler":
            my_sampler = TwoToOneSampler(dset.pos_indices, dset.neg_indices, batch_size)
        elif sampler_type == "PositiveSampler":
            my_sampler = PositiveSampler(dset.pos_indices, shuffle=True)
        elif sampler_type != "default":
            raise ValueError(f"Unknown sampler_type: {sampler_type}")
    
    collate_fn_with_num_nodes = partial(custom_collate_fn, num_node=num_nodes, mode_type=mode_type)

    loader = DataLoader(
                        dset,
                        batch_size=batch_size,
                        sampler=my_sampler,
                        shuffle= my_sampler is None and "train" in mode_type,
                        num_workers=num_workers,
                        collate_fn=collate_fn_with_num_nodes,
                        pin_memory=True  # Set to True if using GPU
        )
    return loader

def get_sg_dataset(name, image_size, mode_type, data_root, sg_root, num_nodes, 
                flip_augment_prob=0.5, 
                color_jitter_augment_prob=0.5,
                grayscale_augment_prob = 0.2,
                gaussian_blur_augment_prob = 0.1,
                augment_flag= False): #####
    
    dset = dataset_dict[name](size=image_size, mode_type=mode_type, data_root=data_root, sg_root=sg_root, num_nodes=num_nodes,
                              flip_augment_prob=flip_augment_prob, 
                              color_jitter_augment_prob=color_jitter_augment_prob,
                              grayscale_augment_prob = grayscale_augment_prob,
                              gaussian_blur_augment_prob =gaussian_blur_augment_prob,
                              augment_flag = augment_flag) #####
    return dset


def custom_collate_fn(batch, num_node, mode_type):
    """
    Custom collate function for handling the batching operation of scene graph data,
    with filtering based on sg_bg_bbox.shape[0].
    
    Args:
        batch: List of samples.
        num_node: Minimum number of nodes required for sg_bg_bbox.
        
    Returns:
        Filtered batch.
    """
    # Filter the batch to exclude samples where sg_bg_bbox.shape[0] < num_node
    # filtered_batch = [sample for sample in batch if sample[-6].shape[0] >= num_node]  # sg_bg_bbox is the 4th last element
    
    # Handle empty filtered batch
    # if not filtered_batch:
    #     return None  # Return

    # Process filtered_batch as before
    if "eval" in mode_type or "sample" in mode_type:

        index_ , bg_size, annid, scid, bg_img_arr, fg_img_arr, fg_msk_arr, bg_feat, fg_feat, fg_msk_feat, fg_bbox, catnm, sg_bg_trans, sg_bg_bbox_org, node_attr,  edge_index, edge_attr = zip(*batch)
        
        # Convert to tensors
        index_ = torch.tensor(index_)
        annid  = torch.tensor(annid)
        scid   = torch.tensor(scid)

        fg_bbox     = torch.stack([torch.tensor(bbox) for bbox in fg_bbox    ])

        bg_img_arr   = torch.stack([torch.tensor(img) for img in bg_img_arr  ])
        fg_img_arr   = torch.stack([torch.tensor(img) for img in fg_img_arr  ])
        fg_msk_arr   = torch.stack([torch.tensor(img) for img in fg_msk_arr  ])

        bg_size           = torch.stack(bg_size)

        bg_feat           = torch.stack(bg_feat)
        fg_feat           = torch.stack(fg_feat)
        fg_msk_feat       = torch.stack(fg_msk_feat)
        sg_bg_trans       = torch.stack(sg_bg_trans)
        sg_bg_bbox_org    = torch.stack(sg_bg_bbox_org)

        # Process and concatenate graph data
        node_attr = torch.cat(node_attr, dim=0)  # Concatenate along the node dimension
        edge_index = torch.cat([ei + i * num_node for i, ei in enumerate(edge_index)], dim=1)  # Offset edge indices for batch
        edge_attr = torch.cat(edge_attr, dim=0)  # Concatenate edge attributes

        return (index_ , bg_size, annid, scid, bg_img_arr, fg_img_arr, fg_msk_arr, bg_feat, fg_feat, fg_msk_feat, fg_bbox, catnm, sg_bg_trans, sg_bg_bbox_org, node_attr,  edge_index, edge_attr)

    else:
        index_,bg_size, bg_feat, fg_feat, fg_msk_feat, fg_bbox, comp_feat, comp_msk_feat, label, trans_label, catnm, sg_bg_trans, sg_bg_bbox_org, node_attr ,edge_index ,edge_attr = zip(*batch)
        
        # Convert to tensors
        index_ = torch.tensor(index_)
        label  = torch.tensor(label)
        fg_bbox     = torch.stack([torch.tensor(bbox) for bbox in fg_bbox])
        trans_label = torch.stack([torch.tensor(t) for t in trans_label])

        bg_size        = torch.stack(bg_size)

        bg_feat        = torch.stack(bg_feat)
        fg_feat        = torch.stack(fg_feat)
        fg_msk_feat    = torch.stack(fg_msk_feat)
        comp_feat      = torch.stack(comp_feat)
        comp_msk_feat  = torch.stack(comp_msk_feat)
        sg_bg_trans    = torch.stack(sg_bg_trans)
        sg_bg_bbox_org = torch.stack(sg_bg_bbox_org)

        # Process and concatenate graph data
        node_attr = torch.cat(node_attr, dim=0)  # Concatenate along the node dimension
        edge_index = torch.cat([ei + i * num_node for i, ei in enumerate(edge_index)], dim=1)  # Offset edge indices for batch
        edge_attr = torch.cat(edge_attr, dim=0)  # Concatenate edge attributes
        
        return (index_,bg_size, bg_feat, fg_feat, fg_msk_feat, fg_bbox, comp_feat, comp_msk_feat, label, trans_label, catnm, sg_bg_trans, sg_bg_bbox_org, node_attr ,edge_index ,edge_attr)
    

# ###############################################################################################
