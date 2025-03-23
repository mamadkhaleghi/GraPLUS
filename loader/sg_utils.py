import torch
from PIL import Image

from torchvision.ops import roi_align
from torchvision.transforms.functional import to_tensor



def dict_extract(sg_dict, readable_category=False ): # returns customized scene graph with specified number of nodes 
                                                                    # in its appropriate form to get used as GNN data

    sg_bbox = torch.tensor( sg_dict["bbox"] , dtype=torch.float32)  
    sg_bbox_labels = torch.tensor(sg_dict["bbox_labels"] , dtype=torch.long)       
    sg_rel_pairs = torch.tensor( sg_dict["rel_pairs"], dtype=torch.long)          
    sg_rel_labels = torch.tensor( sg_dict["rel_labels"] , dtype=torch.long )        

    #-----------------------------------------------------#
    if readable_category:
        sg_bbox_labels, sg_rel_labels = label_index_to_category(sg_bbox_labels, sg_rel_labels)
    #-----------------------------------------------------#

    return sg_bbox, sg_bbox_labels, sg_rel_pairs, sg_rel_labels

#---------------------------------------------------------------#
def label_index_to_category(box_labels, rel_labels):  # converts label indices into human-readable class names

    obj_labels_list = ['__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
    rel_labels_list = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind', 'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for', 'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
    
    for i in range(len(box_labels)):
        box_labels[i] = obj_labels_list[ box_labels[i] ]

    for i in range(len(rel_labels)):
        rel_labels[i] = rel_labels_list[ rel_labels[i] ]

    return torch.Tensor(box_labels) , torch.Tensor(rel_labels)

#==============================================================================================
def adjust_bbox(sg_bbox, original_size, target_size): # Adjust detected objects bboxes by sgg after resizing bg image into (opt.size, opt.size)
     
    """
    Adjust bounding boxes to a fixed target size after resizing.
    
    Args:
    - sg_bbox (Tensor): Bounding boxes in format [x_min, y_min, x_max, y_max] with shape [num_node, 4]
    - original_size (tuple): Original image size (w, h)
    - target_size (tuple): The new fixed size (w,h) to resize the image to.
    
    Returns:
    - adjusted_bboxes (Tensor): Bounding boxes adjusted for the target size. in format [x, y, w, h]

    """
     
    # Size after applying `get_size()` function (w_resized, h_resized)
    resized_size = get_size(original_size)

    # Calculate scaling factors
    scale_w = target_size[0] / resized_size[0]
    scale_h = target_size[1] / resized_size[1]
    
    # Scale the bounding boxes
    adjusted_bboxes = sg_bbox.clone()
    adjusted_bboxes[:, 0] *= scale_w  # x_min
    adjusted_bboxes[:, 1] *= scale_h  # y_min
    adjusted_bboxes[:, 2] *= scale_w  # x_max
    adjusted_bboxes[:, 3] *= scale_h  # y_max

    adjusted_bboxes[:, 2] = adjusted_bboxes[:, 2]-adjusted_bboxes[:, 0] # w
    adjusted_bboxes[:, 3] = adjusted_bboxes[:, 3]-adjusted_bboxes[:, 1] # h

    # Clip bounding boxes to be within the target size (optional)
    # adjusted_bboxes = torch.clamp(adjusted_bboxes, min=0, max=target_size)
    bbox = adjusted_bboxes.clone()
    bbox[:,2]=bbox[:,2]+bbox[:,0]
    bbox[:,3]=bbox[:,3]+bbox[:,1]

    assert(bbox[:,2].max()<target_size[0])
    assert(bbox[:,3].max()<target_size[1])

    return adjusted_bboxes

#==============================================================================================

def get_size(Image_size): # gets image size (w,h) and returns the resized size (new_w, new_h)
    '''
    (the generated bboxes by `SGG` model are for resized images that their resized size gets computed by this function!)
    this method is to ensure that the resized image dimensions fit within a given range, 
    specifically between min_size (600) and max_size (1000) pixels, 
    while maintaining the original aspect ratio of the image.
    '''
    min_size = 600
    max_size = 1000

    w,h = Image_size
    size = min_size

    # Calculate the new size while maintaining the aspect ratio
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))

        # Adjust the size if it exceeds the max size
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size)) # size would be < 1000

    # Return the original size if it already fits within the constraints
    if (w <= h and w == size) or (h <= w and h == size):
        return (w, h)

    # Calculate the new dimensions based on the aspect ratio
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return (ow, oh)


###################################################################################################

def bbox_to_trans(sg_bg_bbox_org, bg_img):       
    '''
    converting adjusted obj bboxes of sg data on bg image on its original size into t[t_r, t_x, t_y] format

    sg_bg_bbox_org shape: [N, 4] 

    t shape: [N, 3]
    '''
    
    bg_w, bg_h = bg_img.size[0], bg_img.size[1]
    AR_obj = sg_bg_bbox_org[:,2]/sg_bg_bbox_org[:,3]
    AR_bg  = bg_w/bg_h

    mask = AR_obj<AR_bg
    
    t_r_y  = sg_bg_bbox_org[:,3] / bg_h
    t_r_x  = sg_bg_bbox_org[:,2] / bg_w
    t_r = torch.where(mask, t_r_y, t_r_x)
    t_x  = sg_bg_bbox_org[:,0] / (bg_w - sg_bg_bbox_org[:,2])
    t_y  = sg_bg_bbox_org[:,1] / (bg_h - sg_bg_bbox_org[:,3])

    t = torch.stack((t_r, t_x, t_y), dim=1)
    assert (t.min() >= 0 and t.max() <= 1)

    return t

###################################################################################################
def convert_opa_bbox(img_path, opa_fg_bbox):  # returns converted opa fg bbox on resized image  
                                              # # opa_bbox: a list of 4 strings
    fg_bbox = opa_fg_bbox
    original_size = Image.open(img_path).size
    resized_size = get_size(original_size)
    
    RS = resized_size[0] / original_size[0]  # Resizing scale

    fg_bbox = [int(x) for x in fg_bbox]

    fg_bbox[2]= fg_bbox[0] + fg_bbox[2]
    fg_bbox[3]= fg_bbox[1] + fg_bbox[3]

    fg_bbox = [int(RS*x) for x in fg_bbox]

    return fg_bbox
