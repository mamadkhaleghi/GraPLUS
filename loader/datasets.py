import math
import random
import numpy as np
from PIL import Image, ImageFilter
import torch 
from torchvision import transforms
from torchvision.transforms import functional as F

if __name__ == 'loader.datasets':
    from .base import OPABasicDataset, SG_OPABasicDataset
    from .utils import img_crop, get_trans_label
    from .sg_utils import adjust_bbox, bbox_to_trans
elif __name__ == '__main__' or __name__ == 'datasets':
    from base import OPABasicDataset, SG_OPABasicDataset
    from utils import img_crop, get_trans_label
    from sg_utils import adjust_bbox, bbox_to_trans
else:
    raise NotImplementedError


class OPADst1(OPABasicDataset):
    def __init__(self, size, mode_type, data_root):
        super().__init__(size, mode_type, data_root)

    def __getitem__(self, index):
        index_, annid, scid, bbox, scale, label, catnm, bg_img, fg_img, fg_msk, comp_img, comp_msk = super().__getitem__(index)

        bg_img_arr = np.array(bg_img, dtype=np.uint8)
        fg_img_arr = np.array(fg_img, dtype=np.uint8)
        fg_msk_arr = np.array(fg_msk, dtype=np.uint8)
        comp_img_arr = np.array(comp_img, dtype=np.uint8)
        comp_msk_arr = np.array(comp_msk, dtype=np.uint8)

        bg_feat = self.img_trans_bg(bg_img)
        fg_feat = self.img_trans_fg(fg_img, 'color', bg_img, fg_img)
        fg_msk_feat = self.img_trans_fg(fg_msk, 'gray', bg_img, fg_img)
        comp_feat = self.img_trans_bg(comp_img)
        comp_msk_feat = self.img_trans_bg(comp_msk)
        comp_crop_feat = self.img_trans_fg(img_crop(comp_img, 'color', bbox), 'color', bg_img, fg_img)
        fg_bbox = self.get_fg_bbox(bg_img, fg_img)
        trans_label = get_trans_label(bg_img, fg_img, bbox)

        if "eval" in self.mode_type:
            return index_, annid, scid, bg_img_arr, fg_img_arr, fg_msk_arr, comp_img_arr, comp_msk_arr, bg_feat, fg_feat, fg_msk_feat, fg_bbox, comp_feat, comp_msk_feat, comp_crop_feat, label, trans_label, catnm
        else:
            return index_, bg_feat, fg_feat, fg_msk_feat, fg_bbox, comp_feat, comp_msk_feat, comp_crop_feat, label, trans_label, catnm

    def img_trans_bg(self, x):
        y = transforms.Resize((self.size, self.size), interpolation=Image.BILINEAR)(x)
        y = transforms.ToTensor()(y)
        return y

    def img_trans_fg(self, x, x_mode, bg_img, fg_img):
        # assert (math.fabs((x.size[0] * fg_img.size[1]) / (x.size[1] * fg_img.size[0]) - 1.0) < self.error_bar)
        assert (x_mode in ['gray', 'color'])
        bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
        if bg_w / bg_h > fg_w / fg_h:
            y = transforms.Resize((self.size, (self.size * fg_w * bg_h) // (fg_h * bg_w)), interpolation=Image.BILINEAR)(x)
            delta_w = self.size - y.size[0]
            delta_w0, delta_w1 = delta_w // 2, delta_w - (delta_w // 2)
            y_arr = np.array(y, dtype=np.uint8)
            if x_mode == 'gray':
                d0 = np.zeros((self.size, delta_w0), dtype=np.uint8)
                d1 = np.zeros((self.size, delta_w1), dtype=np.uint8)
            else:
                d0 = np.zeros((self.size, delta_w0, y_arr.shape[2]), dtype=np.uint8)
                d1 = np.zeros((self.size, delta_w1, y_arr.shape[2]), dtype=np.uint8)
            y_arr = np.concatenate((d0, y_arr, d1), axis=1)
        else:
            y = transforms.Resize(((self.size * fg_h * bg_w) // (fg_w * bg_h), self.size), interpolation=Image.BILINEAR)(x)
            delta_h = self.size - y.size[1]
            delta_h0, delta_h1 = delta_h // 2, delta_h - (delta_h // 2)
            y_arr = np.array(y, dtype=np.uint8)
            if x_mode == 'gray':
                d0 = np.zeros((delta_h0, self.size), dtype=np.uint8)
                d1 = np.zeros((delta_h1, self.size), dtype=np.uint8)
            else:
                d0 = np.zeros((delta_h0, self.size, y_arr.shape[2]), dtype=np.uint8)
                d1 = np.zeros((delta_h1, self.size, y_arr.shape[2]), dtype=np.uint8)
            y_arr = np.concatenate((d0, y_arr, d1), axis=0)
        y = Image.fromarray(y_arr)
        assert (y.size == (self.size, self.size))
        y = transforms.ToTensor()(y)
        return y

    def get_fg_bbox(self, bg_img, fg_img):
        bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
        if bg_w / bg_h > fg_w / fg_h:
            y_w = (self.size * fg_w * bg_h) // (fg_h * bg_w)
            delta_w0 = (self.size - y_w) // 2
            fg_bbox = np.array([delta_w0, 0, y_w, self.size])
        else:
            y_h = (self.size * fg_h * bg_w) // (fg_w * bg_h)
            delta_h0 = (self.size - y_h) // 2
            fg_bbox = np.array([0, delta_h0, self.size, y_h])
        return fg_bbox


class OPADst3(OPABasicDataset):
    def __init__(self, size, mode_type, data_root):
        super().__init__(size, mode_type, data_root)

    def __getitem__(self, index):
        index_, annid, scid, bbox, scale, label, catnm, bg_img, fg_img, fg_msk, comp_img, comp_msk = super().__getitem__(index)

        comp_w, comp_h = comp_img.size[0], comp_img.size[1]

        bg_img_arr = np.array(bg_img, dtype=np.uint8)
        fg_img_arr = np.array(fg_img, dtype=np.uint8)
        fg_msk_arr = np.array(fg_msk, dtype=np.uint8)
        comp_img_arr = np.array(comp_img, dtype=np.uint8)
        comp_msk_arr = np.array(comp_msk, dtype=np.uint8)

        bg_feat = self.img_trans_bg(bg_img)
        fg_feat = self.img_trans_fg(fg_img, 'color', bg_img, fg_img)
        fg_msk_feat = self.img_trans_fg(fg_msk, 'gray', bg_img, fg_img)
        comp_feat = self.img_trans_bg(comp_img)
        comp_msk_feat = self.img_trans_bg(comp_msk)

        comp_bbox = self.get_resized_bbox(bbox, bg_img, fg_img)

        if "eval" in self.mode_type:
            return index_, annid, scid, comp_w, comp_h, bg_img_arr, fg_img_arr, fg_msk_arr, comp_img_arr, comp_msk_arr, bg_feat, fg_feat, fg_msk_feat, comp_feat, comp_msk_feat, comp_bbox, label, catnm
        else:
            return index_, bg_feat, fg_feat, fg_msk_feat, comp_feat, comp_msk_feat, comp_bbox, label, catnm

    def img_trans_bg(self, x):
        y = transforms.Resize((self.size, self.size), interpolation=Image.BILINEAR)(x)
        y = transforms.ToTensor()(y)
        return y

    def img_trans_fg(self, x, x_mode, bg_img, fg_img):
        # assert (math.fabs((x.size[0] * fg_img.size[1]) / (x.size[1] * fg_img.size[0]) - 1.0) < self.error_bar)
        assert (x_mode in ['gray', 'color'])
        bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
        if bg_w / bg_h > fg_w / fg_h:
            y = transforms.Resize((self.size, (self.size * fg_w * bg_h) // (fg_h * bg_w)), interpolation=Image.BILINEAR)(x)
            delta_w = self.size - y.size[0]
            delta_w0, delta_w1 = delta_w // 2, delta_w - (delta_w // 2)
            y_arr = np.array(y, dtype=np.uint8)
            if x_mode == 'gray':
                d0 = np.zeros((self.size, delta_w0), dtype=np.uint8)
                d1 = np.zeros((self.size, delta_w1), dtype=np.uint8)
            else:
                d0 = np.zeros((self.size, delta_w0, y_arr.shape[2]), dtype=np.uint8)
                d1 = np.zeros((self.size, delta_w1, y_arr.shape[2]), dtype=np.uint8)
            y_arr = np.concatenate((d0, y_arr, d1), axis=1)
        else:
            y = transforms.Resize(((self.size * fg_h * bg_w) // (fg_w * bg_h), self.size), interpolation=Image.BILINEAR)(x)
            delta_h = self.size - y.size[1]
            delta_h0, delta_h1 = delta_h // 2, delta_h - (delta_h // 2)
            y_arr = np.array(y, dtype=np.uint8)
            if x_mode == 'gray':
                d0 = np.zeros((delta_h0, self.size), dtype=np.uint8)
                d1 = np.zeros((delta_h1, self.size), dtype=np.uint8)
            else:
                d0 = np.zeros((delta_h0, self.size, y_arr.shape[2]), dtype=np.uint8)
                d1 = np.zeros((delta_h1, self.size, y_arr.shape[2]), dtype=np.uint8)
            y_arr = np.concatenate((d0, y_arr, d1), axis=0)
        y = Image.fromarray(y_arr)
        assert (y.size == (self.size, self.size))
        y = transforms.ToTensor()(y)
        return y

    def get_resized_bbox(self, bbox, bg_img, fg_img):
        bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
        x1, y1, w, h = bbox
        x2, y2 = x1 + w, y1 + h
        xc = ((x1 + x2) / 2) / bg_w
        yc = ((y1 + y2) / 2) / bg_h
        if bg_w / bg_h > fg_w / fg_h:
            r = h / bg_h
        else:
            r = w / bg_w
        bbox_new = np.array([xc, yc, r], dtype=np.float32)
        return bbox_new


class SG_OPADst(SG_OPABasicDataset): 
    def __init__(self, size, mode_type, data_root, sg_root, num_nodes,
                  flip_augment_prob=0.5, 
                  color_jitter_augment_prob=0.5,
                  grayscale_augment_prob = 0.2,
                  gaussian_blur_augment_prob = 0.5,
                  augment_flag = False ): #####
        
        super().__init__(size, mode_type, data_root, sg_root, num_nodes) #####
        
        self.num_nodes = num_nodes
        self.size = size
        #=============================================================== ####
        self.flip_augment_prob = flip_augment_prob                       #### horizontal flipping probability
        self.color_jitter_augment_prob = color_jitter_augment_prob       #### color jittering probability
        self.grayscale_augment_prob = grayscale_augment_prob
        self.gaussian_blur_augment_prob = gaussian_blur_augment_prob
        self.augment_flag = augment_flag
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        #=============================================================== ####

    def __getitem__(self, index):
        (index_, annid, scid, opa_fg_bbox, scale, label, catnm, 
         bg_img, fg_img, fg_msk, comp_img, comp_msk, 
        sg_bg_bbox, sg_bg_bbox_label, sg_bg_rel_pairs, sg_bg_rel_labels) = super().__getitem__(index) 

        # Adjust detected objects bboxes by sgg bg on bg image in its original size into (bg_img.size)
        # format: [x, y, w, h]
        sg_bg_bbox_org = adjust_bbox(sg_bg_bbox, bg_img.size, bg_img.size)

        # Adjust detected objects bboxes by sgg on resized bg image into (opt.size, opt.size)
        # format: [x, y, w, h]
        sg_bg_bbox_res = adjust_bbox(sg_bg_bbox, bg_img.size, (self.size,self.size) )

        #=============================================================== #### Apply augmentation with the specified probability (just for train mode)
        if "train" in self.mode_type and self.augment_flag :
            # horizontal flipping
            if random.random() < self.flip_augment_prob:
                bg_img, fg_img, fg_msk, comp_img, comp_msk, opa_fg_bbox, sg_bg_bbox_res, sg_bg_bbox_org =\
                    self.flip_augmentation(bg_img, fg_img, fg_msk, comp_img, comp_msk, opa_fg_bbox, sg_bg_bbox_res, sg_bg_bbox_org, self.size)
                
            # color jittering    
            if random.random() < self.color_jitter_augment_prob:
                bg_img, fg_img, comp_img = self.color_jitter_augmentation(bg_img, fg_img, comp_img)

            # Random grayscale
            if random.random() < self.grayscale_augment_prob:
                bg_img, fg_img, comp_img = self.random_grayscale_augmentation( bg_img, fg_img, comp_img)

            # Gaussian blur
            if random.random() < self.gaussian_blur_augment_prob:
                bg_img, fg_img, comp_img = self.gaussian_blur_augmentation(bg_img, fg_img, comp_img)


        #=============================================================== ####
        
        # converting adjusted obj bboxes of sg data on bg image on its original size into t[t_r, t_x, t_y] format
        # does the same work that `get_trans_label` does, but for all the detected objects by SGG model 
        # sg_bg_trans shape: (N, 3)
        sg_bg_trans = bbox_to_trans(sg_bg_bbox_org, bg_img)

        # t[t_r, t_x, t_y] computed using original bg/fg images and opa_fg_bbox in the OPA dataset 
        trans_label = get_trans_label(bg_img, fg_img, opa_fg_bbox)
        
        # bbox of fg in `resized and padded` fg image which is in (opt.size, opt.size) shape
        # format: [x, y, w, h]
        fg_bbox = self.get_fg_bbox(bg_img, fg_img) 


        # original_size = bg_img.size  # (width, height) of the original image
        bg_size = torch.tensor([bg_img.size[0],bg_img.size[1]])

        bg_img_arr   = np.array(bg_img, dtype=np.uint8)
        fg_img_arr   = np.array(fg_img, dtype=np.uint8)
        fg_msk_arr   = np.array(fg_msk, dtype=np.uint8)

        bg_feat       = self.img_trans_bg(bg_img)    # torch.Tensor ---> shape: [C,H,W] = [3, opt.size, opt.size ]
        comp_feat     = self.img_trans_bg(comp_img)  # torch.Tensor ---> shape: [C,H,W] = [3, opt.size, opt.size ]
        comp_msk_feat = self.img_trans_bg(comp_msk)  # torch.Tensor ---> shape: [C,H,W] = [1, opt.size, opt.size ]

        fg_feat        = self.img_trans_fg(fg_img, 'color', bg_img, fg_img) # torch.Tensor ---> shape: [C,H,W] = [3, opt.size, opt.size ]
        fg_msk_feat    = self.img_trans_fg(fg_msk, 'gray',  bg_img, fg_img) # torch.Tensor ---> shape: [C,H,W] = [1, opt.size, opt.size ]
        # comp_crop_feat = self.img_trans_fg(img_crop(comp_img, 'color', opa_fg_bbox), 'color', bg_img, fg_img) # torch.Tensor ---> shape: [C,H,W] = [3, opt.size, opt.size ]


        node_attr = sg_bg_bbox_label
        edge_index = sg_bg_rel_pairs.view(2,-1)
        edge_attr = sg_bg_rel_labels

        if "eval" in self.mode_type or "sample" in self.mode_type:
            return (index_ , bg_size, annid, scid, bg_img_arr, fg_img_arr, fg_msk_arr, bg_feat, fg_feat, fg_msk_feat, fg_bbox, catnm, sg_bg_trans, sg_bg_bbox_org, node_attr ,  edge_index , edge_attr) 
        else:
            return (index_,bg_size, bg_feat, fg_feat, fg_msk_feat, fg_bbox, comp_feat, comp_msk_feat, label, trans_label, catnm, sg_bg_trans, sg_bg_bbox_org, node_attr ,edge_index ,edge_attr ) 

    def img_trans_bg(self, x):
        y = transforms.Resize((self.size, self.size), interpolation=Image.BILINEAR)(x)
        y = transforms.ToTensor()(y)
        return y

    def img_trans_fg(self, x, x_mode, bg_img, fg_img):
        # assert (math.fabs((x.size[0] * fg_img.size[1]) / (x.size[1] * fg_img.size[0]) - 1.0) < self.error_bar)
        assert (x_mode in ['gray', 'color'])
        bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
        if bg_w / bg_h > fg_w / fg_h:
            y = transforms.Resize((self.size, (self.size * fg_w * bg_h) // (fg_h * bg_w)), interpolation=Image.BILINEAR)(x)
            delta_w = self.size - y.size[0]
            delta_w0, delta_w1 = delta_w // 2, delta_w - (delta_w // 2)
            y_arr = np.array(y, dtype=np.uint8)
            if x_mode == 'gray':
                d0 = np.zeros((self.size, delta_w0), dtype=np.uint8)
                d1 = np.zeros((self.size, delta_w1), dtype=np.uint8)
            else:
                d0 = np.zeros((self.size, delta_w0, y_arr.shape[2]), dtype=np.uint8)
                d1 = np.zeros((self.size, delta_w1, y_arr.shape[2]), dtype=np.uint8)
            y_arr = np.concatenate((d0, y_arr, d1), axis=1)
        else:
            y = transforms.Resize(((self.size * fg_h * bg_w) // (fg_w * bg_h), self.size), interpolation=Image.BILINEAR)(x)
            delta_h = self.size - y.size[1]
            delta_h0, delta_h1 = delta_h // 2, delta_h - (delta_h // 2)
            y_arr = np.array(y, dtype=np.uint8)
            if x_mode == 'gray':
                d0 = np.zeros((delta_h0, self.size), dtype=np.uint8)
                d1 = np.zeros((delta_h1, self.size), dtype=np.uint8)
            else:
                d0 = np.zeros((delta_h0, self.size, y_arr.shape[2]), dtype=np.uint8)
                d1 = np.zeros((delta_h1, self.size, y_arr.shape[2]), dtype=np.uint8)
            y_arr = np.concatenate((d0, y_arr, d1), axis=0)
        y = Image.fromarray(y_arr)
        assert (y.size == (self.size, self.size))
        y = transforms.ToTensor()(y)
        return y

    def get_fg_bbox(self, bg_img, fg_img):
        bg_w, bg_h, fg_w, fg_h = bg_img.size[0], bg_img.size[1], fg_img.size[0], fg_img.size[1]
        if bg_w / bg_h > fg_w / fg_h:
            y_w = (self.size * fg_w * bg_h) // (fg_h * bg_w)
            delta_w0 = (self.size - y_w) // 2
            fg_bbox = np.array([delta_w0, 0, y_w, self.size])
        else:
            y_h = (self.size * fg_h * bg_w) // (fg_w * bg_h)
            delta_h0 = (self.size - y_h) // 2
            fg_bbox = np.array([0, delta_h0, self.size, y_h])
        return fg_bbox

    #=============================================================== #### Augmentation Implementation
    def flip_augmentation(self,bg_img, fg_img, fg_msk, comp_img, comp_msk, opa_fg_bbox, sg_bg_bbox_res, sg_bg_bbox_org, target_size):

        #-----------------------------------------#
        def flip_bbox(bbox, img_width):
            x, y, w, h = bbox
            flipped_x = img_width - x - w
            flipped_opa_bbox = [flipped_x, y, w, h]
            assert (flipped_opa_bbox[0] + flipped_opa_bbox[2] <= bg_img.size[0] and flipped_opa_bbox[1] + flipped_opa_bbox[3] <= bg_img.size[1])
            return flipped_opa_bbox
        #-----------------------------------------# obj bboxes in sg are defined in bg image in its resized size(opt.size, opt.size)
        def flip_sg_bboxes_res(bboxes, target_size):
            flipped_bboxes = bboxes.clone()
            flipped_bboxes[:, 0] = target_size - bboxes[:, 0] - bboxes[:, 2]  # x' = opt.size - x - w

            # bbox check
            bbox = flipped_bboxes.clone()
            bbox[:,2]=bbox[:,2]+bbox[:,0]
            bbox[:,3]=bbox[:,3]+bbox[:,1]
            assert(bbox.max()<=target_size)

            return flipped_bboxes
        #-----------------------------------------#obj bboxes in sg are defined in bg image in its original size (bg_img.size)
        def flip_sg_bboxes_org(bboxes, bg_img):
            bg_w, bg_h = bg_img.size[0],bg_img.size[1] 

            flipped_bboxes = bboxes.clone()
            flipped_bboxes[:, 0] = bg_w - bboxes[:, 0] - bboxes[:, 2]  # x' = bg_w - x - w

            # bbox check
            bbox = flipped_bboxes.clone()
            bbox[:,2]=bbox[:,2]+bbox[:,0]
            bbox[:,3]=bbox[:,3]+bbox[:,1]
            assert(bbox[:,2].max()<=bg_w)
            assert(bbox[:,3].max()<=bg_h)

            return flipped_bboxes
        #-----------------------------------------#
        # Apply horizontal flip to images and masks
        bg_img   = bg_img.transpose(Image.FLIP_LEFT_RIGHT)
        fg_img   = fg_img.transpose(Image.FLIP_LEFT_RIGHT)
        fg_msk   = fg_msk.transpose(Image.FLIP_LEFT_RIGHT)
        comp_img = comp_img.transpose(Image.FLIP_LEFT_RIGHT)
        comp_msk = comp_msk.transpose(Image.FLIP_LEFT_RIGHT)

        # Update bounding boxes
        bg_w = bg_img.width
        opa_fg_bbox = flip_bbox(opa_fg_bbox, bg_w) # opa_fg_bbox has defined for bg image in original size

        # Update scene graph detected objects bounding boxes
        sg_bg_bbox_res =flip_sg_bboxes_res(sg_bg_bbox_res, target_size)
        sg_bg_bbox_org =flip_sg_bboxes_org(sg_bg_bbox_org, bg_img)


        return bg_img, fg_img, fg_msk, comp_img, comp_msk, opa_fg_bbox, sg_bg_bbox_res,sg_bg_bbox_org

    # Function to apply the color jitter augmentation with shared parameters
    def color_jitter_augmentation(self, bg_img, fg_img, comp_img):
        
        def _get_random_factor(attr):
            if isinstance(attr, tuple):
                # If the attribute is a tuple, select a random value within the range
                return random.uniform(attr[0], attr[1])
            else:
                # Otherwise, use the value itself (assuming it's a float)
                return random.uniform(max(0, 1 - attr), 1 + attr)
    
        # Generate random parameters for brightness, contrast, saturation, and hue
        brightness_factor = _get_random_factor(self.color_jitter.brightness)
        contrast_factor = _get_random_factor(self.color_jitter.contrast)
        saturation_factor = _get_random_factor(self.color_jitter.saturation)
        hue_factor = _get_random_factor(self.color_jitter.hue)

        # Apply the same parameters to all three images
        bg_img = F.adjust_brightness(bg_img, brightness_factor)
        fg_img = F.adjust_brightness(fg_img, brightness_factor)
        comp_img = F.adjust_brightness(comp_img, brightness_factor)

        bg_img = F.adjust_contrast(bg_img, contrast_factor)
        fg_img = F.adjust_contrast(fg_img, contrast_factor)
        comp_img = F.adjust_contrast(comp_img, contrast_factor)

        bg_img = F.adjust_saturation(bg_img, saturation_factor)
        fg_img = F.adjust_saturation(fg_img, saturation_factor)
        comp_img = F.adjust_saturation(comp_img, saturation_factor)

        bg_img = F.adjust_hue(bg_img, hue_factor)
        fg_img = F.adjust_hue(fg_img, hue_factor)
        comp_img = F.adjust_hue(comp_img, hue_factor)

        return bg_img, fg_img, comp_img
    
    def random_grayscale_augmentation(self, bg_img, fg_img, comp_img):
        bg_img = F.rgb_to_grayscale(bg_img, num_output_channels=3)
        fg_img = F.rgb_to_grayscale(fg_img, num_output_channels=3)
        comp_img = F.rgb_to_grayscale(comp_img, num_output_channels=3)

        return bg_img, fg_img, comp_img
    

    def gaussian_blur_augmentation(self, bg_img, fg_img, comp_img):
        blur_radius = random.uniform(0.2, 1)

        bg_img = bg_img.filter(ImageFilter.GaussianBlur(blur_radius))
        fg_img = fg_img.filter(ImageFilter.GaussianBlur(blur_radius))
        comp_img = comp_img.filter(ImageFilter.GaussianBlur(blur_radius))

        return bg_img, fg_img, comp_img
