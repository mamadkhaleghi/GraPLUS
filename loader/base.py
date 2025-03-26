import os
from PIL import Image
from torch.utils.data import Dataset, Sampler
import json
import random 

if __name__ == 'loader.base':
    from .utils import obtain_opa_data
    from .sg_utils import dict_extract
elif __name__ == '__main__' or __name__ == 'base':
    from utils import obtain_opa_data
    from sg_utils import dict_extract
else:
    raise NotImplementedError


class OPABasicDataset(Dataset):
    def __init__(self, size, mode_type, data_root):
        # self.error_bar = 0.15
        self.size = size
        self.mode_type = mode_type
        self.data_root = data_root
        self.bg_dir = os.path.join(data_root, "background")
        self.fg_dir = os.path.join(data_root, "foreground")
        self.fg_msk_dir = os.path.join(data_root, "foreground")

        if mode_type == "train":
            csv_file = os.path.join(data_root, "train_data.csv")
        elif mode_type == "trainpos":
            csv_file = os.path.join(data_root, "train_data_pos.csv")
        elif mode_type == "sample":
            csv_file = os.path.join(data_root, "test_data.csv")
        elif mode_type == "eval":
            csv_file = os.path.join(data_root, "test_data_pos.csv")
        elif mode_type == "evaluni":
            csv_file = os.path.join(data_root, "test_data_pos_unique.csv")
        else:
            raise NotImplementedError
        self.data = obtain_opa_data(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index_, annid, scid, bbox, scale, label, catnm, img_path, msk_path = self.data[index]

        bg_path = os.path.join(self.bg_dir, catnm, "{}.jpg".format(scid))
        fg_path = os.path.join(self.fg_dir, catnm, "{}.jpg".format(annid))
        fg_mask_path = os.path.join(self.fg_msk_dir, catnm, "mask_{}.jpg".format(annid))
        img_path = os.path.join(self.data_root, img_path)
        msk_path = os.path.join(self.data_root, msk_path)

        bg_img = Image.open(bg_path).convert('RGB')
        fg_img = Image.open(fg_path).convert('RGB')
        fg_msk = Image.open(fg_mask_path).convert('L')
        comp_img = Image.open(img_path).convert('RGB')
        comp_msk = Image.open(msk_path).convert('L')

        assert (bg_img.size == comp_img.size and comp_img.size == comp_msk.size and fg_img.size == fg_msk.size)
        # assert (math.fabs((bbox[2] * fg_img.size[1]) / (bbox[3] * fg_img.size[0]) - 1.0) < self.error_bar)
        assert (bbox[0] + bbox[2] <= bg_img.size[0] and bbox[1] + bbox[3] <= bg_img.size[1])

        return index_, annid, scid, bbox, scale, label, catnm, bg_img, fg_img, fg_msk, comp_img, comp_msk



class SG_OPABasicDataset(Dataset):
    def __init__(self, size, mode_type, data_root , sg_root, num_nodes): #####
        self.size = size
        self.mode_type = mode_type
        self.data_root = data_root
        self.bg_dir = os.path.join(data_root, "background")
        self.fg_dir = os.path.join(data_root, "foreground")
        self.fg_msk_dir = os.path.join(data_root, "foreground")

        
        self.num_node = num_nodes                                            

        self.sg_bg_dir = os.path.join(sg_root, f"sg_opa_background_n{num_nodes}")

        if mode_type == "train":
            csv_file = os.path.join(data_root, "train_data.csv")           
        elif mode_type == "trainpos":
            csv_file = os.path.join(data_root, "train_data_pos.csv")       
        elif mode_type == "sample":
            csv_file = os.path.join(data_root, "test_data.csv")            
        elif mode_type == "eval":
            csv_file = os.path.join(data_root, "test_data_pos.csv")        
        elif mode_type == "evaluni":
            csv_file = os.path.join(data_root, "test_data_pos_unique.csv") 

        else:
            raise NotImplementedError
        self.data = obtain_opa_data(csv_file)

        # =============================================== ####
        self.pos_indices = []
        self.neg_indices = []
        for idx, data_item in enumerate(self.data):
            label = data_item[5]  # Label is at index 5
            if label == 1:
                self.pos_indices.append(idx)
            elif label == 0:
                self.neg_indices.append(idx)
        # ===============================================

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        index_, annid, scid, opa_fg_bbox, scale, label, catnm, img_path, msk_path = self.data[index]

        sg_bg_path   = os.path.join(self.sg_bg_dir, catnm, "sg_{}.json".format(scid)) 
        sg_bg_dict   = json.load(open(sg_bg_path))   
        
        sg_bg_bbox, sg_bg_bbox_label, sg_bg_rel_pairs, sg_bg_rel_labels = dict_extract(sg_bg_dict)          
        
        bg_path      = os.path.join(self.bg_dir, catnm, "{}.jpg".format(scid))
        fg_path      = os.path.join(self.fg_dir, catnm, "{}.jpg".format(annid))
        fg_mask_path = os.path.join(self.fg_msk_dir, catnm, "mask_{}.jpg".format(annid))
        img_path     = os.path.join(self.data_root, img_path)
        msk_path     = os.path.join(self.data_root, msk_path)

        bg_img   = Image.open(bg_path).convert('RGB')
        fg_img   = Image.open(fg_path).convert('RGB')
        fg_msk   = Image.open(fg_mask_path).convert('L')
        comp_img = Image.open(img_path).convert('RGB')
        comp_msk = Image.open(msk_path).convert('L')


        assert (bg_img.size == comp_img.size and comp_img.size == comp_msk.size and fg_img.size == fg_msk.size)
        assert (opa_fg_bbox[0] + opa_fg_bbox[2] <= bg_img.size[0] and opa_fg_bbox[1] + opa_fg_bbox[3] <= bg_img.size[1])

        return (index_, annid, scid, opa_fg_bbox, scale, label, catnm, 
                bg_img, fg_img, fg_msk, comp_img, comp_msk,
                sg_bg_bbox, sg_bg_bbox_label, sg_bg_rel_pairs, sg_bg_rel_labels) 
                

class BalancedSampler(Sampler):
    '''
    * All data is used in each epoch.
    * 'Positive' and 'negative' samples are always balanced in each batch.
    * It's okay for some positive samples to be used more than once in each epoch.
    * The positive samples that are repeated should vary across different epochs.
    '''
    def __init__(self, pos_indices, neg_indices, batch_size):
        assert batch_size % 2 == 0, "Batch size must be even."
        self.batch_size = batch_size
        self.half_batch_size = batch_size // 2
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices

    def __iter__(self):
        # Shuffle the indices at the start of each epoch
        random.shuffle(self.pos_indices)
        random.shuffle(self.neg_indices)

        # Calculate the number of batches needed to use all negative samples
        num_batches = len(self.neg_indices) // self.half_batch_size

        # Oversample positive indices to match the number of negative indices
        pos_indices = self.pos_indices.copy()
        if len(pos_indices) < len(self.neg_indices):
            # Randomly sample with replacement from positive indices to match the required length
            pos_indices = random.choices(pos_indices, k=len(self.neg_indices))

        # Truncate or match the positive indices to ensure equal number of batches
        pos_indices = pos_indices[:num_batches * self.half_batch_size]
        neg_indices = self.neg_indices[:num_batches * self.half_batch_size]

        # Build the list of indices for balanced batches
        indices = []
        for i in range(num_batches):
            pos_batch = pos_indices[i * self.half_batch_size:(i + 1) * self.half_batch_size]
            neg_batch = neg_indices[i * self.half_batch_size:(i + 1) * self.half_batch_size]

            batch = pos_batch + neg_batch
            random.shuffle(batch)  # Shuffle within the batch to mix positive and negative samples
            indices.extend(batch)

        return iter(indices)

    def __len__(self):
        # The total number of samples that will be used in each epoch
        return len(self.neg_indices) // self.half_batch_size * self.batch_size


class TwoToOneSampler(Sampler):
    '''
    * All data is used in each epoch.
    * Each batch will have twice as many positive samples as negative samples.
    * It's okay for some positive samples to be used more than once in each epoch.
    * The positive samples that are repeated should vary across different epochs.
    '''
    def __init__(self, pos_indices, neg_indices, batch_size):
        assert batch_size % 3 == 0, "Batch size must be divisible by 3."
        self.batch_size = batch_size
        self.pos_per_batch = 2 * (batch_size // 3)  # Two-thirds of the batch
        self.neg_per_batch = batch_size // 3        # One-third of the batch
        self.pos_indices = pos_indices
        self.neg_indices = neg_indices

    def __iter__(self):
        # Shuffle the indices at the start of each epoch
        random.shuffle(self.pos_indices)
        random.shuffle(self.neg_indices)

        # Calculate the number of batches needed to use all negative samples
        num_batches = len(self.neg_indices) // self.neg_per_batch

        # Oversample positive indices to match the required number
        pos_indices = self.pos_indices.copy()
        if len(pos_indices) < num_batches * self.pos_per_batch:
            # Randomly sample with replacement from positive indices to match the required length
            pos_indices = random.choices(pos_indices, k=num_batches * self.pos_per_batch)

        # Truncate or match the indices to ensure equal number of batches
        pos_indices = pos_indices[:num_batches * self.pos_per_batch]
        neg_indices = self.neg_indices[:num_batches * self.neg_per_batch]

        # Build the list of indices for balanced batches
        indices = []
        for i in range(num_batches):
            pos_batch = pos_indices[i * self.pos_per_batch:(i + 1) * self.pos_per_batch]
            neg_batch = neg_indices[i * self.neg_per_batch:(i + 1) * self.neg_per_batch]
            batch = pos_batch + neg_batch
            random.shuffle(batch)  # Shuffle within the batch to mix positive and negative samples
            indices.extend(batch)

        return iter(indices)

    def __len__(self):
        # The total number of samples that will be used in each epoch
        return len(self.neg_indices) // self.neg_per_batch * self.batch_size



class PositiveSampler(Sampler):
    """
    Sampler that yields only positive samples.

    Args:
        pos_indices (list): List of indices corresponding to positive samples.
        shuffle (bool): Whether to shuffle the positive indices each epoch.
    """
    def __init__(self, pos_indices, shuffle=True):
        self.pos_indices = pos_indices
        self.shuffle = shuffle

    def __iter__(self):
        # Shuffle the positive indices at the start of each epoch if shuffle is True
        indices = self.pos_indices.copy()
        if self.shuffle:
            random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        # Return the total number of positive samples
        return len(self.pos_indices)
