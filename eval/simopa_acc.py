import argparse
import datetime
import os
import torch
from tqdm import tqdm
import numpy as np
import torch

from simopa_cfg import opt
from simopa_dst import ImageDataset
from simopa_net import ObjectPlaceNet
import pandas as pd

def evaluate(args):
    # modify configs
    opt.dataset_path = os.path.join('result', args.expid, args.eval_type, str(args.epoch))
    assert (os.path.exists(opt.dataset_path))
    opt.img_path = opt.dataset_path
    opt.mask_path = opt.dataset_path
    opt.test_data_path = os.path.join(opt.dataset_path, '{}.csv'.format(args.eval_type))
    opt.test_box_dic_path = os.path.join(opt.dataset_path, '{}_bboxes.npy'.format(args.eval_type))
    opt.test_reference_feature_path = os.path.join(opt.dataset_path, '{}_feats.npy'.format(args.eval_type))
    opt.test_target_feature_path = os.path.join(opt.dataset_path, '{}_fgfeats.npy'.format(args.eval_type))

    opt.relation_method = 5
    opt.attention_method = 2
    opt.refer_num = 5
    opt.attention_head = 16
    opt.without_mask = False
    opt.without_global_feature = False

    net = ObjectPlaceNet(backbone_pretrained=False)

    checkpoint_path = args.checkpoint
    print('load pretrained weights from ', checkpoint_path)
    net.load_state_dict(torch.load(checkpoint_path))
    net = net.cuda().eval()

    total = 0
    pred_labels = []
    sample_ids = []

    testset = ImageDataset(istrain=False)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=128,
                                              shuffle=False, num_workers=2,
                                              drop_last=False, pin_memory=True)

    with torch.no_grad():
        for batch_index, (sample_id, img_cat, label, target_box, refer_box, target_feats, refer_feats, target_mask, refer_mask, tar_class, w, h) in enumerate(
                tqdm(test_loader)):
            
            img_cat, label, target_box, refer_box, target_mask, refer_mask, w, h = img_cat.cuda(), label.cuda(), target_box.cuda(), refer_box.cuda(), target_mask.cuda(), refer_mask.cuda(), w.cuda(), h.cuda()
            
            target_feats, refer_feats = target_feats.cuda(), refer_feats.cuda()
            
            logits, weights = net(img_cat, target_box, refer_box, target_feats, refer_feats, target_mask, refer_mask, w, h)
            pred_labels.extend(logits.max(1)[1].cpu().numpy())
            total += label.size(0)
            sample_ids.extend(list(sample_id))


 


    pred_acc = (np.array(pred_labels, dtype=np.int32) == 1).sum() / len(pred_labels)


    #=================================================================###
    dir_path = os.path.join('result', args.expid, 'models')
    eval_metrics_dir = os.path.join(dir_path, f"{args.expid}_eval_metrics")
    os.makedirs(eval_metrics_dir, exist_ok=True)


    save_to_csv(eval_metrics_dir, args.expid, args.epoch, pred_acc)
    save_SimOPA_pred_to_csv(sample_ids, pred_labels, eval_metrics_dir, args.epoch)
    #=================================================================###

    print(f"\nModel Name:{args.expid} - epoch:{args.epoch} =====>  Accuracy = {pred_acc:.3f}\n")
    # mark = 'a' if os.path.exists(os.path.join(opt.dataset_path, "{}_acc.txt".format(args.eval_type))) else 'w'
    # with open(os.path.join(opt.dataset_path, "{}_acc.txt".format(args.eval_type)), mark) as f:
    #     f.write("{}\n".format(datetime.datetime.now()))
    #     f.write(" - Accuracy = {:.3f}\n".format(pred_acc))


#=========================================================================================================================###
def save_to_csv(dir, expid, epoch, pred_acc):
        # Ensure the directory exists
    os.makedirs(dir, exist_ok=True)

    # Round pred_acc to 3 decimal places
    pred_acc = round(pred_acc, 3)

    metrics_file = os.path.join(dir, f'metrics_{expid}.csv')

    # Prepare the metrics DataFrame for the current epoch
    metrics_data = pd.DataFrame({
        'epoch': [epoch],
        'accuracy': [pred_acc],
        'fid': [None],
        'lpips_dist_avg': [None],
        'lpips_stderr': [None]
    })

    if os.path.exists(metrics_file):
        # Read the existing metrics CSV file
        df_metrics_existing = pd.read_csv(metrics_file)

        # Check if the current epoch already exists
        if epoch in df_metrics_existing['epoch'].values:
            # Update the accuracy for the existing row
            df_metrics_existing.loc[df_metrics_existing['epoch'] == epoch, 'accuracy'] = pred_acc
        else:
            # Append the new entry
            df_metrics_existing = pd.concat([df_metrics_existing, metrics_data], ignore_index=True)
    else:
        # If file doesn't exist, create a new DataFrame with the current metrics
        df_metrics_existing = metrics_data

    # Sort the DataFrame by epoch
    df_metrics_existing = df_metrics_existing.sort_values(by='epoch').reset_index(drop=True)

    # Save the updated metrics DataFrame back to CSV
    df_metrics_existing.to_csv(metrics_file, index=False)


#=================================================================###
def save_SimOPA_pred_to_csv(sample_ids, pred_labels, dir, epoch):
    ''' Save predictions to separate CSV files for each epoch '''
    # Ensure the directory exists
    predictions_dir = os.path.join(dir, 'SimOPA_per_image_predictions')
    os.makedirs(predictions_dir, exist_ok=True)

    # Define the CSV file path for the current epoch
    csv_file = os.path.join(predictions_dir, f'predictions_epoch_{epoch}.csv')

    # Create a DataFrame from the current predictions
    data = pd.DataFrame({
        'sample_ids': sample_ids,
        'pred_labels': pred_labels
    })

    # Save the DataFrame to a CSV file
    data.to_csv(csv_file, index=False)
    print(f"Saved predictions for epoch {epoch} to {csv_file}")


#=================================================================###


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="path to loaded checkpoint")
    parser.add_argument("--expid", type=str, required=True, help="experiment name")
    parser.add_argument("--epoch", type=int, required=True, help="epoch for evaluation")
    parser.add_argument("--eval_type", type=str, default="eval", help="evaluation type")

    args = parser.parse_args()
    assert os.path.exists(args.checkpoint)

    evaluate(args)
