import argparse
import os
import torch
from tqdm import tqdm
import numpy as np
import torch

from simopa_cfg import opt
from simopa_dst import ImageDataset
from simopa_net import ObjectPlaceNet
import pandas as pd


def save_to_csv(metrics_csv_path, epoch, pred_acc):
    # Round pred_acc to 3 decimal places
    pred_acc = round(pred_acc, 3)
    
    # Prepare the metrics DataFrame for the current epoch
    metrics_data = pd.DataFrame({
        'epoch': [epoch],
        'accuracy': [pred_acc],
        'fid': [None],
        'lpips_dist_avg': [None],
        'lpips_stderr': [None],

        'mean_iou':[None],
        'percentage_above_50_iou': [None],
        'mean_center_distance': [None],
        'center_distance_under_50px': [None],
        'scale_ratio_over_80': [None]
    })

    if os.path.exists(metrics_csv_path):
        # Read the existing metrics CSV file
        df_metrics_existing = pd.read_csv(metrics_csv_path)

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
    df_metrics_existing.to_csv(metrics_csv_path, index=False)


def save_SimOPA_pred_to_csv(sample_ids, pred_labels, SimOPA_Preds_csv_path):
    ''' Save predictions to separate CSV files for each epoch '''
    # Create a DataFrame from the current predictions
    data = pd.DataFrame({
        'sample_ids': sample_ids,
        'pred_labels': pred_labels
    })

    # Save the DataFrame to a CSV file
    data.to_csv(SimOPA_Preds_csv_path, index=False)


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
    print(f"\n{args.expid} - epoch:{args.epoch} =====>  Accuracy = {pred_acc:.3f}")

    expid_dir = os.path.join('result', args.expid)
    metrics_csv_path = os.path.join(expid_dir, f'eval_metrics_{args.expid}.csv')

    save_to_csv(metrics_csv_path, args.epoch, pred_acc)
    print(f'\nResults of Accuracy Evaluation saved to: {metrics_csv_path}')        

    predictions_dir = os.path.join(expid_dir, f'SimOPA_preds_on_{args.expid}')
    os.makedirs(predictions_dir, exist_ok=True)
    SimOPA_Preds_csv_path = os.path.join(predictions_dir, f'SimOPA_preds_on_{args.expid}_with_epoch_{args.epoch}.csv')

    save_SimOPA_pred_to_csv(sample_ids, pred_labels, SimOPA_Preds_csv_path)
    print(f'SimOPA Per-Image Predictions saved to: {SimOPA_Preds_csv_path}')        

    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="path to loaded checkpoint")
    parser.add_argument("--expid", type=str, required=True, help="experiment name")
    parser.add_argument("--epoch", type=int, required=True, help="epoch for evaluation")
    parser.add_argument("--eval_type", type=str, default="eval", help="evaluation type")

    args = parser.parse_args()
    assert os.path.exists(args.checkpoint)

    evaluate(args)
