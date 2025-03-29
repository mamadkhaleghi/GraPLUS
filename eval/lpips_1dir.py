import argparse
import datetime
from tqdm import tqdm
import os
import numpy as np
import torch
import lpips
import pandas as pd
'''

#------------------------------------------------------------#
current_dir = os.path.dirname(os.path.abspath(__file__))
OPA_path    = os.path.abspath(os.path.join(current_dir, "../../OPA_dataset"))

def create_unique_zero_idx_list(OPA_path):

    test_pos_csv_path = os.path.join(OPA_path, "sg_test_data_pos.csv")
    test_pos_unique_csv_path = os.path.join(OPA_path, "sg_test_data_pos_unique.csv")

    #-----------------------------------------------# zero_idx_list
    eval_result = pd.read_csv(os.path.join(OPA_path, "SimOPA_preds_on_test_pos_data.csv"), sep=',')
    SimOPA_labels = list(eval_result['pred_labels'])

    zero_idx_list = []
    # Iterate through the input list and store the indexes of ones and zeros
    for i, value in enumerate(SimOPA_labels):
        if value == 0:
            zero_idx_list.append(i)

    #-----------------------------------------------# zero_img_id_list
    test_pos_df = pd.read_csv(test_pos_csv_path)

    zero_img_id_list = []
    for idx in zero_idx_list:
        zero_img_id_list.append(int(test_pos_df['imgID'][idx]))

    #-----------------------------------------------# unique_zero_idx_list
    test_pos_unique_df = pd.read_csv(test_pos_unique_csv_path)

    unique_zero_idx_list = test_pos_unique_df[test_pos_unique_df['imgID'].isin(zero_img_id_list)].index.tolist()

    return unique_zero_idx_list


# parser.add_argument("--real_eval", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=False, help="computing LPIPS on real eval test set")

#------------------------------------------------------------#

'''

def save_to_csv( metrics_csv_path, epoch, dist_avg_original, dist_stderr_original):
    
    dist_avg    = round(dist_avg_original   , 3) 
    dist_stderr = round(dist_stderr_original, 6) 
    #----------------------------------------------------------#  Save metrics to 'metrics.csv'

    # Prepare the metrics DataFrame for the current epoch
    metrics_data = pd.DataFrame({
        'epoch': [epoch],
        'accuracy': [None],
        'fid': [None],

        'lpips_dist_avg': [dist_avg],
        'lpips_stderr': [dist_stderr],
        
        'mean_iou':[None],
        'percentage_above_50_iou': [None],
        'mean_center_distance':[None],
        'center_distance_under_50px': [None],
        'scale_ratio_over_80': [None]
   })

    if os.path.exists(metrics_csv_path):
        # Read the existing metrics CSV file
        df_metrics_existing = pd.read_csv(metrics_csv_path)

        # Check if the current epoch already exists
        if epoch in df_metrics_existing['epoch'].values:
            # Update the accuracy for the existing row
            df_metrics_existing.loc[df_metrics_existing['epoch'] == epoch, 'lpips_dist_avg'] = dist_avg
            df_metrics_existing.loc[df_metrics_existing['epoch'] == epoch, 'lpips_stderr'] = dist_stderr

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


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--dir', type=str, default='./imgs/ex_dir')
    parser.add_argument('-v','--version', type=str, default='0.1')
    parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')
    parser.add_argument("--expid", type=str, required=True, help="experiment name")
    parser.add_argument("--epoch", type=int, required=True, help="epoch for evaluation")
    parser.add_argument("--eval_type", type=str, default="evaluni", help="evaluation type")
    parser.add_argument("--repeat", type=int, default=10, help="repeat count for sampling z")
    opt = parser.parse_args()

    assert (opt.repeat > 1)
    data_dir = os.path.join('result', opt.expid, opt.eval_type, str(opt.epoch))
    assert (os.path.exists(data_dir))

    # initialize the model
    loss_fn = lpips.LPIPS(net='alex', version=opt.version)
    if (opt.use_gpu):
        loss_fn.cuda()

    # crawl directory
    files_list = list(sorted(os.listdir(opt.dir)))
    files_dict = {}
    for filename in files_list:
        index = filename.split('_')[0]
        if index in files_dict:
            files_dict[index].append(filename)
        else:
            files_dict[index] = [filename]
    total = len(files_dict)

    # stores distances
    dist_all = {}
    for i, index in enumerate(tqdm(files_dict, total=total)):
        dist_all[index] = []
        files = files_dict[index]
        for ff, file0 in enumerate(files[:-1]):
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir, file0))) # RGB image from [-1,1]
            if (opt.use_gpu):
                img0 = img0.cuda()
            for file1 in files[ff+1:]:
                img1 = lpips.im2tensor(lpips.load_image(os.path.join(opt.dir, file1)))
                if (opt.use_gpu):
                    img1 = img1.cuda()
                # compute distance
                with torch.no_grad():
                    dist01 = loss_fn.forward(img0, img1).squeeze().cpu().item()
                dist_all[index].append(dist01)

    # calculate results
    dist_res = np.zeros((total, 2), dtype=np.float32)
    for i, index in enumerate(dist_all):
        dists = dist_all[index]
        dist_res[i,0] = np.mean(np.array(dists))  # avg of dists for index
        dist_res[i,1] = np.std(np.array(dists))/np.sqrt(len(dists))  # stderr of dists for index

    dist_avg = np.mean(dist_res[:,0])
    dist_stderr = np.mean(dist_res[:,1])

    expid_dir = os.path.join("result", opt.expid)
    metrics_csv_path = os.path.join(expid_dir, f'eval_metrics_{opt.expid}.csv')
    
    metrics_csv_path = os.path.join(expid_dir, f'eval_metrics_{opt.expid}.csv')

    print(f"\nModel Name:{opt.expid} - epoch:{opt.epoch} =====>  LPIPS (Variety): dist = {dist_avg:.3f}, stderr = {dist_stderr:.6f}")

    save_to_csv(metrics_csv_path, opt.epoch, dist_avg, dist_stderr)

    print(f'\nResults of LPIPS Evaluation saved to: {metrics_csv_path}')        


if __name__ == '__main__':
    main()
