import pandas as pd
import numpy as np
import ast
import os
import argparse  

def calculate_iou(box1, box2):
    """
    Calculate IOU between two bounding boxes in [x, y, width, height] format.
    """
    try:
        # Extract coordinates
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Calculate corners
        x1_min, y1_min = x1, y1
        x1_max, y1_max = x1 + w1, y1 + h1
        x2_min, y2_min = x2, y2
        x2_max, y2_max = x2 + w2, y2 + h2

        # Calculate intersection coordinates
        x_min_inter = max(x1_min, x2_min)
        y_min_inter = max(y1_min, y2_min)
        x_max_inter = min(x1_max, x2_max)
        y_max_inter = min(y1_max, y2_max)

        # Calculate areas
        intersection = max(0, x_max_inter - x_min_inter) * max(0, y_max_inter - y_min_inter)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        # Calculate IOU
        iou = intersection / union if union > 0 else 0
        return iou
    except Exception as e:
        print(f"Error in calculate_iou: {e}")
        print(f"box1: {box1}, box2: {box2}")
        return 0

def calculate_center_distance(box1, box2):
    """Calculate the distance between centers of two bounding boxes in [x, y, width, height] format."""
    try:
        # Calculate centers
        center1_x = box1[0] + box1[2]/2
        center1_y = box1[1] + box1[3]/2
        center2_x = box2[0] + box2[2]/2
        center2_y = box2[1] + box2[3]/2

        return np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    except Exception as e:
        print(f"Error in calculate_center_distance: {e}")
        print(f"box1: {box1}, box2: {box2}")
        return float('inf')

def calculate_scale_ratio(box1, box2):
    """Calculate the ratio of areas between two bounding boxes in [x, y, width, height] format."""
    try:
        area1 = max(0.1, box1[2] * box1[3])  # width * height, minimum area of 0.1 to avoid division by zero
        area2 = max(0.1, box2[2] * box2[3])  # width * height
        return min(area1, area2) / max(area1, area2)
    except Exception as e:
        print(f"Error in calculate_scale_ratio: {e}")
        print(f"box1: {box1}, box2: {box2}")
        return 0
    
def evaluate_model_predictions(gt_csv_path, pred_csv_path):
    """
    Evaluate model predictions against ground truth with multiple metrics.
    """
    try:
        # Read CSVs
        gt_df = pd.read_csv(gt_csv_path)
        pred_df = pd.read_csv(pred_csv_path)
        
        # Convert string representation of bbox to list
        gt_df['bbox'] = gt_df['bbox'].apply(ast.literal_eval)
        pred_df['bbox'] = pred_df['bbox'].apply(ast.literal_eval)
        
        # Make sure the DataFrames are aligned
        if len(gt_df) != len(pred_df):
            print(f"Warning: Ground truth ({len(gt_df)}) and prediction ({len(pred_df)}) files have different numbers of entries")
            return None
        
        # Calculate metrics for each image
        ious = []
        center_distances = []
        scale_ratios = []
        
        for idx, (gt_box, pred_box) in enumerate(zip(gt_df['bbox'], pred_df['bbox'])):
            try:
                iou = calculate_iou(gt_box, pred_box)
                center_dist = calculate_center_distance(gt_box, pred_box)
                scale_ratio = calculate_scale_ratio(gt_box, pred_box)
                
                ious.append(iou)
                center_distances.append(center_dist)
                scale_ratios.append(scale_ratio)
            except Exception as e:
                print(f"Error processing pair {idx}: {e}")
                print(f"GT box: {gt_box}")
                print(f"Pred box: {pred_box}")
                continue
        
        if not ious:  # If no valid results were collected
            print("No valid results were collected")
            return None
            
        # Calculate summary statistics
        results = {
            'mean_iou': np.mean(ious),
            'percentage_above_50_iou': (np.array(ious) > 0.5).mean() * 100,
            'mean_center_distance': np.mean(center_distances),
            'center_distance_under_50px': (np.array(center_distances) < 50).mean() * 100,
            'scale_ratio_over_80': (np.array(scale_ratios) > 0.8).mean() * 100,
        }
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None



def compare_models(gt_path, pred_path):
    """
    Compare multiple models against ground truth.
    """
    results = {}

    results = evaluate_model_predictions(gt_path, pred_path)
    
    if results is not None:
        print("\nDetailed Spatial Precision Evaluation Results:")
        print(f"Mean IOU: {results['mean_iou']:.3f}")
        print(f"Percentage above 0.5 IOU: {results['percentage_above_50_iou']:.1f}%")
        print(f"Mean Center Distance: {results['mean_center_distance']:.2f} pixels")
        print(f"Predictions within 50px of GT center: {results['center_distance_under_50px']:.1f}%")
        print(f"Predictions with >80% scale match: {results['scale_ratio_over_80']:.1f}%")
    else:
        print(f"No valid results!")

    return results


def save_to_csv(metrics_csv_path, epoch, results):
   
    results['mean_iou']    = round(results['mean_iou']   , 3) 
    results['percentage_above_50_iou']    = round(results['percentage_above_50_iou']   , 3) 
    results['mean_center_distance']    = round(results['mean_center_distance']   , 3) 
    results['center_distance_under_50px']    = round(results['center_distance_under_50px']  , 3) 
    results['scale_ratio_over_80']    = round(results['scale_ratio_over_80']   , 3) 


    metrics_data = pd.DataFrame({
        'epoch': [epoch],
        'accuracy': [None],
        'fid': [None],
        'lpips_dist_avg': [None],
        'lpips_stderr': [None],

        'mean_iou':[results['mean_iou']],
        'percentage_above_50_iou': [results['percentage_above_50_iou']],
        'mean_center_distance': [results['mean_center_distance']],
        'center_distance_under_50px': [results['center_distance_under_50px']],
        'scale_ratio_over_80': [results['scale_ratio_over_80']]
        })

    if os.path.exists(metrics_csv_path):
        # Read the existing metrics CSV file
        df_metrics_existing = pd.read_csv(metrics_csv_path)

        # Check if the current epoch already exists
        if epoch in df_metrics_existing['epoch'].values:
            # Update the accuracy for the existing row
            df_metrics_existing.loc[df_metrics_existing['epoch'] == epoch, 'mean_iou'] = results['mean_iou']
            df_metrics_existing.loc[df_metrics_existing['epoch'] == epoch, 'percentage_above_50_iou'] = results['percentage_above_50_iou']
            df_metrics_existing.loc[df_metrics_existing['epoch'] == epoch, 'mean_center_distance'] = results['mean_center_distance']
            df_metrics_existing.loc[df_metrics_existing['epoch'] == epoch, 'center_distance_under_50px'] = results['center_distance_under_50px']
            df_metrics_existing.loc[df_metrics_existing['epoch'] == epoch, 'scale_ratio_over_80'] = results['scale_ratio_over_80']
        else:
            # Append the new entry
            df_metrics_existing = pd.concat([df_metrics_existing, metrics_data], ignore_index=True)
    else:
        # If file doesn't exist, create a new DataFrame with the current metrics
        df_metrics_existing = metrics_data


    df_metrics_existing = df_metrics_existing.sort_values(by='epoch').reset_index(drop=True)
    df_metrics_existing.to_csv(metrics_csv_path, index=False)



def main():
    parser = argparse.ArgumentParser(description='Evaluate model predictions against ground truth.')
    parser.add_argument('expid', type=str, help='Name of the experience to evaluate')
    parser.add_argument('epoch', type=str, help='Epoch number to evaluate')
    args = parser.parse_args()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    main_dir = os.path.abspath(os.path.join(current_dir, "../"))

    gt_csv_path = os.path.join(main_dir, "dataset/OPA/test_data_pos.csv")
    expid_dir = os.path.join(main_dir, "result", args.expid)
    eval_csv_path = os.path.join(expid_dir, "eval", args.epoch, "eval.csv")
    metrics_csv_path = os.path.join(expid_dir, f'eval_metrics_{args.expid}.csv')

    # Check if files exist
    if not os.path.exists(gt_csv_path):
        print(f"Error: Ground truth file not found at {gt_csv_path}")
        return
    
    if not os.path.exists(eval_csv_path):
        print(f"Error: Prediction file not found at {eval_csv_path}")
        return
    
    print(f"\nEvaluating experiment {args.expid}, epoch {args.epoch}")
    results = compare_models(gt_csv_path, eval_csv_path)
    
    # Print the results in a clean format
    if results:
        print("\nFinal Evaluation Summary:")
        print(f"Experiment: {args.expid}")
        print(f"Epoch: {args.epoch}")
        print("-" * 50)
        for metric_name, value in results.items():
            if isinstance(value, float):
                print(f"{metric_name.replace('_', ' ').title():<30}: {value:.3f}")
            else:
                print(f"{metric_name.replace('_', ' ').title():<30}: {value}")
    else:
        print("\nNo valid results were obtained from the evaluation")

    save_to_csv(metrics_csv_path, args.epoch, results)
    print(f'\nResults of Spatial Precision Evaluation saved to: {metrics_csv_path}')        


if __name__ == '__main__':
    main()

'''
Usage Example:

python spatial_precision.py graplus 21
'''