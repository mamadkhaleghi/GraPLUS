import os
import pandas as pd
import matplotlib.pyplot as plt
import sys
"""
plot_eval_metrics.py

This module is designed to read a CSV file containing evaluation metrics for a machine learning model
across different epochs and generate a single combined plot with all the specified metrics.

The CSV file is expected to have the following columns:
- 'epoch': The epoch number.
- 'accuracy': The accuracy metric of the model.
- 'fid': The Fréchet Inception Distance (FID) score.
- 'lpips_dist_avg': The average Learned Perceptual Image Patch Similarity (LPIPS) distance.
- 'lpips_stderr': The standard error of the LPIPS distance.

Each metric is plotted on a separate subplot stacked vertically in a single figure. The metrics
'accuracy', 'lpips_dist_avg', and 'lpips_stderr' are plotted in green, while 'fid' is plotted in red.

The resulting plot is saved as a PNG file in the same directory as the input CSV file. The PNG file
is named using the format: `metrics_plot_<EXPID>.png`.

Usage:
    python plot_eval_metrics.py <csv_file_path> <EXPID>

Arguments:
    csv_file_path: The path to the CSV file containing the metrics data.
    expid: A unique experiment identifier used for naming the output plot file.

Example:
    python plot_eval_metrics.py "/path/to/metrics.csv" "experiment_01"

The generated plot will be saved as:
    /path/to/metrics_plot_experiment_01.png
"""

import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_metrics(csv_file_path, expid):
    # Load the CSV file
    try:
        df = pd.read_csv(csv_file_path, sep=',')
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return

    # Print the columns to verify
    print("CSV Columns:", df.columns)

    # Check if the 'epoch' column exists
    if 'epoch' not in df.columns:
        print("Error: 'epoch' column not found in the CSV file.")
        return
    
    # Extract directory and filename information
    csv_dir = os.path.dirname(csv_file_path)

    # Define metrics to plot and their colors
    metrics = {
        'accuracy': 'green',
        'fid': 'red',
        'lpips_dist_avg': 'green',
        'lpips_stderr': 'green'
    }
    
    # Find the epoch with the maximum accuracy value
    max_accuracy_idx = df['accuracy'].idxmax()  # Get the index of the max accuracy
    max_accuracy_epoch = df['epoch'][max_accuracy_idx]  # Get the corresponding epoch
    max_accuracy_value = df['accuracy'][max_accuracy_idx]  # Get the max accuracy value
    
    # Create a single figure with multiple subplots
    fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 8), sharex=True)
    fig.suptitle(f"Model: {expid} - Metrics over Epochs", fontsize=16)

    # Loop through each metric to create subplots
    for idx, (metric, color) in enumerate(metrics.items()):
        if metric not in df.columns:
            print(f"Warning: '{metric}' column not found in the CSV file.")
            continue

        # Plot each metric in a separate subplot
        ax = axes[idx]
        ax.plot(df['epoch'], df[metric], marker='o',markersize=2, color=color, label=metric)
        ax.set_ylabel(metric.capitalize())
        ax.grid(True)
        # ax.legend(loc='upper right')

        # Highlight the maximum accuracy point in all plots
        ax.plot(max_accuracy_epoch, df.loc[max_accuracy_idx, metric], 'bo', markersize=8, label='Max Accuracy')

        # Annotate each point with its value in the same color as the marker
        for epoch, value in zip(df['epoch'], df[metric]):
            ax.text(epoch, value, f"{value:.3f}", ha='center', va='bottom', color="black", rotation=90)  # Set rotation to 90 for vertical text

        # Annotate the maximum accuracy point in blue with vertical text
        ax.text(max_accuracy_epoch, df.loc[max_accuracy_idx, metric], f"{df.loc[max_accuracy_idx, metric]:.3f}", 
                ha='center', va='bottom', color='blue', rotation=90)

        # If plotting accuracy, set the y-axis range from 0 to 1
        if metric == 'accuracy':
            ax.set_ylim(0, 1)

        # Set the x-axis to display the epochs as ticks
        ax.set_xticks(df['epoch'])  # Set x-axis ticks to be the epoch values
        ax.set_xticklabels(df['epoch'], ha='right')  # Display epochs as labels

        # Set aspect ratio to auto for each subplot
        ax.set_aspect('auto')

    # Set the common x-label for all subplots
    plt.xlabel("Epoch")

    # Save the combined plot as a single PNG file
    combined_plot_path = os.path.join(csv_dir, f"metrics_plot_{expid}.png")
    plt.savefig(combined_plot_path, bbox_inches='tight')
    plt.close()

    print(f"Combined plot saved at: {combined_plot_path}")



# Main execution
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_eval_metrics.py <csv_file_path> <EXPID>")
        sys.exit(1)
    
    csv_file_path = sys.argv[1]
    expid = sys.argv[2]
    plot_metrics(csv_file_path, expid)
