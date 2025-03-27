import os
import logging
import datetime
from PIL import Image
import torch

#####
import csv
import pandas as pd
import matplotlib.pyplot as plt
import re

class AccMeter():
    '''Computes and stores the correctness of the discriminator prediction'''
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_correct_num = 0
        self.total_num = 0
        self.avg = 0

    def update(self, correct_num, n):
        if n != 0:
            self.total_correct_num += correct_num 
            self.total_num += n
            self.avg = self.total_correct_num / self.total_num

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def make_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    logfile = log_file
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info('logfile = {}'.format(logfile))
    return logger


# At the beginning of your script where you set up the logger, modify the logger format
def make_logger_new(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create a file handler
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
    
    # Create a console handler with a simpler format
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))  # Only show the message
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def make_dirs(save_dir):
    is_old_exp = os.path.exists(save_dir)

    model_dir = os.path.join(save_dir, 'models')
    sample_dir = os.path.join(save_dir, 'sample')
    tblog_dir = os.path.join(save_dir, 'tblog')
    log_path = os.path.join(save_dir, 'log-{}'.format(datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')))

    if not is_old_exp:
        os.makedirs(save_dir)
        os.mkdir(model_dir)
        os.mkdir(sample_dir)
        os.mkdir(tblog_dir)

    return {
        'save_dir': save_dir,
        'model_dir': model_dir,
        'sample_dir': sample_dir,
        'tblog_dir': tblog_dir,
        'log_path': log_path
    }, is_old_exp

def save(model_dir, model, opt, logger=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    sav_path = os.path.join(model_dir, '{}.pth'.format(opt.epoch))
    if logger is None:
        print("=> saving checkpoint to '{}'".format(sav_path))
    else:
        logger.info("=> saving checkpoint to '{}'".format(sav_path))

    torch.save({
        'epoch': opt.epoch,
        'model': model.state_dict(),
        'opt': opt,
        'optimizer': model.optimizer_dict(),
    }, sav_path)

def resume(path, model, resume_list, strict=False, logger=None):
    if path is None:
        return model, 0

    assert (os.path.exists(path))
    if logger is None:
        print("=> loading {} from checkpoint '{}' with strict={}".format(resume_list, path, strict))
    else:
        logger.info("=> loading {} from checkpoint '{}' with strict={}".format(resume_list, path, strict))
    checkpoint = torch.load(path)

    pretrained_model_dict = checkpoint['model']
    model_dict = model.state_dict()
    for k in pretrained_model_dict:
        if k in resume_list:
            model_dict[k].update(pretrained_model_dict[k])
    model.load_state_dict(model_dict, strict=strict)

    pretrained_opt_dict = checkpoint['optimizer']
    opt_dict = model.optimizer_dict()
    for k in pretrained_opt_dict:
        if k in resume_list:
            opt_dict[k].update(pretrained_opt_dict[k])
    model.load_opt_state_dict(opt_dict)

    epoch = checkpoint['epoch']
    return model, epoch


###################################################################
def plot_loss_from_csv(csv_file_path, exp_name):

    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    loss_columns = df.columns[2:] 
    exp_name = str(exp_name)

    
    epochs = df['Epoch']
    batches = df['Batch']

    m = max(batches)
    df["continuous_batches"] = (batches + 9) + ((epochs - 1) * m)
    continuous_batches = df['continuous_batches']
    
    plots_dir = os.path.join(os.path.dirname(csv_file_path),"loss_plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Individual plots
    for loss in loss_columns:
        # Extract relevant data
        loss_values = df[loss]
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(continuous_batches, loss_values, marker='o', linestyle='-', markersize=3, label=loss)
        
        # Adding vertical lines for each epoch change
        unique_epochs = epochs.unique()
        epoch_start_batches = [df[df['Epoch'] == epoch]['continuous_batches'].min() for epoch in unique_epochs]

        plt.xticks(epoch_start_batches, unique_epochs)

        # Set the title and labels
        plt.title(f'{loss} ------------ {exp_name} ')
        plt.xlabel('Epoch')
        plt.ylabel(loss)
        plt.grid(True)

        # Save the plot as a PNG file beside the CSV file
        plot_file_path = os.path.join(plots_dir, f'loss_plot_{loss}.png')
        plt.savefig(plot_file_path)
        plt.close()

        print(f"Plot saved as {plot_file_path}")
    
    # Combined plot
    plt.figure(figsize=(10, 6))
    for loss in loss_columns:
        loss_values = df[loss]
        plt.plot(continuous_batches, loss_values, marker='o', linestyle='-', markersize=1, label=loss)

    # Adding vertical lines for each epoch change
    plt.xticks(epoch_start_batches, unique_epochs)

    # Set the title and labels for the combined plot
    plt.title(f'model: ( {exp_name} )')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Values')
    plt.legend()
    plt.grid(True)

    # Save the combined plot as a PNG file
    combined_plot_file_path = os.path.join(os.path.dirname(csv_file_path), f'LossPlots_{exp_name}.png')
    plt.savefig(combined_plot_file_path)
    plt.close()

    print(f"Combined plot saved as {combined_plot_file_path}")

################################################################### 
def plot_from_csv(csv_file_path, exp_name, use_mean=False, mean_interval=10):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    exp_name = str(exp_name)
    
    # Extract epochs and batches
    epochs = df['Epoch']
    batches = df['Batch']
    
    # Create a continuous batch counter
    max_batches = max(batches)
    df["continuous_batches"] = (batches + 9) + ((epochs - 1) * max_batches)
    continuous_batches = df['continuous_batches']
    
    # Prepare the directory for saving plots
    plots_dir = os.path.join(os.path.dirname(csv_file_path), f"plots_{exp_name}")
    loss_plots = os.path.join(plots_dir, "loss_plots")
    Disc_Acc_plots = os.path.join(plots_dir, "Disc_Acc_plots")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(loss_plots, exist_ok=True)
    os.makedirs(Disc_Acc_plots, exist_ok=True)
    
    # Extract columns containing "loss" or "avg"
    loss_columns = [col for col in df.columns if 'loss' in col.lower()]
    Disc_Acc_columns = [col for col in df.columns if 'avg' in col.lower()]
    
    def get_mean_data(df, columns, interval):
        """
        Calculate the mean of every `interval` rows for specified columns.
        Retain the 'Epoch' and 'Batch' columns.
        """
        # Calculate the mean for the specified columns
        mean_df = df[columns].groupby(df.index // interval).mean()
        
        # Retain the first 'Epoch' and 'Batch' values for each interval
        mean_df['Epoch'] = df['Epoch'].groupby(df.index // interval).first()
        mean_df['Batch'] = df['Batch'].groupby(df.index // interval).first()
        mean_df['continuous_batches'] = df['continuous_batches'].groupby(df.index // interval).first()
        
        return mean_df


    if use_mean:
        print(f"Using mean over every {mean_interval} rows")
        df_loss = get_mean_data(df, loss_columns, mean_interval)
        df_acc = get_mean_data(df, Disc_Acc_columns, mean_interval)
        continuous_batches = df_loss['continuous_batches']
    else:
        df_loss = df
        df_acc = df
    
    print("Disc_Acc_columns: ", Disc_Acc_columns)
    print("loss_columns: ", loss_columns)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    def acc_plot(df, Disc_Acc_columns):
        """
        Function to plot accuracy-related columns and save both individual and combined plots.
        """
        # Dictionary to map column names to more readable titles
        Acc_title_dict = {}
        # Disc_Acc_columns:  ['Disc_avg', 'g_avg', 'dataset_avg', 'dataset_avg_real', 'dataset_avg_fake']
        for col in Disc_Acc_columns:
            if "Disc" in col:
                continue
            elif "g_a" in col:
                Acc_title_dict[col] = "Generator"
            elif "dataset" in col and "real" not in col and "fake" not in col:
                Acc_title_dict[col] = "Dataset"
            elif "dataset" in col and "real" in col:
                Acc_title_dict[col] = "Dataset_Real"
            elif "dataset" in col and "fake" in col:
                Acc_title_dict[col] = "Dataset_Fake"

        print("Acc_title_dict: ", Acc_title_dict)
        # Plot individual accuracy metrics
        for col in Disc_Acc_columns:
            plt.figure(figsize=(10, 6))
            plt.plot(continuous_batches, df[col], marker='o', linestyle='-', markersize=1, label=col)
            
            # Add vertical lines for each epoch change
            unique_epochs = df['Epoch'].unique()
            epoch_start_batches = [df[df['Epoch'] == epoch]['continuous_batches'].min() for epoch in unique_epochs]
            
            plt.xticks(epoch_start_batches, unique_epochs)
            if "Disc" in col:
                plt.title(f'Total Discriminator Accuracy ------------ {exp_name}')
            else:
                plt.title(f'Discriminator Accuracy on {Acc_title_dict[col]} Data ------------ {exp_name}')
            
            plt.xlabel('Epoch')
            plt.ylabel(col)
            plt.grid(True)
            
            # Save the individual plot
            if "Disc" in col:
                plot_file_path = os.path.join(Disc_Acc_plots, f'Disc_acc.png')
            else:
                plot_file_path = os.path.join(Disc_Acc_plots, f'Disc_acc_{Acc_title_dict[col]}.png')
            
            plt.savefig(plot_file_path)
            plt.close()
            print(f"Plot saved as {plot_file_path}")

        # Combined plot for all accuracy metrics
        plt.figure(figsize=(12, 8))
        for col in Disc_Acc_columns:
            if "Disc" in col:
                plt.plot(continuous_batches, df[col], marker='o', linestyle='--', linewidth=5, markersize=1, label="Discriminator in total")
            else:
                plt.plot(continuous_batches, df[col], marker='o', linestyle='-', markersize=1, label="on "+Acc_title_dict[col]+" data")
        
        # Add vertical lines for each epoch change
        unique_epochs = df['Epoch'].unique()
        epoch_start_batches = [df[df['Epoch'] == epoch]['continuous_batches'].min() for epoch in unique_epochs]
        
        plt.xticks(epoch_start_batches, unique_epochs)
        plt.title(f'Discriminator Accuracy change during training ------------ {exp_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(loc='best')
        plt.grid(True)
        
        # Save the combined plot
        combined_plot_file_path = os.path.join(plots_dir, f'{exp_name}_combined_disc_acc.png')
        plt.savefig(combined_plot_file_path)
        plt.close()
        print(f"Combined plot saved as {combined_plot_file_path}")

    def plot_loss(df, loss_columns):
        """
        Function to plot loss-related columns.
        """
        for loss in loss_columns:
            plt.figure(figsize=(10, 6))
            plt.plot(continuous_batches, df[loss], marker='o', linestyle='-', markersize=1, label=loss)
            
            unique_epochs = epochs.unique()
            epoch_start_batches = [df[df['Epoch'] == epoch]['continuous_batches'].min() for epoch in unique_epochs]
            
            plt.xticks(epoch_start_batches, unique_epochs)
            plt.title(f'{loss} ------ {exp_name}')
            plt.xlabel('Epoch')
            plt.ylabel(loss)
            plt.grid(True)
            
            plot_file_path = os.path.join(loss_plots, f'loss_plot_{loss}.png')
            plt.savefig(plot_file_path)
            plt.close()
            print(f"Plot saved as {plot_file_path}")

        # Combined plot for all loss columns
        plt.figure(figsize=(10, 6))
        for loss in loss_columns:
            plt.plot(continuous_batches, df[loss], marker='o', linestyle='-', markersize=0.5, label=loss)

        plt.xticks(epoch_start_batches, unique_epochs)
        plt.title(f'Losses change during training ------- Model: {exp_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Values')
        plt.legend()
        plt.grid(True)
        
        combined_plot_file_path = os.path.join(plots_dir, f'{exp_name}_LossPlots.png')
        plt.savefig(combined_plot_file_path)
        plt.close()
        print(f"Combined plot saved as {combined_plot_file_path}")

    # Plot accuracy and loss using either raw data or mean data
    acc_plot(df_acc, Disc_Acc_columns)
    plot_loss(df_loss, loss_columns)


###################################################################
def save_loss_to_csv(file_path, epoch, batch, g_gan_loss1, g_gan_loss2, g_rec_loss, g_kld_loss, d_real_loss, d_fake_loss1, d_fake_loss2):
    # Check if the file already exists
    file_exists = os.path.isfile(file_path)
    
    # Format the loss values to 3 decimal places
    g_gan_loss1 = f"{g_gan_loss1:.3f}"
    g_gan_loss2 = f"{g_gan_loss2:.3f}"
    g_rec_loss = f"{g_rec_loss:.3f}"
    g_kld_loss = f"{g_kld_loss:.3f}"
    d_real_loss = f"{d_real_loss:.3f}"
    d_fake_loss1 = f"{d_fake_loss1:.3f}"
    d_fake_loss2 = f"{d_fake_loss2:.3f}"

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['Epoch', 'Batch', 'G_GAN_Loss1', 'G_GAN_Loss2', 'G_Rec_Loss', 'G_KLD_Loss', 'D_Real_Loss', 'D_Fake_Loss1', 'D_Fake_Loss2'])
        
        # Write the data row
        writer.writerow([epoch, batch, g_gan_loss1, g_gan_loss2, g_rec_loss, g_kld_loss, d_real_loss, d_fake_loss1, d_fake_loss2])


###################################################################
def save_loss_to_csv_1(file_path, epoch, batch, 
                         G_loss, g_loss, g_rec_loss,
                         d_real_loss, d_fake_loss, 
                         Disc_corr_num, Disc_tot_num, Disc_avg,
                         g_corr_num, g_tot_num, g_avg,
                         dst_corr_num, dst_tot_num, dst_avg,
                         dst_corr_num_real,dst_tot_num_real, dst_avg_real,
                         dst_corr_num_fake,dst_tot_num_fake, dst_avg_fake  ):
    
    # Check if the file already exists
    file_exists = os.path.isfile(file_path)
    
    # Format the loss values to 3 decimal places
    G_loss = f"{G_loss:.3f}"
    g_loss = f"{g_loss:.3f}"
    g_rec_loss = f"{g_rec_loss:.3f}"

    d_real_loss = f"{d_real_loss:.3f}"
    d_fake_loss = f"{d_fake_loss:.3f}"

    Disc_corr_num = f"{Disc_corr_num}"
    Disc_tot_num = f"{Disc_tot_num}"
    Disc_avg = f"{Disc_avg:.3f}"

    g_corr_num = f"{g_corr_num}"
    g_tot_num = f"{g_tot_num}"
    g_avg = f"{g_avg:.3f}"

    dst_corr_num = f"{dst_corr_num}"
    dst_tot_num = f"{dst_tot_num}"
    dst_avg = f"{dst_avg:.3f}"
    
    dst_corr_num_real = f"{dst_corr_num_real}"
    dst_tot_num_real = f"{dst_tot_num_real}"
    dst_avg_real = f"{dst_avg_real:.3f}"
    
    dst_corr_num_fake = f"{dst_corr_num_fake}"
    dst_tot_num_fake = f"{dst_tot_num_fake}"
    dst_avg_fake = f"{dst_avg_fake:.3f}"

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['Epoch', 'Batch', 
                             'G_Loss', 'g_loss', 'g_rec_loss', 
                             'D_Real_Loss', 'D_Fake_Loss',
                             'Disc_corr_num', 'Disc_tot_num', 'Disc_avg',
                             'g_corr_num', 'g_tot_num', 'g_avg',
                             'dataset_corr_num', 'dataset_tot_num', 'dataset_avg',
                             'dataset_corr_num_real','dataset_tot_num_real', 'dataset_avg_real',
                             'dataset_corr_num_fake','dataset_tot_num_fake', 'dataset_avg_fake'])
        
        # Write the data row
        writer.writerow([epoch, batch,
                          G_loss, g_loss, g_rec_loss,
                          d_real_loss, d_fake_loss, 
                          Disc_corr_num, Disc_tot_num, Disc_avg,
                          g_corr_num, g_tot_num, g_avg,
                          dst_corr_num, dst_tot_num, dst_avg,
                          dst_corr_num_real,dst_tot_num_real, dst_avg_real,
                          dst_corr_num_fake,dst_tot_num_fake, dst_avg_fake])

###################################################################        
def save_loss_to_csv_placenet(file_path, epoch, batch, g_gan_loss, g_ndiv_loss, d_real_loss, d_fake_loss):
    # Check if the file already exists
    file_exists = os.path.isfile(file_path)
    
    # Format the loss values to 3 decimal places
    g_gan_loss  = f"{g_gan_loss:.3f}"
    g_ndiv_loss = f"{g_ndiv_loss:.3f}"
    d_real_loss = f"{d_real_loss:.3f}"
    d_fake_loss = f"{d_fake_loss:.3f}"

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file does not exist, write the header
        if not file_exists:
            writer.writerow(['Epoch', 'Batch', 'G_GAN_Loss', 'G_ndiv_Loss', 'D_Real_Loss', 'D_Fake_Loss'])
        
        # Write the data row
        writer.writerow([epoch, batch, g_gan_loss, g_ndiv_loss, d_real_loss, d_fake_loss])


################################################################### 
def save_loss_to_csv_terse(file_path, epoch, batch, g_gan_loss, d_real_loss, d_fake_loss=None):
    # Check if the file already exists
    file_exists = os.path.isfile(file_path)
    
    if d_fake_loss != None : 
        # Format the loss values to 3 decimal places
        g_gan_loss = f"{g_gan_loss:.3f}"
        d_real_loss = f"{d_real_loss:.3f}"
        d_fake_loss  = f"{d_fake_loss:.3f}"


        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # If the file does not exist, write the header
            if not file_exists:
                writer.writerow(['Epoch', 'Batch', 'G_GAN_Loss','D_Real_Loss', 'D_Fake_Loss'])
            
            # Write the data row
            writer.writerow([epoch, batch, g_gan_loss, d_real_loss, d_fake_loss  ])

    #---------------------------------------------------#
    else:
        # Format the loss values to 3 decimal places
        g_gan_loss = f"{g_gan_loss:.3f}"
        d_real_loss = f"{d_real_loss:.3f}"

        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            # If the file does not exist, write the header
            if not file_exists:
                writer.writerow(['Epoch', 'Batch', 'G_Loss','D_Loss',])
            
            # Write the data row
            writer.writerow([epoch, batch, g_gan_loss, d_real_loss  ])
