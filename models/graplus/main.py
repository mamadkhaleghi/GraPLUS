import os
import sys
import time

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# # Move two levels up
main_dir = os.path.abspath(os.path.join(current_dir, "../../"))
# Add this directory to sys.path
sys.path.append(main_dir)


import argparse
import torch
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorboard_logger as tb_logger
                                                                           
from tool.utils import make_dirs, save, resume, make_logger, AverageMeter, save_loss_to_csv_1 , plot_from_csv, AccMeter
from loader import dataset_dict, get_sg_loader, get_sg_dataset
from model import GAN 
from infer import sample, infer


def parse_args():
    parser = argparse.ArgumentParser()
    #---------------------------------------------------------------------------------------------------------------------# training / dataset
    parser.add_argument("--expid", type=str, required=True, help="experiment name")
    parser.add_argument("--n_epochs", type=int, default=40, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--dst", type=str, choices=list(dataset_dict.keys()), default="SG_OPADst", help="dataloder type")
    parser.add_argument("--data_root", type=str, default="OPA", help="dataset root")
    parser.add_argument("--sg_root", type=str, default="OPA_SG", help="scene graph dataset root")
    parser.add_argument("--num_nodes", type=int, default=20, help="number of nodes in each scene graph")
    parser.add_argument("--img_size", type=int, default=32, help="size of image")
    parser.add_argument("--resume_pth", type=str, default=None, help="specify a .pth path to resume training, or None to train from scratch")
    parser.add_argument("--eval_type", type=str, choices=["train", "trainpos", "sample", "eval", "evaluni"], default="eval", help="evaluation type")
    parser.add_argument("--mode_type", type=str, choices=["train", "trainpos", "sample", "eval", "evaluni"], default="train", help="evaluation type")
    parser.add_argument("--with_infer", action='store_true', default=False, help="action to make inference after each training epoch")
    parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image sampling")
    #---------------------------------------------------------------------------------------------------------------------# sampling / data augmentation
    parser.add_argument("--sampler_type", type=str, choices=["default","balance_sampler","TwoToOne_sampler"], default="balance_sampler", help="type of dataset sampler")
    parser.add_argument("--data_augmentation", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="Enable data augmentation")
    parser.add_argument("--flip_augment_prob", type=float, default=0.5, help="probability of horizontally flip augmentation in dataset")
    parser.add_argument("--color_jitter_augment_prob", type=float, default=0.5, help="probability of color jitter augmentation in dataset")
    parser.add_argument("--grayscale_augment_prob", type=float, default=0.2, help="probability of gray scale augmentation in dataset")
    parser.add_argument("--gaussian_blur_augment_prob", type=float, default=0.5, help="probability of gaussian blur augmentation in dataset")
    #---------------------------------------------------------------------------------------------------------------------# optimizers
    parser.add_argument("--g_lr", type=float, default=0.0001, help="adam: generator learning rate")
    parser.add_argument("--g_weight_decay", type=float, default=0, help="generator weight decay")
    parser.add_argument("--d_lr", type=float, default=0.00002, help="adam: discriminator learning rate")
    parser.add_argument("--d_weight_decay", type=float, default=0, help="discriminator weight decay")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--rec_loss_weight", type=float, default=50.0, help="reconstruction loss weight")
    parser.add_argument("--num_d_up_real", type=int, default=2, help="size of bg image cropped objects")
    #---------------------------------------------------------------------------------------------------------------------# scene graph properties
    #------------------------------------ node/edge Embedding
    parser.add_argument("--embed_dim", type=int, default=768, help="node/edge embeding dimension")
    parser.add_argument("--embed_freeze", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="to freeze the embedding layer or not")
    parser.add_argument("--gpt2_path", type=str, default="gpt2_embeddings", help="path for node/edge gpt2 embeddings ")
    parser.add_argument("--gpt2_node_mode", type=str, choices=['category_embedding', 'description_embedding', 'cat_desc_embedding', 'placement_embedding', 'cat_place_embedding', 'cat_desc_place_embedding'], default='cat_desc_place_embedding', help="Mode for selecting node embedding type")

    parser.add_argument("--use_spatial_info", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="to concatenate spatial information on GTN output")
    parser.add_argument("--spatial_dim", type=int, default=256, help="output dimension of Linear layer for bg image objects spatial_info")
    #---------------------------------------------------------------------------------------------------------------------# GTN properties
    parser.add_argument("--gtn_num_head", type=int, default=8, help="number of attention heads in GTN")
    parser.add_argument("--gtn_num_layer", type=int, default=5, help="number of gtn layers in GTN")
    parser.add_argument("--gtn_hidden_dim", type=int, default=768, help="output dimension of hidden layers in GTN")
    parser.add_argument("--gtn_output_dim", type=int, default=768, help="node/edge dimension of GNN output ")
    #---------------------------------------------------------------------------------------------------------------------# Attention properties
    parser.add_argument("--d_k", type=int, default=64, help="dimension of key in multi-head attention")
    parser.add_argument("--d_v", type=int, default=64, help="dimension of value in multi-head attention")
    parser.add_argument("--n_heads", type=int, default=8, help="number of heads in multi-head attention")
    parser.add_argument("--use_pos_enc", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="to use positional encoding in attention block")
    parser.add_argument("--add_residual", type=lambda x: x.lower() in ['true', '1', 't', 'y', 'yes'], default=True, help="to add residual in attention block")
    #----------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--d_noise", type=int, default=2048, help="vector dimension of random noise")
    opt = parser.parse_args()
    return opt


def main():
    opt = parse_args()
    if not opt.use_spatial_info : opt.spatial_dim=0
    save_dir = os.path.join(main_dir,'result', opt.expid)
    dirs, is_old_exp = make_dirs(save_dir)
    model_dir, sample_dir, tblog_dir, log_path = dirs['model_dir'], dirs['sample_dir'], dirs['tblog_dir'], dirs['log_path']
    assert (is_old_exp or opt.resume_pth is None)

    tb_logger.configure(tblog_dir, flush_secs=5)
    logger = make_logger(log_path)
    logger.info(opt)
             
    Loss_dir = os.path.join(model_dir, "Losses")
    os.makedirs(Loss_dir, exist_ok=True)
    loss_csv_file_path = os.path.join(Loss_dir, f'losses_{opt.expid}.csv')
                    
    train_loader = get_sg_loader(opt.dst, batch_size=opt.batch_size, num_workers=8, image_size=opt.img_size, mode_type="train", data_root=opt.data_root, sg_root=opt.sg_root, num_nodes = opt.num_nodes,
                              flip_augment_prob=opt.flip_augment_prob, 
                              color_jitter_augment_prob=opt.color_jitter_augment_prob,
                              grayscale_augment_prob = opt.grayscale_augment_prob,
                              gaussian_blur_augment_prob = opt.gaussian_blur_augment_prob,
                              augment_flag = opt.data_augmentation,
                              sampler_type = opt.sampler_type)

    sample_dataset = get_sg_dataset(opt.dst, image_size=opt.img_size, mode_type="sample", data_root=opt.data_root, sg_root=opt.sg_root, num_nodes = opt.num_nodes, 
                                flip_augment_prob=opt.flip_augment_prob, 
                                color_jitter_augment_prob=opt.color_jitter_augment_prob,
                                grayscale_augment_prob = opt.grayscale_augment_prob,
                                gaussian_blur_augment_prob = opt.gaussian_blur_augment_prob,
                                augment_flag = False)

    model = GAN(opt)
    model, start_ep = resume(opt.resume_pth, model, resume_list=['generator', 'discriminator'], strict=True, logger=logger)
    assert (start_ep < opt.n_epochs)
    model.Eiters = start_ep * len(train_loader)

    G_loss_meter,g_loss_meter,g_rec_loss_meter, d_loss_real_meter, d_loss_fake_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
    Disc_acc_meter, gen_acc_meter , dataset_acc_meter, dataset_acc_meter_real, dataset_acc_meter_fake = AccMeter(), AccMeter(), AccMeter(), AccMeter(), AccMeter()

    for epoch in range(start_ep, opt.n_epochs):

        for i, batch in enumerate(train_loader): 
            # Skip if the loader returns None
            if batch is None:
                continue

            indices, bg_size, bg_img_feats, fg_img_feats, fg_msk_feats, fg_bboxes, comp_img_feats, comp_msk_feats, labels, trans_labels, catnms, sg_bg_trans, sg_bg_bbox_org, node_attr, edge_index , edge_attr = batch

            model.start_train()

            G_loss, g_loss, g_rec_loss, d_loss_real_data, d_loss_fake, gen_correct, real_correct, fake_correct, num_real, num_fake = \
            model.train_disc_gen(bg_size, catnms, bg_img_feats, fg_img_feats, fg_msk_feats, fg_bboxes, comp_img_feats, comp_msk_feats, labels, trans_labels, node_attr, edge_attr, edge_index, sg_bg_bbox_org, sg_bg_trans)   

            tb_logger.log_value('G_loss', G_loss.item(), step=model.Eiters)                                                                                                                                             
            tb_logger.log_value('g_gan_loss', g_loss.item(), step=model.Eiters)
            tb_logger.log_value('g_rec_loss', g_rec_loss.item(), step=model.Eiters)
            tb_logger.log_value('d_real_loss', d_loss_real_data.item(), step=model.Eiters)
            tb_logger.log_value('d_fake_loss', d_loss_fake.item(), step=model.Eiters)

            bs = len(indices)
            G_loss_meter.update(G_loss.item(), bs)
            g_loss_meter.update(g_loss.item(), bs)
            g_rec_loss_meter.update(g_rec_loss.item(), bs)
            d_loss_real_meter.update(d_loss_real_data.item(), bs)
            d_loss_fake_meter.update(d_loss_fake.item(), bs)
            gen_acc_meter.update(gen_correct, bs)
            dataset_acc_meter_real.update(real_correct, num_real)
            dataset_acc_meter_fake.update(fake_correct, num_fake)
            dataset_acc_meter.update(real_correct + fake_correct , bs)
            Disc_acc_meter.update(gen_correct+real_correct+fake_correct, 2*bs)

            ###===================================================================
            if i!=0 and (epoch * len(train_loader) + i) % 10 == 0:
                logger.info(
                    "[Epoch %d/%d] [Batch %d/%d]"
                    % (epoch + 1, opt.n_epochs, i + 1, len(train_loader))
                )

                logger.info(
                    "[G_loss: %.3f (gen: %.3f) - (rec: %.3f)] [D_loss on (real: %.3f) - (fake: %.3f)]"
                    % (G_loss_meter.avg, g_loss_meter.avg, g_rec_loss_meter.avg, d_loss_real_meter.avg, d_loss_fake_meter.avg))

                #### Save loss values to CSV
                save_loss_to_csv_1(loss_csv_file_path, epoch+1, i+1,
                                 G_loss_meter.avg, g_loss_meter.avg, g_rec_loss_meter.avg, 
                                 d_loss_real_meter.avg, d_loss_fake_meter.avg,
                                 Disc_acc_meter.total_correct_num, Disc_acc_meter.total_num, Disc_acc_meter.avg,
                                 gen_acc_meter.total_correct_num, gen_acc_meter.total_num, gen_acc_meter.avg,
                                 dataset_acc_meter.total_correct_num, dataset_acc_meter.total_num, dataset_acc_meter.avg,
                                 dataset_acc_meter_real.total_correct_num, dataset_acc_meter_real.total_num, dataset_acc_meter_real.avg,
                                 dataset_acc_meter_fake.total_correct_num, dataset_acc_meter_fake.total_num, dataset_acc_meter_fake.avg)
                                
                logger.info("")
                logger.info(
                    "Discriminator Accuracy (%d/%d): %.3f "
                    % (Disc_acc_meter.total_correct_num, Disc_acc_meter.total_num, Disc_acc_meter.avg) )

                logger.info(
                    "[on generated samples (%d/%d): %.3f] "
                    % (gen_acc_meter.total_correct_num, gen_acc_meter.total_num, gen_acc_meter.avg,))
                
                logger.info(                               
                    "[on  dataset  samples (%d/%d): %.3f   (real(%d/%d): %.3f  - fake(%d/%d): %.3f)]"
                    % (dataset_acc_meter.total_correct_num, dataset_acc_meter.total_num, dataset_acc_meter.avg,
                       dataset_acc_meter_real.total_correct_num, dataset_acc_meter_real.total_num, dataset_acc_meter_real.avg,
                       dataset_acc_meter_fake.total_correct_num, dataset_acc_meter_fake.total_num, dataset_acc_meter_fake.avg)) 
                   
                logger.info(f"=========================================================================================== {opt.expid} ")
                G_loss_meter.reset()
                g_loss_meter.reset()
                g_rec_loss_meter.reset()
                d_loss_real_meter.reset()
                d_loss_fake_meter.reset()
                gen_acc_meter.reset()
                dataset_acc_meter_real.reset()
                dataset_acc_meter_fake.reset()
                dataset_acc_meter.reset()
                Disc_acc_meter.reset()

            ################################################################################ sampling
            if (epoch * len(train_loader) + i) % opt.sample_interval == 0:
                with torch.no_grad():
                    print("")
                    print(f" ==========> sampling model at epoch {epoch+1} and batch {i + 1} ... ")
                    start_time = time.time()  # Start timing the sampling
                    sample(sample_dataset, model, epoch * len(train_loader) + i, sample_dir)
                    end_time = time.time()  # End timing the sampling
                    elapsed_time = end_time - start_time  # Calculate the elapsed time
                    print(f" ==========> done! (saved in {sample_dir}/category/{epoch * len(train_loader) + i}.jpg) ... ")
                    print(f"==========> Time spent for sampling: {elapsed_time:.2f} seconds")
                    print("")
            ###############################################################################

        opt.epoch = epoch + 1
        
        if opt.with_infer:
            with torch.no_grad():
                infer(opt.expid, opt.epoch, opt.eval_type, opt, model, repeat=1)

        save(model_dir, model, opt, logger=logger)
        
        plot_from_csv(csv_file_path=loss_csv_file_path, exp_name=opt.expid, use_mean=True, mean_interval=100)

        G_loss_meter.reset()
        g_loss_meter.reset()
        g_rec_loss_meter.reset()
        d_loss_real_meter.reset()
        d_loss_fake_meter.reset()
        gen_acc_meter.reset()
        dataset_acc_meter_real.reset()
        dataset_acc_meter_fake.reset()
        dataset_acc_meter.reset()       
        Disc_acc_meter.reset()
        
    plot_from_csv(csv_file_path=loss_csv_file_path, exp_name=opt.expid, use_mean=True, mean_interval=100)

if __name__ == '__main__':
    main()
