#!/usr/bin/env python


"""Generate bottom-up attention features as a tsv file. Can use cuda and multiple GPUs.
   Modify the load_image_ids script as necessary for your data location. """


# Example:
# python generate_tsv.py --expid ${expid}$ --epoch ${epoch}$ --cuda


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import csv
import json
import torch
import base64
from utils.timer import Timer
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as dset
from imageio import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.roi_layers import nms
from torchvision.ops import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

csv.field_size_limit(sys.maxsize)

FIELDNAMES = ['image_id', 'image_name', 'image_w','image_h','num_boxes', 'boxes', 'pred_scores', 'features', 'fg_feature']
# Settings for the number of features per image. To re-create pretrained features with 36 features
# per image, set both values to 36.
MIN_BOXES = 32
MAX_BOXES = 32
# MIN_BOXES = 10
# MAX_BOXES = 100
#========================================================================================#
def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Generate bbox output from a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='vg', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models',
                        default="models")
    parser.add_argument('--cuda', dest='cuda', default=True,
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images',
                        default="images")
    parser.add_argument('--classes_dir', dest='classes_dir',
                        help='directory to load object classes for classification',
                        default="data/genome/1600-400-20")
    parser.add_argument('--out', dest='outfile',
                        help='output filepath',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/res101.yml', type=str)
    parser.add_argument('--split', dest='data_split',
                        help='dataset to use',
                        default='karpathy_train', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    
    ### START ADD ###
    parser.add_argument("--expid", type=str, required=True, help="experiment name")
    parser.add_argument("--epoch", type=int, required=True, help="epoch for evaluation")
    parser.add_argument("--eval_type", type=str, default="eval", help="evaluation type")
    ### END ADD ###

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()

    return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
#========================================================================================#
def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)
#========================================================================================#
#build [image_path, image_id] for dataset, and you can create your own
def load_image_ids(split_name):
    ''' Load a list of (path,image_id tuples). Modify this to suit your data locations. '''
    split = []
    if split_name == 'coco_test2014':
      with open('/data/coco/annotations/image_info_test2014.json') as f:
        data = json.load(f)
        for item in data['images']:
          image_id = int(item['id'])
          filepath = os.path.join('/data/test2014/', item['file_name'])
          split.append((filepath,image_id))
    elif split_name == 'coco_test2015':
      with open('/data/coco/annotations/image_info_test2015.json') as f:
        data = json.load(f)
        for item in data['images']:
          image_id = int(item['id'])
          filepath = os.path.join('/data/test2015/', item['file_name'])
          split.append((filepath,image_id))
    elif split_name == 'genome':
      with open('/data/visualgenome/image_data.json') as f:
        for item in json.load(f):
          image_id = int(item['image_id'])
          filepath = os.path.join('/data/visualgenome/', item['url'].split('rak248/')[-1])
          split.append((filepath,image_id))
    else:
      print ('Unknown split')
    return split
#========================================================================================#
def compute_iou(cls_boxes, ori_bbox, thresh):
    dets = cls_boxes.cpu().numpy()
    ori_bbox = np.array(ori_bbox)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    area_ori = (ori_bbox[2] -  ori_bbox[0] + 1) * (ori_bbox[3] - ori_bbox[1] + 1)

    xx1 = np.maximum(ori_bbox[0], x1)
    yy1 = np.maximum(ori_bbox[1], y1)
    xx2 = np.minimum(ori_bbox[2], x2)
    yy2 = np.minimum(ori_bbox[3], y2)

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h
    ovr = inter / (areas + area_ori - inter)

    inds = np.where(ovr < thresh)[0]

    return inds
#========================================================================================#
def get_detections_from_im(fasterRCNN, classes, remained_ids, im_file, image_id, ori_bbox, args, conf_thresh=0.05):
    """obtain the image_info for each image,
    im_file: the path of the image

    return: dict of {'image_id', 'image_h', 'image_w', 'num_boxes', 'boxes', 'features'}
    boxes: the coordinate of each box
    """
    # initilize the tensor holder here.
    im_data = torch.FloatTensor(1)
    im_info = torch.FloatTensor(1)
    num_boxes = torch.LongTensor(1)
    gt_boxes = torch.FloatTensor(1)

    # ship to cuda
    if args.cuda > 0:
        im_data = im_data.cuda()
        im_info = im_info.cuda()
        num_boxes = num_boxes.cuda()
        gt_boxes = gt_boxes.cuda()

    # make variable
    with torch.no_grad():
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

    if args.cuda > 0:
        cfg.CUDA = True

    if args.cuda > 0:
        fasterRCNN.cuda()

    fasterRCNN.eval()

    #load images
    # im = cv2.imread(im_file)
    im_in = np.array(imread(im_file))
    if len(im_in.shape) == 2:
      im_in = im_in[:,:,np.newaxis]
      im_in = np.concatenate((im_in,im_in,im_in), axis=2)
    # rgb -> bgr
    im = im_in[:,:,::-1]

    blobs, im_scales = _get_image_blob(im)
    assert len(im_scales) == 1, "Only single-image batch implemented"
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)

    with torch.no_grad():
            im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
            im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
            gt_boxes.resize_(1, 1, 5).zero_()
            num_boxes.resize_(1).zero_()
    # pdb.set_trace()
    det_tic = time.time()

    # for [left, top, width, height] format bbox
    fgroi = [ori_bbox[0], ori_bbox[1], ori_bbox[0]+ori_bbox[2], ori_bbox[1]+ori_bbox[3]]
    # fgroi = [ori_bbox[0], ori_bbox[1], ori_bbox[2], ori_bbox[3]]
    fgroi_resize = [[im_info[0][-1] * x for x in fgroi]]

    # the region features[box_num * 2048] are required.
    rois, cls_prob, bbox_pred, \
    rpn_loss_cls, rpn_loss_box, \
    RCNN_loss_cls, RCNN_loss_bbox, \
    rois_label, pooled_feat, fg_pooled_feat = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, pool_feat=True, fgroi=fgroi_resize)

    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
          if args.class_agnostic:
              if args.cuda > 0:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

              box_deltas = box_deltas.view(1, -1, 4)
          else:
              if args.cuda > 0:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                             + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
              box_deltas = box_deltas.view(1, -1, 4 * len(classes))

        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_scales[0]
    
    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    det_toc = time.time()
    detect_time = det_toc - det_tic
    misc_tic = time.time()

    max_conf = torch.zeros((pred_boxes.shape[0]))
    if args.cuda > 0:
        max_conf = max_conf.cuda()

    roithred = 0.2  # for roithred in [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
    for j in xrange(1, len(classes)):
        if j not in remained_ids:
            continue
        inds = torch.nonzero(scores[:,j]>conf_thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
          cls_scores = scores[:,j][inds]
          _, order = torch.sort(cls_scores, 0, True)
          if args.class_agnostic:
            cls_boxes = pred_boxes[inds, :]
          else:
            cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

          keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
          ano_keep = compute_iou(cls_boxes[order, :], fgroi, thresh=roithred)
          new_keep = list(set([int(x) for x in keep]) & (set([int(x) for x in order[ano_keep]])))
          new_keep = [int(x) for x in new_keep]
          index = inds[order[new_keep]]

          max_conf[index] = torch.where(scores[index, j] > max_conf[index], scores[index, j], max_conf[index])

    if args.cuda > 0:
        keep_boxes = torch.where(max_conf >= conf_thresh, max_conf, torch.tensor(0.0).cuda())
    else:
        keep_boxes = torch.where(max_conf >= conf_thresh, max_conf, torch.tensor(0.0))

    keep_boxes = torch.nonzero(keep_boxes).squeeze(1)
    if len(keep_boxes) < MIN_BOXES:
        keep_boxes = torch.argsort(max_conf, descending = True)[:MIN_BOXES]
    elif len(keep_boxes) > MAX_BOXES:
        keep_boxes = torch.argsort(max_conf, descending = True)[:MAX_BOXES]

    objects = torch.argmax(scores[keep_boxes][:,1:], dim=1)
    box_dets = np.zeros((len(keep_boxes), 6), dtype=np.float32)
    boxes = pred_boxes[keep_boxes]
    label_scores = np.array(scores[keep_boxes].cpu())

    for i in range(len(keep_boxes)):
        kind = objects[i]+1
        bbox = boxes[i, kind * 4: (kind + 1) * 4]
        box_dets[i][:4] = np.array(bbox.cpu())
        box_dets[i][4]  = kind
        box_dets[i][5]  = max_conf[keep_boxes[i]]
   
    return {
        'image_id': image_id,
        'image_name': "/".join(im_file.split("/")[-2:]),
        'image_h': np.size(im, 0),
        'image_w': np.size(im, 1),
        'num_boxes': len(keep_boxes),
        'boxes': base64.b64encode(box_dets),
        'pred_scores': base64.b64encode(label_scores),
        'features': base64.b64encode((pooled_feat[keep_boxes].cpu()).detach().numpy()),
        'fg_feature': base64.b64encode((fg_pooled_feat.cpu()).detach().numpy()),
    }
    # pooled_feat    = pooled_feat[keep_boxes]
    # visualize_results(im_file, ori_bbox, box_dets, (im.shape[1], im.shape[0]), label_scores)
    
    # if base64:
    #   box_dets     = base64.b64encode(box_dets)
    #   label_scores = base64.b64encode(label_scores)
    #   pooled_feat  = base64.b64encode(pooled_feat.cpu().detach().numpy())
    #   fg_pooled_feat = base64.b64encode(fg_pooled_feat.cpu().detach().numpy())
    
    # return {
    #     'image_id': image_id,
    #     'image_name': im_file,
    #     'image_h': np.size(im, 0),
    #     'image_w': np.size(im, 1),
    #     'num_boxes': len(keep_boxes),
    #     'boxes': box_dets,
    #     'pred_scores': label_scores,
    #     'features': pooled_feat,
    #     'fg_feature': fg_pooled_feat,
    # }

#========================================================================================#    
# def visualize_results(im_file, fg_bbox, ref_bboxes, src_size, pred_scores):
# 	save_dir = '../cache/faster_rcnn_results'
# 	os.makedirs(save_dir, exist_ok=True)

# 	img  = cv2.imread(im_file)
# 	x1,y1,x2,y2 = list(map(int, fg_bbox))
# 	cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 4)

# 	for i in range(5):
# 		bbox = ref_bboxes[i][:4].tolist()
# 		x1,y1,x2,y2 = bbox
# 		x_scale = float(img.shape[1]) / src_size[0]
# 		y_scale = float(img.shape[0]  / src_size[1])
# 		x1 = int(x_scale * x1)
# 		y1 = int(y_scale * y1)
# 		x2 = int(x_scale * x2)
# 		y2 = int(y_scale * y2)
# 		cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 4)
# 	res_file = os.path.join(save_dir, os.path.basename(im_file))
# 	cv2.imwrite(res_file, img)
# 	print(res_file)
#========================================================================================#
def load_model(args):
    # set cfg according to the dataset used to train the pre-trained model
    if args.dataset == "pascal_voc":
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "pascal_voc_0712":
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "coco":
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "imagenet":
        args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
    elif args.dataset == "vg":
        args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

    if args.cfg_file is not None:
      cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
      cfg_from_list(args.set_cfgs)

    cfg.USE_GPU_NMS = args.cuda

    print('Using config:')
    pprint.pprint(cfg)
    np.random.seed(cfg.RNG_SEED)

    # Load classes
    classes = ['__background__']
    with open(os.path.join(args.classes_dir, 'objects_vocab.txt')) as f:
        for object in f.readlines():
            classes.append(object.split(',')[0].lower().strip())

    remained_classes = ['__background__']
    with open("object_150_list.txt") as f:
        for object in f.readlines():
            remained_classes.append(object.split(',')[0].lower().strip())

    remained_ids = set(classes.index(x) for x in remained_classes)

    if not os.path.exists(args.load_dir):
        raise Exception('There is no input directory for loading network from ' + args.load_dir)
    load_name = os.path.join(args.load_dir, 'faster_rcnn_{}_{}.pth'.format(args.net, args.dataset))

    # initilize the network here. the network used to train the pre-trained model
    if args.net == 'vgg16':
      fasterRCNN = vgg16(classes, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res101':
      fasterRCNN = resnet(classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res50':
      fasterRCNN = resnet(classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
    elif args.net == 'res152':
      fasterRCNN = resnet(classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
    else:
      print("network is not defined")
      pdb.set_trace()

    fasterRCNN.create_architecture()

    print("load checkpoint %s" % (load_name))
    if args.cuda > 0:
      checkpoint = torch.load(load_name)
    else:
      checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
    fasterRCNN.load_state_dict(checkpoint['model'])
    if 'pooling_mode' in checkpoint.keys():
      cfg.POOLING_MODE = checkpoint['pooling_mode']

    print('load model successfully!')

    print("load model %s" % (load_name))

    return classes, remained_ids, fasterRCNN
#========================================================================================#
def generate_tsv(outfile, image_ids, args):
    # First check if file exists, and if it is complete
    # image_ids: [image_path, image_id]
    wanted_ids = set([int(image_id[1]) for image_id in image_ids])
    found_ids = set()
    if os.path.exists(outfile):
        with open(outfile) as tsvfile:
            reader = csv.DictReader(tsvfile, delimiter='\t', fieldnames = FIELDNAMES)
            for item in reader:
                found_ids.add(int(item['image_id']))
    missing = wanted_ids - found_ids
    if len(missing) == 0:
        print ('Already completed {:d}'.format(len(image_ids)))
    else:
        print ('Missing {:d}/{:d}'.format(len(missing), len(image_ids)))
    if len(missing) > 0:
        classes, remained_ids, fasterRCNN = load_model(args)
        with open(outfile, 'a+') as tsvfile:
            writer = csv.DictWriter(tsvfile, delimiter = '\t', fieldnames = FIELDNAMES)
            _t = {'misc' : Timer()}
            count = 0
            for im_file,image_id, ori_bbox in image_ids:
                if int(image_id) in missing:
                    _t['misc'].tic()
                    writer.writerow(get_detections_from_im(fasterRCNN, classes, remained_ids, im_file, image_id, ori_bbox, args))
                    _t['misc'].toc()
                    if (count % 100) == 0:
                        print ('{:d}/{:d} {:.3f}s (projected finish: {:.2f} hours)' \
                              .format(count+1, len(missing), _t['misc'].average_time,
                              _t['misc'].average_time*(len(missing)-count)/3600))
                    count += 1
#========================================================================================#

def get_imgids(dataset_dir, args):
    img_dir = dataset_dir
    csv_file = os.path.join(dataset_dir, '{}.csv'.format(args.eval_type))
    csv_data = csv.DictReader(open(csv_file, 'r'))
    res_data = [
        [
            os.path.join(img_dir, row['img_path']), i, list(map(int, row['bbox'][1:-1].split(',')))
        ]
        for i, row in enumerate(csv_data)
    ]
    return res_data
#========================================================================================#  
# def get_merged_imgids():
# 	csv_file = './../cache/dataset_merge.json'
# 	samples = json.load(open(csv_file, 'r'))
# 	res_data = [
# 	[
# 		sample['composite'], i, sample['bbox']
# 	]
# 	for i, sample in enumerate(samples)
# 	]
# 	return res_data
#========================================================================================#
  

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    dataset_dir = os.path.join('../result', args.expid, args.eval_type, str(args.epoch))
    assert (os.path.exists(dataset_dir))
    image_ids = get_imgids(dataset_dir, args)
    outfile = os.path.join(dataset_dir, "{}_roiinfos.csv".format(args.eval_type))

    # image_ids = get_merged_imgids()
    # outfile = './../cache/roiinfos.csv'
    generate_tsv(outfile, image_ids, args)
