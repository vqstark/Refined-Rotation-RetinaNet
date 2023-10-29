from __future__ import print_function

import os
import cv2
import torch
import codecs
import zipfile
import shutil
import argparse
import sys
sys.path.append('datasets/DOTA_devkit')

from tqdm import tqdm
from datasets import *
from models.model import RetinaNet
from utils.detect import im_detect
from utils.bbox import rbox_2_aabb, rbox_2_quad
from utils.utils import sort_corners, is_image, hyp_parse
from utils.map import eval_mAP

from datasets.DOTA_devkit.ResultMerge_multi_process import ResultMerge
from datasets.DOTA_devkit.dota_evaluation_task1 import task1_eval


DATASETS = {'DOTA':DOTADataset}

def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    # pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            # arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, filename)
    zipf.close()

def dota_evaluate(model, 
                  target_size, 
                  test_path,
                  merge_img = False,
                  conf = 0.01):
    
    root_data, evaldata = os.path.split(test_path)
    splitdata = evaldata + 'split'
    ims_dir = os.path.join(root_data, splitdata + '/' + 'images')
    root_dir = 'outputs'
    res_dir = os.path.join(root_dir, 'detections')
    integrated_dir = os.path.join(root_dir, 'integrated')
    merged_dir = os.path.join(root_dir, 'merged')
    dota_out = os.path.join(root_dir, 'dota_out')

    if  os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)

    for f in [res_dir, integrated_dir, merged_dir]: 
        if os.path.exists(f):
            shutil.rmtree(f)
        os.makedirs(f)

    ds = DOTADataset()
    # loss = torch.zeros(3)
    ims_list = [x for x in os.listdir(ims_dir) if is_image(x)]
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean')
    nt = 0
    for idx, im_name in enumerate(tqdm(ims_list, desc=s)):
        im_path = os.path.join(ims_dir, im_name)
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dets = im_detect(model, im, target_sizes=target_size, conf = conf)
        nt += len(dets)
        out_file = os.path.join(res_dir,  im_name[:im_name.rindex('.')] + '.txt')
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if dets.shape[0] == 0:
                f.close()
                continue
            res = sort_corners(rbox_2_quad(dets[:, 2:]))
            for k in range(dets.shape[0]):
                f.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {} {} {:.2f}\n'.format(
                    res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                    res[k, 4], res[k, 5], res[k, 6], res[k, 7],
                    ds.return_class(dets[k, 0]), im_name[:-4], dets[k, 1],)
                )
    if not merge_img:
        ResultMerge(res_dir, integrated_dir, merged_dir)
    else:
        ResultMerge(res_dir, integrated_dir, merged_dir, dota_out)

    ## calc mAP
    mAP, classaps = task1_eval(merged_dir, test_path)
    # # display result
    pf = '%20s' + '%10.3g' * 6  # print format    
    print(pf % ('all', len(ims_list), nt, 0, 0, mAP, 0))
    return 0, 0, mAP, 0 


def evaluate(target_size,
             test_path,
             backbone=None, 
             weight=None, 
             model=None,
             hyps=None,
             merge_img = False,
             conf=0.3):
    if model is None:
        model = RetinaNet(backbone=backbone,hyps=hyps)
        if torch.cuda.is_available():
            model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
            
        if weight.endswith('.pth'):
            chkpt = torch.load(weight, map_location=torch.device('cpu'))
            # load model
            if 'model' in chkpt.keys():
                model.load_state_dict(chkpt['model'])
            else:
                model.load_state_dict(chkpt)

    model.eval()
    results = dota_evaluate(model, target_size, test_path, merge_img, conf)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', dest='backbone', default='res50', type=str)
    parser.add_argument('--weight', type=str, default='weights/best.pth')
    parser.add_argument('--target_size', dest='target_size', default=[800], type=int) 
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    parser.add_argument('--test_path', nargs='?', type=str, default="F:/Quan/DOTA/val", help='Run ImgSplit*.py before eval')

    arg = parser.parse_args()
    hyps = hyp_parse(arg.hyp)
    evaluate(arg.target_size,
             arg.test_path,
             arg.backbone,
             arg.weight,
             hyps = hyps)