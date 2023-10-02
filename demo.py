from __future__ import print_function

import os
import cv2
import time
import torch
import random
import shutil
import argparse
import numpy as np
from datasets import *
from models.model import RetinaNet
from utils.detect import im_detect
from utils.bbox import rbox_2_quad
from utils.utils import is_image, draw_caption, hyp_parse
from utils.utils import show_dota_results
from eval import evaluate

def demo(args):
    hyps = hyp_parse(args.hyp)

    ds = DOTADataset(level = 1)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(int(hyps['num_classes']))]

    model = RetinaNet(backbone=args.backbone, hyps=hyps)

    if args.weight.endswith('.pth'):
        if args.device == 'CPU':
            chkpt = torch.load(args.weight, map_location=torch.device('cpu'))
        else:
            chkpt = torch.load(args.weight)
        # load model
        if 'model' in chkpt.keys():
            model.load_state_dict(chkpt['model'])
        else:
            model.load_state_dict(chkpt)
        print('Load weight from: {}'.format(args.weight))
    model.eval()

    t0 = time.time()
    if args.type == 'patch':
        ims_list = [x for x in os.listdir(args.ims_dir) if is_image(x)]
        if args.save_img:
            if os.path.exists('outputs/dota_out'):
                shutil.rmtree('outputs/dota_out')
            os.mkdir('outputs/dota_out')
        for idx, im_name in enumerate(ims_list):
            s = ''
            t = time.time()
            im_path = os.path.join(args.ims_dir, im_name)   
            s += 'image %g/%g %s: ' % (idx, len(ims_list), im_path)
            src = cv2.imread(im_path, cv2.IMREAD_COLOR)
            im = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            cls_dets = im_detect(model, im, target_sizes=args.target_size)
            for j in range(len(cls_dets)):
                cls, scores = cls_dets[j, 0], cls_dets[j, 1]
                bbox = cls_dets[j, 2:]
                if len(bbox) == 4:
                    draw_caption(src, bbox, '{:1.3f}'.format(scores))
                    cv2.rectangle(src, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=2)
                else:
                    pts = np.array([rbox_2_quad(bbox[:5]).reshape((4, 2))], dtype=np.int32)
                    cv2.drawContours(src, pts, 0, thickness=2, color=colors[int(cls-1)])
                    put_label = False
                    plot_anchor = False
                    if put_label:
                        label = ds.return_class(cls) + str(' %.2f' % scores)
                        fontScale = 0.45
                        font = cv2.FONT_HERSHEY_COMPLEX
                        thickness = 1
                        t_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=thickness)[0]
                        c1 = tuple(bbox[:2].astype('int'))
                        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 5
                        # import ipdb;ipdb.set_trace()

                        cv2.rectangle(src, c1, c2, colors[int(cls-1)], -1)  # filled
                        cv2.putText(src, label, (c1[0], c1[1] -4), font, fontScale, [0, 0, 0], thickness=thickness, lineType=cv2.LINE_AA)
                        if plot_anchor:
                            pts = np.array([rbox_2_quad(bbox[5:]).reshape((4, 2))], dtype=np.int32)
                            cv2.drawContours(src, pts, 0, color=(0, 0, 255), thickness=2)
            print('%sDone. (%.3fs) %d objs' % (s, time.time() - t, len(cls_dets)))
            if args.save_img:
                out_path = os.path.join('outputs/dota_out' , os.path.split(im_path)[1])
                cv2.imwrite(out_path, src)
    ## DOTA detect on scene image
    else:
        evaluate(args.target_size,
                args.ims_dir,    
                args.backbone,
                args.weight,
                hyps = hyps,
                conf = 0.05)
        if  os.path.exists('outputs/dota_out'):
            shutil.rmtree('outputs/dota_out')
        os.mkdir('outputs/dota_out')
        exec('cd outputs &&  rm -rf detections && rm -rf integrated  && rm -rf merged')    

        img_path = os.path.join(args.ims_dir,'images')
        label_path = 'outputs/dota_out'
        save_imgs =  True
        if save_imgs:
            show_dota_results(img_path,label_path)
    print('Done. (%.3fs)' % (time.time() - t0))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', type=str, default='res50')
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    parser.add_argument('--weight', type=str, default='weights/last.pth')
    parser.add_argument('--ims_dir', type=str, default='patch_test_imgs')
    parser.add_argument('--type', type=str, default='patch', help = 'Detect on patch or scene')
    parser.add_argument('--save_img', type=str, default=True, help = 'Save detected images or not')
    parser.add_argument('--device', type=str, default='CPU', help = 'Your device')
    parser.add_argument('--target_size', type=int, default=[800])
    demo(parser.parse_args())