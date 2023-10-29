from __future__ import print_function

import os
import cv2
import time
import torch
import random
import shutil
import argparse
import codecs
import numpy as np
from datasets import *
from models.model import RetinaNet
from utils.detect import im_detect
from utils.bbox import rbox_2_quad
from utils.utils import is_image, draw_caption, hyp_parse
from utils.utils import show_dota_results, sort_corners
from eval import evaluate
from datasets.DOTA_devkit.ResultMerge_multi_process import ResultMerge
from datasets.DOTA_devkit.SplitOnlyImage_multi_process import splitbase

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
                merge_img = True,
                conf = 0.05)

        img_path = os.path.join(args.ims_dir,'images')
        label_path = 'outputs/dota_out'
        save_imgs =  True
        if save_imgs:
            show_dota_results(img_path,label_path)
    print('Done. (%.3fs)' % (time.time() - t0))

def demo_for_server(args, model, img, put_label, plot_anchor):

    ds = DOTADataset(level = 1)
    # colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(int(hyps['num_classes']))]
    colors = [[10, 255, 226], [255, 0, 0]]

    model.eval()

    src = img
    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cls_dets = im_detect(model, im, target_sizes=args.target_size)
    total_classes = []
    for j in range(len(cls_dets)):
        cls, scores = cls_dets[j, 0], cls_dets[j, 1]
        total_classes.append(cls)
        bbox = cls_dets[j, 2:]
        if len(bbox) == 4:
            draw_caption(src, bbox, '{:1.3f}'.format(scores))
            cv2.rectangle(src, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(0, 0, 255), thickness=2)
        else:
            pts = np.array([rbox_2_quad(bbox[:5]).reshape((4, 2))], dtype=np.int32)
            cv2.drawContours(src, pts, 0, thickness=2, color=colors[int(cls-1)])
            # put_label = False
            # plot_anchor = False
            if put_label:
                label = ds.return_class(cls) + str(' %.2f' % scores)
                fontScale = 0.45
                font = cv2.FONT_HERSHEY_COMPLEX
                thickness = 1
                t_size = cv2.getTextSize(label, font, fontScale=fontScale, thickness=thickness)[0]
                c1 = tuple(bbox[:2].astype('int'))
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 5

                cv2.rectangle(src, c1, c2, colors[int(cls-1)], -1)  # filled
                cv2.putText(src, label, (c1[0], c1[1] -4), font, fontScale, [0, 0, 0], thickness=thickness, lineType=cv2.LINE_AA)
                if plot_anchor:
                    pts = np.array([rbox_2_quad(bbox[5:]).reshape((4, 2))], dtype=np.int32)
                    cv2.drawContours(src, pts, 0, color=(0, 0, 255), thickness=2)

    return src, total_classes.count(1), total_classes.count(2)

def detect_on_scene_img(model, 
                  target_size, 
                  test_path,
                  conf = 0.01):
    root_data, evaldata = os.path.split(test_path)
    splitdata = evaldata + 'split'
    ims_dir = os.path.join(root_data, splitdata + '/')
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
    nt = 0
    for idx, im_name in enumerate(ims_list):
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
    ResultMerge(res_dir, integrated_dir, merged_dir, dota_out)

def demo_on_video(args, path):
    # Read model
    hyps = hyp_parse(args.hyp)
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

    cap = cv2.VideoCapture(path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    
    # Get video properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(5))


    # Create processed folder
    frame_count = 0
    root_path = 'tmp'
    if os.path.exists(os.path.join(root_path, 'frame')):
        shutil.rmtree(os.path.join(root_path, 'frame'))
    os.makedirs(os.path.join(root_path, 'frame'))

    if os.path.exists(os.path.join(root_path, 'framesplit')):
        shutil.rmtree(os.path.join(root_path, 'framesplit'))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_filename = f'frame_{frame_count:04d}.png'
        cv2.imwrite(os.path.join(root_path, 'frame', frame_filename), frame)

        frame_count += 1
    print(f"Extracted {frame_count} frames from the video.")
    cap.release()

    # Split image to fit target size (optional)
    split = splitbase(os.path.join(root_path, 'frame'),
                    os.path.join(root_path, 'framesplit'),
                    gap=150,  
                    subsize=800,
                    num_process=16
                    )
    split.splitdata(1)


    # Detect on each frames and save to folder
    detect_on_scene_img(model,
        args.target_size,
        os.path.join(root_path, 'frame'),    
        conf = 0.05)

    img_path = os.path.join(root_path, 'frame')
    label_path = 'outputs/dota_out'
    show_dota_results(img_path,label_path)
    

    # Create a VideoWriter object to save the processed frames as a video
    split_path = path.split('/')
    file_name = split_path[-1].split('.')[0] + '_out.mp4'
    output_video_path = '/'.join(split_path[:-1]) + '/' + file_name
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    for file in os.listdir('dota_res'):
        processed_frame = cv2.imread(os.path.join('dota_res',file))

        out.write(processed_frame)

    out.release()
    print("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', type=str, default='res50')
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    parser.add_argument('--weight', type=str, default='weights/deploy_02.pth')
    parser.add_argument('--ims_dir', type=str, default='../DOTA/val', help='[../DOTA/val]')
    parser.add_argument('--type', type=str, default='scene', help = 'Detect on [patch] or [scene] or [from_server]')
    parser.add_argument('--save_img', type=str, default=True, help = 'Save detected images or not')
    parser.add_argument('--device', type=str, default='GPU', help = 'Your device')
    parser.add_argument('--target_size', type=int, default=[800])
    # demo(parser.parse_args())

    path = '../Video/1098492569-preview.mp4'
    demo_on_video(parser.parse_args(), path)