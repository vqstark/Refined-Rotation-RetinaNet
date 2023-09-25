import os
import time
import sys
import glob
import shutil
import argparse
from PIL import Image
from tqdm import tqdm
from DOTA_devkit.ImgSplit_multi_process import splitbase, rm_background

def bmpToJpg(file_path):
   for fileName in tqdm(os.listdir(file_path)):
       newFileName = fileName[0:fileName.find(".bmp")]+".jpg"
       im = Image.open(os.path.join(file_path,fileName))
       rgb = im.convert('RGB')      
       rgb.save(os.path.join(file_path,newFileName))

def del_bmp(root_dir=None):
    file_list = os.listdir(root_dir)
    for f in file_list:
        file_path = os.path.join(root_dir, f)
        if os.path.isfile(file_path):
            if f.endswith(".BMP") or f.endswith(".bmp"):
                os.remove(file_path)
                print( "File removed! " + file_path)
        elif os.path.isdir(file_path):
            del_bmp(file_path)

def remove_non_car_labels(dota_path):
    train_label_path = os.path.join(dota_path, 'train', 'labels')
    val_label_path = os.path.join(dota_path, 'val', 'labels')

    car_train_label_path = os.path.join(dota_path, 'train', 'labelTxt')
    if os.path.exists(car_train_label_path):
      shutil.rmtree(car_train_label_path)
    os.mkdir(car_train_label_path)

    car_val_label_path = os.path.join(dota_path, 'val', 'labelTxt')
    if os.path.exists(car_val_label_path):
      shutil.rmtree(car_val_label_path)
    os.mkdir(car_val_label_path)

    files = sorted(glob.glob(os.path.join(train_label_path, '**.*' )))
    for file in files:
        img_path, filename = os.path.split(file)
        with open(file, 'r') as f:
            lines = f.readlines()
        f.close()
        with open(os.path.join(dota_path, 'train', 'labelTxt', filename), 'w') as f:
            for line in lines[2:]:
                if line.split()[-2] in ['large-vehicle', 'small-vehicle']:
                    f.write(line)
        f.close()
    
    files = sorted(glob.glob(os.path.join(val_label_path, '**.*' )))
    for file in files:
        img_path, filename = os.path.split(file)
        with open(file, 'r') as f:
            lines = f.readlines()
        f.close()
        with open(os.path.join(dota_path, 'val', 'labelTxt', filename), 'w') as f:
            for line in lines[2:]:
                if line.split()[-2] in ['large-vehicle', 'small-vehicle']:
                    f.write(line)
        f.close()
    

def generate_image_ds(train_img_path, set_file):
    files= sorted(glob.glob(os.path.join(train_img_path, '**.*' )))
    with open(set_file,'w') as f:
        for file in files:
            img_path, filename = os.path.split(file)
            name, extension = os.path.splitext(filename)
            if extension in ['.jpg', '.bmp','.png']:
                annotation_path = img_path[:-6] + 'labels'
                annotation_file = name + '.txt'
                with open(os.path.join(annotation_path, annotation_file), 'r') as a:
                    lines = a.readlines()
                a.close()
                # Keep image contains objects
                if len(lines) > 2:
                    f.write(os.path.join(file)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--target_size', type=int, default=500)
    parser.add_argument('--overlap', type=int, default=100)
    parser.add_argument('--dota_path', type=str, default='/content/DOTA')
    args = parser.parse_args()

    dota_path = args.dota_path

    # Remove non-car labels
    remove_non_car_labels(dota_path)

    # Split train images
    print("Splitting train images:")
    start = time.time()
    split = splitbase(os.path.join(dota_path, 'train'),
                      os.path.join(dota_path, 'trainsplit'),
                      gap=args.overlap,  
                      subsize=args.target_size,
                      num_process=8
                      )
    split.splitdata(1)
    rm_background(os.path.join(dota_path, 'trainsplit'))
    elapsed = (time.time() - start)
    print("Time used:", elapsed)

    # Split val images
    print("Splitting val images:")
    start = time.time()
    split = splitbase(os.path.join(dota_path, 'val'),
                      os.path.join(dota_path, 'valsplit'),
                      gap=args.overlap,  
                      subsize=args.target_size,
                      num_process=8
                      )
    split.splitdata(1)
    rm_background(os.path.join(dota_path, 'valsplit'))
    elapsed = (time.time() - start)
    print("Time used:", elapsed)

    # Generate image set
    generate_image_ds(os.path.join(dota_path, 'trainsplit', 'images'), 
                      os.path.join(dota_path, 'trainsplit', 'train.txt'))
    
    generate_image_ds(os.path.join(dota_path, 'valsplit', 'images'), 
                      os.path.join(dota_path, 'valsplit', 'val.txt'))