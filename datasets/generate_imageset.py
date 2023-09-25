import os
import sys
import glob
from PIL import Image
from tqdm import tqdm

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

def generate_image_ds(train_img_path, val_img_path, set_file):
    files= sorted(glob.glob(os.path.join(train_img_path, '**.*' ))) + sorted(glob.glob(os.path.join(val_img_path, '**.*' )))
    with open(set_file,'w') as f:
        for file in files:
            img_path, filename = os.path.split(file)
            name, extension = os.path.splitext(filename)
            if extension in ['.jpg', '.bmp','.png']:
                f.write(os.path.join(file)+'\n')

if __name__ == '__main__':
    train_img_path = r"/content/DOTA/trainsplit/images" 
    val_img_path = r"/content/DOTA/valsplit/images" 
    set_file = r'/content/DOTA/trainval.txt'
    generate_image_ds(train_img_path, val_img_path, set_file)