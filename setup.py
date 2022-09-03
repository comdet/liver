#convert json to YOLO
import glob, os, pickle, json
from tqdm import tqdm
from os import listdir, getcwd
from os.path import join
import cv2

classes = ["cystic","FFS","solid"]

def getImagesInDir(dir_path):
  image_list = []
  for filename in glob.glob(dir_path + '/*.jpg'):
    image_list.append(filename)
  return image_list

def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

def yolo_to_xml_bbox(bbox, w, h, class_name):
    # x_center, y_center width heigth
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax, class_name]

with open('./dataset/TrainSet/label.json','rb') as file:
  anno_data = json.load(file)

output_dir = './dataset/TrainSet/images/'
input_dir = "./dataset/TrainSet/images/"

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

image_paths = getImagesInDir(input_dir)

for image_path in tqdm(image_paths):
  basename = os.path.basename(image_path)
  basename_no_ext = os.path.splitext(basename)[0]
  out_file = open(output_dir + basename_no_ext + '.txt', 'w')
  im = cv2.imread(image_path)
  [h ,w ,ch] = im.shape
  boxes = anno_data[basename]
  if boxes:
    for bbox_name in boxes:
      bb = xml_to_yolo_bbox(bbox_name,w,h)
      class_name = bbox_name[4]
      class_ind = classes.index(class_name)
      out_file.write(str(class_ind) + " " + " ".join([str(a) for a in bb]) + '\n')
  out_file.close()

print("Finished processing : " + input_dir)