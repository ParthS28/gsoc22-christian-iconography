import albumentations as A
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
# from bbaug import policies
import cv2
import os
# from pascal_voc_writer import Writer
from xml.dom import minidom
import imgaug as ia
import imgaug.augmenters as iaa
import math
import random
import copy
import glob
imagespath = 'all_data/'
# random.seed(7)


def readImage(filename):
    # OpenCV uses BGR channels
    img = cv2.imread(imagespath+filename)
    return img


def getCoordinates(filename):
    allbb = []
    xmldoc = minidom.parse(imagespath+filename)
    itemlist = xmldoc.getElementsByTagName('object')

    size = xmldoc.getElementsByTagName('size')[0]
    width = int((size.getElementsByTagName('width')[0]).firstChild.data)
    height = int((size.getElementsByTagName('height')[0]).firstChild.data)

    for item in itemlist:
        classid = (item.getElementsByTagName('name')[0]).firstChild.data
        xmin = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('xmin')[0]).firstChild.data
        ymin = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('ymin')[0]).firstChild.data
        xmax = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('xmax')[0]).firstChild.data
        ymax = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('ymax')[0]).firstChild.data

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)

        b = [xmin, ymin, xmax, ymax, classid]
        allbb.append(b)
    return allbb


def start():
    count = 3000
    for filename in os.listdir(imagespath):
        # print(filename)
        if filename.endswith(".jpg") or filename.endswith(".JPG"):
            title, ext = os.path.splitext(os.path.basename(filename))
            print(title)
            image = readImage(filename)
        # if filename.endswith(".txt"):
            if not os.path.exists('all_data/'+filename.split('.')[0]+'.txt'):
                print('hi')
                continue
            # xmlTitle, txtExt = os.path.splitext(os.path.basename(filename))
            if True:
                
                # bboxes = getCoordinates(filename)
                bboxes = readYolo(imagespath+filename.split('.')[0]+'.txt')
                for i in range(0, 9):
                    img = copy.deepcopy(image)
                    transform = getTransform(i)
                    try:
                        transformed = transform(image=img, bboxes=bboxes)
                        transformed_image = transformed['image']
                        transformed_bboxes = transformed['bboxes']
                        name = 'final/images/'+title+str(count)+'.jpg'
                        cv2.imwrite(name, transformed_image)
                        writeYolo(transformed_bboxes, count, title)
                        count = count+1
                    except:
                        print("bounding box issues")
                        continue

                # bboxes = [[int(float(j)) for j in i] for i in bb]




def readYolo(filename):
    coords = []
    with open(filename, 'r') as fname:
        for file1 in fname:
            x = file1.strip().split(' ')
            x.append(x[0])
            x.pop(0)
            x[0] = float(x[0])
            x[1] = float(x[1])
            x[2] = float(x[2])
            x[3] = float(x[3])
            coords.append(x)
    return coords


def writeYolo(coords, count, name):

    with open('final/labels/'+name+str(count)+'.txt', "w") as f:
        for x in coords:
            f.write("%s %s %s %s %s \n" % (x[-1], x[0], x[1], x[2], x[3]))


def getTransform(loop):
    if loop == 0:
        transform = A.Compose([
            A.HorizontalFlip(p=1),
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 1:
        transform = A.Compose([
            A.RandomBrightnessContrast(p=1),
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 2:
        transform = A.Compose([
            A.MultiplicativeNoise(multiplier=0.5, p=0),
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 3:
        transform = A.Compose([
            A.VerticalFlip(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 4:
        transform = A.Compose([
            A.Blur(blur_limit=(50, 50), p=0)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 5:
        transform = A.Compose([
            A.Transpose(1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 6:
        transform = A.Compose([
            A.RandomRotate90(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 7:
        transform = A.Compose([
            A.JpegCompression(quality_lower=0, quality_upper=1, p=0.2)
        ], bbox_params=A.BboxParams(format='yolo'))
    elif loop == 8:
        transform = A.Compose([
            A.Cutout(num_holes=50, max_h_size=40,
                     max_w_size=40, fill_value=128, p=0)
        ], bbox_params=A.BboxParams(format='yolo'))

    return transform

start()





# import random
# import os

# import cv2
# from matplotlib import pyplot as plt

# import albumentations as A


# BOX_COLOR = (255, 0, 0) # Red
# TEXT_COLOR = (255, 255, 255) # White


# def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=2):
#     """Visualizes a single bounding box on the image"""
#     x_min, y_min, w, h = bbox
#     x_min, x_max, y_min, y_max = int(x_min), int(x_min + w), int(y_min), int(y_min + h)
   
#     cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
    
#     ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)    
#     cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)), (x_min + text_width, y_min), BOX_COLOR, -1)
#     cv2.putText(
#         img,
#         text=class_name,
#         org=(x_min, y_min - int(0.3 * text_height)),
#         fontFace=cv2.FONT_HERSHEY_SIMPLEX,
#         fontScale=0.35, 
#         color=TEXT_COLOR, 
#         lineType=cv2.LINE_AA,
#     )
#     return img


# def visualize(image, bboxes, category_ids, category_id_to_name):
#     img = image.copy()
#     for bbox, category_id in zip(bboxes, category_ids):
#         class_name = category_id_to_name[category_id]
#         img = visualize_bbox(img, bbox, class_name)
#     plt.figure(figsize=(12, 12))
#     plt.axis('off')
#     plt.imshow(img)


# image = cv2.imread('images/0a81ec0f-f103-443b-bfe9-1e3dd05306d5_7bacdafe-bf4c-460f-bd0c-991ac1d2d839.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# if not os.path.exists('labels/'+'0a81ec0f-f103-443b-bfe9-1e3dd05306d5_7bacdafe-bf4c-460f-bd0c-991ac1d2d839'+'.txt'):
#     print('out2/exp/labels/'+image.split('.')[0]+'.txt not present')
        

# with open('labels/'+'0a81ec0f-f103-443b-bfe9-1e3dd05306d5_7bacdafe-bf4c-460f-bd0c-991ac1d2d839'+'.txt', 'r') as f:
#     lines = f.readlines()

# bbox = []
# labels = []
# for line in lines:
#     # labels.append(str(line.split(' ')[0]))
#     bbox.append([line.split(' ')[1], line.split(' ')[2], line.split(' ')[3], line.split(' ')[4], line.split(' ')[0]])

# # ids = [0,1,2,3,4,5,6,7,8,9,10,11,12]
# category_id_to_name = {0: 'baby', 1: 'person', 2: 'angel', 3: 'book', 4: 'jar', 5: 'crown', 6: 'bird', 7: 'crescent', 8: 'flowers', 9: 'crucifict', 10: 'pear', 11: 'skull', 12: 'lamb'}

# ids = ['baby','person','angel','book','jar','crown','bird','crescent','flowers','crucifict','pear','skull', 'lamb']
# transform = A.Compose(
#     [A.HorizontalFlip(p=0.5)],
#     bbox_params=A.BboxParams(format='yolo'),
# )

# random.seed(7)
# transformed = transform(image=image, bboxes=bbox)
# visualize(
#     transformed['image'],
#     transformed['bboxes'],
#     transformed['category_ids'],
#     category_id_to_name,
# )
