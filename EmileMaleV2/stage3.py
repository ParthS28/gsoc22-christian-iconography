import json
import codecs
import math
from scipy import spatial
import pandas as pd
import os

obj_text = codecs.open('embeddings/embeddings.json', 'r', encoding='utf-8').read()
coordinates = json.loads(obj_text)

mary = coordinates['mary']

# Dictionary for distances from Mary
l = {}
for i in coordinates:
    l[i] = spatial.distance.cosine(mary, coordinates[i])


# Create Dataframe
df = pd.DataFrame(columns=["item", "stage_1_prediction", "final_prediction"])
# Read input dataframe
df_in = pd.read_csv('out1.csv')
items = df_in['item']

classes = ['baby','person','angel','book','jar','crown','bird','crescent','flowers','crucifict','pear','skull']
for item in items:
    row = df_in[df_in['item']==item]
    # print(row)
    # print(item)
    # break
    if not os.path.exists('out2/exp/labels/'+item.split('.')[0]+'.txt'):
        print('out2/exp/labels/'+item.split('.')[0]+'.txt not present')
        df.loc[df.shape[0]] = [item, row['predicted'].item(), row['predicted'].item()]
        continue
    # if row['predicted'].item()==row['predicted'].item():
    #     df.loc[df.shape[0]] = [item, row['predicted'].item(), row['predicted'].item()]
    #     continue
    with open('out2/exp/labels/'+item.split('.')[0]+'.txt', 'r') as f:
        lines = f.readlines()

    present_in_image = []
    for line in lines:
        num = int(line.split(' ')[0])
        present_in_image.append(classes[num])
 
    avg_dist = 0
    for p in present_in_image:
        if p in l:
            avg_dist += l[p]
    avg_dist = avg_dist/len(present_in_image)

    if(row['predicted'].item() == 0 and avg_dist>1): # tag mislabel
        df.loc[df.shape[0]] = [item, row['predicted'].item(), 1]
    elif(row['predicted'].item() == 1 and avg_dist<1): # tag mislabel
        df.loc[df.shape[0]] = [item, row['predicted'].item(), 0]
    else:
        df.loc[df.shape[0]] = [item, row['predicted'].item(), row['predicted'].item()]
    # break

df.to_csv('final.csv')