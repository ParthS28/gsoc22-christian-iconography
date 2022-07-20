from cmath import sqrt
import json
import codecs
import math
from scipy import spatial

obj_text = codecs.open('output/embeddings.json', 'r', encoding='utf-8').read()
b_new = json.loads(obj_text)

mary = b_new['mary']

l = []
for i in b_new:
    l.append([i, spatial.distance.cosine(mary, b_new[i])])

li = sorted(l, key = lambda x: x[1])
print(li)
