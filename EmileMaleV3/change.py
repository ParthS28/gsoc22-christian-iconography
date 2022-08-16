import pandas as pd
import os 


df = pd.read_csv('data/info_small.csv')

# df['set']='train'
# df.to_csv('data/info.csv')
items = list(df['item'])

l = os.listdir('data/images/')

for i in l:
    if i.split('.jpg')[0] not in items:
        os.rename('data/images/'+i, 'data/extra/'+i)