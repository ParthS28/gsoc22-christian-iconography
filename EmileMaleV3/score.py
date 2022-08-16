import pandas as pd


df = pd.read_csv('out1.csv')

score = 0
for i, row in df.iterrows():
    if(row['label'] == row['predicted']):
        score+=1

print('Score after stage 1', score/df.shape[0])


df = pd.read_csv('final.csv')

score = 0
for i, row in df.iterrows():
    if(row['label'] == row['final_prediction']):
        score+=1

print('Score after stage 3', score/df.shape[0])