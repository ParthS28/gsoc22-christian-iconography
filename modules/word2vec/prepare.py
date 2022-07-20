import pandas as pd

with open('mary2.txt') as f:
    lines = f.readlines()

lines = list(filter(('\n').__ne__, lines))
l = []
for i in lines:
  i = i.split('\n')[0]
  l.append(i)

df = pd.DataFrame(columns=['text'])
df['text'] = l

df.to_csv('input/data3.csv')