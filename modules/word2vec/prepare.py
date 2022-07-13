import pandas as pd

with open('mary2.txt') as f:
    lines = f.readlines()

lines = list(filter(('\n').__ne__, lines))

l = []

for i in lines:
  t = i.split('. ')
  for j in t:
    l.append(j)

df = pd.DataFrame(columns=['text'])
df['text'] = l

df.to_csv('input/data2.csv')