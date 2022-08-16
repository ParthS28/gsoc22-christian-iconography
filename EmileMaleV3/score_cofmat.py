import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, precision_score

df = pd.read_csv('out1.csv')

conf = confusion_matrix(df['label'], df['predicted'])

print(conf)
# print(precision_score(df['label'], df['predicted']))
df1 = pd.read_csv('final.csv')

conf = confusion_matrix(df['label'], df1['final_prediction'])


print(conf)
# print(precision_score(df['label'], df1['final_prediction']))
