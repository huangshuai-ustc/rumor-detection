import pandas as pd
import numpy as np
from collections import defaultdict

df = pd.read_excel('data.xls')
print(df.shape)
df = df.drop_duplicates()
print(df.shape)
df['text'] = df['text'].apply(lambda x: x.replace(' ', ' '))
df.to_excel('data_cleaned.xlsx', index=False)



