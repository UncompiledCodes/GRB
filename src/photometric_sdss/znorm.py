import numpy as np
import pandas as pd

y_pred=[]
targets=[]
znorm=[]

df_tar=pd.read_csv("data/n=600/targers_test.csv",delimiter='\n')
df_y=pd.read_csv("data/n=600/n=600 m=25.csv",delimiter='\n')

for i in range(89887):
   znorm.append(df_tar.loc[i]-df_y.loc[i])

# for index, row in df_tar.iterrows():
    
#     targets.append(row)
    