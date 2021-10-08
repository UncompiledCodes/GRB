import pandas as pd
dfIn = pd.read_csv("sdss_galaxy_450000.csv")
count=dfIn["specClass"]
j=0
for i in count:
    if i=="QSO":
        j+=1