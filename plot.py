import pandas as pd

df = pd.read_csv('dataset_4weeks_H1W1A1_H2W2A2.csv')

df.plot(kind="bar", figsize = (2, 4))

#print(df.to_string()) 

