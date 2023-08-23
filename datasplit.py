import pandas as pd

df = pd.read_csv('cluster_cpu_util_full.csv')

print(df.head())
print(df.shape)
# get only first 10000k rows
df = df.iloc[:1000000]

# save the first 50k rows to a new csv file
df.to_csv('cluster_cpu_util.csv', index=False)
print("done")