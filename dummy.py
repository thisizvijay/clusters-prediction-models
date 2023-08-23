import pandas as pd

# Load the CSV file into a pandas DataFrame
df = pd.read_csv('cluster_cpu_util_full.csv')

# Convert the 'time' column to datetime type
df['time'] = pd.to_datetime(df['time'], format='mixed')

# Group by 'machine_id' and resample the data to hourly intervals
resampled_df = df.groupby('machine_id').resample('60T', on='time')['cpu_util'].mean().reset_index()

# Save the resampled DataFrame to a new CSV file
resampled_df.to_csv('resampled_cluster_cpu_util.csv', index=False)

print("Resampling and averaging done. Saved to 'resampled_cluster_cpu_util.csv'.")
