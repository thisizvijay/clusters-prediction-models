# Importing the libraries
import pandas as pd

# Importing the dataset
print('Reading the dataset...')
df = pd.read_csv('cluster_cpu_util.csv')
print('Done reading the dataset...')

# Convert time column to datetime objects
print('Converting time column to datetime objects...')
df['time'] = pd.to_datetime(df['time'], format='mixed')
print('Done converting time column to datetime objects...')

# Sort by time
print('Sorting by time...')
df.sort_values(by='time', inplace=True)
print('Done sorting by time...')

# Convert machine_id column to categorical
print('Converting machine_id column to categorical...')
df['machine_id'] = df['machine_id'].astype('category')
print('Done converting machine_id column to categorical...')

# Make a new dataframe with hourly averages of cpu_util
print('Making a new dataframe with hourly averages of cpu_util...')
df_hourly = df.groupby([df['time'].dt.hour, 'machine_id'])['cpu_util'].mean().reset_index()
df_hourly.rename(columns={'cpu_util': 'hourly_cpu_util'}, inplace=True)

# Select the mode (most frequent) machine_id for each hour
print('Selecting the mode machine_id for each hour...')
df_mode_machine_id = df.groupby([df['time'].dt.hour])['machine_id'].apply(lambda x: x.mode().iloc[0]).reset_index()
df_hourly = df_hourly.merge(df_mode_machine_id, on='time', how='left')

# Remove the null values
print('Removing the null values...')
df_hourly.dropna(inplace=True)
print('Done removing the null values...')

# Save the dataframe to a csv file
print('Saving the dataframe to a csv file...')
df_hourly.to_csv('cluster_cpu_util_hourly.csv', index=False)
print('Done saving the dataframe to a csv file...')
