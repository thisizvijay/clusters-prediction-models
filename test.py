import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing



#### DATA INGESTION ####
df = pd.read_csv('cluster_cpu_util.csv')

'''
The dataset is look like this:
time,machine_id,cpu_util
2017-11-27 00:00:00 PST,m29,31.175
2017-11-27 00:01:00 PST,m29,31.97
2017-11-27 00:02:00 PST,m29,31.711666667
2017-11-27 00:03:00 PST,m29,31.8
2017-11-27 00:04:00 PST,m29,31.845
2017-11-27 00:05:00 PST,m29,31.993333333
2017-11-27 00:06:00 PST,m29,31.16
2017-11-27 00:07:00 PST,m29,30.073333333
'''

# convert 'time' column to datetime type
df['time'] = pd.to_datetime(df['time'])

# sort the data by time
df = df.sort_values(by='time')

# normalize the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

df['time'] = scaler_x.fit_transform(df[['time']])
df['cpu_util'] = scaler_y.fit_transform(df[['cpu_util']])


# remove null values for cpu_util or time
df = df.dropna(subset=['cpu_util', 'time'])
df_orig = df.copy()


# plot the time vs cpu_util
plt.figure(figsize=(30,6))
plt.plot(df['time'], df['cpu_util'])
plt.savefig('data.png')

# split the data into training and test sets
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# plot the test data and training data
plt.figure(figsize=(30,6))
plt.plot(train_data['time'], train_data['cpu_util'], label='train')
plt.plot(test_data['time'], test_data['cpu_util'], label='test')
plt.legend(loc='best')
plt.savefig('train_test.png')


# split the data into training and test sets
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size]
test_data = df.iloc[train_size:]

# plot the test data
plt.figure(figsize=(30,6))
plt.plot(test_data['time'], test_data['cpu_util'])
# rotate the x-axis labels
plt.xticks(rotation=45)
plt.savefig('test.png')


# convert the data into PyTorch tensors
x_train = torch.FloatTensor(train_data['time'].values).unsqueeze(-1)
y_train = torch.FloatTensor(train_data['cpu_util'].values).unsqueeze(-1)

x_test = torch.FloatTensor(test_data['time'].values).unsqueeze(-1)
y_test = torch.FloatTensor(test_data['cpu_util'].values).unsqueeze(-1)

# check cuda availability
device = "cpu"

# check mps is enabled
if torch.backends.mps.is_available():
    print("MPS is enabled")
    device = "mps:0"
elif torch.cuda.is_available():
    print("MPS is not enabled")
    device = "cuda"



# make training and test data on the GPU if available
x_train = x_train.to(device)
y_train = y_train.to(device)

x_test = x_test.to(device)
y_test = y_test.to(device)

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Add sequence length dimension
        x = x.unsqueeze(dim=1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_().to(x.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
    
input_dim = 1
hidden_dim = 32
num_layers = 12
output_dim = 1
learning_rate = 0.001

# Instantiate the model
lstm_model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

# Move the model to the GPU if available
lstm_model.to(device)



# Define the loss function and optimizer
criterion = torch.nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)



# Train the model
epochs = 100
for epoch in range(epochs):
    outputs = lstm_model.forward(x_train)
    optimizer.zero_grad()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

y_hat = lstm_model(x_test).detach().numpy()



# Plot the test predictions
plt.figure(figsize=(30,6))
plt.plot(x_test, y_test, 'b', label='Actual')
plt.plot(x_test, y_hat, 'r', label='LSTM')
plt.legend()
plt.xticks(rotation=45)
plt.savefig('prediction.png')
plt.close()


# import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# calculate the mean squared error
lstm_mse = mean_squared_error(y_test, y_hat)

# calculate the mean absolute error
lstm_mae = mean_absolute_error(y_test, y_hat)

# calculate the root mean squared error
lstm_rmse = np.sqrt(lstm_mse)

# calculate the R-squared score
lstm_r2 = r2_score(y_test, y_hat)

print("LSTM Model Metrics")
print("Mean Squared Error:", lstm_mse)
print("Mean Absolute Error:", lstm_mae)
print("Root Mean Squared Error:", lstm_rmse)
print("R-squared:", lstm_r2)

# print the length of total data, training data, and test data
print("Total data length:", len(df))
print("Training data length:", len(train_data))
print("Test data length:", len(test_data))
