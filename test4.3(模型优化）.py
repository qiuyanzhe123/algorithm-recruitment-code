import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
data=pd.read_csv("airline-passengers.csv")
import numpy as np
from sklearn.preprocessing import MinMaxScaler
values=data["Passengers"].values.reshape(-1,1)
split=int(len(values)*0.8)
scaler=MinMaxScaler()
scaler.fit(values[:split])
scaled=scaler.transform(values)
def create_sequences(values,window=12):
    X=[]
    y=[]
    for i in range(len(values)-window):
        X.append(values[i:i+window])
        y.append(values[i+window])
    return np.array(X),np.array(y)
X,y=create_sequences(scaled)
train_count=split-12
X_train=X[:train_count]
y_train=y[:train_count]
X_test=X[train_count:]
y_test=y[train_count:]
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm=nn.LSTM(input_size=1,hidden_size=32,batch_first=True)
        self.attention=nn.Linear(32,1)
        self.fc=nn.Linear(32,1)

    def forward(self,x):
        output,(h,c)=self.lstm(x)
        scores=self.attention(output)
        weights=torch.softmax(scores,dim=1)
        context=torch.sum(output*weights,dim=1)
        prediction=self.fc(context)
        return prediction
model=LSTMModel()
X_train=torch.tensor(X_train,dtype=torch.float32)
y_train=torch.tensor(y_train,dtype=torch.float32)
X_test=torch.tensor(X_test,dtype=torch.float32)
y_test=torch.tensor(y_test,dtype=torch.float32)
loss_fn=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
loss_history=[]
for epoch in range(300):
    model.train()
    outputs=model(X_train)
    loss=loss_fn(outputs,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_history.append(loss.item())
plt.figure()
plt.plot(range(1,301),loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Training Loss improved")
plt.savefig("lstm_loss优化版.jpg",dpi=300,bbox_inches="tight")
plt.show()
model.eval()
with torch.no_grad():
    predictions=model(X_test)
predictions=predictions.numpy()
actual=y_test.numpy()
predictions=scaler.inverse_transform(predictions)
actual=scaler.inverse_transform(actual)
plt.figure(figsize=(10,5))
plt.plot(actual.flatten(),label="Actual",marker="o")
plt.plot(predictions.flatten(),label="Predicted",marker="o")
plt.xlabel("Test Month")
plt.ylabel("Passengers (thousands)")
plt.title("LSTM Prediction vs Actual improved")
plt.legend()
plt.grid(True)
plt.savefig("lstm_prediction优化版.jpg",dpi=300,bbox_inches="tight")
plt.show()
mae=mean_absolute_error(actual,predictions)
mse=mean_squared_error(actual,predictions)
rmse=np.sqrt(mse)
r2=r2_score(actual,predictions)
print(f"MAE={mae:.4f}")
print(f"MSE={mse:.4f}")
print(f"RMSE={rmse:.4f}")
print(f"R²={r2:.4f}")
torch.save(model.state_dict(),"lstm_model优化版.pth")
