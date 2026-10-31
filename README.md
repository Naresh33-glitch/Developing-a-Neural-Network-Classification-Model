# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Import the required libraries such astorch, torch.nn, torch.optim, pandas, numpy, scikit-learn, matplotlib, and seaborn.


### STEP 2: 

Load the dataset from the CSV file (customers.csv), inspect its shape and columns, drop the ID column, and handle missing values.

### STEP 3: 

Preprocess the data and encode categorical columns using LabelEncoder, encode the target segment (A, B, C, D → 0, 1, 2, 3), scale numerical features using StandardScaler, and split into train/test sets (80/20).

### STEP 4: 

Convert the processed data into PyTorch tensors and wrap them in TensorDataset and DataLoader for mini-batch training.



### STEP 5: 

Define the PeopleClassifier neural network (input → 64 → 32 → 4), initialize CrossEntropyLoss and Adam optimizer, and train the model using train_model() for multiple epochs.


### STEP 6: 


Evaluate the trained model using a Confusion Matrix and Classification Report, then predict the customer segment for new sample data.


## PROGRAM

### Name: NARESH S

### Register Number:212224240101

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from torch.utils.data import TensorDataset,DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("/content/customers.csv")
df
df=df.drop(columns=["ID"])
df
df.columns
df.fillna({"Work_Experience":0,"Family_Size":df["Family_Size"].median()},inplace=True)
df
cat_columns=['Gender','Ever_Married', 'Graduated', 'Profession',
    'Spending_Score', 'Var_1']
for col in cat_columns:
    df[col]=LabelEncoder().fit_transform(df[col])
lbe=LabelEncoder()
df["Segmentation"]=lbe.fit_transform(df["Segmentation"])
df
x=df.drop(columns="Segmentation")
y=df["Segmentation"].values
xt,xst,yt,yst=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
xt=scaler.fit_transform(xt)
xst=scaler.transform(xst)
xt=torch.FloatTensor(xt)
xst=torch.FloatTensor(xst)
yt=torch.FloatTensor(yt)
yst=torch.FloatTensor(yst)
tr=TensorDataset(xt,yt)
tst=TensorDataset(xst,yst)
trl=DataLoader(tr,batch_size=16,shuffle=True)
tstl=DataLoader(tst,batch_size=16)
class classifier1(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.l1=nn.Linear(input_size,32)
        self.l2=nn.Linear(32,16)
        self.l3=nn.Linear(16,8)
        self.l4=nn.Linear(8,4)
    def forward(self,x):
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        x=F.relu(self.l3(x))
        x=self.l4(x)
        return x
model=classifier1(input_size=xt.shape[1])
criterion=nn.CrossEntropyLoss()
op=optim.Adam(model.parameters(),lr=0.001)
epochs=100
for i in range(epochs):
    for a,b in trl:
        op.zero_grad()
        pred=model(a)
        loss=criterion(pred,b.long())
        loss.backward()
        op.step()
    if i%10==0:
        print(f"Loss:{i}/{epochs}",loss.item())
pre=[]
act=[]
with torch.no_grad():
    output=model(xst)
    _,predicted=torch.max(output,1)
    pre.extend(predicted.numpy())
    act.extend(yst.numpy())
    
accuracy=accuracy_score(act,pre)
conf_matrix=confusion_matrix(act,pre)
cl_report=classification_report(act,pre,target_names=['A','B','C','D'])
print("Accuracy:",accuracy)
print("confusion_matrix:\n",conf_matrix)
print("classification_report:\n",cl_report)
import seaborn as sns
import matplotlib.pyplot as plt
xl=['A','B','C','D']
sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=xl, yticklabels=xl,fmt='g')
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

```



### OUTPUT
<img width="1453" height="526" alt="image" src="https://github.com/user-attachments/assets/58fce83b-36b8-491a-9dff-1ca3d41fb907" />
<img width="1596" height="531" alt="image" src="https://github.com/user-attachments/assets/f0489c84-2910-40a2-b9d6-fc1de1298557" />
<img width="1507" height="97" alt="image" src="https://github.com/user-attachments/assets/3da14812-955b-4f07-bff9-3a430c1751e2" />
<img width="1425" height="527" alt="image" src="https://github.com/user-attachments/assets/69ff957e-ff38-4e3f-8533-0f91d7920f2f" />
<img width="1402" height="522" alt="image" src="https://github.com/user-attachments/assets/724db250-8e16-489e-ba38-a4be50dd375a" />
<img width="1352" height="226" alt="image" src="https://github.com/user-attachments/assets/4f41a63b-aa8b-4cae-97d1-80a7572a4f8a" />
<img width="1473" height="415" alt="image" src="https://github.com/user-attachments/assets/ee7222b3-ec98-43b9-acae-feffec84595e" />
<img width="1506" height="592" alt="image" src="https://github.com/user-attachments/assets/65a5d0da-c55e-4c35-8d80-8e0bfa98690c" />



## RESULT
Hence, the Neural Network Classification Model was successfully developed and customer segments were predicted with good accuracy.
