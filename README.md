# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Explain the problem statement

## Neural Network Model

![image](https://github.com/gpavithra673/basic-nn-model/assets/93427264/b6c9584b-f659-4b6e-b49f-31c990528bd3)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

worksheet = gc.open('DLExp1').sheet1
data = worksheet.get_all_values()
df = pd.DataFrame(data[1:], columns=data[0])
df = df.astype({'Input':'float'})
df = df.astype({'Output':'float'})
df.head()

X = df[['Input']].values
y = df[['Output']].values
X

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)
X_train1

ai=Sequential([
    Dense(7,activation='relu'),
    Dense(6,activation='relu'),
    Dense(1)
])
ai.compile(optimizer='rmsprop',loss='mse')
ai.fit(X_train1,y_train,epochs=2000)
ai.fit(X_train1,y_train,epochs=2000)

## Plot the loss
loss_df = pd.DataFrame(ai.history.history)
loss_df.plot()

## Evaluate the model
X_test1 = Scaler.transform(X_test)
ai.evaluate(X_test1,y_test)

# Prediction
X_n1 = [[30]]
X_n1_1 = Scaler.transform(X_n1)
ai.predict(X_n1_1)
```
## Dataset Information

![image](https://github.com/gpavithra673/basic-nn-model/assets/93427264/033d7773-0ce0-43bc-81e4-81c93f1536f5)

## OUTPUT

### Training Loss Vs Iteration Plot
![image](https://github.com/gpavithra673/basic-nn-model/assets/93427264/ebcdc981-b3a9-473e-96ea-f5560c6e2e39)

### Test Data Root Mean Squared Error
![image](https://github.com/gpavithra673/basic-nn-model/assets/93427264/6378a156-72ac-4879-a671-1af42e757228)

### New Sample Data Prediction
![image](https://github.com/gpavithra673/basic-nn-model/assets/93427264/0786baab-9f17-4bf5-9d58-dfcb2d3f50a9)

## RESULT:
Thus a neural network regression model for the given dataset is written and executed successfully.
