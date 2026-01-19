import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix #model kitna sahi hai 

#DATASET
data = {
    'Fever': [1,1,0,0,0,1],
    'BP_High': [1,0,0,1,1,0],
    'Age': [45,50,30,35,60,25],
    'Result': [1,1,0,0,1,0]
}
df = pd.DataFrame(data)
print(df)

X = df[['Fever', 'BP_High', 'Age']] #features
y = df['Result']   #label

print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.3,random_state=42 #random_state mein data split hoga par ek constant tareeka se split hoga
)
print(X_train)
print(y_train)

model = DecisionTreeClassifier(
    criterion='gini',
    max_depth=3
)

model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("Predictions:",y_pred)
print("Actual:",y_test.values)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy",accuracy)

cm = confusion_matrix(y_test,y_pred)
print(cm)

new_patient = np.array([[1, 0, 52]])  # Fever, BP, Age
prediction = model.predict(new_patient)
print("New Patient Prediction:", prediction)

if prediction[0] == 1:
    print("Patient is Sick")
else:
    print("Patient is Healthy")



