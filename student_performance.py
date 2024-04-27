import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv("E:\data\datasets\student+performance\student\student-mat.csv", sep=';')

data = data.drop(columns=['reason','guardian', 'schoolsup',
                          'activities', 'nursery', 'higher',
                          'health', 'school', 'address', 
                          'famsize', 'Pstatus', 'Medu',
                          'Fedu', 'Mjob', 'Fjob'])
student = data.drop(columns=['G3'])

student['sex'] = student['sex'].map({'M': 0, 'F': 1})
student['romantic'] = student['romantic'].map({'no': 0, 'yes': 1})
student['internet'] = student['internet'].map({'no': 0, 'yes': 1})
student['paid'] = student['paid'].map({'no': 0, 'yes': 1})
student['famsup'] = student['famsup'].map({'no': 0, 'yes': 1})

X = np.array(student)
y = np.array(data['G3'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.03)

model = LinearRegression()
model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print(score)

joblib.dump(model, 'student_performance.joblib')