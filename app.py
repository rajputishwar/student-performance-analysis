import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("data.csv")

X = data[['hours_studied', 'attendance', 'previous_marks']]
y = data['final_marks']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict([[5, 80, 60]])
print("Predicted Final Marks:", round(prediction[0], 2))

plt.scatter(data['hours_studied'], data['final_marks'])
plt.xlabel("Hours Studied")
plt.ylabel("Final Marks")
plt.title("Study Hours vs Final Marks")
plt.show()
