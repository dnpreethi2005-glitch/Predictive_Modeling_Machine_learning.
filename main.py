import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Dataset
data = {
    'Area': [1000,1200,1500,1800,2000,2200,2500,2800,3000,3500],
    'Price': [30,35,45,50,60,65,70,80,85,95]
}

df = pd.DataFrame(data)

# Features and Target
X = df[['Area']]
y = df['Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Accuracy metrics
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Predict new value
new_house = [[2500]]
prediction = model.predict(new_house)
print("Predicted Price:", prediction[0], "Lakhs")

# Graph
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction")
plt.show()