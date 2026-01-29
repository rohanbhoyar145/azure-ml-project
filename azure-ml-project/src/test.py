import pandas as pd
import joblib

# Load dataset
data = pd.read_csv("../data/iris.csv")

# Load trained model
model = joblib.load("model.pkl")

# Take few samples for testing
X_test = data.iloc[:5, :-1]

# Predict
predictions = model.predict(X_test)

print("Testing completed")
print("Predictions:", predictions)
