import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("Loading dataset...")

data = pd.read_csv("data/iris.csv")

X = data.drop("species", axis=1)
y = data["species"]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training model...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

print(f"Model accuracy: {acc}")

os.makedirs("outputs", exist_ok=True)
joblib.dump(model, "outputs/iris_model.pkl")

print("Training complete. Model saved.")
