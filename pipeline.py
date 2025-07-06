import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Create folders if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# 1️⃣ Load data
df = pd.read_csv("data/cleaned_pipeline_file.csv")
df.columns = df.columns.str.strip()  # clean column names

print("Columns:", df.columns.tolist())
print(df.head())

# 2️⃣ Features & target
X = df.drop('target', axis=1)
y = df['target']

# 3️⃣ Identify types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

# 4️⃣ Preprocessing
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 5️⃣ Pipeline
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# 6️⃣ Split & train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model_pipeline.fit(X_train, y_train)

# 7️⃣ Predict & evaluate
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")

# 8️⃣ Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.show()
print("✅ Confusion matrix saved to outputs/confusion_matrix.png")

# 9️⃣ Save model
joblib.dump(model_pipeline, "models/trained_model_pipeline.pkl")
print("✅ Trained model saved to models/trained_model_pipeline.pkl")