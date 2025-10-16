import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Sample customer data (replace with real dataset or API data)
data = {
    'age': np.random.randint(18, 80, 1000),
    'tenure': np.random.randint(1, 60, 1000),  # Months with service
    'monthly_charges': np.random.uniform(20, 120, 1000),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 1000),
    'support_calls': np.random.randint(0, 10, 1000),
    'churn': np.random.choice([0, 1], 1000, p=[0.8, 0.2])  # 0: Stay, 1: Churn
}
df = pd.DataFrame(data)

# Preprocess data: Encode categorical variables
df['contract_type'] = df['contract_type'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})

# Prepare features (X) and target (y)
X = df[['age', 'tenure', 'monthly_charges', 'contract_type', 'support_calls']]
y = df['churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of churn

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualize churn probability distribution
plt.figure(figsize=(10, 6))
sns.histplot(y_pred_proba, bins=20, kde=True)
plt.title('Distribution of Churn Probability')
plt.xlabel('Churn Probability')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('churn_probability.png')
plt.show()

# Feature importance visualization
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 6))
sns.barplot(x=feature_importance.values, y=feature_importance.index)
plt.title('Feature Importance in Churn Prediction')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Bonus: Example code to load data from a CSV or mock API (uncomment to use)
"""
import requests
def fetch_customer_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        df_api = pd.DataFrame(data)
        return df_api
    return None

# Example usage
# api_url = 'https://example.com/customer-data-api'
# customer_data = fetch_customer_data(api_url)
# if customer_data is not None:
#     customer_data['contract_type'] = customer_data['contract_type'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
#     X_new = customer_data[['age', 'tenure', 'monthly_charges', 'contract_type', 'support_calls']]
#     predicted_churn = model.predict(X_new)
#     print(f'Predicted churn for new customers: {predicted_churn}')
"""
