from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# Generate sample data
X = np.random.rand(100, 1) * 1000  # House sizes
y = 50 * X + 100000 + np.random.randn(100, 1) * 10000  # House prices

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print the model's performance
print(f"Model score: {model.score(X_test, y_test):.2f}")
