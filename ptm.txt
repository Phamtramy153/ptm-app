from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize our model
model = LogisticRegression(max_iter=200)

# Train model
model.fit(X_train, y_train)

# Save model
with open('iris_model.pkl', 'wb') as file:
    pickle.dump(model, file)
