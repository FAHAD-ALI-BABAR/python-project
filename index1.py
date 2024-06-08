import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree


# Load the dataset
df = pd.read_csv("seattle-weather.csv")

# Drop the date column
df.drop("date", axis=1, inplace=True)

# Split features and target
X = df.drop("weather", axis=1)
y = df["weather"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=9)

# Train the decision tree classifier
treemodel = DecisionTreeClassifier()
treemodel.fit(X_train, y_train)

# Make predictions
y_predict = treemodel.predict(X_test)

# Evaluate the model
score = accuracy_score(y_predict, y_test)
print("Accuracy:", score)
print(classification_report(y_test, y_predict))

# Visualize the decision tree
plt.figure(figsize=(20, 15))
plot_tree(treemodel, filled=True, feature_names=X.columns, class_names=treemodel.classes_)
plt.show()