import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
df = pd.read_csv("badminton_dataset.csv")

# Split features and target
X = df.drop("Play_Badminton", axis=1)
y = df["Play_Badminton"]

# One-hot encode categorical features
categorical_features = X.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_features)
    ], remainder='passthrough')

X_encoded = preprocessor.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=9)

# Train the decision tree classifier
treemodel = DecisionTreeClassifier()
treemodel.fit(X_train, y_train)

# Make predictions
y_predict = treemodel.predict(X_test)

# Evaluate the model
score = accuracy_score(y_predict, y_test)
print("Accuracy:", score)
print(classification_report(y_test, y_predict))

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_predict)
print("Confusion Matrix:")
print(conf_matrix)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks([0, 1], labels=treemodel.classes_)
plt.yticks([0, 1], labels=treemodel.classes_)
plt.show()

# Visualize the decision tree
plt.figure(figsize=(15, 10))
plot_tree(treemodel, filled=True, feature_names=preprocessor.get_feature_names_out(), class_names=treemodel.classes_)
plt.show()