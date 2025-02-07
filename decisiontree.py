from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#  Load the wine dataset and split it into training and testing sets
X, y = load_wine(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state = 42)
#  Train a Logistic Regression model on the training data
model_log = LogisticRegression()
#  Make predictions with the Logistic Regression model and calculate its accuracy
model_log.fit(X_train, y_train)
accurcy_score_1 = accuracy_score(y_test, model_log.predict(X_test))


#  Train a Decision Tree model on the training data
model_tree = DecisionTreeClassifier(max_depth=5, min_samples_split=4)


#  Make predictions with the Decision Tree model and calculate its accuracy
model_tree.fit(X_train, y_train)
accuracy_score_2 = accuracy_score(y_test, model_tree.predict(X_test))

#  Print the accuracies of both models
print(f"Logistic Regression accuracy { accurcy_score_1 }")
print(f"Logistic Regression accuracy { accuracy_score_2 }")