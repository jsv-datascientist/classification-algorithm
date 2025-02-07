from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the Iris dataset
X, y = load_iris(return_X_y=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#  Initialize and train the Naive Bayes classifier
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)


# Make and print predictions on the testing set
y_pred = nb_clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Bayes model accuracy: {accuracy * 100:.2f}%")