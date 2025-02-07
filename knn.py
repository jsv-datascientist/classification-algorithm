from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset and split to train and test
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Iterate over different k values
for k in range(2, 10):
    #  initialize the KNeighborsClassifier with n_neighbors=k
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, y_train)
    #  make predictions and calculate the accuracy for the test data. Save the result in the "accuracy" variable
    accuracy = knn_clf.score(X_test, y_test)
    print(f'k={k} -> Accuracy: {accuracy * 100:.2f}%')