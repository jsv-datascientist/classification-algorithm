from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Load the wine dataset
X, y = load_wine(return_X_y=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Hyperparameter ranges
svm_kernels = ['linear', 'rbf']
dt_max_depths = [3, 5, 10, None]
knn_neighbors = [3, 5, 7, 9]

# Decision Tree hyperparameter tuning
best_dt_accuracy = 0
best_dt_model = None
for max_depth in dt_max_depths:
    dt_clf = DecisionTreeClassifier(max_depth=max_depth)
    dt_clf.fit(X_train, y_train)
    accuracy = dt_clf.score(X_test, y_test)
    if accuracy > best_dt_accuracy:
        best_dt_accuracy = accuracy
        best_dt_model = dt_clf

print(f"Best Decision Tree Accuracy: {best_dt_accuracy}")

# KNN hyperparameter tuning
best_knn_accuracy = 0
best_knn_model = None
for n_neighbors in knn_neighbors:
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf.fit(X_train, y_train)
    accuracy = knn_clf.score(X_test, y_test)
    if accuracy > best_knn_accuracy:
        best_knn_accuracy = accuracy
        best_knn_model = knn_clf

print(f"Best KNN Accuracy: {best_knn_accuracy}")


#  Perform the SVM hypertuning in the same manner.
# Hint: use "for kernel in svm_kernels:" loop for iteration
#  find and print the best accuracy for the SVM3670

best_svm_accuracy = 0
best_svm_model = None
for kernels in svm_kernels:
    svc_clf = SVC(kernel= kernels)
    svc_clf.fit(X_train, y_train)
    accuracy = svc_clf.score(X_test, y_test)
    if accuracy > best_svm_accuracy:
        best_svm_accuracy = accuracy
        best_svm_model = svc_clf

print(f"Best SVC Accuracy: {best_svm_accuracy}")