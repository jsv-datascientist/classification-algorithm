from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Load dataset and split to train and test
X, y = load_wine(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Hypertune KNN
best_knn_acc = 0
best_k = 0
for k in range(2, 11):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf.fit(X_train, y_train)
    
    accuracy = knn_clf.score(X_test, y_test)
    #  calculate the accuracy with the given k
    if accuracy > best_knn_acc:
        best_knn_acc = accuracy
        best_k = k
print(f'Best K for KNN: {best_k} -> Accuracy: {best_knn_acc * 100:.2f}%')

# Hypertune Decision Tree
best_dt_acc = 0
best_depth = 0
for depth in range(1, 11):
    #  calculate the accuracy with the given max_depth
    dt_clf = DecisionTreeClassifier(max_depth=depth)
    dt_clf.fit(X_train, y_train)
    
    accuracy = dt_clf.score(X_test, y_test)
    if accuracy > best_dt_acc:
        best_dt_acc = accuracy
        best_depth = depth
print(f'Best Depth for Decision Tree: {best_depth} -> Accuracy: {best_dt_acc * 100:.2f}%')