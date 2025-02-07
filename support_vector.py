from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_wine

#  Load the wine dataset
X, y = load_wine(return_X_y = True )

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

#  Create and train an SVM classifier with a linear kernel
svm_clf = SVC(kernel='linear')
svm_clf.fit(X_train, y_train)
#  Print the accuracy of the trained model
print(f"Accuracy {svm_clf.score(X_test, y_test):.2f}")