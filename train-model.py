import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


X = np.load("features.npy")
y = np.load("labels.npy")

print(" Shape X:", X.shape)
print(" Labels:", np.unique(y))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(kernel='linear')
}


best_accuracy = 0
best_model = None
best_name = ""


for name, model in models.items():

    print("\n==============================")
    print(f" Modèle : {name}")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print(" Accuracy :", acc)
    print("\n Classification Report :")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    print("\n Matrice de confusion :")
    print(cm)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

joblib.dump(best_model, "best_model.pkl")

print("\n==============================")
print(" Meilleur modèle :", best_name)
print(" Accuracy :", best_accuracy)
print(" Modèle sauvegardé dans best_model.pkl")