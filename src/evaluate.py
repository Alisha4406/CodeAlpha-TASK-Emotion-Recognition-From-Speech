# =========================================================
# EVALUATION
# =========================================================

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


def evaluate_model(model, X_test, y_test):
    
    predictions = model.predict(X_test)
    
    pred_classes = np.argmax(predictions, axis=1)
    
    
    accuracy = accuracy_score(y_test, pred_classes)
    
    print("Accuracy:", accuracy)
    
    
    print("\nClassification Report:")
    
    print(classification_report(y_test, pred_classes))
    
    
    cm = confusion_matrix(y_test, pred_classes)
    
    
    plt.figure(figsize=(10,6))
    
    sns.heatmap(
        cm,
        annot=True,
        cmap="Blues"
    )
    
    plt.title("Confusion Matrix")
    
    plt.show()