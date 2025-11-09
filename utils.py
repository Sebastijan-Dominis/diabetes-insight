import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

def load_and_prepare_data(df):
    X = df.drop(columns=["diagnosed_diabetes"])
    y = df["diagnosed_diabetes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42, 
        shuffle=True
    )
    
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std, y_train, y_test

def evaluate_metrics(actual, predicted, title=None):
    if title:
        print(title)

    matrix = confusion_matrix(actual, predicted)
    ax = sns.heatmap(
        matrix,
        annot=True,
        cmap="Blues",
        fmt="g"
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_yticklabels(labels=["No Diabetes", "Diabetes"])
    ax.set_xticklabels(labels=["No Diabetes", "Diabetes"])
    ax.set_title("Diabetes Diagnosis - Confusion Matrix")
    plt.show();

    print(f"Accuracy: {accuracy_score(actual, predicted)}")
    print(f"Precision: {precision_score(actual, predicted)}")
    print(f"Recall: {recall_score(actual, predicted)}")
    print(f"F1 Score: {f1_score(actual, predicted)}")
    
def plot_roc_curve(y_true, y_probs, model_name='Model'):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)

    auc_score = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess (AUC = 0.50)')

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Models')
    plt.legend()
    plt.show()