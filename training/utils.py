import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_prepare_data(df, target="diagnosed_diabetes"):
    X = df.drop(columns=[target])
    y = df[target]

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

def load_and_prepare_data_2(df, target="diagnosed_diabetes"):
    chosen_columns = ['family_history_diabetes', 'hypertension_history',
       'cardiovascular_history', 'waist_to_hip_ratio', 'age_30-39', 'age_40-49', 'age_50-59', 'age_60-69', 'age_70-79',
       'age_80+', 'gender_Male', 'gender_Other', 'ethnicity_Black',
       'ethnicity_Hispanic', 'ethnicity_Other', 'ethnicity_White',
       'education_level_Highschool', 'education_level_No formal',
       'education_level_Postgraduate', 'income_level_Low',
       'income_level_Lower-Middle', 'income_level_Middle',
       'income_level_Upper-Middle', 'employment_status_Retired',
       'employment_status_Student', 'employment_status_Unemployed',
       'smoking_status_Former', 'smoking_status_Never',
       'alcohol_consumption_per_week_Light',
       'alcohol_consumption_per_week_Moderate',
       'alcohol_consumption_per_week_Heavy',
       'physical_activity_minutes_per_week_Light',
       'physical_activity_minutes_per_week_Moderate',
       'physical_activity_minutes_per_week_Active',
       'physical_activity_minutes_per_week_Very Active','sleep_hours_per_day_Short',
       'sleep_hours_per_day_Normal', 'sleep_hours_per_day_Long',
       'screen_time_hours_per_day_Moderate', 'screen_time_hours_per_day_High',
       'screen_time_hours_per_day_Very_High', 'bmi_Normal', 'bmi_Overweight',
       'bmi_Obese_I', 'bmi_Obese_II', target]

    X = df[chosen_columns].drop(columns=[target])
    y = df[target]

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

def optimize_threshold(y_true, y_probs):
    p_curve, r_curve, t_curve = precision_recall_curve(y_true, y_probs)

    plt.plot(t_curve, p_curve[:-1], label='Precision')
    plt.plot(t_curve, r_curve[:-1], label='Recall')
    plt.xlabel('Prediction Threshold')
    plt.ylabel('Scores')
    plt.legend()
    plt.title('Precision & Recall Curves')
    plt.show()


    plt.plot(p_curve[:-1],r_curve[:-1], label='Precision-Recall Curve')
    plt.xlabel('Precision')
    plt.ylabel('Recall')
    plt.legend()
    plt.title('Precision-Recall Curve')
    plt.show()


    f1 = []
    thresholds = np.linspace(0, 1, 101)

    for thresh in thresholds:
        y_pred =(y_probs > thresh)
        f1.append(f1_score(y_true, y_pred))
        
    sns.lineplot(x=thresholds, y=f1)
    plt.xlabel('Prediction Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs. Prediction Threshold')
    plt.show()

    thresh = thresholds[f1.index(max(f1))]

    return thresh