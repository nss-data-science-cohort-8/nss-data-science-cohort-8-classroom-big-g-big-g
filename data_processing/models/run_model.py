# Import statements
import pandas as pd
import numpy as np
import mlflow
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# File name for the dataset
file_name = 'data_processing/preprocessed_data/mlflow_test_05_01'
mlflow.set_experiment("big-g-haulin-oats")
mlflow.autolog()
with mlflow.start_run():
    train_data_file_path = f"{file_name}_train.csv"
    test_data_file_path = f"{file_name}_test.csv"
    mlflow.log_param("train_data_file_path", train_data_file_path)
    mlflow.log_param("test_data_file_path", test_data_file_path)
    mlflow.log_artifact(train_data_file_path, artifact_path="data")
    mlflow.log_artifact(test_data_file_path, artifact_path="data")
    train_df = pd.read_csv(train_data_file_path)
    test_df = pd.read_csv(test_data_file_path)
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    #train_df = train_df.ffill().bfill()
    #test_df = test_df.ffill().bfill()
    # Try imputing (optionally by ffill)
    y_train = train_df.iloc[:, 0]
    X_train = train_df.iloc[:, 1:]
    y_test = test_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    #simple_model = SVC(class_weight='balanced', probability=True, random_state=42)
    #simple_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    simple_model = DecisionTreeClassifier(random_state=42)
                # pipeline = ImbPipeline([
                #     ('xgb', XGBClassifier(random_state=42, scale_pos_weight=10))
                # ])

                # random_search = RandomizedSearchCV(
                #     estimator=pipeline,
                #     param_distributions={
                #         'xgb__n_estimators': [100, 200, 300],
                #         'xgb__learning_rate': [0.01, 0.05, 0.1],
                #         'xgb__max_depth': [3, 5, 7],
                #         'xgb__subsample': [0.8, 1.0],
                #         'xgb__colsample_bytree': [0.8, 1.0],
                #         'xgb__gamma': [0, 0.2],
                #     },
                #     n_iter=10,
                #     scoring='precision',
                #     cv=3,
                #     verbose=1,
                #     random_state=42,
                #     n_jobs=-1
                # )
                # random_search.fit(X_train, y_train)
                # best_params = random_search.best_params_
                # best_model = random_search.best_estimator_
                # print(f"Best hyperparameters: {best_params}")
                # y_pred = best_model.predict(X_test) 

    #simple_model = RidgeClassifier(alpha=1.0, random_state=42) 
    #simple_model = RandomForestClassifier(random_state=42)

    simple_model.fit(X_train, y_train)
    y_pred = simple_model.predict(X_test)


     # Calculate additional metrics on the test set
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    # Log custom metrics for the test set
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)
    mlflow.log_metric("test_accuracy", accuracy)
    print(f"Test precision: {precision}")
    print(f"Test recall: {recall}")
    print(f"Test F1 score: {f1}")
    print(f"Test accuracy: {accuracy}")
    print("Model training and evaluation completed successfully.")


#path for output 
output_dir = Path("data_processing/model_outputs")
output_dir.mkdir(parents=True, exist_ok=True)
predictions_path = output_dir / f"predictions_{Path(file_name).name}.csv"
#Preictions to save for Monetary Evaluation
predictions_df = pd.DataFrame({
    'actual': y_test,
    'predicted': y_pred
})

# Save to CSV
predictions_df.to_csv(predictions_path, index=False)


# Log the predictions file to MLflow for traceability
mlflow.log_artifact(str(predictions_path), artifact_path="outputs")