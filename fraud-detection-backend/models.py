from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import svm
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

def train_and_evaluate_models(df):
    features = df.drop('isFraud', axis=1)
    target = df.isFraud
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)
    
    classifiers = [LogisticRegression, KNeighborsClassifier, RandomForestClassifier, XGBClassifier, svm.SVC]

    def ml_func(algorithm):
        model = algorithm()
        model.fit(X_train, y_train)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)
        train_accuracy = roc_auc_score(y_train, train_preds)
        test_accuracy = roc_auc_score(y_test, test_preds)
        print(f"{algorithm.__name__}: Training Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

    for classifier in classifiers:
        ml_func(classifier)
    
    # Example grid search for RandomForest
    rf_model = RandomForestClassifier()
    param_grid_rf = {'n_estimators': [10, 80, 100], 'criterion': ['gini', 'entropy'], 'max_depth': [10], 'min_samples_split': [2, 3, 4]}
    grid_search = GridSearchCV(rf_model, param_grid_rf, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"RandomForest Best Parameters: {grid_search.best_params_}")

    # SMOTE for handling imbalance
    X_resampled, y_resampled = SMOTE().fit_resample(X_train, y_train)
    rf_model_best = RandomForestClassifier(**grid_search.best_params_)
    rf_model_best.fit(X_resampled, y_resampled)
    y_pred = rf_model_best.predict(X_test)
    print("RandomForest with SMOTE")
    print(classification_report(y_test, y_pred))
    
    # Add XGBoost and other models similarly
