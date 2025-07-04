from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from preprocess import preprocess_data
from sklearn.metrics import classification_report

def train_ensemble(x_train, y_train):
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=50)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ]
    clf = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression())
    clf.fit(x_train, y_train)
    return clf

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = preprocess_data('creditcard.csv')
    model = train_ensemble(x_train, y_train)
    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred, digits=4))