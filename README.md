# -XGBoost-
基于 XGBoost 与集成学习的信用卡欺诈检测研究
import requests

def download_dataset(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {filename}")

if __name__ == '__main__':
    # 以Kaggle公开数据为例
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv'
    filename = 'creditcard.csv'
    doimport pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop(['Class', 'Time'], axis=1)  # 特征
    y = data['Class']  # 标签
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = preprocess_data('creditcard.csv')
    print("训练集样本数:", x_train.shape[0])wnload_dataset(url, filename)
    import xgboost as xgb
from preprocess import preprocess_data

def train_xgboost(x_train, y_train):
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(x_train, y_train)
    return model

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = preprocess_data('creditcard.csv')
    model = train_xgboost(x_train, y_train)
    print("XGBoost模型训练完成")
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
    
