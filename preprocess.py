XGBoost模型训练
import pandas as pd
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
    print("训练集样本数:", x_train.shape[0])
