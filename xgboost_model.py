集成学习（Stacking）
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
