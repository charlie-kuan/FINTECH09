import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score , average_precision_score 
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve ,auc , log_loss ,  classification_report

def rev_df(df):
	df['fraud_ind_0'] = df['fraud_ind_0'].replace({0:1, 1:0})

def show_cm(cm, tag):
	print(tag)
	print(cm)

	tn, fp, fn, tp = cm.ravel()
	print("about fp:")
	print(fp)
	print(tn + fp)

	print("about fn:")
	print(fn)
	print(tp + fn)

	print("so:")
	print('fpr: {0:0.5f}'.format(fp / (tn + fp)))
	print('fnr: {0:0.5f}'.format(fn / (tp + fn)))

df_train_all = pd.read_csv("./data/train_parsed_V2.csv")
# rev_df(df_train_all)
df_y = df_train_all['fraud_ind_0']
df_X = df_train_all.drop(['fraud_ind_0'], axis=1)

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split

#df_X_train, df_X_val = train_test_split(df_X, test_size=0.25, random_state=42)

#df_y_train = df_y.loc[df_X_train.index]
#df_y_val = df_y.loc[df_X_val.index]

#model = LGBMClassifier(boosting_type='goss')
#model = LGBMClassifier(boosting_type='rf', bagging_freq = 12, bagging_fraction = 0.5)

print('Starting training...')
model = LGBMClassifier(boosting_type='gbdt', objective='binary', metric='binary_logloss', max_depth=10, n_jobs=4)
#model = LGBMClassifier(boosting_type='rf', bagging_freq = 5, bagging_fraction = 0.1, max_depth=10)
model.fit(df_X, df_y)

# y_pred_val = model.predict(df_X_val)

#print('auroc(train) = {0:0.4f}'.format(roc_auc_score(df_y_val, y_pred_val)))
#print('f1_score(train) = {0:0.4f}'.format(f1_score(df_y_val, y_pred_val)))

#show_cm(confusion_matrix(df_y_val, y_pred_val, labels=[0, 1]), "[validation]")


# print()

df_test_all = pd.read_csv("./data/test_parsed_V2.csv")
# rev_df(df_test_all)
y_test = df_test_all['fraud_ind_0']
X_test = df_test_all.drop(['fraud_ind_0'], axis=1)

y_pred_test = model.predict(X_test)

print('auroc(test) = {0:0.5f}'.format(roc_auc_score(y_test, y_pred_test)))
print('f1_score(test) = {0:0.5f}'.format(f1_score(y_test, y_pred_test)))

show_cm(confusion_matrix(y_test, y_pred_test, labels=[0, 1]), "[test]")
