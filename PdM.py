import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn import metrics
from sklearn.svm import SVC, LinearSVC
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,Lasso
from math import sqrt
import math
from sklearn.metrics import mean_squared_error as MSE
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute,IterativeImputer #用KNN填補空缺值

data_train=pd.read_csv('PM_train.txt',sep=' ',header=None)
data_test=pd.read_csv('PM_test.txt',sep=' ',header=None)
PM_truth=pd.read_csv('PM_truth.txt',sep=' ', header=None)
#檢查是否有空缺值
print(data_train.isnull().sum())  
print(data_test.isnull().sum())  
#刪除空缺col
data_train.drop(27,1,inplace=True)
data_train.drop(26,1,inplace=True)

data_test.drop(27,1,inplace=True)
data_test.drop(26,1,inplace=True)
#再次確認有無空缺值
print(data_train.isnull().sum())  
print(data_test.isnull().sum()) 

col_names = ['id','time_in_cycles','setting1','setting2','setting3','s1','s2','s3','s4','s5','s6','s7','s8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21']
data_train.columns=col_names
data_test.columns=col_names
#刪掉標準差為0的feature(因為無意義)
for feature in col_names:
    if data_train.std()[feature]==0:
        data_train.drop(feature,1 ,inplace=True)

col_names=data_train.columns

#取time_in_cycles最大值就是該零件的壽命，因此用壽命剪time_in_cycles就是RUL
def prepare_train_data(data, factor = 0):
    df = data.copy()
    fd_RUL = df.groupby('id')['time_in_cycles'].max().reset_index()
    fd_RUL = pd.DataFrame(fd_RUL)
    fd_RUL.columns = ['id','max']
    df = df.merge(fd_RUL, on=['id'], how='left')
    df['RUL'] = df['max'] - df['time_in_cycles']
    df.drop(columns=['max'],inplace = True)
    
    return df[df['time_in_cycles'] > factor]

train_data = prepare_train_data(data_train)

#normlaiztion
RUL_train=train_data['RUL']
ID_train=train_data['id']
Cycle_train=train_data['time_in_cycles']
train_data.drop('RUL',1,inplace=True)
train_data.drop('id',1,inplace=True)
train_data.drop('time_in_cycles',1,inplace=True)
col_name_norm=train_data.columns
scaler=preprocessing.RobustScaler().fit(train_data)
train_data=pd.DataFrame(scaler.transform(train_data),columns=col_name_norm)
train_data['RUL']=RUL_train
train_data['id']=ID_train
train_data['time_in_cycles']=Cycle_train

ID_test=data_test['id']
data_test.drop('id',1,inplace=True)
Cycle_test=data_test['time_in_cycles']
data_test.drop('time_in_cycles',1,inplace=True)
data_test=data_test[col_name_norm]
data_test=pd.DataFrame(scaler.transform(data_test),columns=col_name_norm)
data_test['id']=ID_test
data_test['time_in_cycles']=Cycle_test

#RUL與其他變數的相關變數
sns.heatmap(train_data.corr(),annot=True,cmap='RdYlGn',linewidths=0.2)
fig=plt.gcf()
fig.set_size_inches(20,20)
plt.show()

correlations =train_data.corr().abs()['RUL'].sort_values(ascending=False)[1:]
ax = sns.barplot(x=correlations.values,y=correlations.index).set_title('Most Correlated with SalePrice')

high_cor_feature=[]
for i in range(len(correlations)):
    if correlations[i]>0.3:
        high_cor_feature.append(correlations.index[i])

print(high_cor_feature)

RUL_train=train_data['RUL']
data_train_high_cor=train_data[high_cor_feature]

#cross validation
x_train, x_test, y_train, y_test = train_test_split(data_train_high_cor, RUL_train, test_size=0.2, random_state=0)

# logistic regression
svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr.fit(x_train,y_train)
prediction_svr=svr.predict(x_test)

#XG Boost
xgb_r = xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=10000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)

xgb_r.fit(x_train,y_train)
prediction_xgb=xgb_r.predict(x_test)

#Random Forest Regression
rf_r=RandomForestRegressor(n_estimators = 1000 , oob_score = True, random_state = 42)

rf_r.fit(x_train,y_train)
prediction_rf=rf_r.predict(x_test)

#GradientBoostingRegressor
GB_R = GradientBoostingRegressor(random_state=0)
GB_R.fit(x_train, y_train)
prediction_GB_R=GB_R.predict(x_test)

#Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(x_train, y_train)
prediction_ridge_R=ridge.predict(x_test)

#Lasso Regression
lasso = Lasso(alpha=1.0)
lasso.fit(x_train, y_train)
prediction_lasso_R=lasso.predict(x_test)

#LightGBM Regressor
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
model_lgb.fit(x_train, y_train)
prediction_lgb_R=model_lgb.predict(x_test)


#RMSE function
def RMSE(target,prediction):
    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])
    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)#target-prediction之差平方 
        absError.append(abs(val))#誤差絕對值
    return sqrt(sum(squaredError) / len(squaredError)) #sum(absError) / len(absError))#平均絕對誤差MAE


rmse_svr=RMSE(y_test.values,prediction_svr)
rmse_xgb=RMSE(y_test.values,prediction_xgb)
rmse_rf=RMSE(y_test.values,prediction_rf)
rmse_GB_R=RMSE(y_test.values,prediction_GB_R)
rmse_ridge_R=RMSE(y_test.values,prediction_ridge_R)
rmse_lasso_R=RMSE(y_test.values,prediction_lasso_R)
rmse_lgb_R=RMSE(y_test.values,prediction_lgb_R)
#np.sqrt(MSE(y_test.values,prediction_svr)) #較快算出RMSE的方法


models = pd.DataFrame({
    'Model': ['Support Vector Regression','XGBoost Regression',
              'Random Forest Regression','Gradient Boosting Regressor',
              'Ridge Regression','Lasso Regression','LightGBM Regressor'],
    'RMSE': [rmse_svr,rmse_xgb,rmse_rf,rmse_GB_R,rmse_ridge_R,rmse_lasso_R,rmse_lgb_R]
    })

models.sort_values(by='RMSE', ascending=True)

# R square: if the value more closer to 1, it means better outcome 
#print('Variance score: %.2f' % r2_score(y_test.values, prediction_lgb_R))
r2_score_svr=r2_score(y_test.values, prediction_svr)
r2_score_xgb=r2_score(y_test.values, prediction_xgb)
r2_score_rf=r2_score(y_test.values, prediction_rf)
r2_score_GB_R=r2_score(y_test.values, prediction_GB_R)
r2_score_ridge_R=r2_score(y_test.values, prediction_ridge_R)
r2_score_lasso_R=r2_score(y_test.values, prediction_lasso_R)
r2_score_lgb_R=r2_score(y_test.values, prediction_lgb_R)

model_r2 = pd.DataFrame({
    'Model': ['Support Vector Regression','XGBoost Regression',
              'Random Forest Regression','Gradient Boosting Regressor',
              'Ridge Regression','Lasso Regression','LightGBM Regressor'],
    'R^2': [r2_score_svr,r2_score_xgb,r2_score_rf,r2_score_GB_R,r2_score_ridge_R,r2_score_lasso_R,r2_score_lgb_R]
    })

model_r2.sort_values(by='R^2', ascending=False)   
# data visulization
plt.figure(figsize=(30,10),dpi=100,linewidth = 2)
plt.plot(y_test.values[0:200],color = 'r', label="Real RUL")
plt.plot(prediction_lgb_R[0:200],color = 'g', label="LightGBM")
plt.plot(prediction_svr[0:200],color = 'b', label="Support Vector Regression")
plt.legend(loc = "best", fontsize=20)
plt.title("RUL", x=0.5, y=1.03,fontsize=20)  
plt.show()

###########################################################
#處理test data
rul_max = pd.DataFrame(data_test.groupby('id')['time_in_cycles'].max()).reset_index()
rul_max.columns = ['id', 'max']

PM_truth.columns=['more','id']
PM_truth['id']=PM_truth.index+1
PM_truth.head()

PM_truth['total']=PM_truth['more']+rul_max['max']

data_test=data_test.merge(PM_truth,on=['id'],how='left')
data_test['total']=data_test['total'] - data_test['time_in_cycles']

RUL_test=data_test['total']
data_test_high_cor=data_test[high_cor_feature]
####################################################################
#use original all training data to build model
#LightGBM Regressor
model_lgb.fit(data_train_high_cor, RUL_train)
prediction_lgb_R=model_lgb.predict(data_test_high_cor)

rmse_lgb_R=RMSE(RUL_test,prediction_lgb_R)
print(rmse_lgb_R)
r2_score_lgb_R=r2_score(RUL_test, prediction_lgb_R)
print('RMSE: ',rmse_lgb_R,'; R^2: ',r2_score_lgb_R)

# data visulization
plt.figure(figsize=(30,10),dpi=100,linewidth = 2)
plt.plot(RUL_test,color = 'r', label="Real RUL")
plt.plot(prediction_lgb_R,color = 'g', label="LightGBM")
plt.legend(loc = "best", fontsize=20)
plt.title("RUL", x=0.5, y=1.03,fontsize=20)  
plt.show()
