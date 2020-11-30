# -*- coding: utf-8 -*-
import pandas  as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV as GSCV
import time 

#Custom module
import model.load_data_module as LM
import model.StockPriceForecast_Model_XG as XG_model
import plot_figure 
import util.util as tool
import appraisal.apprisal_result as ar

start = time.time()
config_obj = LM.load_config()
prev_factor = config_obj["prev_factor"]
data_df = LM.load_stock_data(config_obj["data_path"])


df = XG_model.stockFeature_process(data_df)

colList = ["turn","range_hl","range_oc","volume","close"]
for col in colList:
    df = XG_model.get_mean_Output(df, col, prev_factor)


num_test = int(config_obj["test_size"] * len(df))
num_train = len(df) - num_test
train = df[:num_train]
test = df[-20:]
cols_to_scale = [
    "close"
]

for i in range(1, prev_factor + 1):
    tool.append_lagStr(cols_to_scale,i)

scaler = StandardScaler()
train_scaled = scaler.fit_transform(train[cols_to_scale])
train_scaled = pd.DataFrame(train_scaled, columns=cols_to_scale)
train_scaled[['date', 'month']] = train.reset_index()[['date', 'month']]

test_scaled = test[['date']]
for col in tqdm(colList):
    feat_list = [col + '_lag_' + str(shift) for shift in range(1, prev_factor + 1)]
    temp = test.apply(lambda row: XG_model.narrow_row_feat(row[feat_list], row[col + '_mean'], row[col + '_std']), axis=1)
    test_scaled = pd.concat([test_scaled, temp], axis=1)

features = []
for i in range(1, prev_factor + 1):
    features.append("turn_lag_" + str(i))
    features.append("range_hl_lag_" + str(i))
    features.append("range_oc_lag_" + str(i))
    features.append("volume_lag_" + str(i))
    features.append("close_lag_" + str(i))

target = "close"

X_train = train[features]
y_train = train[target]
X_sample = test[features]
y_sample = test[target]

X_train_scaled = train_scaled[features]
y_train_scaled = train_scaled[target]
X_sample_scaled = test_scaled[features]


parameters = {'n_estimators':[90],
            'max_depth':[9],
            'learning_rate': [0.2],
            'min_child_weight':range(5, 21, 1),
            }
model = XG_model.get_model()

GS = GSCV(estimator= model,param_grid=parameters,cv=5,refit= True,scoring='neg_mean_squared_error')

ndarray = plot_figure.plot_chart(GS,df,test,X_train_scaled,y_train_scaled,X_sample_scaled)
#calculate time cost
end = time.time()
print('total time cost {:.2f} sec.'.format(end-start))

test_track = list(zip(range(len(test)),ndarray))
pre_y_track = list(zip(range(len(test)),test['predict_y_Value'].values))
distans = ar.frechet_distance(test_track, pre_y_track)
print('appraisal:\nfrechet_distance =',distans)


