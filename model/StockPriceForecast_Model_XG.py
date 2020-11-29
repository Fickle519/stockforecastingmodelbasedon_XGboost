import numpy as np
from xgboost import XGBRegressor

import model.load_data_module as LM


config_obj = LM.load_config()
prev_factor = 3  #default number is 3 but you can alter it in config(./config.json)

def stockFeature_process(dF):

    dF['range_hl'] = dF['high'] - dF['low']
    dF['range_oc'] = dF['open'] - dF['close']

    lag_cols = ['turn', 'range_hl', 'range_oc', 'volume', 'close']
    shift_range = [x + 1 for x in range(prev_factor)]

    for col in lag_cols:
        for i in shift_range:
            new_col='{}_lag_{}'.format(col, i)
            dF[new_col]=dF[col].shift(i)

    return dF[prev_factor:]

def narrow_row_feat(row, feat_mean, feat_std):
    feat_std = 0.001 if feat_std == 0 else feat_std
    row_scaled = (row - feat_mean) / feat_std

    return row_scaled

def get_mean_Output(dF, col, prev_factor):
    mean_list = dF[col].rolling(window=prev_factor, min_periods=1).mean()  
    std_list = dF[col].rolling(window=prev_factor, min_periods=1).std()  
    mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
    std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))

    output = dF.copy()
    output[col + '_mean'] = mean_list
    output[col + '_std'] = std_list

    return output

def get_peak_value(num_list):
    index_list = []
    index_flag = 0
    index_list.append(index_flag)
    for i in range(1,len(num_list)):
        if abs(num_list[i] - num_list[index_flag]) > config_obj["threshold_value"]:
            index_flag = i
            index_list.append(i)
    return index_list


def get_model():
    return XGBRegressor(seed=config_obj["SEED"],
                     n_estimators=config_obj["XGBRegressor"]["n_estimators"],
                     max_depth=config_obj["XGBRegressor"]["max_depth"],
                     eval_metric=config_obj["XGBRegressor"]["eval_metric"],
                     learning_rate=config_obj["XGBRegressor"]["learning_rate"],
                     min_child_weight=config_obj["XGBRegressor"]["min_child_weight"],
                     subsample=config_obj["XGBRegressor"]["subsample"],
                     colsample_bytree=config_obj["XGBRegressor"]["colsample_bytree"],
                     colsample_bylevel=config_obj["XGBRegressor"]["colsample_bylevel"],
                     gamma=config_obj["XGBRegressor"]["gamma"])