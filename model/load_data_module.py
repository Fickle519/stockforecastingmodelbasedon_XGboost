# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import json

def load_config():
    f = open('./config.json','r')
    config_str = f.read()
    f.close()
    config_obj = json.loads(config_str)
    return config_obj

def load_stock_data(stk_path):
    df = pd.read_csv(stk_path, encoding="utf-8", dtype={'open':np.float64,'high':np.float64,'low':np.float64,'close':np.float64,'volume':np.float64,'turn':np.float64})
    df.loc[:, 'date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
    df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
    df['month'] = df['date'].dt.month
    df.sort_values(by='date', inplace=True, ascending=True)
    return df