# -*- coding: utf-8 -*-

def append_lagStr(cols_to_scale,i):
    cols_to_scale.append("turn_lag_" + str(i))
    cols_to_scale.append("range_hl_lag_" + str(i))
    cols_to_scale.append("range_oc_lag_" + str(i))
    cols_to_scale.append("volume_lag_" + str(i))
    cols_to_scale.append("close_lag_" + str(i))
    
