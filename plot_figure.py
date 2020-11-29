# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import model.load_data_module as LM

config_obj = LM.load_config()

def plot_ax(test):
    ax = test.plot(x='date', y='close', grid=True,color='#008c8c',alpha=0.65)
    ax = test.plot(x='date', y='predict_y_Value', grid=True, ax=ax,color='#cc00cc')
    
    ax.set_title(config_obj["diagram_title"])
    ax.set_xlabel('date')
    ax.set_ylabel('stock price')
    
    
def plot_ax2(test):
    ax2 = test.plot(x='date', y='close',  grid=True,color='#008c8c')
    ax2.set_xlabel('date')
    ax2.set_ylabel('stock price')
    ax2.set_title(config_obj["diagram_title"])
    
    
def plot_ax3(extre):
    ax3 = extre.plot(x='date', y='close', grid=True,color='#cc00cc')
    ax3.set_xlabel('date')
    ax3.set_ylabel('stock price')
    ax3.set_title(config_obj["diagram_title"])
    
    
def plot_chart(gs,test,X_train_scaled,y_train_scaled,X_sample_scaled):
    gs.fit(X_train_scaled, y_train_scaled)
    
    pre_scaled_Y_value = gs.predict(X_sample_scaled)
    test['pre_scaled_Y_value'] = pre_scaled_Y_value
    test['predict_y_Value'] = test['pre_scaled_Y_value'] * test['close_std'] + test['close_mean']
    #figure the diagrams by matplotlib[ax/plt]
    plt.figure(dpi=50)
    plot_ax(test)
    plt.savefig('./XGBoost_result_pre/pre_result_{}_1.png'.format(config_obj["diagram_title"]))
    import model.StockPriceForecast_Model_XG as XG_model
    ndarray = test['close'].values
    indexs = XG_model.get_peak_value(ndarray)
    #print(indexs)
    extre = test.iloc[indexs]
    
    plot_ax2(test)
    plt.savefig('./XGBoost_result_pre/pre_result_{}_2.png'.format(config_obj["diagram_title"]))
    
    plot_ax3(extre)
    plt.savefig('./XGBoost_result_pre/pre_result_{}_3.png'.format(config_obj["diagram_title"]))
    
    plt.show()
    