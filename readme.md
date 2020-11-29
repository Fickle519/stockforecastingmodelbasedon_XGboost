## Stockforecastingmodelbasedon_XGboost(B question) 
### [https://github.com/Fickle519/stockforecastingmodelbasedon_XGboost.git]
[Author:Xuanhe Er, Xiaohan Huang, Yanbo Cheng. ]from ** Chengdu University of Information Technology**

## Background

The change of the stock price of the listed company can directly reflect the operation status of the listed company and the recognition degree of the market. The modeling and forecasting of stock price is always a difficult problem. The most important factor is the stock price trend. Therefore, the stock market is a very typical nonlinear complex system. In the aspect of solving nonlinear complex system modeling, practice has proved that chaos theory is an effective theory, and has achieved certain theoretical and application effects in power, communication and other fields.

## Abstract

This code for the stock market with nonlinear complex system, using the idea and theory of chaos theory to establish a prediction model based on xgboost. This paper analyzes the daily trend, weekly trend and monthly trend of the three stocks; forecasts the future rise and fall of the price trend obtained from the data of different stocks; demonstrates the cyclical changes of the stock through the rise and fall data.

## Dataset

Stock Code: 000400 daily opening price, closing price, highest price, lowest price, trading volume and turnover rate from 2016 to 2020.
Stock Code: 002281, 2016-2020 daily opening price, closing price, highest price, lowest price, trading volume, turnover rate.
Stock Code: 600519, 2016-2020, daily opening price, closing price, highest price, lowest price, trading volume, turnover rate.

## Quick Test

#### Dependencies

- Python 3.x
- Python packages:  `pip install numpy pandas matplotlib sklearn tqdm py-xgboost`

- pandas > 0.24.2 && xgboost > 1.2.0 

#### Data

data is in the project(./data/)

 as follows:
- 600519.SH.csv
- 000400.SZ.csv
- 002281.SZ.csv

 (the .xls data file was transfered to .csv file for convenience.)


#### Run

1. chooose the one(.csv dataset file) you want to use then alter the config file(./config.json)

```json
{
    "data_path" : "./data/600519.SH.csv",
    "diagram_title" : "600519.SH diagram",
    "test_size" : 0.2,
    "N" : 3,
    "SEED" : 300,
    "threshold_value" : 1
}

```
stk_path:origin .csv data file location
title: the diagrams' title
you can see other parameters' meaning in the paper
**threshold's value depends on the diagrams' Y ticks and you can chooose the suitable value depend on your dataset you are using**
(here we recommend you the value of N should equal 3,otherwise you should change some of the structure of code)

2. check the data is already existed in the project path(if it is not in the project ,just create a new dir in project and put the .csv data file in it.)

3. simply run the run.py (without command-line argument parameters）
```
python run.py
```

## Citation

XGBoost: A Scalable Tree Boosting System 1603.02754 [https://arxiv.org/pdf/1603.02754.pdf]