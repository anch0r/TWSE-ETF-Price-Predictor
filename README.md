# TWSE-ETF-Price-Predictor
台股ETF股價預測

使用ARIMA模型，預測未來一週5個交易日(星期一到星期五)台股ETF的漲跌和股價

input : CSV檔，欄位包含：股票代號、日期、股票名稱、當日開盤價、當日最高價、當日最低價、收盤價、成交張數

output： CSV檔，欄位包含每檔ETF代號、預測每個交易日的漲或跌以及股價 

### 執行環境

Python 3.6

需要安裝 statsmodels： https://www.statsmodels.org

