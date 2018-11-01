from __future__ import print_function
from scipy import stats
from datetime import timedelta
from decimal import Decimal
import pandas as pd
import statsmodels.api as sm
import codecs
import numpy as np
import csv

with codecs.open('./input.csv','r',encoding='big5') as f:
    dta = pd.read_csv(f)

#columns name assigned
dta.columns = ['CODE','YEAR','NAME','OPRICE','HIGH','LOW','VALUE','TRADE_NUM']

#convert str into date format
dta['YEAR'] = pd.to_datetime(dta.YEAR, format='%Y%m%d', errors='ignore')

#set table index
dta = dta.set_index("YEAR")

#list of the stock# which will be predicted
code_list=[50,51,52,53,54,55,56,57,58,
           59,690,692,701,713,6201,6203,6204,6208]

output_data = []

# choose p,d,q = (2,1,2) for minimum AIC as default
pVal,dVal,qVal = (2,1,2)

for code in code_list:
    train_data = dta.loc[dta['CODE'] == code]
    first_date = train_data.index[-1]
    end_date = first_date + timedelta(days=6)
    output_data.append(train_data['CODE'][0])
    try : 
        #freq must be given 
        arima_mod_VALUE = sm.tsa.ARIMA(train_data["VALUE"], 
                                       order=(pVal,dVal,qVal),
                                       exog=np.column_stack((train_data["OPRICE"],
                                                             train_data["HIGH"],
                                                             train_data["LOW"])),
                                       freq='D').fit(trend="nc")
    except ValueError :
        print('not stationary,try larger p:')
        arima_mod_VALUE = sm.tsa.ARIMA(train_data["VALUE"], 
                                       order=(pVal+1,1,2),
                                       exog=np.column_stack((train_data["OPRICE"],
                                                             train_data["HIGH"],
                                                             train_data["LOW"])),
                                       freq='D').fit(trend="nc")

    resid = arima_mod_VALUE.resid
    stats.normaltest(resid)
    r,q,p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    data = np.c_[range(1,41), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', "AC", "Q", "Prob(>Q)"])
    
    #predict
    predict_diff = arima_mod_VALUE.predict(first_date,
                                           end_date, 
                                           dynamic=True,
                                           exog=np.column_stack((train_data["OPRICE"],
                                                                 train_data["HIGH"],
                                                                 train_data["LOW"])))                   
    lastKnownPrice = train_data['VALUE'][-1] # + predict_diff[list(predict_diff).index(predict_diff.loc[dta.index[-1]])]
    lastPredictPrice = lastKnownPrice + predict_diff[list(predict_diff).index(predict_diff.loc[train_data.index[-1]])+1]
    for i in range(1,6):
        # round to 2 digits of float
        # trend = ith predict difference; if >0 -> rise, <0 -> down , =0 ->unchanged
        trend = float(predict_diff[list(predict_diff).index(predict_diff.loc[train_data.index[-1]])+ i])    
        if (trend > 0.00) :
            print(train_data['NAME'][0] + 
                  str(float(Decimal(lastPredictPrice).quantize(Decimal('0.00'))))+
                  ',1')
            output_data.append('1')
            output_data.append(str(float(Decimal(lastPredictPrice).quantize(Decimal('0.00')))))
        elif (trend < 0.00):
            print(train_data['NAME'][0] + 
                  str(float(Decimal(lastPredictPrice).quantize(Decimal('0.00'))))+
                  ',-1')
            output_data.append('-1')
            output_data.append(str(float(Decimal(lastPredictPrice).quantize(Decimal('0.00')))))
        else :
            print(train_data['NAME'][0] + 
                  str(float(Decimal(lastPredictPrice).quantize(Decimal('0.00'))))+
                  ',0')
            output_data.append('0')
            output_data.append(str(float(Decimal(lastPredictPrice).quantize(Decimal('0.00')))))
        lastPredictPrice += predict_diff[list(predict_diff).index(predict_diff.loc[train_data.index[-1]])+ (i+1)]
#output result to csv file
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['ETFid', 'Mon_ud', 'Mon_cprice', 'Tue_ud', 'Tue_cprice',
                     'Wed_ud', 'Wed_cprice','Thu_ud', 'Thu_cprice','Fri_ud', 'Fri_cprice'])
    for i in range(0,18) :
        output_data_index = i*11
        writer.writerow(output_data[output_data_index:output_data_index+11])
    csvfile.close()
