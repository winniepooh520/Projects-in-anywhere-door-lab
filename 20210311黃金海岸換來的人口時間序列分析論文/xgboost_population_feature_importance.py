# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:33:19 2020

@author: 2070
"""

#%%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import xgboost
import matplotlib.pyplot as plt

f_birth_df = pd.read_excel('E:/Hank/population_prediction/Population_Forcasting/result/birth/six_city_birth_forcasting.xlsx')
f_city_annaul_expend_df = pd.read_excel('E:/Hank/population_prediction/Population_Forcasting/result/city_annual_expend/six_city_city_annual_expend_forcasting.xlsx')
f_death_df = pd.read_excel('E:/Hank/population_prediction/Population_Forcasting/result/death/six_city_death_forcasting.xlsx')
f_immigration_df = pd.read_excel('E:/Hank/population_prediction/Population_Forcasting/result/immigration/six_city_immigration_forcasting.xlsx')
f_per_capital_income_df = pd.read_excel('E:/Hank/population_prediction/Population_Forcasting/result/per_capital_income/six_city_per_capital_income_forcasting.xlsx')
f_population_df = pd.read_excel('E:/Hank/population_prediction/Population_Forcasting/result/population/six_city_population_forcasting.xlsx')

#birth_df = pd.read_excel('E:/Hank/population_prediction/Population_Forcasting/data/birth_df.xlsx')
#city_annaul_expend_df = pd.read_excel('E:/Hank/population_prediction/Population_Forcasting/data/city_annual_expend_df.xlsx')
#death_df = pd.read_excel('E:/Hank/population_prediction/Population_Forcasting/data/death_df.xlsx')
#immigration_df = pd.read_excel('E:/Hank/population_prediction/Population_Forcasting/data/immigration_df.xlsx')
#per_capital_income_df = pd.read_excel('E:/Hank/population_prediction/Population_Forcasting/data/per_capital_income_df.xlsx')
#population_df = pd.read_excel('E:/Hank/population_prediction/Population_Forcasting/data/population_df.xlsx')

def preprocessing(df, name):
    
    t_df = df.transpose()
    t_df.columns = t_df.iloc[0]
    t_df = t_df[1:-1]
    t_df_columns = t_df.columns.tolist()
    for i in range(len(t_df_columns)):
        t_df_columns[i] = t_df_columns[i] + ' ' + name
    t_df.columns = t_df_columns
    t_df = t_df.reset_index(drop = True)
    
    return t_df

t_birth_df = preprocessing(f_birth_df, 'birth')
t_city_annual_expend_df = preprocessing(f_city_annaul_expend_df, 'city annaul expend')
t_death_df = preprocessing(f_death_df, 'death')
t_immigration_df = preprocessing(f_immigration_df, 'immigration')
t_per_capital_income_df = preprocessing(f_per_capital_income_df, 'per capital income')
t_population_df = preprocessing(f_population_df, 'population')

def collect_seperate(city):
    
    df = pd.concat([t_birth_df['{0} birth'.format(city)],
               t_city_annual_expend_df['{0} city annaul expend'.format(city)],
               t_death_df['{0} death'.format(city)],
               t_immigration_df['{0} immigration'.format(city)],
               t_per_capital_income_df['{0} per capital income'.format(city)],
               t_population_df['{0} population'.format(city)]
               ],axis=1)
    
    df_cols = df.columns.tolist()
    
    for i in range(len(df_cols)):
        df_cols[i] = df_cols[i].replace(city+' ', '')
    
    df.columns = df_cols
    df.to_excel('E:/Hank/population_prediction/Population_Forcasting/data/4ksting_data_4_feature_imp/{0}.xlsx'.format(city), index = False)
    
    return df

df_new_taipei = collect_seperate('New Taipei City')
df_taipei = collect_seperate('Taipei City')
df_taoyuan = collect_seperate('Taoyuan City')
df_taichung = collect_seperate('Taichung City')
df_tainan = collect_seperate('Tainan City')
df_kaohsiung = collect_seperate('Kaohsiung City')
df_taiwan = collect_seperate('Taiwan')

#%%
def feature_imp(df, location):
    
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    x = scaled[:, :-1]
    y = scaled[:, -1]
    
    xgb = xgboost.XGBRegressor()
    xgb.fit(x, y)
    
    c, s = [], []
    for col,score in zip(df.columns[:-1],xgb.feature_importances_):
        c.append(col)
        s.append(score)
        print(col,score)
    
    f_imp = pd.DataFrame(
        {'col': c,
         'score': s,
         })
    f_imp=f_imp.sort_values(by='score', ascending=False)
    plot_data = f_imp.sort_values(by='score', ascending=False)
    f_imp=f_imp.reset_index(drop=True)
    
    plt.figure(figsize=(10, 20))
    plt.ylabel('Importance score')
    plt.xlabel('Feature')
    x_plot = plot_data['score']
    plt.barh(range(len(plot_data)), x_plot, tick_label = plot_data['col'])
    plt.grid()
    plt.title('{0} Feature Importance in Forcasting'.format(location))
    plt.savefig('E:/Hank/population_prediction/Population_Forcasting/feature_imp/forcasting/{0}_feature_imp.png'.format(location))
    f_imp.to_excel('E:/Hank/population_prediction/Population_Forcasting/feature_imp/forcasting/{0}_feature_imp.xlsx'.format(location), index = False)

feature_imp(df_new_taipei,'New Taipei City')
feature_imp(df_taipei,'Taipei City')
feature_imp(df_taoyuan,'Taoyuan City')
feature_imp(df_taichung,'Taichung City')
feature_imp(df_tainan,'Tainan City')
feature_imp(df_kaohsiung,'Kaohsiung City')
feature_imp(df_taiwan,'Taiwan')