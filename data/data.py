# -*- coding: utf-8 -*-
import pandas as pd
from numpy import nan
from numpy import isnan
from pandas import read_csv
from datetime import datetime
import os.path

# fill missing values with a value at the same time one day ago
def fill_missing(values):
	one_day = 60 * 24
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if isnan(values[row, col]):
				values[row, col] = values[row - one_day, col]
                
def clean_raw_data():
    # load all data
    dataset = read_csv('raw/household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
    # mark all missing values
    dataset.replace('?', nan, inplace=True)
    # make dataset numeric
    dataset = dataset.astype('float32')
    # fill missing dataset values
    fill_missing(dataset.values)
    # save cleaned dataset
    dataset.to_csv('cleaned/household_power_consumption.csv')

def generate_training_data(ref_date, duration=1, time_step=60, resample=False,
                           path='cleaned/household_power_consumption.csv'):
    if not os.path.isfile(path):
        clean_raw_data()
    
    #load cleaned dataset
    dataset = read_csv("{0}".format(path), header=0, infer_datetime_format=True, 
                       parse_dates=['datetime'], index_col='datetime')
    dataset.index = pd.to_datetime(dataset.index)
    dataset.rename(columns={'Global_active_power':'Power'}, inplace=True)

    #determine number of training samples for given training duration
    training_data_samples = int((86400 * duration)/pd.Timedelta(dataset.index[1]-dataset.index[0]).seconds)

    # TODO add logic to check if returned dataframe is empty
    training_df=dataset[dataset.index <= datetime.strptime(ref_date, '%d/%m/%y %H:%M:%S')]

    # TODO refactor to 'try-except' logic
    if len(training_df) > training_data_samples:
        training_df = training_df[-training_data_samples:]
    else:
        training_df = training_df[:]
    
    if resample:
        training_df.resample("{0}S".format(time_step)).mean()
    training_df.to_csv("training_data/{0}_day_training_data.csv".format(duration))
    return training_df

df = generate_training_data('21/02/08 00:00:00')