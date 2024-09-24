import numpy as np
import pandas as pd

def extract_dataset_features(dataset):
    # initialise dataframe to contain only training features
    feature_df = pd.DataFrame({'Power': dataset['Power']})
    feature_df.index = pd.to_datetime(dataset.index)
    
    # set column for epoch in seconds
    feature_df['Epoch Seconds'] = feature_df.index.map(pd.Timestamp.timestamp)

    #conversion constants for time of day,week and year
    day = 60*60*24
    week = day * 7
    year = day * 365.2425

    #NUMERICALLY ENCODE TEMPORAL FEATURES
    # map time of day epoch seconds to sinusoids
    feature_df['Day sin'] = np.sin(feature_df['Epoch Seconds'] * (2 * np.pi / day))
    feature_df['Day cos'] = np.cos(feature_df['Epoch Seconds'] * (2 * np.pi / day))
    # map time of the week epoch seconds to sinusoids
    feature_df['Week sin'] = np.sin(feature_df['Epoch Seconds'] * (2 * np.pi / week))
    feature_df['Week cos'] = np.cos(feature_df['Epoch Seconds'] * (2 * np.pi / week))
    # map time of the year epoch seconds to sinusoids
    feature_df['Year sin'] = np.sin(feature_df['Epoch Seconds'] * (2 * np.pi / year))
    feature_df['Year cos'] = np.cos(feature_df['Epoch Seconds'] * (2 * np.pi / year))

    # remove epoch seconds now that sinusoid mapping is complete
    feature_df = feature_df.drop('Epoch Seconds', axis=1)
     
    return feature_df, len(feature_df.columns)
