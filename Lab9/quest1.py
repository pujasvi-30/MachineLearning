import numpy as np
import pandas as pd


df=pd.read_csv('simulated_data_multiple_linear_regression_for_ML.csv')
    # X = df.drop(columns=['disease_score', 'disease_score_fluct']).values
    # y = df['disease_score'].values
    # return X, y

def partition_data(df, threshold):
    left_partition = df[df["BP"] <= threshold]
    right_partition = df[df["BP"] > threshold]
    print(left_partition)
    print(right_partition)
    return left_partition, right_partition
thresholds=[78, 82]
for threshold in thresholds:
    left_partition, right_partition = partition_data(df, threshold)
    print(left_partition.shape)
    print(right_partition.shape)

left_partition, right_partition = partition_data(df, 80)






