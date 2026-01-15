import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

california_housing = fetch_california_housing(as_frame=True)
print("DESCRIPTION", california_housing.DESCR)
print("data set", california_housing.frame.head())
print("first five smaple inputs",california_housing.data.head())
print("first 5 target values", california_housing.target.head())
print("INFORMATION", california_housing.frame.info())
california_housing.frame.hist(figsize=(12, 10), bins=30, edgecolor="black")
plt.subplots_adjust(hspace=0.7, wspace=0.4)
plt.show()

