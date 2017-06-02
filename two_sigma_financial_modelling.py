# Model portfolio returns using time series analysis
__author__ = 'Mizio'

# import csv as csv
import numpy as np
import pandas as pd
# import matplotlib
# matplotlib.use('TkAgg')
import pylab as plt
from fancyimpute import MICE
# import sys
# sys.path.append('/custom/path/to/modules')
import random
# from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import skew
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LassoCV
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn_pandas import DataFrameMapper
import xgboost as xgb
from matplotlib.backends.backend_pdf import PdfPages
import datetime
from sklearn.cluster import FeatureAgglomeration
import seaborn as sns

pd.set_option('display.max_columns', 120)

# Load the data from the HDF5 file instead of csv. The file consist only of training data.
# For upload.
# with pd.HDFStore("../input/train.h5", "r") as train:
# For local run.
with pd.HDFStore("/home/mizio/Documents/Kaggle/TwoSigmaFinancialModelling/input/train.h5", "r") as train:
    df = train.get("train")

# Test with house sale price data
# df = pd.read_csv('/home/mizio/Documents/Kaggle/HousePrices/train.csv', header=0)

# Overview of train data
print('\n TRAINING DATA:----------------------------------------------- \n')
# print(df.head(3))
# print('\n')
# print(df.info())
# print('\n')
# print(df.describe())
# print('\n')
# print(df.dtypes)
# print(df.get_dtype_counts())


# Histogram of features in the portfolio
# Each asset is identified by it's 'id'.
print(len(df.id.unique()))  # Shows the number of asset (financial instruments) that are being tracked.
print(len(df.timestamp.unique()))  # shows the number of periods in time
features = ['timestamp', 'derived_0', 'derived_1', 'derived_2', 'derived_3', 'derived_4']
# features = ['timestamp', 'derived_1']
# df[features].groupby('timestamp').agg([np.mean]).reset_index().apply(np.log1p).hist(bins='auto', alpha=0.5)
# plt.show()

df_derived_features = df[features].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()
print(df_derived_features.head())
print(df_derived_features.describe())

# Examine individual assets that are identified by the 'id'
print('\n Assets:----------------------------------------------- \n')
df_assets = df.groupby('id')['y'].agg(['mean', 'std', len]).reset_index()
print(df_assets.head())

# Plot target value of asset id=0 as function of timestamp
asset_0 = df[df.id == 0]
# asset_0 = df.loc[df.id == 0, ('timestamp', 'y')].groupby('timestamp')
plt.figure()
plt.plot(asset_0.timestamp.values, asset_0.y.values, '.')
plt.plot(asset_0.timestamp.values, asset_0.y.values.cumsum())
plt.legend(('asset val', 'cumulative asset value'), loc=1, borderaxespad=0.)
plt.xlabel('timestamp')
plt.ylabel('asset value')
plt.show()

# Visualize market run over the time period
market_return_df = df[['timestamp', 'y']].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()
# print(market_return_df.head())

# How does the mean and std of the target value 'y' vary as function of timestamp?
# How does size of the portfolio vary as function of timestamp?
timestamp = market_return_df['timestamp']
y_mean = np.array(market_return_df['y']['mean'])
y_std = np.array(market_return_df['y']['std'])
size_of_portfolio = np.array(market_return_df['y']['len'])

# Todo: make subplots
plt.figure()
plt.plot(timestamp, y_mean, '.')
plt.xlabel('timestamp')
plt.ylabel('y mean')


plt.figure()
plt.plot(timestamp, y_std, '.')
plt.xlabel('timestamp')
plt.ylabel('y std')


plt.figure()
plt.plot(timestamp, size_of_portfolio, '.')
plt.xlabel('timestamp')
plt.ylabel('size of portfolio')
plt.show()
# Comm.: we see that timestamp 250 and 1550 has high variation in mean value.


# Plot correlations between mean, std of 'y' and size of portfolio.
sns.set()
columns = ['mean', 'std', 'len']
sns.pairplot(market_return_df['y'][columns], size=2.5)
plt.show()



