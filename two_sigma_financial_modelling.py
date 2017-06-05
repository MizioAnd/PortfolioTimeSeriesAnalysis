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

class TwoSigmaFinModTools:
    def __init__(self):
        pass

    def locate_intermediate_sale_of_asset(self):
        pass

    def assets_with_intermediate_sales(self, df, is_with_intermediate_sale):
        df_grouped_by_id = df[['id', 'timestamp', 'y']].groupby('id').agg([np.min, np.max, len]).reset_index()
        # Todo: identify intermediate sale positions
        # what are the length differences
        # 1) make cutting strategi that checks on amax until the intermediate amax is found.
        # 2) After first cut, decide on the left cutted part if amax has length len.
        # 3) If True continue by making additional cut on right hand part,
        # 3i) then make check on left part of new cut to see if amax equals len. If True iterate from 3) if False iterate 4).
        # 4) If False continue with new cut on same left part,
        # 4i) then make check on left part of new cut to see if amax equals len. If True iterate from 3) if False iterate 4).

        # id of assets with intermediate trades
        id_with_intermediate_trades = np.where(is_with_intermediate_sale)[0]

        # Timestamp length diffs with len for assets with intermediate sale
        timestamp_length_and_len_diffs = (df_grouped_by_id[('timestamp', 'amax')]
                                          - df_grouped_by_id[('timestamp', 'amin')]
                                          - (df_grouped_by_id[('timestamp', 'len')] - 1))[is_with_intermediate_sale]
        print('\n')
        print('timestamp length and len diffs:', '\n')
        print(timestamp_length_and_len_diffs)

        # Cut in two and check if amax is equal to length of cutted part
        id = id_with_intermediate_trades[0]
        # df_grouped_by_id_with_intermediate_trade = df_grouped_by_id[df.id == id]
        amin = df_grouped_by_id[df_grouped_by_id.id == id][('timestamp', 'amin')]
        amax = df_grouped_by_id[df_grouped_by_id.id == id][('timestamp', 'amax')]
        asset_timestamps = df[['timestamp', 'id']][df.id == id].groupby('timestamp').timestamp
        asset_timestamps_length = len(asset_timestamps)
        midway_timestamps = round(asset_timestamps/2)
        # Check left part
        amin_left = df_grouped_by_id[(df.id == id & df.timestamp < midway_timestamps)][('timestamp', 'amin')]
        amax_left = df_grouped_by_id[(df.id == id & df.timestamp < midway_timestamps)][('timestamp', 'amax')]
        timestamp_diff_with_len_left = (df_grouped_by_id[df_grouped_by_id.id == id][('timestamp', 'amax')]
                                        - df_grouped_by_id[df_grouped_by_id.id == id][('timestamp', 'amin')]).values \
                                       != (df_grouped_by_id[df_grouped_by_id.id == id][('timestamp', 'len')] - 1)
        # Define recursive function


def main():
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

    df_derived_features = df[features].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()
    print(df_derived_features.head())
    print(df_derived_features.describe())

    # Examine individual assets that are identified by the 'id'
    print('\n Assets:----------------------------------------------- \n')
    df_assets = df.groupby('id')['y'].agg(['mean', 'std', len]).reset_index()
    print(df_assets.head())

    # Plot target value of asset id=0 as function of timestamp
    asset_id = 0
    asset_0 = df[df.id == asset_id]
    # asset_0 = df.loc[df.id == 0, ('timestamp', 'y')].groupby('timestamp')
    plt.figure()
    plt.plot(asset_0.timestamp.values, asset_0.y.values, '.')
    plt.plot(asset_0.timestamp.values, asset_0.y.values.cumsum())
    plt.legend(('asset value', 'cumulative asset value'), loc=1, borderaxespad=0.)
    plt.xlabel('timestamp')
    plt.ylabel('asset value')
    plt.title(''.join(['Asset ', str(asset_id)]))

    # When are the assets sold and bought?
    # how can we be sure that they are not sold in between and hold for less time? checking on amax of timestamp just
    # indicates first time the asset is bought and last time indicates last time the asset is sold.

    # Todo: make check on intermediate sale of asset
    df_grouped_by_id = df[['id', 'timestamp', 'y']].groupby('id').agg([np.min, np.max, len]).reset_index()
    df_grouped_by_id.sort_values([('timestamp', 'amax')], inplace=True, ascending=False)
    print(df_grouped_by_id.head())

    # Plot without check on intermediate sales
    plt.figure()
    plt.plot(df_grouped_by_id[('timestamp', 'amin')], df_grouped_by_id.id, '.', label='bought')
    plt.plot(df_grouped_by_id[('timestamp', 'amax')], df_grouped_by_id.id, '.', color='r', label='sold')
    plt.xlabel('timestamp')
    plt.ylabel('asset id')
    plt.legend()

    # Check on intermediate sales
    # check if len - 1 of timestamps equals amax - amin
    is_with_intermediate_sale = (df_grouped_by_id[('timestamp', 'amax')] - df_grouped_by_id[('timestamp', 'amin')]).values \
                                != (df_grouped_by_id[('timestamp', 'len')] - 1)
    print(''.join(['Number of intermediate sold assets: ', str(is_with_intermediate_sale.sum())]))

    # Visualize market run over the time period
    market_return_df = df[['timestamp', 'y']].groupby('timestamp').agg([np.mean, np.std, len]).reset_index()
    # print(market_return_df.head())

    # How does the mean and std of the target value 'y' vary as function of timestamp?
    # How does size of the portfolio vary as function of timestamp?
    timestamp = market_return_df['timestamp']
    y_mean = np.array(market_return_df['y']['mean'])
    y_std = np.array(market_return_df['y']['std'])
    # Number of assets traded for each unique timestamp
    size_of_portfolio = np.array(market_return_df['y']['len'])

    f, axarr = plt.subplots(3, sharex=True)
    axarr[0].plot(timestamp, y_mean, '.')
    axarr[0].set_ylabel('y mean')

    axarr[1].plot(timestamp, y_std, '.')
    axarr[1].set_ylabel('y std')

    axarr[2].plot(timestamp, size_of_portfolio, '.')
    axarr[2].set_ylabel('size of portfolio')

    axarr[2].set_xlabel('timestamp')
    # Comm.: we see that timestamp 250 and 1550 has high variation in mean value.

    # Plot correlations between mean, std of 'y' and size of portfolio.
    sns.set()
    columns = ['mean', 'std', 'len']
    sns.pairplot(market_return_df['y'][columns], size=2.5)

    # Price chart for returns of portfolio. This corresponds to the mean of y of the portfolio.
    # Plot is together with mean of y of portfolio.
    plt.figure()
    plt.plot(timestamp, y_mean, '.')
    plt.plot(timestamp, y_mean.cumsum())
    plt.legend(('portfolio value', 'cumulative portfolio value'), loc=1, borderaxespad=0.)
    plt.xlabel('timestamp')
    plt.ylabel('y mean')
    plt.title('Portfolio returns')
    plt.show()


if __name__ == '__main__':
    main()
