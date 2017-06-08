# Model portfolio returns using time series analysis
__author__ = 'Mizio'

# Not so often used imports
# import csv as csv
# import matplotlib
# matplotlib.use('TkAgg')
# import sys
# sys.path.append('/custom/path/to/modules')
# from sklearn.model_selection import cross_val_score

# Used imports
import numpy as np
# import pandas as pd
# import pylab as plt
# from fancyimpute import MICE
# import random
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
# from scipy.stats import skew
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import KFold, train_test_split
# from sklearn.linear_model import LassoCV
# from sklearn.ensemble import IsolationForest
# from sklearn.preprocessing import StandardScaler, LabelBinarizer
# from sklearn_pandas import DataFrameMapper
# import xgboost as xgb
# from matplotlib.backends.backend_pdf import PdfPages
# import datetime
# from sklearn.cluster import FeatureAgglomeration
# import seaborn as sns

class TwoSigmaFinModTools:
    def __init__(self):
        pass

    def locate_intermediate_sale_of_asset(self):
        pass

    def assets_with_intermediate_sales(self, df, is_with_intermediate_sale):
        df_grouped_by_id = df[['id', 'timestamp', 'y']].groupby('id').agg([np.min, np.max, len]).reset_index()
        df_grouped_by_id.sort_values([('timestamp', 'amax')], inplace=True, ascending=False)
        # Todo: identify intermediate sale positions
        # what are the length differences
        # 1) make cutting strategi that checks on amax until the intermediate amax is found.
        # 2) After first cut, decide on the left cutted part if amax has length len.
        # 3) If True continue by making additional cut on right hand part,
        # 3i) then make check on left part of new cut to see if amax equals len. If True iterate from 3) if False iterate 4).
        # 4) If False continue with new cut on same left part,
        # 4i) then make check on left part of new cut to see if amax equals len. If True iterate from 3) if False iterate 4).

        # id of assets with intermediate trades
        is_with_intermediate_sale = is_with_intermediate_sale.drop(['index'], axis=1)
        indices_with_intermediate_trades = np.where(is_with_intermediate_sale)[0]
        id_for_intermediate_trades = df_grouped_by_id.reset_index().loc[indices_with_intermediate_trades,].id.values

        # Timestamp length diffs with len for assets with intermediate sale
        timestamp_length_and_len_diffs = (df_grouped_by_id[('timestamp', 'amax')]
                                          - df_grouped_by_id[('timestamp', 'amin')]
                                          - (df_grouped_by_id[('timestamp', 'len')] - 1)).reset_index().loc[indices_with_intermediate_trades]
        print('\n')
        print('timestamp length and len diffs:', '\n')
        print(timestamp_length_and_len_diffs)

        # Cut in two and check if amax is equal to length of cutted part

        # Assuming only one intermediate sale exists
        intermediate_trade_timestamp_of_assets = np.zeros((len(timestamp_length_and_len_diffs), 2))
        for ite in np.arange(0, len(id_for_intermediate_trades)):
            id = id_for_intermediate_trades[ite]
            # df_grouped_by_id_with_intermediate_trade = df_grouped_by_id[df.id == id]
            amin = df_grouped_by_id[df_grouped_by_id.id == id][('timestamp', 'amin')]
            amax = df_grouped_by_id[df_grouped_by_id.id == id][('timestamp', 'amax')]
            # asset_timestamps = df[['timestamp', 'id']][df.id == id].groupby('timestamp').timestamp
            # asset_timestamps_length = len(asset_timestamps)
            # midway_timestamps = round(asset_timestamps/2)

            # Todo: More general case: What if there are several intermediate trades?

            intermediate_trade_timestamp_of_assets[ite] = self.recursive_left_right_check(df, amin, amax, id)

        # return np.array([id_for_intermediate_trades, intermediate_trade_timestamp_of_assets]).transpose()
        return id_for_intermediate_trades, intermediate_trade_timestamp_of_assets

    def recursive_left_right_check(self, df, amin, amax, id):
        '''
        # method structure
        # 1)compute left part
        # 2a) check if left part is unique, if True compute right part
        # 2a,i) return amax and amin of right part
        # 2a,ii) divide right part in two and compute again left part starting in 1)
        # 2b) else if left part is not unique then return amax and amin of left part
        # 2b,i) divide left part in two and compute again left part starting in 1)
        :param df:
        :param df_grouped_by_id:
        :param amin:
        :param amax:
        :return:
        '''
        asset_timestamps = df[['timestamp', 'id']][(df.id == id) & (df.timestamp >= amin.values[0]) & (df.timestamp <= amax.values[0])].groupby('timestamp').timestamp
        # Find midway timestamp of particular id
        midway_timestamp = asset_timestamps.apply(int).values[round(len(asset_timestamps.apply(int).values)/2)]

        is_timestamp_diff_equal_len_left, amin_left, amax_left, lenght_left = self.check_timestamps_left_part(df, midway_timestamp, amin, id)
        if is_timestamp_diff_equal_len_left.values[0]:
            is_timestamp_diff_equal_len_right, amin_right, amax_right, lenght_right = self.check_timestamps_right_part(df, midway_timestamp, amax, id)
            if lenght_right.values[0]:
                return amin_right, amax_right
            else:
                if lenght_left.values[0] == 2:
                    return amin_left, amax_left
                else:
                    return self.recursive_left_right_check(df, amin_right, amax_right, id)
        else:
            return self.recursive_left_right_check(df, amin_left, amax_left, id)

    def check_timestamps_left_part(self, df, midway_timestamps, amin, id):
        '''
        Check left part

        :param df:
        :param df_grouped_by_id:
        :param midway_timestamps:
        :return: True if intermediate sale is in left part False otherwise.
        '''
        # Todo : correct code as done for right part
        df = df[df.id == id]
        # amin_left = df_grouped_by_id[(df.timestamp >= amin.values[0])
        #                              & (df.timestamp <= midway_timestamps)][('timestamp', 'amin')]
        # amax_left = df_grouped_by_id[(df.timestamp >= amin.values[0])
        #                              & (df.timestamp <= midway_timestamps)][('timestamp', 'amax')]
        # is_timestamp_diff_equal_len_left = (amax_left- amin_left).values \
        #                                    == (df_grouped_by_id[(df.timestamp >= amin.values[0])
        #                                                          & (df.timestamp <= midway_timestamps)][('timestamp',
        #                                                                                                'len')] - 1)

        df_timestamp_interval = df[(df.timestamp >= amin.values[0]) & (df.timestamp <= midway_timestamps)]
        df_timestamp_interval_aggregated = df_timestamp_interval.groupby('id').agg([np.min, np.max, len])
        amin_left = df_timestamp_interval_aggregated[('timestamp', 'amin')]
        amax_left = df_timestamp_interval_aggregated[('timestamp', 'amax')]
        lenght_left = df_timestamp_interval_aggregated[('timestamp', 'len')]
        is_timestamp_diff_equal_len_left = (amax_left - amin_left).values \
                                           == (lenght_left - 1)

        return is_timestamp_diff_equal_len_left, amin_left, amax_left, lenght_left

    def check_timestamps_right_part(self, df, midway_timestamps, amax, id):
        '''
        Check right part

        :param df:
        :param df_grouped_by_id:
        :param midway_timestamps:
        :return: True if intermediate sale is in left part False otherwise.
        '''
        df = df[df.id == id]
        # amin_right = df_grouped_by_id[(df_grouped_by_id.id == id) & (df_grouped_by_id.timestamp > midway_timestamps)
        #                               & (df_grouped_by_id.timestamp <= amax.values[0])][('timestamp', 'amin')]
        # amin_right = df_grouped_by_id[(df.id == id) & (df.timestamp > midway_timestamps)
        #                                & (df.timestamp <= amax.values[0])][('timestamp', 'amin')]
        df_timestamp_interval = df[(df.timestamp > midway_timestamps) & (df.timestamp <= amax.values[0])]
        # amin_right# [('timestamp', 'amin')]
        df_timestamp_interval_aggregated = df_timestamp_interval.groupby('id').agg([np.min, np.max, len])
        amin_right = df_timestamp_interval_aggregated[('timestamp', 'amin')]
        amax_right = df_timestamp_interval_aggregated[('timestamp', 'amax')]
        lenght_right = df_timestamp_interval_aggregated[('timestamp', 'len')]

        # amax_right = df_grouped_by_id[(df.timestamp > midway_timestamps)
        #                               & (df.timestamp <= amax.values[0])][('timestamp', 'amax')]
        # is_timestamp_diff_equal_len_right = (amax_right - amin_right).values \
        #                                     == (df_grouped_by_id[(df.timestamp > midway_timestamps)
        #                                                          & (df.timestamp <= amax.values[0])][('timestamp', 'len')] - 1)
        is_timestamp_diff_equal_len_right = (amax_right - amin_right).values \
                                            == (lenght_right - 1)

        # df_grouped_by_id = df[['id', 'timestamp', 'y']].groupby('id').agg([np.min, np.max, len]).reset_index()
        # df_grouped_by_id.sort_values([('timestamp', 'amax')], inplace=True, ascending=False)
        return is_timestamp_diff_equal_len_right, amin_right, amax_right, lenght_right


def main():
    # Not so often used imports
    # import csv as csv
    # import matplotlib
    # matplotlib.use('TkAgg')
    # import sys
    # sys.path.append('/custom/path/to/modules')
    # from sklearn.model_selection import cross_val_score

    # Used imports
    import numpy as np
    import pandas as pd
    import pylab as plt
    import seaborn as sns
    # from fancyimpute import MICE
    # import random
    # from sklearn.preprocessing import LabelEncoder
    # from sklearn.preprocessing import OneHotEncoder
    # from scipy.stats import skew
    # from sklearn.model_selection import cross_val_score
    # from sklearn.model_selection import KFold, train_test_split
    # from sklearn.linear_model import LassoCV
    # from sklearn.ensemble import IsolationForest
    # from sklearn.preprocessing import StandardScaler, LabelBinarizer
    # from sklearn_pandas import DataFrameMapper
    # import xgboost as xgb
    # from matplotlib.backends.backend_pdf import PdfPages
    # import datetime
    # from sklearn.cluster import FeatureAgglomeration

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
    is_with_intermediate_sale = ((df_grouped_by_id[('timestamp', 'amax')] - df_grouped_by_id[('timestamp', 'amin')])
                                 != (df_grouped_by_id[('timestamp', 'len')] - 1)).reset_index()
    print(''.join(['Number of intermediate sold assets: ', str(int(is_with_intermediate_sale.sum()[0]))]))
    print(df_grouped_by_id.reset_index().loc[np.where(is_with_intermediate_sale.drop(['index'], axis=1))[0],])
    two_sigma_fin_mod_tools = TwoSigmaFinModTools()
    intermediate_sales = two_sigma_fin_mod_tools.assets_with_intermediate_sales(df, is_with_intermediate_sale)
    print(intermediate_sales)


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
