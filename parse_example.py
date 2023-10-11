#%%
import pandas as pd

df = pd.read_csv("f:/projects/ann_calsim_ec_estimator/Inputs/inputs.csv",
                 index_col=0,parse_dates=[1],header=0)
print(df)

#%%

x = df.loc[df.date==pd.to_datetime("2015-09-28")]


# %%
print(x)
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


num_feature_dims = {"sac" : 18, 
                    "exports" : 18, 
                    "dcc": 18, 
                    "net_dcd" : 18, 
                    "sjr": 18, 
                    "tide" : 18, 
                    "smscg" : 18}

lags_feature = None


def feature_names():
    return list(num_feature_dims.keys())


def calc_lags_feature(df):
    global lags_feature
    lags_feature = {feature: df.loc[:, pd.IndexSlice[feature,:]].columns.get_level_values(level='lag')[0:num_feature_dims[feature]] 
                    for feature in feature_names()}

def df_by_variable(df):
    """ Convert a dataset with a single index with var_lag as column names and convert to MultiIndex with (var,ndx)
        This facilitates queries that select only lags or only variables. As a side effect this routine will store
        the name of the active lags for each feature, corresponding to the number of lags in the dictionary num_feature_dims)
        into the module variable lag_features.

        Parameters
        ----------
        df : pd.DataFrame 
            The DataFrame to be converted

        Returns
        -------
        df_var : A DataFrame with multiIndex based on var,lag  (e.g. 'sac','4d')
    """
    indextups = []
    df = df.copy()
    for col in list(df.columns):
        var = col
        lag = ""
        for key in num_feature_dims.keys():
            if col.startswith(key):
                var = key
                lag = col.replace(key,"").replace("_","")
                if lag is None or lag == "": 
                    lag = "0d"
                continue
        if var == "EC": lag = "0d"
        indextups.append((var,lag))
 
    ndx = pd.MultiIndex.from_tuples(indextups, names=('var', 'lag'))
    df.columns = ndx
    calc_lags_feature(df)
    return df


# %%
df2 = df_by_variable(df)
dfndx = df2.loc[:,('date',)]
print(dfndx)
selected = []
request = pd.to_datetime("2013-11-15")

varnames = feature_names()
for varb in varnames:
    subdf = df2.loc[:, pd.IndexSlice[varb,:]].droplevel(level="var",axis=1)
    subdf["datetime"] = dfndx
    subdf.set_index("datetime",inplace=True)
    sel = subdf.loc[request,:].transpose()
    sel.name=varb
    selected.append(sel)
    print("Original: ",df.loc[df["date"]==request,"EC"])
allselect = pd.concat(selected,axis=1)
print(allselect.transpose())
allselect.transpose().to_csv("f:/projects/ann_calsim_ec_estimator/Inputs/selected.csv",
         header=False,index=False,float_format="%.3f")


# %%
