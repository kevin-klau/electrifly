import pandas as pd
import numpy as np

# Gaussian filter 
from scipy.ndimage import gaussian_filter1d

def add_altitude(df):
    """
    Add the actual altitude --> alt
    """
    df[" PRESSURE_ALT"] = (df[" PRESSURE_ALT"] - df[" PRESSURE_ALT"].iloc[-1]) * 3.28084
    df[" PRESSURE_ALT"] = df[" PRESSURE_ALT"].where(df[" PRESSURE_ALT"] >= 0, 0)
    df.rename(columns={" PRESSURE_ALT":"alt"}, inplace=True)
    return df

def add_RoC(df):
    """
    Add the rate of climb --> taking gradient of altitude
    """
    df["RoC"] = np.gradient(df["alt"], df[" time(min)"]) # feet per min
    return df

def add_smoothed_alt(df,sigma):
    df[f"smoothed_alt_{sigma}"] = gaussian_filter1d(df["alt"], sigma)
    return df

def add_smoothed_RoC(df,sigma):
    df[f"smoothed_RoC_{sigma}"] = np.gradient(df[f"smoothed_alt_{sigma}"], df[" time(min)"]) # feet per min
    return df

def add_rolling_mean(df,feature,window_size=250):
    df[f'rollingMean_{window_size}_{feature}'] = df[feature].rolling(window=window_size).mean()
    return df


def add_features(df):
    df = add_altitude(df)
    df = add_RoC(df)
    df = add_smoothed_alt(df,15)
    df = add_smoothed_RoC(df,15)
    df = add_rolling_mean(df,'smoothed_RoC_15',window_size=25)

