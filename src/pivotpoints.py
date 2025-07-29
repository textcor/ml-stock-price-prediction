import pandas as pd


wick_threshold = 0.0001

# Function to detect support levels
def support(df1, l, n1, n2):  # n1 and n2 are the number of candles before and after candle l
    if (df1.low[l-n1:l].min() < df1.low[l] or df1.low[l+1:l+n2+1].min() < df1.low[l]):
        return 0

    candle_body = abs(df1.open[l] - df1.close[l])
    lower_wick = min(df1.open[l], df1.close[l]) - df1.low[l]
    if (lower_wick > candle_body) and (lower_wick > wick_threshold):
        return 1

    return 0

# Function to detect resistance levels
def resistance(df1, l, n1, n2):  # n1 and n2 are the number of candles before and after candle l
    if (df1.high[l-n1:l].max() > df1.high[l] or df1.high[l+1:l+n2+1].max() > df1.high[l]):
        return 0

    candle_body = abs(df1.open[l] - df1.close[l])
    upper_wick = df1.high[l] - max(df1.open[l], df1.close[l])
    if (upper_wick > candle_body) and (upper_wick > wick_threshold):
        return 1

    return 0


# Function to identify if the current candle is close to an existing resistance level
def closeResistance(l, levels, lim, df):
    if len(levels) == 0:
        return 0
    closest_level = min(levels, key=lambda x: abs(x - df.high[l]))
    c1 = abs(df.high[l] - closest_level) <= lim
    c2 = abs(max(df.open[l], df.close[l]) - closest_level) <= lim
    c3 = min(df.open[l], df.close[l]) < closest_level
    c4 = df.low[l] < closest_level
    if (c1 or c2) and c3 and c4:
        return closest_level
    else:
        return 0

# Function to identify if the current candle is close to an existing support level
def closeSupport(l, levels, lim, df):
    if len(levels) == 0:
        return 0
    closest_level = min(levels, key=lambda x: abs(x - df.low[l]))
    c1 = abs(df.low[l] - closest_level) <= lim
    c2 = abs(min(df.open[l], df.close[l]) - closest_level) <= lim
    c3 = max(df.open[l], df.close[l]) > closest_level
    c4 = df.high[l] > closest_level
    if (c1 or c2) and c3 and c4:
        return closest_level
    else:
        return 0
    

# Function to check if the high prices of recent candles are below the resistance level
def is_below_resistance(l, level_backCandles, level, df):
    return df.loc[l-level_backCandles:l-1, 'high'].max() < level

# Function to check if the low prices of recent candles are above the support level
def is_above_support(l, level_backCandles, level, df):
    return df.loc[l-level_backCandles:l-1, 'low'].min() > level

'''
Bullish Signal Conditions:

    Close to a resistance level (cR).
    Recent high prices are below the resistance level.
    Minimum RSI value of recent candles is less than 45.

Bearish Signal Conditions:

    Close to a support level (cS).
    Recent low prices are above the support level.
    Maximum RSI value of recent candles is greater than 55.
'''
def check_candle_signal(l, n1, n2, backCandles, df):
    ss = []
    rr = []

    # Identify support and resistance levels within the given range
    for subrow in range(l - backCandles, l - n2):
        if support(df, subrow, n1, n2):
            ss.append(df.low[subrow])
        if resistance(df, subrow, n1, n2):
            rr.append(df.high[subrow])
    
    # Merge close support levels
    ss.sort()  # Keep lowest support when merging
    for i in range(1, len(ss)):
        if i >= len(ss):
            break
        if abs(ss[i] - ss[i - 1]) <= 0.0001:  # Merging close distance levels
            ss.pop(i)

    # Merge close resistance levels
    rr.sort(reverse=True)  # Keep highest resistance when merging
    for i in range(1, len(rr)):
        if i >= len(rr):
            break
        if abs(rr[i] - rr[i - 1]) <= 0.0001:  # Merging close distance levels
            rr.pop(i)

    # Merge close support and resistance levels
    rrss = rr + ss
    rrss.sort()
    for i in range(1, len(rrss)):
        if i >= len(rrss):
            break
        if abs(rrss[i] - rrss[i - 1]) <= 0.0001:  # Merging close distance levels
            rrss.pop(i)

    # Check for close resistance and support levels
    cR = closeResistance(l, rrss, 150e-5, df)
    cS = closeSupport(l, rrss, 150e-5, df)

    # Determine if there's a signal based on conditions
    if (cR and is_below_resistance(l, 6, cR, df) and df.RSI[l-1:l].min() < 45):  # and df.RSI[l] > 65
        return 1
    elif (cS and is_above_support(l, 6, cS, df) and df.RSI[l-1:l].max() > 55):  # and df.RSI[l] < 35
        return 2
    else:
        return 0
    
from tqdm import tqdm

def generate_trading_signals(df):
    
# Parameters for support and resistance detection
    n1 = 8
    n2 = 6
    backCandles = 140

    # Initialize signal array
    signal = [0 for i in range(len(df))]

    # Generate trading signals
    for row in tqdm(range(backCandles + n1, len(df) - n2)):
        signal[row] = check_candle_signal(row, n1, n2, backCandles, df)

    # Add signals to the dataframe
    df["signal"] = signal