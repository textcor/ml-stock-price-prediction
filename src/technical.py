import ta.trend
import talib
import pandas as pd
import ta
import numpy as np

header_color = "\033[0m"
label_color = "\033[34m"
data_color = "\033[33m"
call_color = "\033[32m"
put_color = "\033[31m"
neutral_color = "\033[93m"
explanation_color = "\033[37m"
signal_color = {
    'Call' : call_color,
    'Put': put_color,
    'Neutral': neutral_color,
    'Buy' : call_color,
    'Sell': put_color,
    'Hold': neutral_color,
    'Stable': call_color,
    'Volatile': put_color
}


sma_short_period = 5
sma_long_period = 20
ema_short_period = 9
ema_mid_period = 20
ema_long_period = 200
rsi_period = 14
atr_period =14


def format_ind_number(num):  # Helper function for formatting
        if isinstance(num, (int, float)):
            return "{:,.2f}".format(num)  # Format with commas and no decimals
        return num  # Return as is if not a number

def calculate_pivot_points(data):
    """
    Calculates pivot points and support/resistance levels based on historical OHLC data.

    Parameters:
    data (pandas.DataFrame): DataFrame containing historical OHLC data, as returned by yfinance.

    Returns:
    pandas.DataFrame: DataFrame with columns for Pivot Point (PP),
        Resistance 1 (R1), Support 1 (S1), R2, S2, R3, S3.
    """
    # Check for required columns
    required = ['High', 'Low', 'Close']
    if not all(col in data.columns for col in required):
        missing = [col for col in required if col not in data.columns]
        raise ValueError(f"Missing required columns: {missing}")

    # Sort data chronologically
   # data = data.copy()
   # data = data.sort_index(ascending=True)
    
    last_session = data.iloc[-1]

    # Calculate previous period's values
    prev_high = last_session['High']
    prev_low = last_session['Low']
    prev_close = last_session['Close']

    # Calculate pivot point
    pp = (prev_high + prev_low + prev_close) / 3

    # Calculate support and resistance levels
    return {  
        'pivot_point' : round(pp, 2),
        'resistance_1' : round(2 * pp - prev_low, 2),
        'resistance_2' : round(pp + (prev_high - prev_low) ,2),
        'resistance_3' : round(pp + 2 * (prev_high - prev_low), 2),
        'support_1' : round(2 * pp - prev_high, 2),
        'support_2' : round(pp - (prev_high - prev_low), 2),
        'support_3' : round(pp - 2 * (prev_high - prev_low), 2) 
    }

def print_pivot_points(analysis):

    if analysis['pivot_points']:
        print(f"{header_color}\nPIVOT POINTS:")
        print(f"{label_color}Pivot Point: {data_color}${analysis['pivot_points']['pivot_point']}")
        print(f"{label_color}Resistance 1: {data_color}${analysis['pivot_points']['resistance_1']}")
        print(f"{label_color}Support 1: {data_color}${analysis['pivot_points']['support_1']}")
        print(f"{label_color}Resistance 2: {data_color}${analysis['pivot_points']['resistance_2']}")
        print(f"{label_color}Support 2: {data_color}${analysis['pivot_points']['support_2']}")
        print(f"{label_color}Resistance 3: {data_color}${analysis['pivot_points']['resistance_3']}")
        print(f"{label_color}Support 3: {data_color}${analysis['pivot_points']['support_3']}")


def analyze_pattern_performance(df, pattern):
    """Calculate historical performance of a specific pattern"""
    try:
        pattern_dates = df[df[pattern]].index
        returns = []

        for date in pattern_dates:
            if date + pd.DateOffset(weeks=1) in df.index:
                # Proper scalar value extraction
                current_price = df.loc[date, 'Close'].item()  # Use .item() instead of float()
                future_price = df.loc[date + pd.DateOffset(weeks=1), 'Close'].item()

                returns.append((future_price / current_price) - 1)

        if not returns:
            return {}

        return {
            'count': len(returns),
            'win_rate': len([r for r in returns if r > 0])/len(returns),
            'avg_return': float(np.mean(returns)),
            'volatility': float(np.std(returns))
        }
    except Exception as e:
        print(f"Pattern analysis failed for {pattern}: {str(e)}")
        return {}

def detect_pattern(df):
    # Pattern detection
    open_prices = df['Open'].to_numpy().flatten()
    high_prices = df['High'].to_numpy().flatten()
    low_prices = df['Low'].to_numpy().flatten()
    close_prices = df['Close'].to_numpy().flatten()

    patterns = {
        'hammer': talib.CDLHAMMER(open_prices,high_prices,low_prices,close_prices),
        'engulfing_bull': talib.CDLENGULFING(open_prices,high_prices,low_prices,close_prices),
        'doji': talib.CDLDOJI(open_prices,high_prices,low_prices,close_prices),
        'morning_star': talib.CDLMORNINGSTAR(open_prices,high_prices,low_prices,close_prices),
        'three_white_soldiers' : talib.CDL3WHITESOLDIERS(open_prices,high_prices,low_prices,close_prices),
        'inverted_hammer' :  talib.CDLINVERTEDHAMMER(open_prices,high_prices,low_prices,close_prices),
        'piercing_line': talib.CDLPIERCING(open_prices, high_prices, low_prices, close_prices),
        'rising_three_methods': talib.CDLRISEFALL3METHODS(open_prices, high_prices, low_prices, close_prices),
        'three_black_crows' : talib.CDL3BLACKCROWS(open_prices, high_prices, low_prices, close_prices),
        'three_lines_strike' :talib.CDL3LINESTRIKE(open_prices, high_prices, low_prices, close_prices),
        'tasuki_gap': talib.CDLTASUKIGAP(open_prices, high_prices, low_prices, close_prices),
        'evening_star': talib.CDLEVENINGSTAR(open_prices,high_prices,low_prices,close_prices),
        'dragon_fly_doji' : talib.CDLDRAGONFLYDOJI(open_prices,high_prices,low_prices,close_prices),
        'grave_stone_doji': talib.CDLGRAVESTONEDOJI(open_prices,high_prices,low_prices,close_prices),
        'two_crows': talib.CDL2CROWS(open_prices,high_prices,low_prices,close_prices),
    }

    pattern_explanations = {  # Dictionary to store pattern explanations
        'hammer': {
            'explanation': "A Hammer is a bullish reversal pattern that occurs at the bottom of a downtrend. It signals a potential price reversal to the upside.",
            'bullish': True,
            'bearish': False
        },
        'engulfing_bull': {
            'explanation': "A Bullish Engulfing pattern is a bullish reversal pattern that signals the end of a downtrend. It's characterized by a small bearish candle followed by a larger bullish candle that completely engulfs the bearish candle.",
            'bullish': True,
            'bearish': False
        },
        'doji': {
            'explanation': "A Doji is a candlestick pattern characterized by a small body and long upper and/or lower shadows. It indicates indecision in the market and can signal a potential trend reversal.",
            'bullish': None, # Could be bullish or bearish depending on context
            'bearish': None
        },
        'morning_star': {
            'explanation': "A Morning Star is a bullish reversal pattern that occurs at the bottom of a downtrend. It signals a potential price reversal to the upside.",
            'bullish': True,
            'bearish': False
        },
        'three_white_soldiers': {
            'explanation': "Three White Soldiers is a bullish reversal pattern that consists of three consecutive long white (or green) candlesticks that open within the previous candle's body and close progressively higher.",
            'bullish': True,
            'bearish': False
        },
        'inverted_hammer': {
            'explanation': "An Inverted Hammer is a bullish reversal pattern that occurs at the bottom of a downtrend. It signals a potential price reversal to the upside.",
            'bullish': True,
            'bearish': False
        },
        'piercing_line': {
            'explanation': "The Piercing Line is a bullish reversal pattern that occurs at the bottom of a downtrend. It signals a potential price reversal to the upside.",
            'bullish': True,
            'bearish': False
        },
        'rising_three_methods': {
            'explanation': "The Rising Three Methods pattern is a bullish continuation pattern. It suggests that the prevailing uptrend is likely to resume after a brief pause.",
            'bullish': True,
            'bearish': False
        },
        'three_black_crows': {
            'explanation': "Three Black Crows is a bearish reversal pattern that consists of three consecutive long black (or red) candlesticks that open within the previous candle's body and close progressively lower.",
            'bullish': False,
            'bearish': True
        },
        'three_lines_strike': {
            'explanation': "The Three-Line Strike pattern is a bearish reversal pattern. It signals a potential price reversal to the downside.",
            'bullish': False,
            'bearish': True
        },
        'tasuki_gap': {
            'explanation': "The Tasuki Gap is a bearish continuation pattern that typically occurs during a downtrend. It suggests that the prevailing downtrend is likely to continue after a brief pause..",
            'bullish': False,
            'bearish': True
        },
        'evening_star': {
            'explanation': "The Evening Star is a bearish reversal pattern that occurs at the top of an uptrend. It signals a potential price reversal to the downside.",
            'bullish': False,
            'bearish': True
        },
        'dragon_fly_doji': {
            'explanation': "A Dragonfly Doji is a bullish candlestick pattern that occurs at the bottom of a downtrend. It signals a potential price reversal to the upside.",
            'bullish': True,
            'bearish': False
        },
        'grave_stone_doji': {
            'explanation': "A Gravestone Doji is a bearish candlestick pattern that occurs at the top of an uptrend. It signals a potential price reversal to the downside.",
            'bullish': False,
            'bearish': True
        },
        'two_crows': {
            'explanation': "The Two Crows pattern is a bearish reversal pattern. It signals a potential price reversal to the downside.",
            'bullish': False,
            'bearish': True
        },
        # Add explanations for other patterns
    }

    for pattern, values in patterns.items():
        df[pattern] = values > 0

     # Historical performance analysis

    pattern_stats = {}
    for pattern in patterns:
        try:
            stats = analyze_pattern_performance(df, pattern)
            # Validate stats structure before using
            if isinstance(stats, dict) and stats.get('count', 0) > 0:
                pattern_stats[pattern] = {
                    'count': int(stats['count']),
                    'win_rate': float(stats['win_rate']),
                    'avg_return': float(stats['avg_return']),
                    'volatility': float(stats['volatility'])
                }
        except Exception as e:
            print(f"Skipping {pattern} due to error: {str(e)}")

    print(f"{header_color}\n🔍 Detected Patterns (Current Week):")
    current_patterns = [p for p in patterns if df[p].iloc[-1]]
    for p in current_patterns:
        pattern_name = p.replace('_', ' ').title()
        print(f"- {pattern_name}")
        stats = pattern_stats.get(p) # Get stats for the pattern
        if stats: # Check if stats exist
            print(f"  Historical Performance: {stats['win_rate']:.0%} win rate | Avg Return: {stats['avg_return']:.2%}")
        explanation = pattern_explanations.get(p)
        if explanation:
            print(f"  Explanation: {explanation['explanation']}")
            if explanation['bullish'] is True:
                print("  Implication: This is generally considered a bullish signal.")
            elif explanation['bearish'] is True:
                print("  Implication: This is generally considered a bearish signal.")
            elif explanation['bullish'] is None:
                print("  Implication: The implications of this pattern can vary depending on the context.")
        else:
            print("  Explanation: No explanation available for this pattern.") # Handle missing explanations

    print(f"PATERNS: {current_patterns}")
    if current_patterns:
        print("Note: This is based on a limited number of historical occurrences and may not be indicative of future results.")
    return current_patterns
    
    
    
def calculate_indicators(df, fast_length=12, slow_length=26, signal_length=9): # Add parameters
    # Trend Indicators
    df[f"SMA_{sma_short_period}"] = ta.trend.sma_indicator(df['Close'], sma_short_period)
    df[f"SMA_{sma_long_period}"] = ta.trend.sma_indicator(df['Close'], sma_long_period)
    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(
        df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
    
    df[f"EMA_{ema_short_period}"] = ta.trend.ema_indicator(df['Close'], ema_short_period)
    df[f"EMA_{ema_mid_period}"] = ta.trend.ema_indicator(df['Close'], ema_mid_period)
    df[f"EMA_{ema_long_period}"] = ta.trend.ema_indicator(df['Close'], ema_long_period)

    # Momentum Indicators
    df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
    macd = ta.trend.MACD(df['Close'], window_fast=fast_length, window_slow=slow_length, window_sign=signal_length) # Use parameters
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
   
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 14).average_true_range()


    return df.dropna()
    
    
def sma_signal(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Bullish only if crossover happened recently
    if (latest[f"SMA_{sma_short_period}"] > latest[f"SMA_{sma_long_period}"]) and (prev[f"SMA_{sma_short_period}"] <= prev[f"SMA_{sma_long_period}"]):
        return 'Buy'
    # Bearish only if crossunder happened recently   
    elif (latest[f'SMA_{sma_short_period}'] < latest[f'SMA_{sma_long_period}']) and (prev[f'SMA_{sma_short_period}'] >= prev[f'SMA_{sma_long_period}']):
        return 'Sell'

    return 'Neutral'

def rsi_signal(rsi):
    if rsi < 25: return 'Buy'
    if rsi > 75: return 'Sell'
    return 'Neutral'

def atr_signal(current_atr, avg_atr):
    return 'Volatile' if current_atr > 1.2 * avg_atr else 'Stable'


def ema_signal(df):
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    # Bullish only if crossover happened recently
    if (latest[f"EMA_{ema_short_period}"] > latest[f"EMA_{ema_mid_period}"]) and (prev[f"EMA_{ema_short_period}"] <= prev[f"EMA_{ema_mid_period}"]):
        return 'Buy'
    # Bearish only if crossunder happened recently   
    elif (latest[f'EMA_{ema_short_period}'] < latest[f'EMA_{ema_mid_period}']) and (prev[f'EMA_{ema_short_period}'] >= prev[f'EMA_{ema_mid_period}']):
        return 'Sell'

    return 'Neutral'


def macd_signal(df):
    latest = df.iloc[-1]
    if latest['MACD'] > latest['MACD_Signal']:
        return 'Buy'
    elif (latest['MACD'] < latest['MACD_Signal']) and (latest['MACD'] < 0):
        return 'Sell'
    
    return 'Neutral'

          
def vwap_signal(df):
    latest = df.iloc[-1]
    if latest['Close'] > latest['VWAP']:
        return 'Buy'
    else: 
        return'Sell'  # VWAP: Bullish if price is above VWAP, Bearish otherwise.
       

def get_recommendations(df):
    latest = df.iloc[-1]
    # Technical Indicators Analysis
    return  {
        'SMA': sma_signal(df),  # SMA: Call if short-term SMA crosses above long-term SMA (bullish crossover), Put if it crosses below (bearish crossover), Neutral otherwise.
        'RSI': rsi_signal(latest['RSI']),  # RSI: Call if RSI is oversold (< 25), Put if overbought (> 75), Neutral otherwise.
        'MACD': macd_signal(df),
        'VWAP': vwap_signal(df),
        'ATR': atr_signal(latest['ATR'], df['ATR'].mean()),  # ATR: Volatile if current ATR is significantly higher than average, Stable otherwise.
        'EMA' : ema_signal(df)
    }

def get_indicators_values(df, fast_length=12, slow_length=26, signal_length=9):
    latest = df.iloc[-1]
    recommendations = {}
    recommendations['indicators'] = get_recommendations(df)
    indicators_values = {
            #'SMA': {f"SMA_{sma_short_period}": format_ind_number(latest[f"SMA_{sma_short_period}"]), f"SMA_{sma_long_period}": format_ind_number(latest[f"SMA_{sma_long_period}"])},
            f"SMA_{sma_short_period}": format_ind_number(latest[f"SMA_{sma_short_period}"]),
            f"SMA_{sma_long_period}": format_ind_number(latest[f"SMA_{sma_long_period}"]),
            'RSI': format_ind_number(latest['RSI']),
            'MACD': format_ind_number(latest['MACD']),
            'VWAP': "${:,.2f}".format(latest['VWAP']), # format_ind_number(latest['VWAP']), 
            'ATR': format_ind_number(latest['ATR']),
            f"EMA_{ema_short_period}": format_ind_number(latest[f"EMA_{ema_short_period}"]),
            f"EMA_{ema_mid_period}": format_ind_number(latest[f"EMA_{ema_mid_period}"]),
            f"EMA_{ema_long_period}": format_ind_number(latest[f"EMA_{ema_long_period}"]),
            
        
    }

    vwap_explanations = f"The VWAP is lower than the current price ${df['Close'].iloc[-1]:.2f}"
    if df['VWAP'].iloc[-1] > df['Close'].iloc[-1]:
        vwap_explanations = f"The VWAP is higher than the current price ${df['Close'].iloc[-1]:.2f}"

    # Store indicator explanations for the report
    indicator_explanations = {
        f"SMA_{sma_short_period}": f"Simple Moving Average ({sma_short_period} periods): Average price over the last {sma_short_period} periods.",
        f"SMA_{sma_long_period}": f"Simple Moving Average ({sma_long_period} periods): Average price over the last {sma_long_period} periods.",
        "VWAP": f"{vwap_explanations}. Volume Weighted Average Price: The average price weighted by volume.  It represents the average price a stock has traded at throughout the day, based on both volume and price. Current price: ${df['Close'].iloc[-1]:.2f}",
        "RSI": "Relative Strength Index (14 periods): Measures the speed and change of price movements.  Values below 30 are often considered oversold, while values above 70 are often considered overbought.",
        "MACD": f"Moving Average Convergence Divergence ({fast_length}, {slow_length}, {signal_length}): A trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.",
        "MACD_Signal": f"MACD Signal Line ({signal_length} periods): A moving average of the MACD, used to identify buy/sell signals.",
        "ATR": "Average True Range (14 periods): Measures the degree of price volatility.",
        f"EMA_{ema_short_period}": f"Exponential Moving Average ({ema_short_period} periods): Average price over the last {ema_short_period} periods.",
        f"EMA_{ema_mid_period}": f"Exponential Moving Average ({ema_mid_period} periods): Average price over the last {ema_mid_period} periods.",
        f"EMA_{ema_long_period}": f"Exponential Moving Average ({ema_long_period} periods): Average price over the last {ema_long_period} periods.",
    }

    recommendations['indicators_values'] = indicators_values    
    recommendations['indicator_explanations'] = indicator_explanations # Store explanations in the recommendation dict.
    return recommendations


def print_recommendations(analysis, show_explanations = True):
    print(f"{header_color}\n{' TECHNICAL SIGNALS VALUES ':-^60}")
    for ind, val in analysis['indicators_values'].items():
        print(f"{label_color}{ind:<10} {data_color}{val}")
        if show_explanations: 
            explanation = analysis['indicator_explanations'].get(ind) # Retrieve explanation
            print(f"{explanation_color}  → {explanation}")
        #print(f"{ind:15} {val}")
    print(f"{header_color}\n{' TECHNICAL SIGNALS RECOMMENDATIONS':-^60}")  
    for indicator, signal in analysis['indicators'].items():
        print(f"{label_color}{indicator:<10} {signal_color[signal]}{signal:<10}") # Include explanation in the report