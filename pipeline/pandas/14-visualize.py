#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df.drop(columns=['Weighted_Price'], inplace=True)

df = df.rename(columns={'Timestamp': 'Date'})

df['Date'] = pd.to_datetime(df['Date'], unit='s')

df = df.set_index('Date')

df['Close'].fillna(method='ffill', inplace=True)

df['High'].fillna(df['Close'], inplace=True)
df['Low'].fillna(df['Close'], inplace=True)
df['Open'].fillna(df['Close'], inplace=True)

df['Volume_(BTC)'].fillna(0, inplace=True)
df['Volume_(Currency)'].fillna(0, inplace=True)

df = df['2017':].resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Open'], label='Open', color='skyblue')
plt.plot(df.index, df['High'], label='High', color='orange')
plt.plot(df.index, df['Low'], label='Low', color='green')
plt.plot(df.index, df['Close'], label='Close', color='red')
plt.plot(df.index, df['Volume_(BTC)'], label='Volume_(BTC)', color='purple')
plt.plot(df.index, df['Volume_(Currency)'], label='Volume_(Currency)', color='brown')
plt.xlabel('Date')
plt.legend()
plt.show()