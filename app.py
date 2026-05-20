import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
import os

from plotly.subplots import make_subplots

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import load_model

from textblob import TextBlob

st.set_page_config(
    page_title="Hybrid AI Stock Predictor",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

st.title("🤖 HYBRID AI STOCK PREDICTOR")

stocks = {
    "RELIANCE":"RELIANCE.NS",
    "TCS":"TCS.NS",
    "INFY":"INFY.NS",
    "HDFCBANK":"HDFCBANK.NS",
    "ICICIBANK":"ICICIBANK.NS",
    "SBIN":"SBIN.NS",
    "ITC":"ITC.NS",
    "LT":"LT.NS",
    "AXISBANK":"AXISBANK.NS",
    "KOTAKBANK":"KOTAKBANK.NS",
    "HCLTECH":"HCLTECH.NS",
    "WIPRO":"WIPRO.NS",
    "TECHM":"TECHM.NS",
    "SUNPHARMA":"SUNPHARMA.NS",
    "CIPLA":"CIPLA.NS",
    "DRREDDY":"DRREDDY.NS",
    "TATASTEEL":"TATASTEEL.NS",
    "JSWSTEEL":"JSWSTEEL.NS"
}

st.sidebar.title("📊 Navigation")

search_stock = st.sidebar.text_input(
    "🔍 Search NSE Stock"
)

filtered = stocks.copy()

if search_stock:

    stock_name = search_stock.upper()

    filtered[stock_name] = stock_name + ".NS"

stock = st.sidebar.selectbox(
    "Select Stock",
    list(filtered.keys())
)

period = st.sidebar.selectbox(
    "Select Period",
    ["1y","2y","5y"]
)

future_days = st.sidebar.selectbox(
    "Future Prediction Days",
    [1,3,7,15,30]
)

@st.cache_data

def load_data(symbol, period):

    df = yf.download(
        symbol,
        period=period,
        progress=False,
        auto_adjust=False
    )

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)

    return df

df = load_data(
    filtered[stock],
    period
)

if df.empty:

    st.error("Stock Data Not Found")

    st.stop()

close = df["Close"]

df["MA20"] = close.rolling(20).mean()

df["MA50"] = close.rolling(50).mean()

delta = close.diff()

gain = delta.clip(lower=0)

loss = -delta.clip(upper=0)

rs = gain.rolling(14).mean() / (
    loss.rolling(14).mean() + 1e-10
)

df["RSI"] = 100 - (100 / (1 + rs))

df["Volatility"] = (
    df["Close"].rolling(20).std()
)

url = f"https://news.google.com/rss/search?q={stock}+stock"

news = feedparser.parse(url)

sentiments = []

headlines = []

for item in news.entries[:10]:

    title = item.title

    polarity = TextBlob(
        title
    ).sentiment.polarity

    sentiments.append(polarity)

    headlines.append(title)

if len(sentiments) > 0:

    avg_sentiment = np.mean(sentiments)

else:

    avg_sentiment = 0

df["NewsSentiment"] = avg_sentiment

current_price = float(
    df["Close"].iloc[-1]
)

prev_price = float(
    df["Close"].iloc[-2]
)

change = (
    (current_price - prev_price)
    / prev_price
) * 100

volume = int(
    df["Volume"].iloc[-1]
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Current Price",
    round(current_price, 2)
)

c2.metric(
    "Day Change %",
    round(change, 2)
)

c3.metric(
    "Volume",
    volume
)

c4.metric(
    "News Sentiment",
    round(avg_sentiment, 2)
)

fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    row_heights=[0.6,0.2,0.2]
)

fig.add_trace(go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    name=stock
),row=1,col=1)

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["MA20"],
    name="MA20"
),row=1,col=1)

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["MA50"],
    name="MA50"
),row=1,col=1)

fig.add_trace(go.Bar(
    x=df.index,
    y=df["Volume"],
    name="Volume"
),row=2,col=1)

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["RSI"],
    name="RSI"
),row=3,col=1)

fig.update_layout(
    template="plotly_dark",
    height=850,
    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

features = df[[
    "Close",
    "Volume",
    "MA20",
    "MA50",
    "RSI",
    "Volatility",
    "NewsSentiment"
]].copy()

features.dropna(inplace=True)

if len(features) < 100:

    st.error(
        "Not enough data for training"
    )

    st.stop()

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(
    features
)

training_data_len = int(
    np.ceil(len(scaled_data) * 0.8)
)

train_data = scaled_data[
    0:int(training_data_len), :
]

x_train = []

y_train = []

for i in range(60, len(train_data)):

    x_train.append(
        train_data[i-60:i]
    )

    y_train.append(
        train_data[i,0]
    )

x_train = np.array(x_train)

y_train = np.array(y_train)

if len(x_train) == 0:

    st.error(
        "Not enough stock data for training"
    )

    st.stop()

model_name = f"{stock}_hybrid_model.keras"

if os.path.exists(model_name):

    model = load_model(model_name)

else:

    model = Sequential(
        name="hybrid_stock_model"
    )

    model.add(LSTM(
        64,
        return_sequences=True,
        input_shape=(
            x_train.shape[1],
            x_train.shape[2]
        ),
        name="lstm_layer_1"
    ))

    model.add(LSTM(
        64,
        return_sequences=False,
        name="lstm_layer_2"
    ))

    model.add(Dense(
        32,
        name="dense_layer_1"
    ))

    model.add(Dense(
        1,
        name="output_layer"
    ))

    model.compile(
        optimizer="adam",
        loss="mean_squared_error"
    )

    with st.spinner(
        "Training Hybrid AI Model..."
    ):

        model.fit(
            x_train,
            y_train,
            batch_size=16,
            epochs=5,
            verbose=0
        )

    model.save(model_name)

test_data = scaled_data[
    training_data_len - 60:, :
]

x_test = []

y_test = scaled_data[
    training_data_len:, 0
]

for i in range(60, len(test_data)):

    x_test.append(
        test_data[i-60:i]
    )

x_test = np.array(x_test)

predictions = model.predict(
    x_test,
    verbose=0
)

dummy = np.zeros(
    (
        predictions.shape[0],
        scaled_data.shape[1]
    )
)

dummy[:,0] = predictions[:,0]

predictions = scaler.inverse_transform(
    dummy
)[:,0]

actual_dummy = np.zeros(
    (
        len(y_test),
        scaled_data.shape[1]
    )
)

actual_dummy[:,0] = y_test

actual_prices = scaler.inverse_transform(
    actual_dummy
)[:,0]

rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predictions
    )
)

mae = mean_absolute_error(
    actual_prices,
    predictions
)

future_predictions = []

last_sequence = scaled_data[-60:]

temp_input = list(last_sequence)

for i in range(future_days):

    x_input = np.array(
        temp_input[-60:]
    )

    x_input = x_input.reshape(
        (
            1,
            x_input.shape[0],
            x_input.shape[1]
        )
    )

    pred = model.predict(
        x_input,
        verbose=0
    )

    next_row = temp_input[-1].copy()

    next_row[0] = pred[0][0]

    temp_input.append(next_row)

    future_predictions.append(
        pred[0][0]
    )

future_dummy = np.zeros(
    (
        len(future_predictions),
        scaled_data.shape[1]
    )
)

future_dummy[:,0] = future_predictions

future_predictions = scaler.inverse_transform(
    future_dummy
)[:,0]

predicted_price = float(
    future_predictions[-1]
)

future_change = (
    (predicted_price - current_price)
    / current_price
) * 100

trend = (
    "Bullish 📈"
    if predicted_price > current_price
    else "Bearish 📉"
)

st.subheader("🤖 Hybrid AI Prediction")

m1, m2, m3, m4 = st.columns(4)

m1.metric(
    "Predicted Price",
    round(predicted_price, 2)
)

m2.metric(
    "Expected Change %",
    round(future_change, 2)
)

m3.metric(
    "RMSE",
    round(rmse, 2)
)

m4.metric(
    "MAE",
    round(mae, 2)
)

st.success(f"AI Trend: {trend}")

future_df = pd.DataFrame({
    "Day": list(range(1, future_days + 1)),
    "Predicted Price": future_predictions
})

st.subheader("📅 Future Forecast")

st.dataframe(
    future_df,
    use_container_width=True
)

fig2 = go.Figure()

fig2.add_trace(go.Scatter(
    x=future_df["Day"],
    y=future_df["Predicted Price"],
    mode="lines+markers",
    name="Forecast"
))

fig2.update_layout(
    template="plotly_dark",
    height=500,
    title="Future AI Prediction"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

comparison = pd.DataFrame({
    "Actual": actual_prices,
    "Predicted": predictions
})

fig3 = go.Figure()

fig3.add_trace(go.Scatter(
    y=comparison["Actual"],
    name="Actual"
))

fig3.add_trace(go.Scatter(
    y=comparison["Predicted"],
    name="Predicted"
))

fig3.update_layout(
    template="plotly_dark",
    height=600,
    title="Actual vs AI Prediction"
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

st.subheader("📰 Latest News Sentiment")

news_df = pd.DataFrame({
    "Headline": headlines,
    "Sentiment": sentiments
})

st.dataframe(
    news_df,
    use_container_width=True
)

csv = df.to_csv().encode()

st.download_button(
    "⬇ Download CSV",
    csv,
    f"{stock}.csv",
    "text/csv"
)