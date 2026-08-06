import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ===== 頁面設定 =====
st.set_page_config(page_title="公版決策器", layout="wide")
st.title("📊 公版決策器 - MJ 策略回測")

# ===== 側邊欄參數 =====
st.sidebar.header("📌 參數設定")

ticker = st.sidebar.text_input("股票代碼", "2330.TW")
start_date = st.sidebar.date_input("開始日期", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("結束日期", datetime(2024, 12, 31))

st.sidebar.subheader("📈 MACD 參數")
macd_fast = st.sidebar.slider("快線", 6, 20, 12)
macd_slow = st.sidebar.slider("慢線", 18, 36, 26)
macd_signal = st.sidebar.slider("訊號線", 5, 13, 9)

st.sidebar.subheader("📉 J 值參數")
j_period = st.sidebar.slider("J 值週期", 5, 15, 9)
j_threshold = st.sidebar.slider("J 值門檻", 25, 75, 50)

st.sidebar.subheader("⚙️ 條件組合")
use_hist = st.sidebar.checkbox("使用柱狀體條件", value=True)
use_dif_slope = st.sidebar.checkbox("使用 DIF 斜率條件", value=False)

# ===== 載入資料 =====
@st.cache_data
def load_data(ticker, start, end):
    raw = yf.download(ticker, start=start, end=end, progress=False)
    data = pd.DataFrame(index=raw.index)
    data['Open'] = raw['Open'].values
    data['High'] = raw['High'].values
    data['Low'] = raw['Low'].values
    data['Close'] = raw['Close'].values
    data['Volume'] = raw['Volume'].values
    return data

data = load_data(ticker, start_date, end_date)

if len(data) == 0:
    st.error("❌ 無法下載資料，請檢查股票代碼")
    st.stop()

# ===== 計算指標 =====
def calc_macd(close, fast, slow, signal):
    close = np.array(close)
    def ema(data, span):
        alpha = 2 / (span + 1)
        result = np.zeros_like(data)
        result[0] = data[0]
        for i in range(1, len(data)):
            result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd = ema_fast - ema_slow
    signal_line = ema(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist

def calc_j(high, low, close, period):
    high = np.array(high)
    low = np.array(low)
    close = np.array(close)
    n = len(close)
    rsv = np.zeros(n)
    k = np.zeros(n)
    d = np.zeros(n)
    j = np.zeros(n)
    for i in range(period-1, n):
        low_min = low[i-period+1:i+1].min()
        high_max = high[i-period+1:i+1].max()
        if high_max != low_min:
            rsv[i] = (close[i] - low_min) / (high_max - low_min) * 100
        else:
            rsv[i] = 50
        if i == period-1:
            k[i] = rsv[i]
        else:
            k[i] = (2/3) * k[i-1] + (1/3) * rsv[i]
        if i == period-1:
            d[i] = k[i]
        else:
            d[i] = (2/3) * d[i-1] + (1/3) * k[i]
        j[i] = 3 * k[i] - 2 * d[i]
    return j

# ===== 計算 =====
macd, signal_line, hist = calc_macd(data['Close'].values, macd_fast, macd_slow, macd_signal)
j_values = calc_j(data['High'].values, data['Low'].values, data['Close'].values, j_period)

# 計算斜率
dif_slope = np.zeros_like(macd)
dif_slope[1:] = macd[1:] - macd[:-1]

# ===== 生成訊號 =====
buy_signals = []
sell_signals = []
positions = []
position = False

for i in range(1, len(data)):
    j_prev = j_values[i-1]
    j_now = j_values[i]
    hist_now = hist[i]
    dif_slope_now = dif_slope[i]
    
    buy_cond = [j_prev < j_threshold <= j_now]
    if use_hist:
        buy_cond.append(hist_now > 0)
    if use_dif_slope:
        buy_cond.append(dif_slope_now > 0)
    
    sell_cond = [j_prev > j_threshold >= j_now]
    if use_hist:
        sell_cond.append(hist_now < 0)
    if use_dif_slope:
        sell_cond.append(dif_slope_now < 0)
    
    if all(buy_cond) and not position:
        buy_signals.append(i)
        position = True
    elif all(sell_cond) and position:
        sell_signals.append(i)
        position = False

# ===== 計算績效 =====
returns = data['Close'].pct_change()
cumulative_return = (1 + returns).cumprod() - 1

buy_prices = data['Close'].iloc[buy_signals].values if buy_signals else []
sell_prices = data['Close'].iloc[sell_signals].values if sell_signals else []

trade_returns = []
if len(buy_signals) > 0 and len(sell_signals) > 0:
    for b, s in zip(buy_signals, sell_signals):
        if s > b:
            trade_returns.append((data['Close'].iloc[s] - data['Close'].iloc[b]) / data['Close'].iloc[b])

sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
total_return = cumulative_return.iloc[-1] * 100
max_drawdown = (cumulative_return - cumulative_return.cummax()).min() * 100
win_rate = (np.array(trade_returns) > 0).mean() * 100 if trade_returns else 0

# ===== 顯示結果 =====
col1, col2, col3, col4 = st.columns(4)
col1.metric("夏普比率", f"{sharpe:.3f}", delta="✅ 正" if sharpe > 0 else "⚠️ 負")
col2.metric("總報酬", f"{total_return:.2f}%")
col3.metric("最大回撤", f"{max_drawdown:.2f}%")
col4.metric("勝率", f"{win_rate:.1f}%", delta=f"交易 {len(buy_signals)} 次")

# ===== 繪圖 =====
fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=('股價與買賣點', 'MACD', 'J 值'),
                    vertical_spacing=0.08)

# 股價
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='收盤價', line=dict(color='blue')), row=1, col=1)
if buy_signals:
    fig.add_trace(go.Scatter(x=data.index[buy_signals], y=data['Close'].iloc[buy_signals], 
                             mode='markers', name='買進', marker=dict(color='green', size=12, symbol='triangle-up')), row=1, col=1)
if sell_signals:
    fig.add_trace(go.Scatter(x=data.index[sell_signals], y=data['Close'].iloc[sell_signals], 
                             mode='markers', name='賣出', marker=dict(color='red', size=12, symbol='triangle-down')), row=1, col=1)

# MACD
fig.add_trace(go.Scatter(x=data.index, y=macd, name='MACD', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=data.index, y=signal_line, name='訊號線', line=dict(color='orange')), row=2, col=1)

# 柱狀體
colors = ['red' if h < 0 else 'green' for h in hist]
fig.add_trace(go.Bar(x=data.index, y=hist, name='柱狀體', marker_color=colors), row=2, col=1)

# J 值
fig.add_trace(go.Scatter(x=data.index, y=j_values, name='J 值', line=dict(color='purple')), row=3, col=1)
fig.add_hline(y=j_threshold, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=50, line_dash="dash", line_color="gray", row=3, col=1)

fig.update_layout(height=800, showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# ===== 交易記錄 =====
with st.expander("📋 交易記錄"):
    trade_df = pd.DataFrame({
        '買進日期': data.index[buy_signals] if buy_signals else [],
        '買進價格': buy_prices if buy_prices else [],
        '賣出日期': data.index[sell_signals[:len(buy_signals)]] if sell_signals else [],
        '賣出價格': sell_prices[:len(buy_signals)] if sell_prices else [],
        '報酬%': np.array(trade_returns) * 100 if trade_returns else []
    })
    if not trade_df.empty:
        st.dataframe(trade_df)
    else:
        st.write("無交易記錄")
