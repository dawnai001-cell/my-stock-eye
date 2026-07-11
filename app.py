import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# 設定網頁標題與排版
st.set_page_config(page_title="台股籌碼天眼網頁版", layout="wide")

st.title("📊 台股籌碼天眼 - 雲端網頁版系統")
st.write("利用雲端伺服器進行數據渲染，完美避開手機閃退問題。")

# =======================================================
# 側邊欄控制面板
# =======================================================
st.sidebar.header("🛠️ 交易控制台")

ticker_input = st.sidebar.text_input("請輸入台股代號（如 2317 或 6788）：", value="2317").strip()
check_days = st.sidebar.slider("請選擇觀測天數：", min_value=5, max_value=90, value=20, step=5)

st.sidebar.write("---")
st.sidebar.info("💡 智慧上市櫃識別：\n系統已升級！不論是上市（.TW）還是上櫃（.TWO）的股票代號，現在都能自動偵測並正確抓取數據囉。")

# ==========================================
# 📊 數據獲取與計算（自動判斷上市/上櫃）
# ==========================================
@st.cache_data(ttl=3600)
def load_stock_data(ticker):
    # 策略 1：先嘗試用上市後綴 (.TW) 抓取
    try:
        df_yf = yf.download(f"{ticker}.TW", period="6mo", progress=False)
        if not df_yf.empty and len(df_yf) > 2:
            return process_df(df_yf)
    except:
        pass
        
    # 策略 2：失敗的話，嘗試用上櫃後綴 (.TWO) 抓取
    try:
        df_yf = yf.download(f"{ticker}.TWO", period="6mo", progress=False)
        if not df_yf.empty and len(df_yf) > 2:
            return process_df(df_yf)
    except:
        pass
        
    return None

def process_df(df_yf):
    if isinstance(df_yf.columns, pd.MultiIndex):
        df_yf.columns = df_yf.columns.get_level_values(0)
    df = pd.DataFrame(index=df_yf.index)
    df['Open'] = df_yf['Open'].astype(float)
    df['High'] = df_yf['High'].astype(float)
    df['Low'] = df_yf['Low'].astype(float)
    df['Close'] = df_yf['Close'].astype(float)
    df['Volume'] = df_yf['Volume'].astype(float) / 1000
    return df.dropna()

df_all = load_stock_data(ticker_input)

if df_all is None or len(df_all) == 0:
    st.error("❌ 無法取得該股票數據，請檢查代號是否輸入正確（如：2317 或 6788）。")
    st.stop()

# ==========================================
# 📈 核心指標計算 (KDJ & OBV)
# ==========================================
df_all['Price_Change'] = df_all['Close'].diff()
df_all['Is_Up'] = df_all['Close'] >= df_all['Open']

obv_list = [0.0]
for i in range(1, len(df_all)):
    if df_all['Close'].iloc[i] > df_all['Close'].iloc[i-1]:
        obv_list.append(obv_list[-1] + df_all['Volume'].iloc[i])
    elif df_all['Close'].iloc[i] < df_all['Close'].iloc[i-1]:
        obv_list.append(obv_list[-1] - df_all['Volume'].iloc[i])
    else:
        obv_list.append(obv_list[-1])
df_all['OBV'] = obv_list
df_all['OBV_MA5'] = df_all['OBV'].rolling(window=5).mean()

low_9 = df_all['Low'].rolling(window=9).min()
high_9 = df_all['High'].rolling(window=9).max()
rsv = (df_all['Close'] - low_9) / (high_9 - low_9) * 100
rsv = rsv.fillna(50.0)

k_list, d_list = [50.0], [50.0]
for val in rsv:
    current_k = (2/3) * k_list[-1] + (1/3) * val
    current_d = (2/3) * d_list[-1] + (1/3) * current_k
    k_list.append(current_k)
    d_list.append(current_d)

df_all['K'] = k_list[1:]
df_all['D'] = d_list[1:]
df_all['J'] = 3 * df_all['K'] - 2 * df_all['D']

df = df_all.tail(check_days).copy()

# 籌碼牆 POC 計算
price_min = float(df['Low'].min())
price_max = float(df['High'].max())
bins = 12
bin_edges = np.linspace(price_min, price_max, bins + 1)
df['Bin'] = pd.cut(df['Close'], bins=bin_edges, labels=False, include_lowest=True)
volume_profile = df.groupby('Bin', observed=False)['Volume'].sum().fillna(0)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
top_bins = volume_profile.sort_values(ascending=False).index

try:
    poc_1 = bin_centers[top_bins[0]]
    poc_2 = bin_centers[top_bins[1]] if len(top_bins) > 1 else poc_1
    poc_3 = bin_centers[top_bins[2]] if len(top_bins) > 2 else poc_1
except:
    poc_1 = df['Close'].mean()
    poc_2 = poc_1
    poc_3 = poc_1

df['x_index'] = np.arange(len(df))
date_labels = df.index.strftime('%m-%d').tolist()
step = max(1, len(df) // 6)
tick_indices = df['x_index'].iloc[::step].tolist()
tick_labels = [date_labels[i] for i in tick_indices]

# ==========================================
# 🎨 繪圖與雲端網頁渲染
# ==========================================
plt.style.use('dark_background')

fig_main, ax1 = plt.subplots(figsize=(11, 4.5))
ax1.vlines(df['x_index'], df['Low'], df['High'], color='#aaaaaa', linewidth=1.2)
colors = ['#ff3333' if up else '#00cc66' for up in df['Is_Up']]
ax1.vlines(df['x_index'], df['Open'], df['Close'], color=colors, linewidth=6, alpha=1.0)

ax1.axhline(y=poc_1, color='#ff1a1a', linestyle='-', linewidth=2.5, alpha=0.8, label=f'POC 1 (Max): {poc_1:.1f}')
ax1.axhline(y=poc_2, color='#ff6600', linestyle='--', linewidth=1.5, alpha=0.7, label=f'POC 2: {poc_2:.1f}')
ax1.axhline(y=poc_3, color='#ffcc00', linestyle=':', linewidth=1.5, alpha=0.6, label=f'POC 3: {poc_3:.1f}')

ax1.set_title(f"TW Stock {ticker_input} ({check_days} Days Real Price Chart)", color='yellow', fontsize=14)
ax1.grid(True, color='#222222', alpha=0.5)
ax1.legend(loc='upper left', fontsize=9)
ax1.set_xticks(tick_indices)
ax1.set_xticklabels(tick_labels, rotation=0, fontsize=9)

st.pyplot(fig_main)

st.write("### 📈 副圖指標控制艙")
tab1, tab2, tab3 = st.tabs(["📊 經典成交量", "⚡ 專業 KDJ 指標", "🌊 OBV 籌碼動能"])

with tab1:
    fig_vol, ax_vol = plt.subplots(figsize=(11, 2.5))
    ax_vol.bar(df['x_index'], df['Volume'], color=colors, width=0.6, alpha=0.9)
    ax_vol.set_ylabel('Volume (張)', color='white', fontsize=9)
    ax_vol.grid(True, color='#222222', alpha=0.5)
    ax_vol.set_xticks(tick_indices)
    ax_vol.set_xticklabels(tick_labels, rotation=0, fontsize=9)
    st.pyplot(fig_vol)

with tab2:
    fig_kdj, ax_kdj = plt.subplots(figsize=(11, 2.5))
    ax_kdj.plot(df['x_index'], df['K'], color='white', linewidth=1.5, label='K (9)')
    ax_kdj.plot(df['x_index'], df['D'], color='yellow', linewidth=1.5, label='D (3)')
    ax_kdj.plot(df['x_index'], df['J'], color='magenta', linewidth=1.2, linestyle='--', label='J (3)')
    ax_kdj.axhline(y=80, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax_kdj.axhline(y=20, color='green', linestyle=':', linewidth=1, alpha=0.5)
    ax_kdj.set_ylabel('KDJ Value', color='yellow', fontsize=9)
    ax_kdj.grid(True, color='#222222', alpha=0.5)
    ax_kdj.legend(loc='upper left', fontsize=8)
    ax_kdj.set_xticks(tick_indices)
    ax_kdj.set_xticklabels(tick_labels, rotation=0, fontsize=9)
    st.pyplot(fig_kdj)

with tab3:
    fig_obv, ax_obv = plt.subplots(figsize=(11, 2.5))
    ax_obv.plot(df['x_index'], df['OBV'], color='#00ffff', linewidth=2, label='OBV Flow')
    ax_obv.plot(df['x_index'], df['OBV_MA5'], color='#ffff00', linestyle=':', linewidth=1.5, label='OBV MA5')
    ax_obv.set_ylabel('OBV Volume', color='#00ffff', fontsize=9)
    ax_obv.grid(True, color='#222222', alpha=0.5)
    ax_obv.legend(loc='upper left', fontsize=8)
    ax_obv.set_xticks(tick_indices)
    ax_obv.set_xticklabels(tick_labels, rotation=0, fontsize=9)
    st.pyplot(fig_obv)

st.write("### 📝 近期交易數據明細")
st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10))
