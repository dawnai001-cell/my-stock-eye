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
check_days = st.sidebar.slider("請選擇觀測天數：", min_value=5, max_value=200, value=90, step=5)

st.sidebar.write("---")
st.sidebar.subheader("📈 主圖疊加武器庫")
# 💡 這裡擴充了選單，加入了肯特納通道！
overlay_options = st.sidebar.multiselect(
    "請勾選欲顯示的主圖指標：",
    ["5日均線 (MA5)", "20日均線 (MA20)", "60日均線 (MA60)", "布林通道 (Bollinger)", "肯特納通道 (Keltner)"],
    default=[]
)

st.sidebar.write("---")
st.sidebar.info("💡 頂級看盤體驗：\n現在你可以將「布林」與「肯特納」分開單獨看、同時看，或者都不看。看不懂精髓時就一個一個點開，慢慢培養敏銳度！")

# ==========================================
# 📊 數據獲取與計算（自動判斷上市/上櫃）
# ==========================================
@st.cache_data(ttl=3600)
def load_stock_data(ticker):
    try:
        df_yf = yf.download(f"{ticker}.TW", period="1y", progress=False)
        if not df_yf.empty and len(df_yf) > 2:
            return process_df(df_yf)
    except:
        pass
        
    try:
        df_yf = yf.download(f"{ticker}.TWO", period="1y", progress=False)
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
# 📈 核心與擴充指標計算
# ==========================================
# 1. 均線計算
df_all['MA5'] = df_all['Close'].rolling(window=5).mean()
df_all['MA20'] = df_all['Close'].rolling(window=20).mean()
df_all['MA60'] = df_all['Close'].rolling(window=60).mean()

# 2. 布林通道計算 (標準月線 20 MA +/- 2倍標準差)
df_all['BB_Mid'] = df_all['MA20']
df_all['BB_Std'] = df_all['Close'].rolling(window=20).std()
df_all['BB_Up'] = df_all['BB_Mid'] + (2 * df_all['BB_Std'])
df_all['BB_Low'] = df_all['BB_Mid'] - (2 * df_all['BB_Std'])

# 3. 肯特納通道計算 (20 EMA 中軌 +/- 2倍 ATR 真實波動幅度)
df_all['KC_Mid'] = df_all['Close'].ewm(span=20, adjust=False).mean()
# 計算 ATR (真實活動幅度)
high_low = df_all['High'] - df_all['Low']
high_close = (df_all['High'] - df_all['Close'].shift()).abs()
low_close = (df_all['Low'] - df_all['Close'].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df_all['ATR'] = tr.rolling(window=20).mean()
df_all['KC_Up'] = df_all['KC_Mid'] + (2 * df_all['ATR'])
df_all['KC_Low'] = df_all['KC_Mid'] - (2 * df_all['ATR'])

# 4. KDJ & OBV 計算
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

step = max(1, len(df) // 8)
tick_indices = df['x_index'].iloc[::step].tolist()
tick_labels = [date_labels[i] for i in tick_indices]

# ==========================================
# 🎨 繪圖與動態指標疊加
# ==========================================
plt.style.use('dark_background')

k_width = max(1.2, 7.0 - (check_days / 35.0))
line_width = max(0.6, 1.5 - (check_days / 200.0))

fig_main, ax1 = plt.subplots(figsize=(11, 4.5))

# 1. 繪製 K 線基本引線與實體
ax1.vlines(df['x_index'], df['Low'], df['High'], color='#aaaaaa', linewidth=line_width)
colors = ['#ff3333' if up else '#00cc66' for up in df['Is_Up']]
ax1.vlines(df['x_index'], df['Open'], df['Close'], color=colors, linewidth=k_width, alpha=1.0)

# 2. 動態疊加均線系統 (MA)
if "5日均線 (MA5)" in overlay_options:
    ax1.plot(df['x_index'], df['MA5'], color='#17becf', linewidth=1.5, label='MA5', alpha=0.8)
if "20日均線 (MA20)" in overlay_options:
    ax1.plot(df['x_index'], df['MA20'], color='#e377c2', linewidth=1.8, label='MA20', alpha=0.8)
if "60日均線 (MA60)" in overlay_options:
    ax1.plot(df['x_index'], df['MA60'], color='#9467bd', linewidth=2.0, label='MA60', alpha=0.8)

# 3. 動態疊加布林通道 (BB) — 紅色調點虛線
if "布林通道 (Bollinger)" in overlay_options:
    ax1.plot(df['x_index'], df['BB_Up'], color='#ff4d4d', linestyle=':', linewidth=1.3, label='BB Upper')
    ax1.plot(df['x_index'], df['BB_Low'], color='#ff4d4d', linestyle=':', linewidth=1.3, label='BB Lower')
    ax1.fill_between(df['x_index'], df['BB_Up'], df['BB_Low'], color='#ff0000', alpha=0.02)

# 4. 動態疊加肯特納通道 (KC) — 藍色調線實線
if "肯特納通道 (Keltner)" in overlay_options:
    ax1.plot(df['x_index'], df['KC_Up'], color='#00bfff', linestyle='-', linewidth=1.2, label='KC Upper')
    ax1.plot(df['x_index'], df['KC_Low'], color='#00bfff', linestyle='-', linewidth=1.2, label='KC Lower')
    ax1.fill_between(df['x_index'], df['KC_Up'], df['KC_Low'], color='#00bfff', alpha=0.02)

# 5. 繪製籌碼成本牆（POC）
ax1.axhline(y=poc_1, color='#ff1a1a', linestyle='-', linewidth=2.5, alpha=0.8, label=f'POC 1: {poc_1:.1f}')
ax1.axhline(y=poc_2, color='#ff6600', linestyle='--', linewidth=1.5, alpha=0.7, label=f'POC 2: {poc_2:.1f}')
ax1.axhline(y=poc_3, color='#ffcc00', linestyle=':', linewidth=1.5, alpha=0.6, label=f'POC 3: {poc_3:.1f}')

ax1.set_title(f"TW Stock {ticker_input} ({check_days} Days Real Price Chart)", color='yellow', fontsize=14)
ax1.grid(True, color='#222222', alpha=0.5)
ax1.legend(loc='upper left', fontsize=9)
ax1.set_xticks(tick_indices)
ax1.set_xticklabels(tick_labels, rotation=0, fontsize=9)

st.pyplot(fig_main)

# ==========================================
# 📈 副圖指標控制艙
# ==========================================
st.write("### 📈 副圖指標控制艙")
tab1, tab2, tab3 = st.tabs(["📊 經典成交量", "⚡ 專業 KDJ 指標", "🌊 OBV 籌碼動能"])

with tab1:
    fig_vol, ax_vol = plt.subplots(figsize=(11, 2.5))
    v_width = max(0.2, 0.7 - (check_days / 400.0))
    ax_vol.bar(df['x_index'], df['Volume'], color=colors, width=v_width, alpha=0.9)
    ax_vol.set_ylabel('Volume (張)', color='white', fontsize=9)
    ax_vol.grid(True, color='#222222', alpha=0.5)
    ax_vol.set_xticks(tick_indices)
    ax_vol.set_xticklabels(tick_labels, rotation=0, fontsize=9)
    st.pyplot(fig_vol)

with tab2:
    fig_kdj, ax_kdj = plt.subplots(figsize=(11, 2.5))
    ax_kdj.plot(df['x_index'], df['K'], color='white', linewidth=1.2, label='K (9)')
    ax_kdj.plot(df['x_index'], df['D'], color='yellow', linewidth=1.2, label='D (3)')
    ax_kdj.plot(df['x_index'], df['J'], color='magenta', linewidth=1.0, linestyle='--', label='J (3)')
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
    ax_obv.plot(df['x_index'], df['OBV'], color='#00ffff', linewidth=1.5, label='OBV Flow')
    ax_obv.plot(df['x_index'], df['OBV_MA5'], color='#ffff00', linestyle=':', linewidth=1.0, label='OBV MA5')
    ax_obv.set_ylabel('OBV Volume', color='#00ffff', fontsize=9)
    ax_obv.grid(True, color='#222222', alpha=0.5)
    ax_obv.legend(loc='upper left', fontsize=8)
    ax_obv.set_xticks(tick_indices)
    ax_obv.set_xticklabels(tick_labels, rotation=0, fontsize=9)
    st.pyplot(fig_obv)

st.write("### 📝 近期交易數據明細")
st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10))
