import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# 設定網頁標題與排版
st.set_page_config(page_title="台股籌碼天眼網頁版", layout="wide")

st.title("📊 台股籌碼天眼 - 雲端網頁版系統")
st.write("利用雲端伺服器進行數據渲染，完美避開手機閃退問題。")

# =======================================================
# 側邊欄控制面板
# =======================================================
st.sidebar.header("🛠️ 交易控制台")

ticker_input = st.sidebar.text_input("請輸入台股代號：", value="2356").strip()
check_days = st.sidebar.slider("請選擇觀測天數：", min_value=5, max_value=90, value=20, step=5)

st.sidebar.write("---")
st.sidebar.info("💡 判讀小提示：\n1. 主圖 K 棒已修正為正統開高低收比例（有長短實體與上下影線）。\n2. 週末模擬數據時間軸已同步校正至最新交易日（2026-07-10）。")

# ==========================================
# 網路數據抓取
# ==========================================
def fetch_data(ticker):
    try:
        url = f"https://www.twse.com.tw/exchangeReport/STOCK_DAY?response=json&stockNo={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=8)
        raw_data = response.json()
        
        if raw_data['stat'] == 'OK':
            raw_fields = raw_data['data']
            actual_cols_num = len(raw_fields[0]) if raw_fields else 9
            columns = [f'col_{id}' for id in range(actual_cols_num)]
            columns[0], columns[1], columns[3], columns[4], columns[5], columns[6] = 'Date', 'Volume', 'Open', 'High', 'Low', 'Close'
            
            df_raw = pd.DataFrame(raw_fields, columns=columns)
            
            def convert_taiwan_date(date_str):
                try:
                    parts = date_str.split('/')
                    year = int(parts[0]) + 1911
                    return f"{year}-{parts[1]}-{parts[2]}"
                except:
                    return None
            
            converted_dates = df_raw['Date'].apply(convert_taiwan_date)
            df = pd.DataFrame(index=pd.to_datetime(converted_dates, errors='coerce'))
            
            df['Open'] = df_raw['Open'].str.replace(',', '').astype(float)
            df['High'] = df_raw['High'].str.replace(',', '').astype(float)
            df['Low'] = df_raw['Low'].str.replace(',', '').astype(float)
            df['Close'] = df_raw['Close'].str.replace(',', '').astype(float)
            df['Volume'] = df_raw['Volume'].str.replace(',', '').astype(float) / 1000
            
            if len(df.dropna()) > 0:
                return df.dropna(), None
            else:
                return None, "資料解析後為空值"
        else:
            return None, "證交所目前非交易時間或代號錯誤"
    except Exception as e:
        return None, str(e)

# 執行抓取
df_all, error_msg = fetch_data(ticker_input)

# 強制進入週末模擬數據（如果證交所沒回傳正確資料）
if error_msg is not None or df_all is None:
    st.info("💡 證交所伺服器目前休息中，系統已自動為您接軌至『週末高仿真模擬數據』。")
    np.random.seed(42)
    base_price = 68.0
    prices = np.random.normal(0, 1.2, 100).cumsum() + base_price
    
    # 💡 關鍵校正：將模擬數據的結束時間精準對齊到 2026-07-10，向前推算 100 個交易日
    df_all = pd.DataFrame({
        'Open': prices - 0.3,
        'High': prices + 0.8,
        'Low': prices - 0.7,
        'Close': prices,
        'Volume': np.random.randint(10000, 50000, 100).astype(float)
    }, index=pd.date_range(end="2026-07-10", periods=100, freq='B'))

# ==========================================
# 📊 核心指標計算 (包含 KDJ 與 OBV)
# ==========================================
df_all['Price_Change'] = df_all['Close'].diff()
df_all['Is_Up'] = df_all['Price_Change'] >= 0

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

# 計算經典 KDJ (9, 3, 3)
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

# 根據使用者拉桿的天數切取最終顯示數據
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

# 建立連續流水號索引
df['x_index'] = np.arange(len(df))
date_labels = df.index.strftime('%Y-%m-%d').tolist()
step = max(1, len(df) // 6)
tick_indices = df['x_index'].iloc[::step].tolist()
tick_labels = [date_labels[i] for i in tick_indices]

# ==========================================
# 🎨 繪圖與雲端網頁渲染
# ==========================================
plt.style.use('dark_background')

# --- 👑 1. 上層主圖：正統有長短、有影線的 K 線圖 ---
fig_main, ax1 = plt.subplots(figsize=(11, 4.5))
k_width = 0.6

ax1.vlines(df['x_index'], df['Low'], df['High'], color='#999999', linewidth=1.2)
colors = ['#ff3333' if up else '#00cc66' for up in df['Is_Up']]
ax1.vlines(df['x_index'], df['Open'], df['Close'], color=colors, linewidth=5, alpha=1.0)

ax1.axhline(y=poc_1, color='#ff1a1a', linestyle='-', linewidth=2.5, alpha=0.8, label=f'POC 1 (Max): {poc_1:.1f}')
ax1.axhline(y=poc_2, color='#ff6600', linestyle='--', linewidth=1.5, alpha=0.7, label=f'POC 2: {poc_2:.1f}')
ax1.axhline(y=poc_3, color='#ffcc00', linestyle=':', linewidth=1.5, alpha=0.6, label=f'POC 3: {poc_3:.1f}')

ax1.set_title(f"TW Stock {ticker_input} ({check_days} Days Analysis)", color='yellow', fontsize=14)
ax1.grid(True, color='#222222', alpha=0.5)
ax1.legend(loc='upper left', fontsize=9)
ax1.set_xticks(tick_indices)
ax1.set_xticklabels(tick_labels, rotation=15, fontsize=9)

st.pyplot(fig_main)

# --- 🎛️ 2. 下層副圖：三竹式分頁切換艙 ---
st.write("### 📈 副圖指標控制艙")
tab1, tab2, tab3 = st.tabs(["📊 經典成交量", "⚡ 專業 KDJ 指標", "🌊 OBV 籌碼動能"])

with tab1:
    fig_vol, ax_vol = plt.subplots(figsize=(11, 2.5))
    ax_vol.bar(df['x_index'], df['Volume'], color=colors, width=0.6, alpha=0.9)
    ax_vol.set_ylabel('Volume (張)', color='white', fontsize=9)
    ax_vol.grid(True, color='#222222', alpha=0.5)
    ax_vol.set_xticks(tick_indices)
    ax_vol.set_xticklabels(tick_labels, rotation=15, fontsize=9)
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
    ax_kdj.set_xticklabels(tick_labels, rotation=15, fontsize=9)
    st.pyplot(fig_kdj)

with tab3:
    fig_obv, ax_obv = plt.subplots(figsize=(11, 2.5))
    ax_obv.plot(df['x_index'], df['OBV'], color='#00ffff', linewidth=2, label='OBV Flow')
    ax_obv.plot(df['x_index'], df['OBV_MA5'], color='#ffff00', linestyle=':', linewidth=1.5, label='OBV MA5')
    ax_obv.set_ylabel('OBV Volume', color='#00ffff', fontsize=9)
    ax_obv.grid(True, color='#222222', alpha=0.5)
    ax_obv.legend(loc='upper left', fontsize=8)
    ax_obv.set_xticks(tick_indices)
    ax_obv.set_xticklabels(tick_labels, rotation=15, fontsize=9)
    st.pyplot(fig_obv)

# 交易數據明細
st.write("### 📝 近期交易數據明細")
st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10))
