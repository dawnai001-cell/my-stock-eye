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
st.sidebar.info("💡 判讀小提示：\n1. 主圖 POC 1 是大庄家核心成本防線。\n2. 已經移除休市日空白，圖表完全緊密連續呈現。")

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
    
    df_all = pd.DataFrame({
        'Open': prices - 0.3,
        'High': prices + 0.8,
        'Low': prices - 0.7,
        'Close': prices,
        'Volume': np.random.randint(10000, 50000, 100).astype(float)
    }, index=pd.date_range(start="2026-01-01", periods=100, freq='B'))

# 根據使用者拉桿的天數切取數據
df = df_all.tail(check_days).copy()

# ==========================================
# 核心計算
# ==========================================
df['Price_Change'] = df['Close'].diff()
if len(df) > 0:
    df.iloc[0, df.columns.get_loc('Price_Change')] = df['Close'].iloc[0] - df['Open'].iloc[0]
df['Is_Up'] = df['Price_Change'] >= 0

obv_series = pd.Series(index=df.index, dtype='float64').fillna(0.0)
current_obv = 0.0
for i in range(1, len(df)):
    if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
        current_obv += df['Volume'].iloc[i]
    elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
        current_obv -= df['Volume'].iloc[i]
    obv_series.iloc[i] = current_obv

df['OBV'] = obv_series
df['OBV_MA5'] = df['OBV'].rolling(window=5).mean()

# 籌碼牆計算
price_min = float(df['Low'].min())
price_max = float(df['High'].max())
if price_min == price_max or np.isnan(price_min) or np.isnan(price_max):
    price_min = 50.0
    price_max = 100.0

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

# 💡 建立連續的流水號索引，徹底抽離休市日的物理空白 gap
df['x_index'] = np.arange(len(df))
# 格式化日期標籤 (只要年-月-日)
date_labels = df.index.strftime('%Y-%m-%d').tolist()

# 根據觀測天數自動調整日期標籤的顯示密度，避免文字重疊擠爆
step = max(1, len(df) // 6)
tick_indices = df['x_index'].iloc[::step].tolist()
tick_labels = [date_labels[i] for i in tick_indices]

# ==========================================
# 🎨 繪圖與雲端網頁渲染 (無縫連續排版)
# ==========================================
plt.style.use('dark_background')

# --- 👑 1. 上層主圖：純淨 K 線與大庄家成本牆 ---
fig_main, ax1 = plt.subplots(figsize=(11, 4.5))
k_width = 0.6  # 改用相對寬度，不受時間軸物理距離影響

ax1.vlines(df['x_index'], df['Low'], df['High'], color='#777777', linewidth=1)
ax1.vlines(df['x_index'][df['Is_Up']], df['Open'][df['Is_Up']], df['Close'][df['Is_Up']], color='#ff3333', linewidth=k_width*10, label='漲')
ax1.vlines(df['x_index'][~df['Is_Up']], df['Open'][~df['Is_Up']], df['Close'][~df['Is_Up']], color='#00cc66', linewidth=k_width*10, label='跌')

ax1.axhline(y=poc_1, color='#ff1a1a', linestyle='-', linewidth=2.5, alpha=0.8, label=f'POC 1 (Max): {poc_1:.1f}')
ax1.axhline(y=poc_2, color='#ff6600', linestyle='--', linewidth=1.5, alpha=0.7, label=f'POC 2: {poc_2:.1f}')
ax1.axhline(y=poc_3, color='#ffcc00', linestyle=':', linewidth=1.5, alpha=0.6, label=f'POC 3: {poc_3:.1f}')

ax1.set_title(f"TW Stock {ticker_input} ({check_days} Days Analysis)", color='yellow', fontsize=14)
ax1.grid(True, color='#222222', alpha=0.5)
ax1.legend(loc='upper left', fontsize=9)

# 強制換上我們的連續日期標籤
ax1.set_xticks(tick_indices)
ax1.set_xticklabels(tick_labels, rotation=15, fontsize=9)

st.pyplot(fig_main)

# --- 🎛️ 2. 下層副圖：三竹式分頁切換艙 ---
st.write("### 📈 副圖指標控制艙")
tab1, tab2 = st.tabs(["📊 經典成交量", "🌊 OBV 籌碼動能"])

# 【分頁一：經典成交量】
with tab1:
    fig_vol, ax_vol = plt.subplots(figsize=(11, 2.5))
    colors = ['#ff3333' if up else '#00cc66' for up in df['Is_Up']]
    ax_vol.bar(df['x_index'], df['Volume'], color=colors, width=0.6, alpha=0.9)
    ax_vol.set_ylabel('Volume (張)', color='white', fontsize=9)
    ax_vol.grid(True, color='#222222', alpha=0.5)
    
    ax_vol.set_xticks(tick_indices)
    ax_vol.set_xticklabels(tick_labels, rotation=15, fontsize=9)
    st.pyplot(fig_vol)

# 【分頁二：OBV籌碼動能】
with tab2:
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
