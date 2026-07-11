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
# 側邊欄控制面板 (Dropdown / Input / Slider)
# =======================================================
st.sidebar.header("🛠️ 交易控制台")

ticker_input = st.sidebar.text_input("請輸入台股代號：", value="2356").strip()
check_days = st.sidebar.slider("請選擇觀測天數：", min_value=5, max_value=90, value=20, step=5)

st.sidebar.write("---")
st.sidebar.info("💡 判讀小提示：\n1. POC 1 是大庄家核心成本防線。\n2. 當 OBV (青線) 跌破 MA5 (黃線)，代表短線資金加速撤退。")

# ==========================================
# 網路數據抓取
# ==========================================
# @st.cache_data(ttl=3600)  # 快取機制：一小時內重複查詢不用重新連證交所
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
            return df.dropna(), None
        else:
            return None, "證交所代號不存在或目前非交易時間"
    except Exception as e:
        return None, str(e)

# 執行抓取
df_all, error_msg = fetch_data(ticker_input)

if error_msg is not None:
    st.error(f"❌ 數據獲取失敗: {error_msg}")
    st.warning("⚠️ 目前可能為非交易時段或證交所伺服器維護中，已自動為您無縫接軌至『週末高仿真模擬數據』進行系統測試。")
    
    # 修正：確保模擬數據生成的內容完美符合新版核心邏輯
    np.random.seed(42)
    base_price = 68.0
    prices = np.random.normal(0, 1.5, 100).cumsum() + base_price
    
    df_all = pd.DataFrame({
        'Open': prices - 0.5,
        'High': prices + 1.2,
        'Low': prices - 1.0,
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

p_min, p_max = df['Close'].min(), df['Close'].max()
o_min, o_max = obv_series.min(), obv_series.max()
if o_max != o_min:
    df['OBV_Scaled'] = p_min + (obv_series - o_min) * (p_max - p_min) / (o_max - o_min)
else:
    df['OBV_Scaled'] = df['Close']
df['OBV_MA5_Scaled'] = df['OBV_Scaled'].rolling(window=5).mean()

# 籌碼牆計算 (防禦邊界處理：確保即使範圍為0也能順利分組)
price_min, price_max = df['Low'].min(), df['High'].max()
if price_min == price_max:
    price_min -= 1.0
    price_max += 1.0

bins = 12
df['Bin'] = pd.cut(df['Close'], bins=np.linspace(price_min, price_max, bins+1), labels=False, include_lowest=True)
volume_profile = df.groupby('Bin', observed=False)['Volume'].sum().fillna(0)
bin_centers = (np.linspace(price_min, price_max, bins+1)[:-1] + np.linspace(price_min, price_max, bins+1)[1:]) / 2
top_bins = volume_profile.sort_values(ascending=False).index

poc_1 = bin_centers[top_bins[0]]
poc_2 = bin_centers[top_bins[1]] if len(top_bins) > 1 else poc_1
poc_3 = bin_centers[top_bins[2]] if len(top_bins) > 2 else poc_1

# ==========================================
# 繪圖與雲端網頁渲染
# ==========================================
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(11, 5.5))

k_width = 8 if check_days <= 10 else 4
ax.vlines(df.index, df['Low'], df['High'], color='#777777', linewidth=1)
ax.vlines(df.index[df['Is_Up']], df['Open'][df['Is_Up']], df['Close'][df['Is_Up']], color='#ff3333', linewidth=k_width, label='漲')
ax.vlines(df.index[~df['Is_Up']], df['Open'][~df['Is_Up']], df['Close'][~df['Is_Up']], color='#00cc66', linewidth=k_width, label='跌')

ax.plot(df.index, df['OBV_Scaled'], color='#00ffff', linewidth=2, label='OBV Flow')
ax.plot(df.index, df['OBV_MA5_Scaled'], color='#ffff00', linestyle=':', linewidth=1.5, label='OBV MA5')

ax.axhline(y=poc_1, color='#ff1a1a', linestyle='-', linewidth=2.5, alpha=0.8, label=f'POC 1 (Max): {poc_1:.1f}')
ax.axhline(y=poc_2, color='#ff6600', linestyle='--', linewidth=1.5, alpha=0.7, label=f'POC 2: {poc_2:.1f}')
ax.axhline(y=poc_3, color='#ffcc00', linestyle=':', linewidth=1.5, alpha=0.6, label=f'POC 3: {poc_3:.1f}')

ax.set_title(f"TW Stock {ticker_input} ({check_days} Days Analysis)", color='yellow', fontsize=14)
ax.grid(True, color='#222222', alpha=0.5)

handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left', fontsize=10)

st.pyplot(fig)

st.write("### 📝 近期交易數據明細")
st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10))
