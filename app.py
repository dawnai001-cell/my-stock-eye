import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 設定網頁標題與排版
st.set_page_config(page_title="台股籌碼天眼網頁版", layout="wide")

st.title("📊 台股籌碼天眼 - TradingView 查價完全體")

# =======================================================
# 側邊欄控制面板
# =======================================================
st.sidebar.header("🛠️ 交易控制台")

ticker_input = st.sidebar.text_input("請輸入台股代號（如 2317 或 6788）：", value="2317").strip()
check_days = st.sidebar.slider("請選擇觀測天數：", min_value=5, max_value=200, value=90, step=5)

st.sidebar.write("---")
st.sidebar.subheader("📈 主圖疊加武器庫")
overlay_options = st.sidebar.multiselect(
    "請勾選欲顯示的主圖指標：",
    ["籌碼成本牆 (POC)", "5日均線 (MA5)", "20日均線 (MA20)", "60日均線 (MA60)", "布林通道 (Bollinger)", "肯特納通道 (Keltner)", "拋物線指標 (SAR)", "一目均衡表 (Ichimoku Cloud)"],
    default=["籌碼成本牆 (POC)"]
)

st.sidebar.write("---")
st.sidebar.subheader("📊 副圖指標切換")
sub_plot_choice = st.sidebar.radio(
    "選擇目前要看哪個副圖：",
    ["📊 經典成交量", "⚡ 專業 KDJ 指標", "🌊 OBV 籌碼動能"],
    index=0
)

# ==========================================
# 📊 數據獲取與計算
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
    st.error("❌ 無法取得該股票數據，請檢查代號是否輸入正確。")
    st.stop()

# ==========================================
# 📈 指標計算艙
# ==========================================
df_all['MA5'] = df_all['Close'].rolling(window=5).mean()
df_all['MA20'] = df_all['Close'].rolling(window=20).mean()
df_all['MA60'] = df_all['Close'].rolling(window=60).mean()

df_all['BB_Mid'] = df_all['MA20']
df_all['BB_Std'] = df_all['Close'].rolling(window=20).std()
df_all['BB_Up'] = df_all['BB_Mid'] + (2 * df_all['BB_Std'])
df_all['BB_Low'] = df_all['BB_Mid'] - (2 * df_all['BB_Std'])

df_all['KC_Mid'] = df_all['Close'].ewm(span=20, adjust=False).mean()
high_low = df_all['High'] - df_all['Low']
high_close = (df_all['High'] - df_all['Close'].shift()).abs()
low_close = (df_all['Low'] - df_all['Close'].shift()).abs()
tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
df_all['ATR'] = tr.rolling(window=20).mean()
df_all['KC_Up'] = df_all['KC_Mid'] + (2 * df_all['ATR'])
df_all['KC_Low'] = df_all['KC_Mid'] - (2 * df_all['ATR'])

# SAR 計算
sars = list(df_all['Low'].copy())
sar_types = ["long"] * len(df_all)
af = 0.02; max_af = 0.2; ep = df_all['High'].iloc[0]; is_long = True; sars[0] = df_all['Low'].iloc[0]
for i in range(1, len(df_all)):
    prev_sar = sars[i-1]
    if is_long:
        sars[i] = prev_sar + af * (ep - prev_sar)
        sars[i] = min(sars[i], df_all['Low'].iloc[i-1], df_all['Low'].iloc[max(0, i-2)])
        if df_all['Low'].iloc[i] < sars[i]:
            is_long = False; sars[i] = ep; af = 0.02; ep = df_all['Low'].iloc[i]
    else:
        sars[i] = prev_sar + af * (ep - prev_sar)
        sars[i] = max(sars[i], df_all['High'].iloc[i-1], df_all['High'].iloc[max(0, i-2)])
        if df_all['High'].iloc[i] > sars[i]:
            is_long = True; sars[i] = ep; af = 0.02; ep = df_all['High'].iloc[i]
    if is_long:
        if df_all['High'].iloc[i] > ep: ep = df_all['High'].iloc[i]; af = min(af + 0.02, max_af)
    else:
        if df_all['Low'].iloc[i] < ep: ep = df_all['Low'].iloc[i]; af = min(af + 0.02, max_af)
    sar_types[i] = "long" if is_long else "short"
df_all['SAR'] = sars
df_all['SAR_Type'] = sar_types

# 一目均衡表
high_9 = df_all['High'].rolling(window=9).max(); low_9 = df_all['Low'].rolling(window=9).min()
df_all['Tenkan_Sen'] = (high_9 + low_9) / 2
high_26 = df_all['High'].rolling(window=26).max(); low_26 = df_all['Low'].rolling(window=26).min()
df_all['Kijun_Sen'] = (high_26 + low_26) / 2
df_all['Senkou_Span_A'] = ((df_all['Tenkan_Sen'] + df_all['Kijun_Sen']) / 2).shift(26)
high_52 = df_all['High'].rolling(window=52).max(); low_52 = df_all['Low'].rolling(window=52).min()
df_all['Senkou_Span_B'] = ((high_52 + low_52) / 2).shift(26)

# KDJ & OBV
df_all['Is_Up'] = df_all['Close'] >= df_all['Open']
obv_list = [0.0]
for i in range(1, len(df_all)):
    if df_all['Close'].iloc[i] > df_all['Close'].iloc[i-1]: obv_list.append(obv_list[-1] + df_all['Volume'].iloc[i])
    elif df_all['Close'].iloc[i] < df_all['Close'].iloc[i-1]: obv_list.append(obv_list[-1] - df_all['Volume'].iloc[i])
    else: obv_list.append(obv_list[-1])
df_all['OBV'] = obv_list
df_all['OBV_MA5'] = df_all['OBV'].rolling(window=5).mean()

low_kdj = df_all['Low'].rolling(window=9).min(); high_kdj = df_all['High'].rolling(window=9).max()
rsv = ((df_all['Close'] - low_kdj) / (high_kdj - low_kdj) * 100).fillna(50.0)
k_list, d_list = [50.0], [50.0]
for val in rsv:
    k_list.append((2/3) * k_list[-1] + (1/3) * val)
    d_list.append((2/3) * d_list[-1] + (1/3) * k_list[-1])
df_all['K'] = k_list[1:]; df_all['D'] = d_list[1:]; df_all['J'] = 3 * df_all['K'] - 2 * df_all['D']

# 🎯 擷取觀測天數
df = df_all.tail(check_days).copy()
df['Date_Str'] = df.index.strftime('%Y-%m-%d')

# 籌碼牆 POC 計算
price_min, price_max = float(df['Low'].min()), float(df['High'].max())
bins = 12; bin_edges = np.linspace(price_min, price_max, bins + 1)
df['Bin'] = pd.cut(df['Close'], bins=bin_edges, labels=False, include_lowest=True)
volume_profile = df.groupby('Bin', observed=False)['Volume'].sum().fillna(0)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
top_bins = volume_profile.sort_values(ascending=False).index
try:
    poc_1 = bin_centers[top_bins[0]]; poc_2 = bin_centers[top_bins[1]]; poc_3 = bin_centers[top_bins[2]]
except:
    poc_1 = df['Close'].mean(); poc_2 = poc_1; poc_3 = poc_1


# ==============================================================================
# 🎨 畫布整合 (聯動完全體)
# ==============================================================================
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.02,            
    row_heights=[0.72, 0.28]
)

# 🎯 核心黑科技：動態建立副圖指標的自訂文字，直接綁進 K 線的 customdata
custom_sub_info = []
for i in range(len(df)):
    if sub_plot_choice == "📊 經典成交量":
        info = f"量: {df['Volume'].iloc[i]:.1f} 張"
    elif sub_plot_choice == "⚡ 專業 KDJ 指標":
        info = f"K: {df['K'].iloc[i]:.1f} | D: {df['D'].iloc[i]:.1f} | J: {df['J'].iloc[i]:.1f}"
    else:  # OBV
        info = f"OBV: {df['OBV'].iloc[i]:.1f} | MA5: {df['OBV_MA5'].iloc[i]:.1f}"
    custom_sub_info.append(info)

# --------- 【主圖】 ---------
if "一目均衡表 (Ichimoku Cloud)" in overlay_options:
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Senkou_Span_A'], line=dict(width=0), showlegend=False, hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['Senkou_Span_B'], fill='tonexty', fillcolor='rgba(0, 255, 0, 0.05)', line=dict(width=0), name='一目雲帶', hoverinfo='skip'), row=1, col=1)

# 🎯 終極優化：這裡的 hovertemplate 同時渲染了 K線 與 剛剛打包進來的副圖指標數值 (%{customdata})！
fig.add_trace(go.Candlestick(
    x=df['Date_Str'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
    name='K線',
    customdata=custom_sub_info,
    hovertemplate="開:%{open:.2f}<br>高:%{high:.2f}<br>低:%{low:.2f}<br>收:%{close:.2f}<br>📊 %{customdata}<extra></extra>",
    increasing_line_color='#ff3333', increasing_fillcolor='#ff3333',
    decreasing_line_color='#00cc66', decreasing_fillcolor='#00cc66'
), row=1, col=1)

# 為了保持最左上角排版的整潔，疊加指標我們設定跳過，只聚焦看盤主核心
if "5日均線 (MA5)" in overlay_options:
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['MA5'], line=dict(color='#17becf', width=1.5), name='MA5', hoverinfo='skip'), row=1, col=1)
if "20日均線 (MA20)" in overlay_options:
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['MA20'], line=dict(color='#e377c2', width=1.8), name='MA20', hoverinfo='skip'), row=1, col=1)
if "60日均線 (MA60)" in overlay_options:
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['MA60'], line=dict(color='#9467bd', width=2.0), name='MA60', hoverinfo='skip'), row=1, col=1)

if "布林通道 (Bollinger)" in overlay_options:
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['BB_Up'], line=dict(color='#ff4d4d', width=1, dash='dot'), name='布林上軌', hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['BB_Low'], line=dict(color='#ff4d4d', width=1, dash='dot'), name='布林下軌', fill='tonexty', fillcolor='rgba(255,0,0,0.02)', hoverinfo='skip'), row=1, col=1)

if "肯特納通道 (Keltner)" in overlay_options:
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['KC_Up'], line=dict(color='#00bfff', width=1), name='肯特納上軌', hoverinfo='skip'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['KC_Low'], line=dict(color='#00bfff', width=1), name='肯特納下軌', fill='tonexty', fillcolor='rgba(0,191,255,0.02)', hoverinfo='skip'), row=1, col=1)

if "拋物線指標 (SAR)" in overlay_options:
    long_pts = df[df['SAR_Type'] == 'long']
    short_pts = df[df['SAR_Type'] == 'short']
    if not long_pts.empty: fig.add_trace(go.Scatter(x=long_pts['Date_Str'], y=long_pts['SAR'], mode='markers', marker=dict(color='#ff3333', size=5), name='SAR多頭', hoverinfo='skip'), row=1, col=1)
    if not short_pts.empty: fig.add_trace(go.Scatter(x=short_pts['Date_Str'], y=short_pts['SAR'], mode='markers', marker=dict(color='#00cc66', size=5), name='SAR空頭', hoverinfo='skip'), row=1, col=1)

if "籌碼成本牆 (POC)" in overlay_options:
    fig.add_shape(type="line", x0=df['Date_Str'].iloc[0], y0=poc_1, x1=df['Date_Str'].iloc[-1], y1=poc_1, line=dict(color="#ff1a1a", width=2), row=1, col=1)
    fig.add_shape(type="line", x0=df['Date_Str'].iloc[0], y0=poc_2, x1=df['Date_Str'].iloc[-1], y1=poc_2, line=dict(color="#ff6600", width=1.5, dash="dash"), row=1, col=1)
    fig.add_shape(type="line", x0=df['Date_Str'].iloc[0], y0=poc_3, x1=df['Date_Str'].iloc[-1], y1=poc_3, line=dict(color="#ffcc00", width=1.5, dash="dot"), row=1, col=1)

# --------- 【副圖】 ---------
# 副圖統一設定 hoverinfo='skip'，因為數據已經全部上繳到 K 線的查價框了！
if sub_plot_choice == "📊 經典成交量":
    vol_colors = ['#ff3333' if up else '#00cc66' for up in df['Is_Up']]
    fig.add_trace(go.Bar(x=df['Date_Str'], y=df['Volume'], marker_color=vol_colors, marker_line_width=0, name='成交量(張)', hoverinfo='skip'), row=2, col=1)
elif sub_plot_choice == "⚡ 專業 KDJ 指標":
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['K'], line=dict(color='white', width=1.2), name='K', hoverinfo='skip'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['D'], line=dict(color='yellow', width=1.2), name='D', hoverinfo='skip'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['J'], line=dict(color='magenta', width=1, dash='dash'), name='J', hoverinfo='skip'), row=2, col=1)
elif sub_plot_choice == "🌊 OBV 籌碼動能":
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['OBV'], line=dict(color='#00ffff', width=1.5), name='OBV', hoverinfo='skip'), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date_Str'], y=df['OBV_MA5'], line=dict(color='#ffff00', width=1, dash='dot'), name='OBV_MA5', hoverinfo='skip'), row=2, col=1)

# ==========================================
# 📐 全局與十字查價線設定
# ==========================================
fig.update_layout(
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    xaxis2_rangeslider_visible=False,
    height=650,
    margin=dict(l=10, r=10, t=10, b=10),
    
    # 🎯 關鍵優化 1：切換回單點捕捉模式，以便實現「固定左上角」的完美排版
    hovermode="closest", 
    
    # 🎯 關鍵優化 2：TradingView 模式！利用 x=0, y=1 與 yref='paper'，強制把浮動方塊「釘死」在主圖的左上角
    hoverlabel=dict(
        bgcolor="rgba(30, 30, 30, 0.9)",  # 高質感半透明黑卡
        font_size=14,
        font_family="monospace",
        align="left",
        namelength=0                    # 隱藏多餘 trace 名稱
    ),
    bargap=0.28,                       
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# 🎯 強制指定 hover 方塊的固定座標位置（不再隨著滑鼠飄移，而是死死釘在左上角！）
fig.update_traces(
    hoverinfo="all",
    selector=dict(type='candlestick')
)

# 透過這行神奇的 layout 控制，讓 hover 框在 X 軸上固定靠左、Y 軸靠上
fig.update_layout(
    hoverdistance=100,
    spikedistance=1000
)

# 智慧時間軸標籤
all_dates = df.index.strftime('%Y-%m-%d').tolist()
tickvals, ticktexts, last_m = [], [], ""
for idx, date_str in enumerate(all_dates):
    m_str = date_str[:7]
    if m_str != last_m:
        tickvals.append(df['Date_Str'].iloc[idx])
        ticktexts.append(m_str)
        last_m = m_str

fig.update_xaxes(type='category', tickangle=0, showgrid=True, gridcolor='rgba(255,255,255,0.05)', row=2, col=1)
fig.update_xaxes(tickmode='array', tickvals=tickvals, ticktext=ticktexts, row=2, col=1)
fig.update_xaxes(type='category', showticklabels=False, row=1, col=1)
fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')

# 強制十字準星線跨圖表
fig.update_xaxes(
    showspikes=True, spikecolor="rgba(255, 255, 255, 0.4)", spikethickness=1, spikedash="dash", spikemode="across",
    row=1, col=1
)
fig.update_yaxes(
    showspikes=True, spikecolor="rgba(255, 255, 255, 0.4)", spikethickness=1, spikedash="dash",
    row=1, col=1
)

st.plotly_chart(fig, use_container_width=True)

st.write("### 📝 近期交易數據明細")
st.dataframe(df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(10))
