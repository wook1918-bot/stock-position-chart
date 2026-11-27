import math
import re
import urllib.parse
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

st.set_page_config(
    page_title="주가위치분석시스템 차트",
    page_icon="📊",
    layout="wide",
)

st.title("📊 주가위치분석시스템 차트 (KRX)")
st.caption("종목명 또는 6자리 종목코드를 입력하면 일봉/주봉 로그 차트와 발목~상투 위치 밴드를 그려줍니다.")

HEADERS = {"User-Agent": "Mozilla/5.0"}

BANDS_SPEC = [
    ("연보라(하)", -1.28, "#cc99ff", 1.2),
    ("얇은빨강", -0.90, "#ff5555", 1.2),
    ("연보라(상)", -0.60, "#cc99ff", 1.2),
    ("무릎(파랑)", 0.00, "#0000ff", 2.2),
    ("허벅지(주황)", 0.82, "#ff6600", 2.2),
    ("허리(검정)", 1.66, "#000000", 2.6),
    ("어깨(초록)", 2.40, "#008f32", 2.2),
    ("상투(빨강)", 3.06, "#ff0000", 2.2),
]


@st.cache_data(ttl=3600)
def search_stock(query: str):
    q = query.strip()
    if re.fullmatch(r"\d{6}", q):
        # Fetch name from Naver item page or autocomplete
        url = f"https://ac.stock.naver.com/ac?q={urllib.parse.quote(q)}&target=stock"
        r = requests.get(url, headers=HEADERS, timeout=10).json()
        for item in r.get("items", []):
            if item.get("code") == q:
                return q, item.get("name", q)
        return q, q

    url = f"https://ac.stock.naver.com/ac?q={urllib.parse.quote(q)}&target=stock"
    r = requests.get(url, headers=HEADERS, timeout=10).json()
    items = [it for it in r.get("items", []) if it.get("nationCode") == "KOR" and re.fullmatch(r"\d{6}", str(it.get("code", "")))]
    if not items:
        return None, None
    # Exact name match first
    for it in items:
        if it.get("name", "").replace(" ", "").lower() == q.replace(" ", "").lower():
            return it["code"], it["name"]
    return items[0]["code"], items[0]["name"]


@st.cache_data(ttl=600)
def fetch_ohlcv(code: str, timeframe: str = "day") -> pd.DataFrame:
    # Fetch ~5 years of daily data (for weekly or daily long MA=300 / 120)
    from datetime import datetime, timedelta
    end_dt = datetime.now().strftime("%Y%m%d")
    start_dt = (datetime.now() - timedelta(days=365 * 6)).strftime("%Y%m%d")
    url = (
        f"https://api.finance.naver.com/siseJson.naver?symbol={code}"
        f"&requestType=1&startTime={start_dt}&endTime={end_dt}&timeframe={timeframe}"
    )
    resp = requests.get(url, headers=HEADERS, timeout=15)
    rows = re.findall(
        r'\["(\d{8})",\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)',
        resp.text,
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def compute_indicators(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    df = df.copy()
    for n in [5, 20, 40, 50, 60, 120]:
        df[f"sma_{n}"] = df["close"].rolling(n).mean()
        df[f"vsma_{n}"] = df["volume"].rolling(n).mean()

    # Slow Stochastics (12, 5, 5)
    hh = df["high"].rolling(12).max()
    ll = df["low"].rolling(12).min()
    fast_k = np.where(hh == ll, 50.0, (df["close"] - ll) / (hh - ll) * 100.0)
    df["slow_k"] = pd.Series(fast_k).rolling(5).mean()
    df["slow_d"] = df["slow_k"].rolling(5).mean()

    # Position bands (long geometric MA + volatility band)
    n_band = 300 if timeframe == "day" else 100
    n_band = min(n_band, max(60, len(df) // 2))
    log_c = np.log(df["close"].replace(0, np.nan))
    gma = np.exp(log_c.rolling(n_band).mean())
    lstd = log_c.rolling(n_band).std()
    eff_sig = 0.25 + 0.50 * lstd

    for name, z, _, _ in BANDS_SPEC:
        df[name] = gma * np.exp(z * eff_sig)

    return df


with st.sidebar:
    st.header("조회 설정")
    query = st.text_input("종목명 또는 종목코드", value="삼성전자", placeholder="예: 삼성전자, 세방전지, 005930")
    tf_label = st.radio("봉 구분", ["일봉", "주봉"], horizontal=True)
    timeframe = "day" if tf_label == "일봉" else "week"
    bars = st.slider("표시 봉수", min_value=120, max_value=500, value=300, step=20)
    use_log = st.checkbox("로그 스케일", value=True)
    st.markdown("---")
    st.caption("※ 본 차트는 공개된 블로그 차트 이미지들의 패턴을 역산·재현한 참고용 지표입니다.")

if query.strip():
    code, name = search_stock(query)
    if not code:
        st.error(f"'{query}'에 해당하는 국내 상장 종목을 찾지 못했습니다.")
    else:
        raw_df = fetch_ohlcv(code, timeframe)
        if raw_df.empty or len(raw_df) < 60:
            st.warning(f"{name}({code})의 주가 데이터가 충분하지 않습니다.")
        else:
            df = compute_indicators(raw_df, timeframe)
            view = df.tail(bars).reset_index(drop=True)
            last = view.iloc[-1]
            prev = view.iloc[-2] if len(view) >= 2 else last
            chg = (last["close"] - prev["close"]) / prev["close"] * 100.0

            # Determine current zone
            cur_p = last["close"]
            zone = "발목 이하"
            for b_name, _, _, _ in BANDS_SPEC:
                if pd.notna(last[b_name]) and cur_p >= last[b_name]:
                    zone = f"{b_name} 이상"

            c1, c2, c3 = st.columns(3)
            c1.metric(f"{name} ({code})", f"{int(cur_p):,}원", f"{chg:+.2f}%")
            c2.metric("현재 추정 위치", zone)
            c3.metric("기준일", last["date_str"])

            fig = make_subplots(
                rows=3, cols=1, shared_xaxes=True,
                vertical_spacing=0.03,
                row_heights=[0.62, 0.19, 0.19],
            )

            # Candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=view["date_str"],
                    open=view["open"], high=view["high"],
                    low=view["low"], close=view["close"],
                    increasing_line_color="#e61919", increasing_fillcolor="#e61919",
                    decreasing_line_color="#1955e6", decreasing_fillcolor="#1955e6",
                    name="주가",
                ),
                row=1, col=1,
            )

            # Moving Averages 40, 50, 60, 120
            for n, col in [(40, "#cc66cc"), (50, "#e6a800"), (60, "#22aa55"), (120, "#666699")]:
                fig.add_trace(
                    go.Scatter(
                        x=view["date_str"], y=view[f"sma_{n}"],
                        mode="lines", line=dict(color=col, width=1),
                        name=f"MA{n}", opacity=0.8,
                    ),
                    row=1, col=1,
                )

            # Position Bands
            for b_name, _, col, lw in BANDS_SPEC:
                fig.add_trace(
                    go.Scatter(
                        x=view["date_str"], y=view[b_name],
                        mode="lines", line=dict(color=col, width=lw),
                        name=b_name,
                    ),
                    row=1, col=1,
                )

            # Slow Stochastics
            fig.add_trace(
                go.Scatter(x=view["date_str"], y=view["slow_k"], mode="lines", line=dict(color="#ff6600", width=1.2), name="Slow %K(12,5)"),
                row=2, col=1,
            )
            fig.add_trace(
                go.Scatter(x=view["date_str"], y=view["slow_d"], mode="lines", line=dict(color="#7b2cbf", width=1.2), name="Slow %D(5)"),
                row=2, col=1,
            )

            # Volume
            v_colors = ["#e61919" if c >= o else "#1955e6" for o, c in zip(view["open"], view["close"])]
            fig.add_trace(
                go.Bar(x=view["date_str"], y=view["volume"], marker_color=v_colors, name="거래량", opacity=0.8),
                row=3, col=1,
            )

            fig.update_layout(
                height=780,
                xaxis_rangeslider_visible=False,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                margin=dict(l=10, r=20, t=40, b=20),
            )
            if use_log:
                fig.update_yaxes(type="log", row=1, col=1)
            fig.update_yaxes(range=[0, 100], row=2, col=1)
            fig.update_xaxes(type="category", nticks=8, row=3, col=1)

            st.plotly_chart(fig, use_container_width=True)

            # Band table
            st.subheader("단계별 가격표 (최신 기준)")
            table_rows = []
            for b_name, _, _, _ in reversed(BANDS_SPEC):
                val = last[b_name]
                if pd.notna(val):
                    diff_pct = (val - cur_p) / cur_p * 100.0
                    table_rows.append({
                        "구간": b_name,
                        "가격": f"{int(val):,}원",
                        "현재가 대비": f"{diff_pct:+.1f}%",
                    })
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
