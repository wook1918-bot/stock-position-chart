from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
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
st.caption("종목 검색 차트와 '무릎(파랑) 아래' 저위치 종목 스캐너를 제공합니다.")

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
        url = f"https://ac.stock.naver.com/ac?q={urllib.parse.quote(q)}&target=stock"
        r = requests.get(url, headers=HEADERS, timeout=10).json()
        for item in r.get("items", []):
            if item.get("code") == q:
                return q, item.get("name", q)
        return q, q

    url = f"https://ac.stock.naver.com/ac?q={urllib.parse.quote(q)}&target=stock"
    r = requests.get(url, headers=HEADERS, timeout=10).json()
    items = [
        it for it in r.get("items", [])
        if it.get("nationCode") == "KOR" and re.fullmatch(r"\d{6}", str(it.get("code", "")))
    ]
    if not items:
        return None, None
    for it in items:
        if it.get("name", "").replace(" ", "").lower() == q.replace(" ", "").lower():
            return it["code"], it["name"]
    return items[0]["code"], items[0]["name"]


@st.cache_data(ttl=600)
def fetch_ohlcv(code: str, timeframe: str = "day", years: int = 6) -> pd.DataFrame:
    end_dt = datetime.now().strftime("%Y%m%d")
    start_dt = (datetime.now() - timedelta(days=365 * years)).strftime("%Y%m%d")
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

    hh = df["high"].rolling(12).max()
    ll = df["low"].rolling(12).min()
    fast_k = np.where(hh == ll, 50.0, (df["close"] - ll) / (hh - ll) * 100.0)
    df["slow_k"] = pd.Series(fast_k).rolling(5).mean()
    df["slow_d"] = df["slow_k"].rolling(5).mean()

    n_band = 300 if timeframe == "day" else 100
    n_band = min(n_band, max(60, len(df) // 2))
    log_c = np.log(df["close"].replace(0, np.nan))
    gma = np.exp(log_c.rolling(n_band).mean())
    lstd = log_c.rolling(n_band).std()
    eff_sig = 0.25 + 0.50 * lstd

    for name, z, _, _ in BANDS_SPEC:
        df[name] = gma * np.exp(z * eff_sig)

    return df


@st.cache_data(ttl=3600)
def fetch_top_market_stocks(market: str = "ALL", pages_per_market: int = 2):
    """Fetch top market-cap stocks from Naver mobile stock JSON API (50 items per page)."""
    mkts = ["KOSPI", "KOSDAQ"] if market == "ALL" else [market]
    seen = set()
    stocks = []
    for m_name in mkts:
        for page in range(1, pages_per_market + 1):
            url = f"https://m.stock.naver.com/api/stocks/marketValue/{m_name}?page={page}&pageSize=50"
            data = requests.get(url, headers=HEADERS, timeout=10).json()
            for it in data.get("stocks", []):
                code = str(it.get("itemCode", "")).strip()
                name = str(it.get("stockName", "")).strip()
                if re.fullmatch(r"\d{6}", code) and name and code not in seen:
                    seen.add(code)
                    stocks.append((code, name, m_name))
    return stocks


def analyze_single_stock(item):
    code, name, m_name = item
    try:
        df = fetch_ohlcv(code, "day", years=2)
        if df.empty or len(df) < 310:
            return None
        log_c = np.log(df["close"].replace(0, np.nan))
        w = log_c.tail(300)
        gma = float(np.exp(w.mean()))
        lstd = float(w.std())
        eff_sig = 0.25 + 0.50 * lstd

        cur_p = float(df["close"].iloc[-1])
        prev_p = float(df["close"].iloc[-2])
        chg = (cur_p - prev_p) / prev_p * 100.0

        knee = gma
        purple_hi = gma * np.exp(-0.60 * eff_sig)
        thin_red = gma * np.exp(-0.90 * eff_sig)
        purple_lo = gma * np.exp(-1.28 * eff_sig)

        if cur_p > knee:
            return None

        if cur_p <= purple_lo:
            stage = "연보라(하) 이하"
        elif cur_p <= thin_red:
            stage = "얇은빨강~연보라(하)"
        elif cur_p <= purple_hi:
            stage = "연보라(상)~얇은빨강"
        else:
            stage = "무릎~연보라(상)"

        knee_gap = (cur_p - knee) / knee * 100.0
        return {
            "종목명": name,
            "코드": code,
            "시장": m_name,
            "현재가": int(cur_p),
            "등락률(%)": round(chg, 2),
            "무릎(파랑)": int(knee),
            "무릎 대비(%)": round(knee_gap, 1),
            "현재 구간": stage,
        }
    except Exception:
        return None


if "selected_query" not in st.session_state:
    st.session_state["selected_query"] = "삼성전자"

with st.sidebar:
    st.header("조회 설정")
    query = st.text_input(
        "종목명 또는 종목코드",
        value=st.session_state["selected_query"],
        placeholder="예: 삼성전자, 세방전지, 005930",
    )
    if query != st.session_state["selected_query"]:
        st.session_state["selected_query"] = query
    tf_label = st.radio("봉 구분", ["일봉", "주봉"], horizontal=True)
    timeframe = "day" if tf_label == "일봉" else "week"
    bars = st.slider("표시 봉수", min_value=120, max_value=500, value=300, step=20)
    use_log = st.checkbox("로그 스케일", value=True)
    st.markdown("---")
    st.caption("※ 본 차트와 스캐너는 공개된 블로그 차트 패턴을 역산·재현한 참고용 지표이며 투자 권유가 아닙니다.")

tab_chart, tab_scan = st.tabs(["📈 종목 차트", "🔍 무릎 아래 종목 스캐너"])

with tab_chart:
    q_now = st.session_state["selected_query"].strip()
    if q_now:
        code, name = search_stock(q_now)
        if not code:
            st.error(f"'{q_now}'에 해당하는 국내 상장 종목을 찾지 못했습니다.")
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

                cur_p = last["close"]
                zone = "연보라(하) 이하"
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

                for n, col in [(40, "#cc66cc"), (50, "#e6a800"), (60, "#22aa55"), (120, "#666699")]:
                    fig.add_trace(
                        go.Scatter(
                            x=view["date_str"], y=view[f"sma_{n}"],
                            mode="lines", line=dict(color=col, width=1),
                            name=f"MA{n}", opacity=0.8,
                        ),
                        row=1, col=1,
                    )

                for b_name, _, col, lw in BANDS_SPEC:
                    fig.add_trace(
                        go.Scatter(
                            x=view["date_str"], y=view[b_name],
                            mode="lines", line=dict(color=col, width=lw),
                            name=b_name,
                        ),
                        row=1, col=1,
                    )

                fig.add_trace(
                    go.Scatter(x=view["date_str"], y=view["slow_k"], mode="lines", line=dict(color="#ff6600", width=1.2), name="Slow %K(12,5)"),
                    row=2, col=1,
                )
                fig.add_trace(
                    go.Scatter(x=view["date_str"], y=view["slow_d"], mode="lines", line=dict(color="#7b2cbf", width=1.2), name="Slow %D(5)"),
                    row=2, col=1,
                )

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

with tab_scan:
    st.subheader("🔍 현재가 ≤ 무릎(파랑) 종목 스캐너 (일봉 기준)")
    st.caption("시가총액 상위 종목 중 현재 주가가 무릎(파랑) 선 아래로 내려온 종목을 찾아 무릎 대비 괴리율 순으로 정렬합니다.")

    col_m, col_n, col_btn = st.columns([2, 2, 1.5])
    with col_m:
        market_opt = st.selectbox("대상 시장", ["ALL (코스피+코스닥)", "KOSPI", "KOSDAQ"])
    with col_n:
        pages_opt = st.selectbox("스캔 범위 (시장별 시총 상위)", [1, 2, 3], index=1, format_func=lambda p: f"상위 {p * 50}종목씩")
    with col_btn:
        st.write("")
        st.write("")
        run_scan = st.button("스캔 시작", type="primary", use_container_width=True)

    market_key = market_opt.split()[0]

    if run_scan:
        with st.spinner("종목 목록과 일봉 데이터를 불러와 계산하는 중..."):
            stock_list = fetch_top_market_stocks(market_key, pages_opt)
            found = []
            with ThreadPoolExecutor(max_workers=10) as ex:
                futures = [ex.submit(analyze_single_stock, s) for s in stock_list]
                for f in as_completed(futures):
                    res = f.result()
                    if res is not None:
                        found.append(res)
            if found:
                res_df = pd.DataFrame(found).sort_values("무릎 대비(%)", ascending=True).reset_index(drop=True)
                st.session_state["scan_results"] = res_df
                st.session_state["scan_total"] = len(stock_list)
            else:
                st.session_state["scan_results"] = pd.DataFrame()
                st.session_state["scan_total"] = len(stock_list)

    if "scan_results" in st.session_state:
        res_df = st.session_state["scan_results"]
        total_cnt = st.session_state.get("scan_total", 0)
        if res_df.empty:
            st.info(f"스캔한 {total_cnt}개 종목 중 무릎(파랑) 아래에 있는 종목이 없습니다.")
        else:
            st.success(f"총 {total_cnt}개 종목 중 **{len(res_df)}개**가 무릎(파랑) 아래에 있습니다. (왼쪽 '종목명 또는 종목코드'에 입력하면 바로 차트를 볼 수 있어요!)")
            st.dataframe(res_df, use_container_width=True, hide_index=True)

            pick = st.selectbox("바로 차트로 볼 종목 선택", res_df["종목명"].tolist())
            if st.button("선택 종목 차트 열기"):
                st.session_state["selected_query"] = pick
                st.rerun()
