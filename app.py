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
st.caption("종목 검색 차트(신호 마커 포함)와 '무릎(파랑) 아래' 저위치 종목 스캐너를 제공합니다.")

HEADERS = {"User-Agent": "Mozilla/5.0"}

BANDS_SPEC = [
    ("바닥(빨강)", -1.10, "#d62828", 1.6),
    ("무릎(파랑)", 0.00, "#0000ff", 2.2),
    ("허벅지(주황)", 0.82, "#ff6600", 2.2),
    ("허리(검정)", 1.66, "#000000", 2.6),
    ("어깨(초록)", 2.25, "#008f32", 2.2),
    ("상투(빨강)", 2.85, "#ff0000", 2.2),
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

    n_band = 260 if timeframe == "day" else 60
    n_vol = 250 if timeframe == "day" else 52
    ann_factor = np.sqrt(250) if timeframe == "day" else np.sqrt(52)

    log_c = np.log(df["close"].replace(0, np.nan))
    gma = np.exp(log_c.ewm(span=n_band, adjust=False).mean())
    ann_vol = (log_c.diff().rolling(n_vol).std() * ann_factor).fillna(0.55)
    eff_sig = np.clip(0.30 + 0.25 * ann_vol, 0.35, 0.58)

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
        df = fetch_ohlcv(code, "day", years=3)
        if df.empty or len(df) < 320:
            return None
        df = compute_indicators(df, "day")
        last = df.iloc[-1]
        prev = df.iloc[-2]

        cur_p = float(last["close"])
        prev_p = float(prev["close"])
        chg = (cur_p - prev_p) / prev_p * 100.0

        knee = float(last["무릎(파랑)"])
        if np.isnan(knee) or cur_p > knee:
            return None

        if cur_p <= float(last["바닥(빨강)"]):
            stage = "바닥 이하"
        else:
            stage = "바닥~무릎"

        # Check 1-year (250 trading days) high and which band it reached at that peak
        recent = df.tail(250)
        peak_idx = recent["high"].idxmax()
        peak_row = recent.loc[peak_idx]
        peak_p = float(peak_row["high"])
        drawdown_pct = (cur_p - peak_p) / peak_p * 100.0

        peak_zone = "무릎 아래"
        for b_name in ["무릎(파랑)", "허벅지(주황)", "허리(검정)", "어깨(초록)", "상투(빨강)"]:
            b_val = float(peak_row[b_name])
            if not np.isnan(b_val) and peak_p >= b_val:
                peak_zone = f"{b_name.split('(')[0]} 이상"

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
            "1년 고점 위치": peak_zone,
            "고점 대비(%)": round(drawdown_pct, 1),
        }
    except Exception:
        return None


if "selected_query" not in st.session_state:
    st.session_state["selected_query"] = "신세계"


def set_selected_stock(stock_name: str):
    st.session_state["selected_query"] = stock_name


with st.sidebar:
    st.header("조회 설정")
    query = st.text_input(
        "종목명 또는 종목코드",
        key="selected_query",
        placeholder="예: 삼성전자, 신세계, 004170",
    )
    tf_label = st.radio("봉 구분", ["일봉", "주봉"], horizontal=True)
    timeframe = "day" if tf_label == "일봉" else "week"
    bars = st.slider("표시 봉수", min_value=120, max_value=500, value=300, step=20)
    use_log = st.checkbox("로그 스케일", value=True)
    allow_zoom = st.checkbox("차트 드래그/확대 켜기 (모바일 스크롤 시 끄기 권장)", value=False)
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
                zone = "바닥(빨강) 이하"
                for b_name, _, _, _ in BANDS_SPEC:
                    if pd.notna(last[b_name]) and cur_p >= last[b_name]:
                        zone = f"{b_name} 이상"

                c1, c2, c3 = st.columns(3)
                c1.metric(f"{name} ({code})", f"{int(cur_p):,}원", f"{chg:+.2f}%")
                c2.metric("현재 추정 위치", zone)
                c3.metric("기준일", last["date_str"])

                st.markdown("[⬇️ 아래 단계별 가격표로 바로 이동](#price-table)")

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

                for b_name, _, col, lw in BANDS_SPEC:
                    fig.add_trace(
                        go.Scatter(
                            x=view["date_str"], y=view[b_name],
                            mode="lines", line=dict(color=col, width=lw),
                            name=b_name,
                        ),
                        row=1, col=1,
                    )

                # Signal markers: Knee breakout & Top (Sangtoo) overheat
                knee_cross = (view["close"].shift(1) < view["무릎(파랑)"].shift(1)) & (view["close"] >= view["무릎(파랑)"])
                if knee_cross.any():
                    pts = view[knee_cross]
                    fig.add_trace(
                        go.Scatter(
                            x=pts["date_str"], y=pts["low"] * 0.95,
                            mode="markers",
                            marker=dict(symbol="triangle-up", size=10, color="#ff0000"),
                            name="무릎통과신호",
                        ),
                        row=1, col=1,
                    )

                top_cross = (view["high"] >= view["상투(빨강)"])
                if top_cross.any():
                    pts = view[top_cross]
                    fig.add_trace(
                        go.Scatter(
                            x=pts["date_str"], y=pts["high"] * 1.05,
                            mode="markers",
                            marker=dict(symbol="triangle-down", size=10, color="#0000ff"),
                            name="상투신호",
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
                    height=680,
                    xaxis_rangeslider_visible=False,
                    hovermode="x unified",
                    dragmode="zoom" if allow_zoom else False,
                    legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
                    margin=dict(l=15, r=65, t=40, b=20),
                )
                if not allow_zoom:
                    fig.update_xaxes(fixedrange=True)
                    fig.update_yaxes(fixedrange=True)
                if use_log:
                    y_min = float(min(view["low"].min(), view["바닥(빨강)"].min())) * 0.9
                    y_max = float(max(view["high"].max(), view["상투(빨강)"].max())) * 1.1
                    cand_ticks = []
                    for exp in range(2, 8):
                        base = 10 ** exp
                        for mult in [1, 1.5, 2, 3, 4, 5, 7, 8]:
                            v = int(base * mult)
                            if y_min <= v <= y_max:
                                cand_ticks.append(v)
                    fig.update_yaxes(
                        type="log",
                        tickvals=cand_ticks,
                        ticktext=[f"{v:,}" for v in cand_ticks],
                        showgrid=True,
                        gridcolor="#e5e5e5",
                        row=1, col=1,
                    )
                else:
                    fig.update_yaxes(
                        tickformat=",",
                        showgrid=True,
                        gridcolor="#e5e5e5",
                        row=1, col=1,
                    )
                fig.add_hline(
                    y=cur_p, line_dash="dot", line_color="#d62828", line_width=1,
                    annotation_text=f" {int(cur_p):,}원",
                    annotation_position="right",
                    annotation_font_color="#d62828",
                    row=1, col=1,
                )
                fig.update_yaxes(range=[0, 100], row=2, col=1)
                fig.update_xaxes(type="category", nticks=8, row=3, col=1)

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"scrollZoom": allow_zoom, "displayModeBar": allow_zoom},
                )

                st.markdown('<div id="price-table"></div>', unsafe_allow_html=True)
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
    st.caption("시가총액 상위 종목 중 현재 주가가 무릎(파랑) 선 아래에 있는 종목을 찾고, 최근 1년 고점 위치와 고점 대비 하락률까지 함께 표시합니다.")

    col_m, col_n, col_f, col_btn = st.columns([1.6, 1.6, 2.0, 1.3])
    with col_m:
        market_opt = st.selectbox("대상 시장", ["ALL (코스피+코스닥)", "KOSPI", "KOSDAQ"])
    with col_n:
        pages_opt = st.selectbox("스캔 범위", [1, 2, 3], index=1, format_func=lambda p: f"시장별 상위 {p * 50}종목")
    with col_f:
        filter_opt = st.selectbox("필터 조건", ["무릎 아래 전체", "허리·어깨 이상 찍고 무릎 아래 하락"])
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
        if not res_df.empty and filter_opt == "허리·어깨 이상 찍고 무릎 아래 하락":
            res_df = res_df[res_df["1년 고점 위치"].isin(["허리 이상", "어깨 이상", "상투 이상"])].reset_index(drop=True)

        if res_df.empty:
            st.info(f"스캔한 {total_cnt}개 종목 중 해당 조건에 맞는 종목이 없습니다.")
        else:
            st.success(f"총 {total_cnt}개 종목 중 조건에 맞는 **{len(res_df)}개** 종목을 찾았습니다.")
            st.dataframe(res_df, use_container_width=True, hide_index=True)

            pick = st.selectbox("바로 차트로 볼 종목 선택", res_df["종목명"].tolist())
            st.button("선택 종목 차트 열기", on_click=set_selected_stock, args=(pick,))
