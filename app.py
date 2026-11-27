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
st.caption(
    "종목·지수 검색 차트(신호 마커 포함)와 '무릎(파랑) 아래' 저위치 종목 스캐너를 제공합니다. "
    "기준선은 고가 120일·650일 이동평균을 혼합(0.3/0.7)한 값이며, 단계는 댕기왕자 블로그 차트 대조로 "
    "역산한 비율(바닥 0.20 ~ 상투 4.16)을 씁니다."
)

HEADERS = {"User-Agent": "Mozilla/5.0"}

# ── 단계 정의 ───────────────────────────────────────────────────────────────
# 댕기왕자 블로그 차트 34개 지점(12종목) 대조로 역산한 계수.
# 중심선(무릎)에서 변동성 s배 만큼 exp(z*s) 만큼 떨어진 위치.
BAND_Z = [
    ("바닥(빨강)", -1.62, "#d62828", 1.6, "solid"),
    ("발목(보라)", -1.10, "#7a3fa8", 1.4, "dot"),
    ("무릎(파랑)", 0.00, "#0000ff", 2.4, "solid"),
    ("허벅지(주황)", 0.82, "#ff6600", 2.2, "solid"),
    ("허리(검정)", 1.66, "#000000", 2.4, "solid"),
    ("어깨(초록)", 2.25, "#008f32", 2.2, "solid"),
    ("상투(빨강)", 2.85, "#ff0000", 2.4, "solid"),
]
BAND_NAMES = [b[0] for b in BAND_Z]
PRICE_BANDS = ["바닥(빨강)", "발목(보라)", "무릎(파랑)", "허벅지(주황)", "허리(검정)", "어깨(초록)", "상투(빨강)"]

BASE_SCALE = 1.05          # 기준선 보정 계수
BASE_SHORT, BASE_LONG = 120, 650
BASE_W = 0.30              # 단기 이동평균 가중치
SIG_C0, SIG_C1 = 0.44, 0.30
SIG_MIN, SIG_MAX = 0.25, 0.75

# ── 차트 조회 기간 ──────────────────────────────────────────────────────────
# 1년 단위로 표시 봉수를 정확히 계산한다(일봉 250·주봉 52 기준).
PERIOD_LABELS = ["1년", "2년", "3년", "5년"]
PERIOD_BARS = {
    "day": {"1년": 250, "2년": 500, "3년": 750, "5년": 1250},
    "week": {"1년": 52, "2년": 104, "3년": 156, "5년": 260},
}

INDEX_ALIASES = {
    "코스피": ("KOSPI", "코스피 지수"),
    "kospi": ("KOSPI", "코스피 지수"),
    "코스피지수": ("KOSPI", "코스피 지수"),
    "종합주가지수": ("KOSPI", "코스피 지수"),
    "코스닥": ("KOSDAQ", "코스닥 지수"),
    "kosdaq": ("KOSDAQ", "코스닥 지수"),
    "코스닥지수": ("KOSDAQ", "코스닥 지수"),
}
INDEX_CODES = {"KOSPI", "KOSDAQ"}


def is_index(code: str) -> bool:
    return (code or "").upper() in INDEX_CODES


def pick_x_ticks(view: pd.DataFrame, timeframe: str, n: int = 7):
    """차트 X축(카테고리 축) 눈금.

    표시 구간의 첫 봉과 마지막 봉, 그리고 그 사이 지점을 고르게 골라
    tickvals(날짜 문자열)와 ticktext(YY.MM)를 만든다. 일봉은 월초 봉에
    눈금을 맞춰 달 단위로 읽기 쉽게 한다.
    """
    dates = list(view["date_str"])
    m = len(dates)
    if m == 0:
        return [], []

    def label_at(i: int) -> str:
        return pd.to_datetime(dates[i]).strftime("%y.%m")

    if m <= n:
        picked = list(range(m))
    else:
        picked = [0, m - 1]
        inner = n - 2
        if timeframe == "day" and inner > 0:
            dts = pd.to_datetime(pd.Series(dates))
            # 양끝 봉과 같은 달의 월초는 중간 눈금에서 제외해 라벨 중복을 막는다.
            edge_months = {label_at(0), label_at(m - 1)}
            starts = [
                i for i in range(1, m - 1)
                if dts.iloc[i].day <= 3 and label_at(i) not in edge_months
            ]
            if len(starts) >= inner:
                for k in range(1, n - 1):
                    target = k * (m - 1) / (n - 1)
                    picked.append(min(starts, key=lambda s: abs(s - target)))
            else:
                picked += [int(round(k * (m - 1) / (n - 1))) for k in range(1, n - 1)]
        else:
            picked += [int(round(k * (m - 1) / (n - 1))) for k in range(1, n - 1)]
        picked = sorted(set(picked))

    tickvals, ticktext = [], []
    for i in picked:
        label = label_at(i)
        if label in ticktext:
            # 라벨이 겹치면 마지막 봉 눈금을 우선한다(구간 끝이 항상 보이도록).
            if i == m - 1:
                k = ticktext.index(label)
                tickvals[k], ticktext[k] = dates[i], label
            continue
        tickvals.append(dates[i])
        ticktext.append(label)
    return tickvals, ticktext


def fmt_price(v: float, index: bool = False) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    return f"{v:,.2f}p" if index else f"{int(round(v)):,}원"


@st.cache_data(ttl=3600)
def search_stock(query: str):
    """종목명·종목코드·지수명을 (코드, 표시명)으로 변환."""
    q = (query or "").strip()
    if not q:
        return None, None

    key = q.replace(" ", "").lower()
    if key in INDEX_ALIASES:
        return INDEX_ALIASES[key]

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
def fetch_ohlcv(code: str, timeframe: str = "day", years: int = 8) -> pd.DataFrame:
    """일봉·주봉 OHLCV. 5년 차트에서도 650봉 장기 이동평균이 앞부분부터 계산되도록
    기본 8년치를 받아온다."""
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
    """주가위치 밴드 계산.

    기준선(무릎) = 1.05 × (고가 120봉 평균)^0.30 × (고가 650봉 평균)^0.70
    단계 = 기준선 × exp(z × s),  s = clip(0.44 + 0.30×(연변동성-0.5), 0.25, 0.75)
    (주봉은 봉 수를 1/5로 환산: 24/130봉, 변동성 52봉)
    """
    df = df.copy()
    for n in [5, 20, 40, 50, 60, 120]:
        df[f"sma_{n}"] = df["close"].rolling(n).mean()
        df[f"vsma_{n}"] = df["volume"].rolling(n).mean()

    hh = df["high"].rolling(12).max()
    ll = df["low"].rolling(12).min()
    fast_k = np.where(hh == ll, 50.0, (df["close"] - ll) / (hh - ll) * 100.0)
    df["slow_k"] = pd.Series(fast_k).rolling(5).mean()
    df["slow_d"] = df["slow_k"].rolling(5).mean()

    short_n = BASE_SHORT if timeframe == "day" else max(6, BASE_SHORT // 5)
    long_n = BASE_LONG if timeframe == "day" else max(30, BASE_LONG // 5)
    vol_n = 250 if timeframe == "day" else 52
    ann_factor = np.sqrt(vol_n)

    hi = df["high"]
    ma_short = hi.rolling(short_n, min_periods=max(10, short_n // 3)).mean()
    ma_long = hi.rolling(long_n, min_periods=max(40, long_n // 5)).mean()
    ma_long = ma_long.fillna(ma_short)
    ma_short = ma_short.fillna(ma_long)
    base = BASE_SCALE * np.exp(BASE_W * np.log(ma_short) + (1 - BASE_W) * np.log(ma_long))

    log_c = np.log(df["close"].replace(0, np.nan))
    ann_vol = (log_c.diff().rolling(vol_n).std() * ann_factor).fillna(0.5)
    sig = (SIG_C0 + SIG_C1 * (ann_vol - 0.5)).clip(SIG_MIN, SIG_MAX)

    df["기준선"] = base
    for name, z, _, _, _ in BAND_Z:
        df[name] = base * np.exp(z * sig)
    df["변동성"] = ann_vol
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


# ── 연간 실적·밸류에이션 ──────────────────────────────────────────────
# 네이버 증권 연간 실적 API의 rowList를 표시용·계산용으로 파싱한다.
# 단위: 매출액·영업이익·당기순이익 억원 / EPS·BPS·주당배당금 원 / 비율 % / PER·PBR 배
_FIN_ROWS = [
    ("매출액", "매출액(억원)", "{:,.0f}"),
    ("영업이익", "영업이익(억원)", "{:,.0f}"),
    ("영업이익률", "영업이익률(%)", "{:.2f}"),
    ("당기순이익", "당기순이익(억원)", "{:,.0f}"),
    ("ROE", "ROE(%)", "{:.2f}"),
    ("부채비율", "부채비율(%)", "{:.2f}"),
    ("EPS", "EPS(원)", "{:,.0f}"),
    ("PER", "PER(배)", "{:.2f}"),
    ("BPS", "BPS(원)", "{:,.0f}"),
    ("PBR", "PBR(배)", "{:.2f}"),
    ("주당배당금", "주당배당금(원)", "{:,.0f}"),
]


def _to_num(cell: dict) -> float:
    """API 셀값('1,234' / '-' / cx='minus')을 float로 변환. 없으면 NaN."""
    if not isinstance(cell, dict):
        return np.nan
    s = str(cell.get("value", "")).strip().replace(",", "")
    if s in ("", "-", "N/A", "None"):
        return np.nan
    try:
        v = float(s)
    except ValueError:
        return np.nan
    if cell.get("cx") == "minus" and v > 0:
        v = -v
    return v


@st.cache_data(ttl=3600)
def fetch_finance_summary(code: str) -> dict:
    """네이버 증권 연간 실적·밸류에이션 요약.

    지수(KOSPI/KOSDAQ)나 실적 미제공 종목은 빈 딕셔너리를 돌려준다(예외 없음).
    반환: table(표시용 DataFrame), raw(지표별 연도 값), labels, eps_options,
          eps/per/pbr/roe/op/revenue/op_margin(최근 실제 연도), eps_consensus/per_consensus
    """
    if not code or is_index(code):
        return {}
    try:
        url = f"https://m.stock.naver.com/api/stock/{code}/finance/annual"
        resp = requests.get(url, headers=HEADERS, timeout=10)
        if resp.status_code != 200:
            return {}
        data = resp.json()
    except Exception:
        return {}

    fi = data.get("financeInfo") or {}
    cols = fi.get("trTitleList") or []
    if not cols:
        return {}

    # 컬럼 순서는 trTitleList 순서를 따른다(JSON의 columns 딕셔너리 순서는 연도순이 아님).
    keys, labels, is_actual = [], [], []
    for c in cols:
        k = str(c.get("key") or "").strip()
        if not k:
            continue
        title = str(c.get("title") or k).replace(".", "").strip()
        year = title[:4] if title[:4].isdigit() else title
        consensus = str(c.get("isConsensus", "N")).upper() == "Y"
        keys.append(k)
        labels.append(f"{year}(E)" if consensus else year)
        is_actual.append(not consensus)
    if not keys:
        return {}

    row_map = {r.get("title"): (r.get("columns") or {}) for r in (fi.get("rowList") or [])}
    raw = {}
    for title, _, _ in _FIN_ROWS:
        cells = row_map.get(title) or {}
        raw[title] = [_to_num(cells.get(k)) for k in keys]

    table = pd.DataFrame(
        {
            label: ["-" if np.isnan(v) else fmt.format(v) for v in raw[title]]
            for title, label, fmt in _FIN_ROWS
        },
        index=labels,
    ).T

    n = len(labels)
    actual_with_eps = [i for i in range(n) if is_actual[i] and not np.isnan(raw["EPS"][i])]
    if actual_with_eps:
        latest_i = actual_with_eps[-1]
    else:
        cand = [
            i for i in range(n)
            if is_actual[i] and (not np.isnan(raw["영업이익"][i]) or not np.isnan(raw["PER"][i]))
        ]
        latest_i = cand[-1] if cand else None
    cons_with_eps = [i for i in range(n) if not is_actual[i] and not np.isnan(raw["EPS"][i])]
    cons_i = cons_with_eps[-1] if cons_with_eps else None

    def pick(title: str, i):
        if i is None:
            return np.nan
        v = raw[title][i]
        return np.nan if np.isnan(v) else float(v)

    eps_options = []
    for i in range(n):
        e = raw["EPS"][i]
        if np.isnan(e) or e == 0:
            continue
        tag = "예상" if not is_actual[i] else "실적"
        eps_options.append({"label": f"{labels[i]} {tag}", "eps": float(e), "idx": i})
    default_opt = 0
    for pos, opt in enumerate(eps_options):
        if latest_i is not None and opt["idx"] == latest_i:
            default_opt = pos

    return {
        "labels": labels,
        "table": table,
        "raw": raw,
        "latest_label": labels[latest_i] if latest_i is not None else "-",
        "eps": pick("EPS", latest_i),
        "per": pick("PER", latest_i),
        "pbr": pick("PBR", latest_i),
        "roe": pick("ROE", latest_i),
        "op": pick("영업이익", latest_i),
        "revenue": pick("매출액", latest_i),
        "op_margin": pick("영업이익률", latest_i),
        "debt_ratio": pick("부채비율", latest_i),
        "eps_options": eps_options,
        "default_eps_option": default_opt,
        "consensus_label": labels[cons_i] if cons_i is not None else None,
        "eps_consensus": pick("EPS", cons_i),
        "per_consensus": pick("PER", cons_i),
    }


def describe_price_zone(price: float, row) -> str:
    """가격이 차트의 어느 단계 사이에 있는지 문장으로 돌려준다."""
    if price is None or (isinstance(price, float) and np.isnan(price)):
        return "-"
    bands = [
        (name.split("(")[0], float(row[name]))
        for name, _, _, _, _ in BAND_Z
        if name in row.index and pd.notna(row[name])
    ]
    if not bands:
        return "-"
    if price < bands[0][1]:
        return f"{bands[0][0]} 이하"
    label = f"{bands[0][0]} 이하"
    for i, (nm, v) in enumerate(bands):
        if price >= v:
            label = f"{nm} ~ {bands[i + 1][0]} 사이" if i + 1 < len(bands) else f"{nm} 이상"
    return label


def enrich_scan_with_finance(rows: list):
    """무릎 아래로 걸러진 종목에만 연간 실적을 병렬 조회해 컬럼을 붙인다."""
    if not rows:
        return rows
    fin_map = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_finance_summary, r["코드"]): r["코드"] for r in rows}
        for f in as_completed(futures):
            c = futures[f]
            try:
                fin_map[c] = f.result() or {}
            except Exception:
                fin_map[c] = {}

    for r in rows:
        fin = fin_map.get(r["코드"]) or {}
        eps = fin.get("eps", np.nan)
        op = fin.get("op", np.nan)
        roe = fin.get("roe", np.nan)
        cur_p = float(r["현재가"])
        r["영업이익(억)"] = None if pd.isna(op) else int(round(op))
        r["PER"] = round(cur_p / eps, 1) if pd.notna(eps) and eps > 0 else None
        r["ROE(%)"] = None if pd.isna(roe) else round(float(roe), 1)
        r["실적연도"] = fin.get("latest_label", "-") if fin else "-"
    return rows


def analyze_single_stock(item):
    code, name, m_name = item
    try:
        df = fetch_ohlcv(code, "day", years=4)
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
        "종목명·종목코드·지수",
        key="selected_query",
        placeholder="예: 삼성전자, 신세계, 004170, 코스피, 코스닥",
    )
    tf_label = st.radio("봉 구분", ["일봉", "주봉"], horizontal=True)
    timeframe = "day" if tf_label == "일봉" else "week"
    use_log = st.checkbox("로그 스케일", value=True)
    allow_zoom = st.checkbox("차트 드래그/확대 켜기 (모바일 스크롤 시 끄기 권장)", value=False)
    st.markdown("---")
    st.caption(
        "※ 본 차트와 스캐너는 공개된 블로그 차트 패턴을 역산·재현한 참고용 지표이며 투자 권유가 아닙니다. "
        "댕기왕자 블로그 차트 34개 지점 대조 기준 평균 절대오차 약 7.4%, 중앙값 3.8%."
    )

tab_chart, tab_scan = st.tabs(["📈 종목·지수 차트", "🔍 무릎 아래 종목 스캐너"])

with tab_chart:
    period_label = st.radio(
        "조회 기간",
        PERIOD_LABELS,
        horizontal=True,
        index=0,
        help="선택한 기간만큼의 일봉·주봉을 차트에 표시합니다(일봉 1년=250봉, 주봉 1년=52봉).",
    )
    bars = PERIOD_BARS[timeframe][period_label]
    q_now = st.session_state["selected_query"].strip()
    if q_now:
        code, name = search_stock(q_now)
        if not code:
            st.error(f"'{q_now}'에 해당하는 국내 상장 종목·지수를 찾지 못했습니다. (지수는 '코스피', '코스닥'으로 입력)")
        else:
            idx_flag = is_index(code)
            raw_df = fetch_ohlcv(code, timeframe)
            if raw_df.empty or len(raw_df) < 60:
                st.warning(f"{name}({code})의 주가 데이터가 충분하지 않습니다.")
            else:
                df = compute_indicators(raw_df, timeframe)
                view = df.tail(bars).reset_index(drop=True)
                st.caption(
                    f"차트 표시 구간: {view['date_str'].iloc[0]} ~ {view['date_str'].iloc[-1]} "
                    f"· {len(view):,}봉 ({period_label}, {tf_label})"
                )
                last = view.iloc[-1]
                prev = view.iloc[-2] if len(view) >= 2 else last
                chg = (last["close"] - prev["close"]) / prev["close"] * 100.0

                cur_p = last["close"]
                zone = "바닥(빨강) 이하"
                for b_name, _, _, _, _ in BAND_Z:
                    if pd.notna(last[b_name]) and cur_p >= last[b_name]:
                        zone = f"{b_name} 이상"

                c1, c2, c3 = st.columns(3)
                c1.metric(f"{name} ({code})", fmt_price(cur_p, idx_flag), f"{chg:+.2f}%")
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

                for b_name, _, col, lw, dash in BAND_Z:
                    fig.add_trace(
                        go.Scatter(
                            x=view["date_str"], y=view[b_name],
                            mode="lines", line=dict(color=col, width=lw, dash=dash),
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
                    annotation_text=f" {fmt_price(cur_p, idx_flag)}",
                    annotation_position="right",
                    annotation_font_color="#d62828",
                    row=1, col=1,
                )
                fig.update_yaxes(range=[0, 100], row=2, col=1)

                # X축 눈금: 표시 구간 시작~끝을 고르게 7개 지점으로 직접 지정
                tickvals, ticktext = pick_x_ticks(view, timeframe)
                fig.update_xaxes(
                    type="category",
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={"scrollZoom": allow_zoom, "displayModeBar": allow_zoom},
                )

                st.markdown('<div id="price-table"></div>', unsafe_allow_html=True)
                st.subheader("단계별 가격표 (최신 기준)")
                table_rows = []
                for b_name, _, _, _, _ in reversed(BAND_Z):
                    val = last[b_name]
                    if pd.notna(val):
                        diff_pct = (val - cur_p) / cur_p * 100.0
                        table_rows.append({
                            "구간": b_name,
                            "가격": fmt_price(val, idx_flag),
                            "현재가 대비": f"{diff_pct:+.1f}%",
                        })
                st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
                st.caption(
                    f"연변동성 {last['변동성']:.0%} · 기준선 {fmt_price(last['기준선'], idx_flag)} · "
                    f"단계 폭 s={float((SIG_C0 + SIG_C1 * (last['변동성'] - 0.5))):.2f} "
                    f"(허리·어깨·상투는 기준선의 약 2.07·2.74·4.16배)"
                )

                # ── 연간 실적 및 밸류에이션 ─────────────────────────────
                fin = {} if idx_flag else fetch_finance_summary(code)
                if fin:
                    st.subheader("📊 연간 실적 및 밸류에이션")
                    st.dataframe(fin["table"], use_container_width=True)
                    st.caption(
                        "단위: 매출액·영업이익·당기순이익 억원, EPS·BPS·주당배당금 원, 비율 %. "
                        "출처: 네이버 증권 연간 실적. (E)는 증권가 예상(컨센서스) 값입니다."
                    )

                    # ── 목표 PER 적정주가 계산기 ─────────────────────────
                    st.subheader("🎯 목표 PER 적정주가 계산기")
                    opts = fin["eps_options"]
                    if not opts:
                        st.info("EPS 데이터가 없어 적정주가를 계산할 수 없습니다.")
                    else:
                        c_eps, c_per = st.columns([2, 1])
                        with c_eps:
                            sel_idx = st.selectbox(
                                "기준 EPS",
                                options=list(range(len(opts))),
                                index=fin["default_eps_option"],
                                format_func=lambda i: f"{opts[i]['label']} · EPS {opts[i]['eps']:,.0f}원",
                            )
                        with c_per:
                            per_now = fin["per"]
                            if pd.notna(per_now) and per_now > 0:
                                per_default = float(min(max(round(float(per_now)), 1), 200))
                            else:
                                per_default = 10.0
                            target_per = st.number_input(
                                "목표 PER(배)",
                                min_value=0.1,
                                max_value=200.0,
                                value=per_default,
                                step=0.5,
                            )

                        eps_sel = float(opts[sel_idx]["eps"])
                        fair = eps_sel * target_per
                        upside = (fair - cur_p) / cur_p * 100.0
                        fair_zone = describe_price_zone(fair, last)

                        m1, m2, m3 = st.columns(3)
                        m1.metric("적정주가", fmt_price(fair, idx_flag), f"{upside:+.1f}% (현재가 대비)")
                        m2.metric("계산식", f"{eps_sel:,.0f}원 × {target_per:g}배")
                        m3.metric("차트상 위치", fair_zone)

                        knee_now = float(last["무릎(파랑)"]) if pd.notna(last["무릎(파랑)"]) else np.nan
                        knee_txt = (
                            f"무릎(파랑) 대비 {(fair - knee_now) / knee_now * 100:+.1f}%"
                            if pd.notna(knee_now) and knee_now
                            else "-"
                        )
                        cur_per_txt = f"{cur_p / eps_sel:.2f}배" if eps_sel else "-"
                        st.caption(
                            f"현재가 {fmt_price(cur_p, idx_flag)} ÷ 기준 EPS {eps_sel:,.0f}원 = PER {cur_per_txt} · "
                            f"적정주가 {fmt_price(fair, idx_flag)}은 {knee_txt} 지점입니다. "
                            f"목표 PER을 바꾸면 적정주가와 상승여력이 바로 갱신됩니다."
                        )
                        st.caption(
                            "※ 적정주가는 목표 PER 가정에 따른 참고치이며 투자 권유가 아닙니다. "
                            "EPS가 마이너스인 기업은 계산값이 의미를 갖지 않습니다."
                        )
                elif not idx_flag:
                    st.info("이 종목의 연간 실적 데이터를 네이버 증권에서 가져오지 못했습니다.")

with tab_scan:
    st.subheader("🔍 현재가 ≤ 무릎(파랑) 종목 스캐너 (일봉 기준)")
    st.caption(
        "시가총액 상위 종목 중 현재 주가가 무릎(파랑) 선 아래에 있는 종목을 찾고, 최근 1년 고점 위치와 고점 대비 하락률, "
        "그 종목의 연간 실적(영업이익·PER·ROE)까지 함께 표시합니다. PER은 현재가 ÷ 최근 연간 EPS입니다."
    )

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

    col_p, col_r = st.columns([1.6, 1.6])
    with col_p:
        only_profit = st.checkbox("흑자 기업만 보기 (최근 영업이익 > 0)", value=False)
    with col_r:
        per_opt = st.selectbox("PER 조건", ["전체", "PER ≤ 15", "PER ≤ 10"])

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
            with st.spinner(f"무릎 아래 {len(found)}개 종목의 연간 실적을 병렬 조회하는 중..."):
                found = enrich_scan_with_finance(found)

        if found:
            res_df = pd.DataFrame(found)
            col_order = [
                "종목명", "코드", "시장", "현재가", "등락률(%)", "무릎(파랑)", "무릎 대비(%)",
                "영업이익(억)", "PER", "ROE(%)", "실적연도",
                "현재 구간", "1년 고점 위치", "고점 대비(%)",
            ]
            res_df = res_df[[c for c in col_order if c in res_df.columns]]
            res_df = res_df.sort_values("무릎 대비(%)", ascending=True).reset_index(drop=True)
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
        if not res_df.empty and only_profit and "영업이익(억)" in res_df.columns:
            op_num = pd.to_numeric(res_df["영업이익(억)"], errors="coerce")
            res_df = res_df[op_num > 0].reset_index(drop=True)
        if not res_df.empty and per_opt != "전체" and "PER" in res_df.columns:
            limit = 15.0 if per_opt == "PER ≤ 15" else 10.0
            per_num = pd.to_numeric(res_df["PER"], errors="coerce")
            res_df = res_df[per_num.notna() & (per_num <= limit)].reset_index(drop=True)

        if res_df.empty:
            st.info(f"스캔한 {total_cnt}개 종목 중 해당 조건에 맞는 종목이 없습니다.")
        else:
            st.success(f"총 {total_cnt}개 종목 중 조건에 맞는 **{len(res_df)}개** 종목을 찾았습니다.")
            st.dataframe(res_df, use_container_width=True, hide_index=True)

            pick = st.selectbox("바로 차트로 볼 종목 선택", res_df["종목명"].tolist())
            st.button("선택 종목 차트 열기", on_click=set_selected_stock, args=(pick,))
