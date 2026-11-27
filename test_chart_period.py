"""차트 조회 기간(1년·2년·3년·5년)·봉 구분별 표시 구간 검증.

Streamlit AppTest로 실제 앱을 실행해 그려진 Plotly 차트 스펙에서
주가 캔들 개수·첫/끝 날짜·X축 눈금(tickvals·ticktext)을 직접 확인한다.

실행: python test_chart_period.py
"""
import json
from datetime import datetime

from streamlit.testing.v1 import AppTest

# (조회 대상, 봉 구분 라벨, [검증할 조회 기간])
CASES = [
    ("삼성전자", "일봉", ["1년", "2년", "3년", "5년"]),
    ("삼성전자", "주봉", ["1년", "2년", "3년", "5년"]),
    ("코스피", "일봉", ["1년", "5년"]),
]
EXPECT_BARS = {
    ("일봉", "1년"): 250, ("일봉", "2년"): 500, ("일봉", "3년"): 750, ("일봉", "5년"): 1250,
    ("주봉", "1년"): 52, ("주봉", "2년"): 104, ("주봉", "3년"): 156, ("주봉", "5년"): 260,
}


def radio_by_label(at, label):
    for r in at.radio:
        if r.label == label:
            return r
    raise AssertionError(f"'{label}' 라디오 없음: {[r.label for r in at.radio]}")


def chart_spec(at):
    charts = at.get("plotly_chart")
    assert len(charts) == 1, f"차트 개수 {len(charts)}"
    spec = json.loads(charts[0].proto.spec)
    return spec.get("figure", spec)


def check(query, tf, period):
    at = AppTest.from_file("app.py", default_timeout=240).run()
    assert not at.exception, [e.value for e in at.exception]
    at.text_input[0].set_value(query).run()
    radio_by_label(at, "봉 구분").set_value(tf)
    radio_by_label(at, "조회 기간").set_value(period)
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    fig = chart_spec(at)
    candle = fig["data"][0]
    xs = candle["x"]
    first, last = xs[0], xs[-1]
    want = EXPECT_BARS[(tf, period)]

    caps = [c.value for c in at.caption if c.value.startswith("차트 표시 구간")]
    assert caps, "표시 구간 캡션이 없습니다"
    print(f"\n[{query} / {tf} / {period}] {caps[0]}")
    print(f"  캔들 {len(xs)}봉 (기대 {want}) · {first} ~ {last}")
    print(f"  X축 눈금 {len(fig['layout'].get('xaxis', {}).get('ticktext', []))}개")

    assert len(xs) == want, f"봉 수 {len(xs)} != {want}"
    assert want == (260 if tf == "주봉" and period == "5년" else want)

    # 기간이 1년 단위인지: 표시 봉수의 첫 날짜가 요청 기간만큼 과거인지 확인
    years = int(period[0])
    d0, d1 = datetime.strptime(first, "%Y-%m-%d"), datetime.strptime(last, "%Y-%m-%d")
    span_years = (d1 - d0).days / 365.25
    assert years - 0.35 <= span_years <= years + 0.35, f"기간 {span_years:.2f}년 (요청 {years}년)"

    # X축: 첫·마지막 봉이 눈금에 포함되고, 6~8개 지점이 구간 전체에 고르게 퍼져 있는지
    xaxis = fig["layout"]["xaxis"]
    tv = xaxis.get("tickvals", [])
    tt = xaxis.get("ticktext", [])
    assert xaxis.get("tickmode") == "array", xaxis.get("tickmode")
    assert xaxis.get("type") == "category"
    assert 6 <= len(tv) <= 8, f"눈금 {len(tv)}개"
    assert len(tv) == len(tt)
    assert tv[0] == first and tv[-1] == last, f"눈금 양끝 {tv[0]} / {tv[-1]}"
    positions = [xs.index(v) for v in tv]
    gaps = [b - a for a, b in zip(positions, positions[1:])]
    assert all(g > 0 for g in gaps), positions
    assert max(gaps) <= 2.2 * min(gaps), f"눈금 간격 불균등 {gaps}"
    print(f"  눈금 위치 {positions} · 라벨 {tt}")


if __name__ == "__main__":
    for q, tf, periods in CASES:
        for p in periods:
            check(q, tf, p)

    # 상장 이력이 짧은 종목(2022-01 상장): 요청 봉수보다 데이터가 적어도 오류 없이 전체 표시
    at = AppTest.from_file("app.py", default_timeout=240).run()
    at.text_input[0].set_value("373220").run()   # LG에너지솔루션
    radio_by_label(at, "조회 기간").set_value("5년")
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    fig = chart_spec(at)
    xs = fig["data"][0]["x"]
    cap = [c.value for c in at.caption if c.value.startswith("차트 표시 구간")][0]
    print(f"\n[373220 / 일봉 / 5년] {cap}")
    print(f"  캔들 {len(xs)}봉 · X축 눈금 {len(fig['layout']['xaxis'].get('ticktext', []))}개")
    assert len(xs) < 1250, len(xs)
    assert 5 <= len(fig["layout"]["xaxis"]["tickvals"]) <= 8

    print("\n모든 검증 통과")
