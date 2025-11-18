# -*- coding: utf-8 -*-
"""
미국 시장 리스크 체크포인트 – 자동 모니터링

이 스크립트는 Streamlit 기반 대시보드입니다.

구성:
1) 입력값 (Forward EPS, 경고 기준치)
2) 실시간 근접 시장 지표 (S&P 500, VIX, HY OAS, Term Premium, UST10Y)
3) Auto check: 각 지표를 기준값과 비교해 OK/경고 표시
4) 시장 심리 지수 (공포·탐욕 스타일 요약)
5) 뉴스 리스크 모니터 (핵심 테마별 뉴스 스코어링)

※ API 키는 .env 파일에서 불러옵니다.
"""

import os
from math import isnan
from datetime import datetime, timedelta

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

# yfinance가 Yahoo 응답 문제로 verbose한 경고를 많이 찍을 수 있어
# 로그 레벨을 줄여 콘솔을 깔끔하게 유지.
import logging
logging.getLogger("yfinance").setLevel(logging.ERROR)

# ---------------------
# 환경 변수 로드
# ---------------------
load_dotenv()  # .env 파일이 있으면 FRED_API_KEY, NEWS_API_KEY 등을 읽어옴

# ---------------------
# Config
# ---------------------
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# FRED에서 가져올 시계열 ID 매핑
FRED_SERIES = {
    "HY_OAS": "BAMLH0A0HYM2",          # 하이일드 회사채 OAS (신용위험 지표)
    "TERM_PREMIUM_10Y": "THREEFYTP10",  # 10년물 Term Premium
    "UST10Y": "DGS10",                  # 미국 10년 국채금리
    "SP500": "SP500",                   # S&P 500 지수 (백업용)
    "VIX": "VIXCLS",                    # VIX 지수 (백업용)
}

# Yahoo Finance 티커
VIX_TICKER = "^VIX"
SPX_TICKER = "^GSPC"

# Auto check에서 사용할 기준값
ALERTS = {
    "VIX": 25.0,
    "HY_OAS": 4.5,
    "TERM_PREMIUM_10Y": 0.9,
}

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# ---------------------
# Streamlit 기본 설정
# ---------------------
st.set_page_config(page_title="미국 시장 리스크 대시보드", layout="wide")
st.title("미국 시장 리스크 체크포인트 – 자동 모니터링")
st.caption(
    "데이터 출처: FRED, Yahoo Finance. Forward P/E와 ERP는 사용자가 입력한 EPS를 기반으로 단순 계산됩니다."
)

# ---------------------
# Helper Functions
# ---------------------
@st.cache_data(ttl=60 * 30)
def fred_series(series_id: str) -> pd.DataFrame:
    """
    FRED에서 시계열을 불러와 DataFrame으로 반환.
    - FRED_API_KEY 없거나 / 오류 발생 시 빈 DataFrame 반환.
    - 여기서 실패해도 앱 전체가 죽지 않도록 예외를 내부 처리.
    """
    if not FRED_API_KEY:
        return pd.DataFrame()

    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": "2010-01-01",
    }

    try:
        r = requests.get(FRED_BASE, params=params, timeout=20)
        r.raise_for_status()
    except requests.exceptions.HTTPError as e:
        st.warning(f"{series_id} 불러오는 중 FRED API 오류: {e} (API 키/쿼터 확인 필요).")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"{series_id} FRED 데이터 로딩 오류: {e}")
        return pd.DataFrame()

    data = r.json().get("observations", [])
    df = pd.DataFrame(data)
    if df.empty:
        return pd.DataFrame()

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.set_index("date")[["value"]]
    df = df.rename(columns={"value": series_id})
    return df.dropna()


@st.cache_data(ttl=60 * 10)
def yf_latest(ticker: str) -> float:
    """
    Yahoo Finance에서 최근 종가를 가져옴. 실패 시 NaN 반환.

    참고:
    - 회사 네트워크/방화벽/VPN 때문에 Yahoo 응답이 HTML/빈값으로 올 때가 있어
      yfinance 내부에서 JSONDecodeError를 출력할 수 있음.
    - 여기서는 예외를 먹고 NaN을 돌려주며,
      이후 로직에서 FRED 데이터로 백업(fallback) 시도.
    """
    try:
        data = yf.download(
            ticker,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
    except Exception:
        return float("nan")

    if data.empty:
        return float("nan")
    return float(data["Close"].dropna().iloc[-1])


@st.cache_data(ttl=60 * 10)
def yf_hist(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Yahoo Finance에서 히스토리컬 데이터. 실패 시 빈 DataFrame."""
    try:
        return yf.download(
            ticker,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
    except Exception:
        return pd.DataFrame()


def traffic_light(value, threshold, higher_is_risk: bool = True):
    """
    단일 값과 기준값을 비교해 간단한 '정상/경고' 아이콘을 반환.
    - higher_is_risk = True  → 값이 크면 위험 (예: VIX, HY OAS)
    - higher_is_risk = False → 값이 작으면 위험 (예: ERP)
    """
    if value is None or isnan(value):
        return "⚪", "데이터 없음"
    if higher_is_risk:
        return ("🟥", "경고") if value >= threshold else ("🟩", "정상")
    else:
        return ("🟥", "경고") if value <= threshold else ("🟩", "정상")


def vol_to_level(value: float) -> str:
    """
    VIX / VXN 수준을 공포·탐욕 단계로 변환
    35↑      : 공포(매수)
    25~35    : 공포
    18~25    : 중립
    12~18    : 탐욕
    12 미만  : 탐욕(매도)
    """
    if value >= 35:
        return "공포(매수)"
    elif value >= 25:
        return "공포"
    elif value >= 18:
        return "중립"
    elif value >= 12:
        return "탐욕"
    else:
        return "탐욕(매도)"


# ---------------------
# Inputs: 가정치 및 임계값
# ---------------------
st.markdown("---")
colA, colB, colC = st.columns(3)

with colA:
    # Forward EPS는 기본적으로 사용자가 입력하지만,
    # 추후 자동화(외부 API 연동)를 붙일 수 있도록 구조만 단순하게 유지.
    eps_forward = st.number_input(
        "S&P 500 향후 12개월 EPS (직접 입력)",
        min_value=0.0,
        value=260.0,
        step=5.0,
        help="컨센서스 Forward EPS 추정치를 입력해 주세요. "
             "추후 외부 데이터 연동 시 이 값을 자동으로 채울 수 있습니다.",
    )

with colB:
    pe_threshold = st.number_input(
        "Forward P/E 경고 기준값",
        min_value=10.0,
        value=23.0,
        step=0.5,
        help="이 값 이상이면 밸류에이션 부담(경고)으로 표시됩니다.",
    )

with colC:
    erp_floor = st.number_input(
        "ERP (E/P - 10Y) 경고 기준값 (%)",
        min_value=-5.0,
        value=0.0,
        step=0.1,
        help="이 값 이하이면 '주식 위험 보상이 부족하다'는 신호로 해석합니다.",
    )

# ---------------------
# Live Data: 시장 지표
# ---------------------
st.markdown("---")
left, right = st.columns([2, 1])

with left:
    st.subheader("주요 시장 지표 (실시간에 근접)")
    st.caption("미국 주식·채권·신용 스프레드 핵심 지표를 모아 현재 시장 환경을 한눈에 보여줍니다.")

    # FRED 데이터 미리 로드 (동일 요청 캐시됨)
    fred_dfs = {name: fred_series(sid) for name, sid in FRED_SERIES.items()}

    def last_value(name: str) -> float:
        """특정 시계열의 가장 최근 값만 꺼내는 헬퍼."""
        df = fred_dfs.get(name)
        if df is None or df.empty:
            return float("nan")
        col = df.columns[0]
        return float(df[col].dropna().iloc[-1])

    # S&P 500: 1차는 Yahoo, 실패 시 FRED SP500으로 백업
    spx = yf_latest(SPX_TICKER)
    if isnan(spx):
        v = last_value("SP500")
        if not isnan(v):
            spx = v  # Yahoo 막혀도 지수 수준은 FRED로 보정

    # VIX: 1차는 Yahoo, 실패 시 FRED VIXCLS로 백업
    vix = yf_latest(VIX_TICKER)
    if isnan(vix):
        v = last_value("VIX")
        if not isnan(v):
            vix = v

    # 그 외 지표는 FRED에서만 가져옴
    hy_oas = last_value("HY_OAS")
    tp10 = last_value("TERM_PREMIUM_10Y")
    ust10 = last_value("UST10Y")

    # Forward P/E & ERP 계산 (eps_forward가 유효할 때만)
    pe_forward = None
    erp = None
    if eps_forward and eps_forward > 0 and not isnan(spx):
        pe_forward = spx / eps_forward
        if not isnan(ust10):
            # ERP ≈ E/P - 10Y (단순 근사)
            erp = (eps_forward / spx) * 100.0 - ust10

    # 상단 핵심 숫자 5개
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("S&P 500", f"{spx:,.0f}" if not isnan(spx) else "N/A")
    m2.metric("VIX", f"{vix:.1f}" if not isnan(vix) else "N/A")
    m3.metric("HY OAS (%)", f"{hy_oas:.2f}" if not isnan(hy_oas) else "N/A")
    m4.metric("10Y Term Premium (%)", f"{tp10:.2f}" if not isnan(tp10) else "N/A")
    m5.metric("UST 10Y (%)", f"{ust10:.2f}" if not isnan(ust10) else "N/A")

    st.write("—")
    c1, c2 = st.columns(2)

    with c1:
        st.caption("VIX (최근 1년) — 변동성이 구조적으로 높아지는지 체크.")
        vix_hist = yf_hist(VIX_TICKER, period="1y")
        if not vix_hist.empty:
            st.line_chart(vix_hist["Close"])

    with c2:
        st.caption("HY OAS (최근 수년) — 신용 스프레드 확대 여부로 위험 선호 변화 확인.")
        df_hy = fred_dfs.get("HY_OAS")
        if df_hy is not None and not df_hy.empty:
            st.line_chart(df_hy.tail(2600))

with right:
    st.subheader("자동 리스크 체크")
    st.caption("각 지표를 사전 기준과 비교해 '정상/경고' 신호를 단순화해 보여줍니다.")

    # VIX
    if not isnan(vix):
        icon, msg = traffic_light(vix, ALERTS["VIX"], higher_is_risk=True)
        st.write(f"**VIX**: {vix:.1f} → {icon} {msg} (기준: {ALERTS['VIX']})")
    else:
        st.write("**VIX**: 데이터 없음")

    # HY OAS
    if not isnan(hy_oas):
        icon, msg = traffic_light(hy_oas, ALERTS["HY_OAS"], higher_is_risk=True)
        st.write(f"**HY OAS**: {hy_oas:.2f}% → {icon} {msg} (기준: {ALERTS['HY_OAS']}%)")
    else:
        st.write("**HY OAS**: 데이터 없음")

    # Term Premium
    if not isnan(tp10):
        icon, msg = traffic_light(tp10, ALERTS["TERM_PREMIUM_10Y"], higher_is_risk=True)
        st.write(
            f"**10Y Term Premium**: {tp10:.2f}% → {icon} {msg} "
            f"(기준: {ALERTS['TERM_PREMIUM_10Y']}%)"
        )
    else:
        st.write("**10Y Term Premium**: 데이터 없음")

    # Forward P/E
    if pe_forward is not None:
        icon, msg = traffic_light(pe_forward, pe_threshold, higher_is_risk=True)
        st.write(
            f"**Forward P/E**: {pe_forward:.1f}배 → {icon} {msg} "
            f"(기준: {pe_threshold}배)"
        )
    else:
        st.write("**Forward P/E**: EPS 입력 필요")

    # ERP
    if erp is not None:
        icon, msg = traffic_light(erp, erp_floor, higher_is_risk=False)
        st.write(
            f"**ERP 추정치 (E/P - 10Y)**: {erp:.2f}% → {icon} {msg} "
            f"(기준: {erp_floor}%)"
        )
    else:
        st.write("**ERP 추정치**: UST10Y 및 EPS 필요")

# ---------------------
# 지수별 시장 심리 지수 (공포지수 기반)
# ---------------------
st.markdown("---")
st.subheader("지수별 시장 심리 지수 (공포지수 기반)")

st.caption(
    "S&P 500은 VIX, 나스닥 100은 VXN 공포지수를 사용해 "
    "공포(매수) ↔ 공포 ↔ 중립 ↔ 탐욕 ↔ 탐욕(매도) 단계로 요약합니다."
)

# VIX 값은 위에서 이미 계산한 vix 사용 (Yahoo → FRED 백업 로직)
vix_value = None if isnan(vix) else float(vix)

# VXN은 FRED에서 직접 호출 (VXNCLS)
df_vxn = fred_series("VXNCLS")
if df_vxn is not None and not df_vxn.empty:
    vxn_value = float(df_vxn["VXNCLS"].dropna().iloc[-1])
else:
    vxn_value = float("nan")

sentiment_data = {
    "S&P 500 (VIX)": vix_value,
    "나스닥 100 (VXN)": None if isnan(vxn_value) else vxn_value,
}

cols = st.columns(2)

for (index_name, value), col in zip(sentiment_data.items(), cols):
    with col:
        # 지수 이름
        st.markdown(
            f"<div style='font-size:16px; color:#555;'>{index_name}</div>",
            unsafe_allow_html=True,
        )

        if value is None:
            # 데이터 없을 때
            st.markdown(
                "<div style='font-size:32px; font-weight:600; margin-top:8px;'>데이터 없음</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='font-size:24px; color:#e53935; margin-top:4px;'>&darr;</div>",
                unsafe_allow_html=True,
            )
        else:
            level = vol_to_level(value)
            st.markdown(
                f"<div style='font-size:32px; font-weight:700; margin-top:8px;'>{level}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:18px; color:#666; margin-top:4px;'>지수 값: {value:.2f}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<div style='font-size:24px; color:#e53935; margin-top:4px;'>&darr;</div>",
                unsafe_allow_html=True,
            )

st.markdown(
    """
    <div style='font-size:13px; color:#777;'>
    단계: 공포(매수) &larr; 공포 &larr; 중립 &larr; 탐욕 &larr; 탐욕(매도)  
    *구간 예시: 35↑ 공포(매수), 25~35 공포, 18~25 중립, 12~18 탐욕, 12↓ 탐욕(매도)
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------
# 뉴스 리스크 모니터 (Market News Risk Radar)
# ---------------------
st.markdown("---")
st.subheader("뉴스 리스크 모니터 (Market News Risk Radar)")
st.caption(
    "AI 투자, 거시 환경, 신용 스트레스, 지정학 리스크 등 핵심 테마별로 "
    "최근 3일간 글로벌 뉴스 톤을 간단히 스코어링해 어디에 리스크가 쌓이는지 보여줍니다."
)


def fetch_news(keywords, days=3, page_size=30):
    """
    NewsAPI를 이용해 키워드 기반 뉴스 검색.
    - NEWS_API_KEY 미설정 또는 오류 시 빈 리스트 반환.
    - UI 쪽에서 조용히 처리하기 위해 여기서 예외를 삼킵니다.
    """
    if not NEWS_API_KEY:
        return []

    from_date = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    query = " OR ".join([f'"{kw}"' for kw in keywords])

    params = {
        "q": query,
        "from": from_date,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": NEWS_API_KEY,
    }

    try:
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("articles", [])
    except Exception:
        return []


def analyze_sentiment(articles):
    """
    매우 단순한 키워드 기반 부정/긍정 스코어.
    - 부정 단어가 많을수록 score ↑ → 위험 증가.
    - 긍정 단어는 일부 상쇄.
    """
    NEGATIVE_WORDS = ["cut", "slowdown", "delay", "freeze", "reduce", "weak", "down", "outflow", "default"]
    POSITIVE_WORDS = ["growth", "expansion", "increase", "record", "raise", "inflow", "strong"]

    total_score = 0
    for art in articles:
        text = (art.get("title", "") + " " + (art.get("description") or "")).lower()
        neg = sum(w in text for w in NEGATIVE_WORDS)
        pos = sum(w in text for w in POSITIVE_WORDS)
        total_score += max(0, neg - 0.5 * pos)

    if total_score <= 0:
        status = "✅ 양호"
        color = "green"
    elif total_score < 4:
        status = "🟡 주의"
        color = "orange"
    else:
        status = "🔴 경고"
        color = "red"

    return status, color, total_score


# 모니터링 테마 설정: 필요시 키워드 수정해서 운용
NEWS_TOPICS = {
    "AI·데이터센터 CapEx":
        ["AI capex slowdown", "data center capex cut", "hyperscaler capex", "NVIDIA order cut"],
    "거시·금리·유동성":
        ["rate hike", "rate cut", "bond market turmoil", "liquidity squeeze", "yield curve inversion"],
    "신용·부도·유동성 경색":
        ["credit spread widening", "high yield stress", "default wave", "bank failure", "fund redemption"],
    "지정학·원자재·에너지":
        ["geopolitical tension", "war", "sanctions", "oil price spike", "shipping disruption", "Red Sea crisis"],
}

if not NEWS_API_KEY:
    st.warning("⚠️ NEWS_API_KEY가 설정되지 않았습니다. .env 파일에 NewsAPI 키를 입력해 주세요.")
else:
    cols = st.columns(len(NEWS_TOPICS))
    topic_results = []

    # 1) 테마별 요약 박스
    for (topic, keywords), col in zip(NEWS_TOPICS.items(), cols):
        articles = fetch_news(keywords)
        if articles:
            status, color, score = analyze_sentiment(articles)
            topic_results.append((topic, status, color, score, articles))

            with col:
                st.markdown(f"**{topic}**")
                st.markdown(
                    f"<span style='color:{color}; font-weight:bold'>{status}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(f"관련 기사 {len(articles)}건 · 리스크 스코어 {score}")
        else:
            topic_results.append((topic, "데이터 부족", "gray", 0, []))
            with col:
                st.markdown(f"**{topic}**")
                st.caption("최근 3일간 뚜렷한 관련 키워드 기사가 없습니다.")

    # 2) 가장 스코어가 높은(부정 뉴스가 많은) 테마 상세 표시
    if any(tr[4] for tr in topic_results):
        worst_topic, worst_status, worst_color, worst_score, worst_articles = max(
            topic_results, key=lambda x: x[3]
        )

        if worst_articles and worst_score > 0:
            st.markdown("---")
            st.markdown(
                f"**현재 가장 주의해야 할 이슈 영역:** {worst_topic} — "
                f"<span style='color:{worst_color}; font-weight:bold'>{worst_status}</span> "
                f"(스코어 {worst_score})",
                unsafe_allow_html=True,
            )
            st.caption("해당 영역에서 선별한 대표 기사입니다. 톤과 빈도, 맥락을 함께 확인하세요.")

            for art in worst_articles[:5]:
                st.markdown(
                    f"- [{art['title']}]({art['url']}) — "
                    f"{art['source']['name']} ({art['publishedAt'][:10]})"
                )
    else:
        st.info("모니터링된 주요 리스크 테마에서 뚜렷한 부정적 뉴스 축적이 감지되지 않았습니다.")

# ---------------------
# 주요 지수별 상위 10개 종목 지표 (Finnhub 기반, 시가총액 순서 + 한글 이름)
# ---------------------
st.markdown("---")
st.subheader("🏛️ 주요 지수별 상위 10개 종목 현황 (Finnhub, 시가총액 순)")
st.caption(
    "각 지수별 시가총액 기준 상위 10개 대표 종목을 고정 리스트로 두고, "
    "Finnhub 무료 API의 /quote 데이터를 사용해 가격·등락률을 표시합니다. "
    "종목 이름은 한국어(영문티커) 형태로 표기합니다."
)

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "").strip()

# ✅ 시가총액 순서로 정렬된 대표 종목 리스트
INDEX_TOP10 = {
    "S&P 500": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META",
        "GOOGL", "BRK.B", "TSLA", "UNH", "XOM"
    ],
    "Nasdaq 100": [
        "AAPL", "MSFT", "NVDA", "AMZN", "META",
        "GOOGL", "AVGO", "TSLA", "PEP", "COST"
    ],
    "Dow Jones": [
        "UNH", "MSFT", "GS", "HD", "MCD",
        "V", "CAT", "AMGN", "CRM", "AAPL"
    ],
    "Russell 2000": [
        "SMCI", "CELH", "APPF", "INSM", "RPD",
        "TMDX", "ENPH", "RUN", "BLDR", "IOT"
    ],
}

# ✅ 한국어 이름 매핑
KOREAN_NAME_MAP = {
    "AAPL": "애플",
    "MSFT": "마이크로소프트",
    "AMZN": "아마존닷컴",
    "NVDA": "엔비디아",
    "GOOGL": "알파벳A",
    "META": "메타 플랫폼스",
    "BRK.B": "버크셔 해서웨이 B",
    "TSLA": "테슬라",
    "UNH": "유나이티드헬스그룹",
    "XOM": "엑손모빌",
    "AVGO": "브로드컴",
    "PEP": "펩시코",
    "COST": "코스트코",
    "GS": "골드만삭스",
    "HD": "홈디포",
    "MCD": "맥도날드",
    "AMGN": "암젠",
    "V": "비자",
    "CAT": "캐터필러",
    "CRM": "세일즈포스",
    "SMCI": "슈퍼마이크로 컴퓨터",
    "CELH": "셀시어스 홀딩스",
    "APPF": "앱폴리오",
    "INSM": "인스메드",
    "RPD": "래피드7",
    "TMDX": "트랜스메딕스",
    "ENPH": "엔페이즈 에너지",
    "RUN": "선런",
    "BLDR": "빌더스 퍼스트소스",
    "IOT": "솜포나우(IOT 기업)"
}

@st.cache_data(ttl=60)
def finnhub_quotes(symbols, token):
    """여러 종목의 /quote 데이터 호출"""
    if not token:
        return pd.DataFrame()

    rows = []
    for sym in symbols:
        url = "https://finnhub.io/api/v1/quote"
        params = {"symbol": sym, "token": token}
        try:
            r = requests.get(url, params=params, timeout=6)
            r.raise_for_status()
            q = r.json()
        except Exception:
            continue

        c = q.get("c")  # current / close
        pc = q.get("pc")
        d = q.get("d")
        dp = q.get("dp")
        h = q.get("h")
        l = q.get("l")

        if not c and not pc:
            continue

        price = c if c not in (None, 0) else pc

        if d is None and price is not None and pc not in (None, 0):
            d = price - pc
        if dp is None and d is not None and pc not in (None, 0):
            dp = (d / pc) * 100

        display_name = KOREAN_NAME_MAP.get(sym, sym)
        rows.append({
            "종목": f"{display_name} ({sym})",
            "가격": round(price, 2) if price is not None else None,
            "등락(USD)": round(d, 2) if d is not None else None,
            "등락률(%)": round(dp, 2) if dp is not None else None,
            "고가": round(h, 2) if h not in (None, 0) else None,
            "저가": round(l, 2) if l not in (None, 0) else None,
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df


if not FINNHUB_API_KEY:
    st.error("⚠️ FINNHUB_API_KEY가 설정되지 않았습니다. .env 파일에 키를 넣어주세요.")
else:
    for index_name, tickers in INDEX_TOP10.items():
        st.markdown(f"### 📈 {index_name} 상위 10개 종목 (시가총액 순)")

        df = finnhub_quotes(tickers, FINNHUB_API_KEY)
        if df.empty:
            st.warning(f"{index_name} 종목 데이터 로딩 실패. Finnhub 키, 레이트리밋 또는 네트워크를 확인하세요.")
            continue

        # 시가총액 순서 유지 (INDEX_TOP10 순서 그대로)
        df["정렬"] = df["종목"].apply(
            lambda x: next((i for i, t in enumerate(tickers) if t in x), 999)
        )
        df = df.sort_values("정렬").drop(columns="정렬")

        st.dataframe(
            df.style.format({
                "가격": "{:,.2f}",
                "등락(USD)": "{:+.2f}",
                "등락률(%)": "{:+.2f}",
                "고가": "{:,.2f}",
                "저가": "{:,.2f}",
            }),
            use_container_width=True,
        )
