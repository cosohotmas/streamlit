import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# 1. 스트림릿 페이지 레이아웃 설정 (넓은 화면 사용)
st.set_page_config(page_title="영화 데이터 & AI 챗봇 대시보드", layout="wide")
st.title("🎬 영화 인사이트 대시보드 & AI 챗봇")

# ------------------------------------------------------------------
# [임시 데이터] 네이버/Kaggle 형태의 영화 데이터셋 구성
# 실제 파일이 있다면: df = pd.read_csv("your_movie_data.csv")
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    data = {
        "title": ["괴물", "기생충", "인셉션", "인터스텔라", "올드보이", "부산행", "범죄도시"],
        "genre": ["스릴러", "드라마", "SF", "SF", "스릴러", "액션", "액션"],
        "rating": [8.6, 9.1, 8.8, 8.9, 9.0, 8.0, 8.3],
        "audience_count": [1300, 1030, 580, 1030, 320, 1150, 680], # 단위: 만 명
        "release_year": [2006, 2019, 2010, 2014, 2003, 2016, 2017]
    }
    return pd.DataFrame(data)

df = load_data()

# ------------------------------------------------------------------
# 레이아웃 분할: 왼쪽은 데이터 시각화, 오른쪽은 AI 챗봇
# ------------------------------------------------------------------
col1, col2 = st.columns([1, 1]) # 50:50 비율로 화면 분할

# ==========================================
# COLUMN 1: 네이버/Kaggle 데이터셋 시각화 영역
# ==========================================
with col1:
    st.subheader("📊 영화 데이터 트렌드 시각화")
    
    # 인터랙티브 필터 (장르 선택)
    genres = ["전체"] + list(df["genre"].unique())
    selected_genre = st.selectbox("🎯 분석할 장르 선택", genres)
    
    # 필터링 데이터 적용
    if selected_genre != "전체":
        filtered_df = df[df["genre"] == selected_genre]
    else:
        filtered_df = df

    # 시각화 1: 평점 vs 관객수 산점도 (Plotly 라이브러리 사용)
    st.write("**🎬 평점 및 관객수 분포**")
    fig_scatter = px.scatter(
        filtered_df, 
        x="rating", 
        y="audience_count", 
        text="title", 
        size="audience_count", 
        color="genre",
        labels={"rating": "네티즌 평점", "audience_count": "관객수 (만명)"},
        hover_name="title"
    )
    fig_scatter.update_traces(textposition='top center')
    st.plotly_chart(fig_scatter, use_container_width=True)

    # 시각화 2: 장르별 평균 평점 비교 바 차트
    st.write("**📌 장르별 평균 평점**")
    avg_rating = df.groupby("genre")["rating"].mean().reset_index()
    fig_bar = px.bar(
        avg_rating, 
        x="genre", 
        y="rating", 
        color="genre",
        labels={"genre": "장르", "rating": "평균 평점"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)


# ==========================================
# COLUMN 2: LLM 기반 영화 추천 AI 챗봇 영역
# ==========================================
with col2:
    st.subheader("🤖 AI 영화 추천 챗봇")
    st.caption("원하는 스타일의 영화나 대시보드 데이터에 대해 질문해 보세요.")
    
    # OpenAI API 키 입력 받기 (사이드바 혹은 텍스트 입력창)
    api_key = st.text_input("OpenAI API Key를 입력하세요", type="password")
    
    # 주피터와 달리 스트림릿은 새로고침 시 대화 기록이 날아가므로 Session State에 저장
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "너는 영화 전문가야. 사용자의 기분이나 취향을 듣고 영화를 추천해 줘."}
        ]

    # 기존 대화 기록 화면에 출력
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # 사용자 입력 창 만들기
    if user_input := st.chat_input("예시: 오늘 우울한데 통쾌한 액션 영화 추천해줘!"):
        if not api_key:
            st.error("API Key를 먼저 입력해야 대화를 시작할 수 있습니다.")
        else:
            # 1. 유저 메시지 화면 표시 및 저장
            with st.chat_message("user"):
                st.write(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})
            
            # 2. OpenAI LLM 모델 호출
            try:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini", # 가성비가 가장 좋은 최신 기본 모델
                    messages=st.session_state.messages
                )
                ai_response = response.choices[0].message.content
                
                # 3. AI 답변 화면 표시 및 저장
                with st.chat_message("assistant"):
                    st.write(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.error(f"에러가 발생했습니다: {e}")