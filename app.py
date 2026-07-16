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
# COLUMN 2: LLM 기반 영화 추천 AI 챗봇 영역 (OpenRouter)
# ==========================================
with col2:
    st.subheader("🤖 OpenRouter 영화 추천 챗봇")
    st.caption("OpenRouter API를 통해 영화 추천을 받습니다.")
    
    # [왜] 주피터와 달리 스트림릿은 새로고침 시 대화 기록이 날아가므로 Session State에 저장
    # Session State는 브라우저 세션 동안 데이터를 유지해주는 스트림릿의 메모리 기능
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "system", "content": "너는 영화 전문가야. 사용자의 기분이나 취향을 듣고 영화를 추천해 줘. 한국어로 답변해줘."}
        ]

    # [왜] 기존 대화 기록을 화면에 출력 - 대화 흐름을 유지하기 위해
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # [왜] st.chat_input()은 채팅 형식의 입력창을 제공하는 스트림릿 전용 위젯
    # 사용자가 입력하면 조건이 True가 되어 코드 블록 실행
    if user_input := st.chat_input("예시: 오늘 우울한데 통쾌한 액션 영화 추천해줘!"):
        # 1. 유저 메시지 화면 표시 및 저장
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # 2. OpenRouter API 모델 호출
        # [왜] st.spinner()로 로딩 표시 - AI 응답 생성은 1~10초가 걸릴 수 있음
        # 사용자에게 "잠시만요"라는 피드백을 주어 좋은 UX 제공
        # with 문 안의 코드가 실행되는 동안 스피너가 화면에 표시됨
        with st.spinner("🤖 AI가 영화를 추천하는 중입니다... (잠시만요!)"):
            try:
                # [왜] OpenAI 호환 API를 사용 - OpenRouter도 OpenAI와 같은 API 형식 지원
                # base_url을 OpenRouter API 주소로 변경
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=st.secrets["OPENROUTER_KEY"]  # .streamlit/secrets.toml에 저장된 API 키
                )
                
                # [왜] model 파라미터에 OpenRouter 모델명 입력
                # meta-llama/llama-3.1-8b-instruct:free는 OpenRouter에서 제공하는 무료 모델
                response = client.chat.completions.create(
                    model="openrouter/free",
                    messages=st.session_state.messages
                )
                ai_response = response.choices[0].message.content
                
                # 3. AI 답변 화면 표시 및 저장
                with st.chat_message("assistant"):
                    st.write(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                # [왜] 예외 처리로 사용자 친화적인 에러 메시지 표시
                # API 키가 없거나 네트워크 오류 등의 경우 대비
                st.error(f"OpenRouter API 연결에 실패했습니다. API 키를 확인하거나 네트워크 연결을 확인하세요. 에러 내용: {e}")
