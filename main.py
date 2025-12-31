# 필요한 라이브러리 설치 명령어
# uv add streamlit langchain langchain-openai python-dotenv

import streamlit as st
import os
from dotenv import load_dotenv
from chatbot_logic import get_restaurant_recommendation, get_chat_response, PERSONAS

# 0. 환경 변수 로드
load_dotenv()

# 1. 페이지 기본 설정
st.set_page_config(page_title="흑백요리사 맛집 추천 🍽️", layout="wide")
st.title("🍽️ 흑백요리사 맛집 AI 비서")

# 2. 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "persona" not in st.session_state:
    st.session_state.persona = "백종원" # 기본값

# 3. 사이드바 설정
with st.sidebar:
    st.header("설정")
    
    # [수정] 처음부터 다시 시작 버튼 (최상단)
    if st.button("🔄 처음부터 다시 시작", use_container_width=True):
        st.session_state.messages = []
        st.session_state.form_submitted = False
        st.rerun()
        
    st.markdown("---")
    st.subheader("🧑‍🍳 셰프 선택")
    
    # [수정] 인격 교체 버튼들
    # 현재 선택된 셰프 표시
    current_persona = PERSONAS[st.session_state.persona]
    st.info(f"현재 셰프: **{current_persona['name']}** {current_persona['emoji']}")
    
    st.markdown("셰프를 선택하면 **새로운 대화**가 시작됩니다.")
    
    # col1, col2 = st.columns(2)
    
    chefs = list(PERSONAS.keys())
    
    for i, chef_name in enumerate(chefs):
        chef = PERSONAS[chef_name]
        # 2열로 배치
        # with (col1 if i % 2 == 0 else col2):
        if st.button(f"{chef['emoji']} {chef['name']}", key=f"btn_{chef_name}", use_container_width=True):
            st.session_state.persona = chef_name
            st.session_state.messages = [] # 대화 내용 초기화
            st.session_state.form_submitted = False # 폼으로 돌아가기
            st.rerun()

    st.markdown("---")
    
    # [수정] API Key Input 제거
    # 환경변수 확인
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        st.success("✅ API Key가 로드되었습니다.")
    else:
        st.error("⚠️ .env 파일에 OPENAI_API_KEY가 없습니다.")
        st.info("프로젝트 폴더의 .env.example 파일을 .env로 변경하고 키를 입력해주세요.")

# 4. Form 보여주기 (아직 제출 안 했을 경우)
if not st.session_state.form_submitted:
    # 현재 선택된 페르소나의 설명 보여주기
    persona_info = PERSONAS[st.session_state.persona]
    st.markdown(f"### {persona_info['emoji']} {persona_info['name']} 셰프가 맛집을 찾아그려유~")
    st.caption(f"\"{persona_info['description']}\"")
    
    with st.form("restaurant_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            location = st.text_input("📍 위치 (예: 제주도, 강남역)", placeholder="어디서 드시나요?")
            genre = st.text_input("🍕 장르 (예: 흑돼지, 파스타)", placeholder="어떤 음식을 선호하시나요?")
        
        with col2:
            people = st.number_input("👥 인원", min_value=1, max_value=50, value=2, step=1)
            price = st.selectbox("💰 가격대", ["상관없음", "1만원 이하", "1~3만원", "3~5만원", "5~10만원", "10만원 이상"])
        
        notes = st.text_area("📝 특이사항", placeholder="예: 뷰가 좋은 곳, 주차 필수, 노키즈존 제외 등...", height=100)
        
        submitted = st.form_submit_button("맛집 추천받기 🚀")
        
        if submitted:
            if not api_key:
                st.error("⚠️ OpenAI API Key가 설정되지 않았습니다. .env 파일을 확인해주세요.")
            else:
                with st.spinner(f"{st.session_state.persona} 셰프가 맛집을 고민 중입니다... 🤔"):
                    try:
                        # 1. 추천 받기 (페르소나 전달)
                        response = get_restaurant_recommendation(
                            api_key, location, people, genre, price, notes, 
                            persona_name=st.session_state.persona
                        )
                        
                        # 2. 상태 업데이트
                        st.session_state.form_submitted = True
                        
                        # 3. 메시지 기록에 추가
                        user_summary = f"위치: {location}, 인원: {people}명, 장르: {genre}, 가격: {price}, 특이사항: {notes}"
                        st.session_state.messages.append({"role": "user", "content": f"맛집 추천 요청: {user_summary}"})
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"오류 발생: {str(e)}")

# 5. Chat Interface (폼 제출 후)
else:
    # 상단에 현재 셰프 표시
    st.caption(f"현재 대화 중인 셰프: {st.session_state.persona}")
    
    # 이전 대화 내용 출력
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력 대기
    if prompt := st.chat_input("추가로 궁금한 점이 있나요?"):
        if not api_key:
            st.error("API Key가 필요합니다.")
        else:
            # 사용자 메시지 표시 및 저장
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("생각 중..."):
                    try:
                        # 페르소나 전달
                        response = get_chat_response(
                            st.session_state.messages, 
                            api_key, 
                            persona_name=st.session_state.persona
                        )
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"오류 발생: {str(e)}")
