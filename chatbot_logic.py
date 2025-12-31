from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.document_loaders import CSVLoader
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ---------------------------------------------------------
# 2. 데이터 로드 및 Vector DB 구축 (Indexing) - 캐싱 적용
# ---------------------------------------------------------
@st.cache_resource
def get_retriever():
    # CSVLoader는 각 행(Row)을 하나의 문서(Document)로 변환합니다.
    loader = CSVLoader(file_path="DATA/restaurant.csv", encoding="utf-8")
    documents = loader.load()

    # 임베딩 모델 준비 (텍스트 -> 벡터 변환)
    embeddings = OpenAIEmbeddings()

    # Vector DB(FAISS)에 저장
    # 실무에서는 이 vectorstore를 로컬에 파일로 저장해두고 불러와서 씁니다.
    vectorstore = FAISS.from_documents(documents, embeddings)

    # 검색기(Retriever) 생성 (유사도 높은 상위 3개 추출)
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3},
        # verbose=True
    )
    print("✅ Vector DB 로드/생성 완료")
    return retriever

MODEL_NAME = "gpt-4o-mini"

PERSONAS = {
    "백종원": {
       "name": "백종원",
       "emoji": "👨‍🍳", 
       "description": "친근하고 대중적인 맛 표현, '재밌쥬?', '그렇쥬?' 말투",
       "system_prompt": """
       당신은 대한민국의 요리 연구가 '백종원'입니다.
       구수한 충청도 사투리를 사용하며, 친근하고 털털한 말투를 써주세요. "~했쥬?", "~그렇쥬?", "아이고~" 같은 표현을 자연스럽게 섞어주세요.
       음식의 '가성비'와 '대중적인 맛'을 중요하게 생각합니다.
       어려운 용어보다는 누구나 이해하기 쉬운 표현으로 설명해 주세요.
       """
    },
    "안성재": {
        "name": "안성재",
        "emoji": "🤵",
        "description": "엄격하고 디테일한 평가, '의도', '익힘 정도' 강조",
        "system_prompt": """
        당신은 국내 유일 미슐랭 3스타 셰프 '안성재'입니다.
        매우 정중하지만, 음식에 대해서는 타협하지 않는 엄격하고 진지한 말투를 사용합니다.
        "요리의 의도가 무엇인지", "채소의 익힘 정도", "간이 맞는지" 등 디테일에 집착하며 평가합니다.
        추천할 때도 셰프의 테크닉과 재료의 본질을 중요하게 설명해 주세요.
        """
    },
    "최강록": {
        "name": "최강록",
        "emoji": "🍳",
        "description": "독특한 화법, '~인데 이제... ~를 곁들인'",
        "system_prompt": """
        당신은 '마스터 셰프 코리아' 우승자 '최강록'입니다.
        다소 어눌하지만 매력적인 말투를 사용합니다. 
        "~인데 이제... ~를 곁들인...", "나야, 들기름." 같은 당신만의 독특한 화법이나 밈을 자연스럽게 사용해 주세요.
        조림 요리나 일식 베이스의 퓨전 요리에 대해 깊은 조예를 보여주세요.
        """
    },
    "이모카세": {
        "name": "이모카세 1호",
        "emoji": "👵",
        "description": "푸근한 이모님 스타일, 정감 있는 말투",
        "system_prompt": """
        당신은 시장에서 오랫동안 장사를 해온 '이모카세 1호'입니다.
        손님을 "우리 아들", "우리 딸" 처럼 부르며 매우 정감 있고 푸근하게 대해주세요.
        "맛있게 먹고 가~", "써비스 좀 더 줬어~" 같은 멘트를 사용하며, 한국적인 정(情)을 듬뿍 담아 추천해 주세요.
        안주 맛집이나 노포 감성을 잘 살려 답변해 주세요.
        """
    }
}

def get_restaurant_recommendation(api_key: str, location: str, people: int, genre: str, price: str, notes: str, persona_name: str = "백종원") -> str:

    # 캐시된 Retriever 가져오기
    retriever = get_retriever()

    # ---------------------------------------------------------
    # 3. RAG 체인 구성 (LCEL 방식)
    # ---------------------------------------------------------

    # LLM 모델 준비
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8, api_key=api_key)
    persona = PERSONAS.get(persona_name, PERSONAS["백종원"])
    system_instruction = persona["system_prompt"]

    # 프롬프트 템플릿 작성
    # 이미 변수들을 문자열 안에 포맷팅해서 넣어버립니다 (간소화)
    template_str = f"""
    {system_instruction}
    
    당신은 {location} 지역의 맛집 추천 전문가입니다.
    사용자의 요청에 맞춰 최고의 식당을 3곳 추천해 주세요.
    
    <사용자 요청 정보>
    - 위치: {location}
    - 인원: {people}명
    - 메뉴/장르: {genre}
    - 예산: {price}
    - 특이사항: {notes}
    
    <출력 형식>
    당신의 말투({persona_name})로 친절하게 설명해 주세요.
    각 식당에 대해 다음 정보를 포함해야 합니다:
    1. 식당 이름:
    2. 추천 이유 (당신의 관점에서 설명):
    3. 대표 메뉴 및 가격대:
    4. 한줄 평:
    
    [식당 목록 (Context)]:
    {{context}}

    [사용자 요구사항]:
    {{question}}

    마지막에는 당신의 캐릭터에 맞는 끝인사로 마무리해 주세요.
    """

    # PromptTemplate 생성
    # input_variables에는 실제로 chain.invoke() 할 때 들어올 변수만 남깁니다.
    prompt = PromptTemplate(
        template=template_str,
        input_variables=["context", "question"]
    )

    # 체인 연결 (Retrieval -> Prompt -> LLM -> Parser)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
        
    # 사용자의 개별 입력을 하나의 '검색 쿼리' 문장으로 만듭니다.
    # 이렇게 해야 Vector DB에서 의미적으로 유사한 데이터를 잘 찾아옵니다.
    query = f"""
        당신은 {location} 지역의 맛집 추천 전문가입니다.
        사용자의 요청에 맞춰 최고의 식당을 3곳 추천해 주세요.
            
        <사용자 요청 정보>
        - 위치: {location} 근처
        - 인원: {people}명 내외
        - 메뉴/장르: {genre}와 비슷한 음식
        - 예산: {price} 정도
        - 특이사항: {notes} 참고하여서 답변. 
    """
    
    result = rag_chain.invoke(query)

    return result

def get_chat_response(messages: list, api_key: str, persona_name: str = "백종원") -> str:
    """
    Get a response from the LLM based on conversation history with a specific persona.
    """
    llm = ChatOpenAI(
        model_name=MODEL_NAME,
        temperature=0.8,
        api_key=api_key
    )
    
    # 선택된 페르소나 가져오기
    persona = PERSONAS.get(persona_name, PERSONAS["백종원"])

    # Convert message history to LangChain format
    lc_messages = [
        SystemMessage(content=persona["system_prompt"])
    ]
    
    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))
            
    # Simple chat chain
    return llm.invoke(lc_messages).content