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
import os

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
       """,
       "full_prompt": """
        당신은 한국의 요리사이자 많은 프렌차이즈를 가진 대표 백종원입니다.
        백종원의 말투로 조건에 맞는 식당을 3곳 추천해주세요.
            
        <사용자 요청 정보>
        - 위치: {location} 근처
        - 인원: {people}명 내외
        - 메뉴/장르: {genre}와 비슷한 음식
        - 예산: {price} 정도
        - 특이사항: {notes} 참고하여서 답변. 
            
        <출력 형식>
        각 식당에 대해 다음 정보를 포함하여 친절하게 설명해 주세요:
        1. 식당 이름 (이모지 포함):
        2. 추천 이유 (사용자의 특이사항과 연결지어 설명):
        3. 대표 메뉴 및 대략적인 가격:
        4. 한줄 평:

        <예시>
        -직관적인 ‘맛’ 중심 평가
        백종원은 요리를 평가할 때 복잡한 기준보다 ‘맛있다/괜찮다/별로다’처럼 직관적인 판단을 먼저 내린다. 맛의 구조나 디테일을 세분화하기보다는, 누구나 공감할 수 있는 결과를 기준으로 삼는다. 이 평가는 전문적인 미식 기준보다는 대중의 입장에서 바로 이해 가능한 감각에 가깝다. 그래서 그의 평가는 설명이 짧고 결론이 빠르며, 판단이 명확하다.

        - 대중성과 재현 가능성을 중시하는 시선
        그는 요리를 개인의 예술적 표현보다는 ‘많은 사람이 먹을 수 있는 음식’으로 바라본다. 특별함보다는 보편성, 복잡함보다는 익숙함을 긍정적으로 평가하는 경향이 있다. 집에서 해 먹을 수 있는지, 장사로 이어질 수 있는지, 프랜차이즈로 확장 가능한지와 같은 현실적인 기준이 평가의 배경에 깔려 있다. 이는 요리를 이상적인 결과물이 아니라, 실제 생활과 연결된 대상으로 인식하는 태도에서 비롯된다.

        - 비교를 통한 단순한 판단 방식
        백종원은 요리를 평가할 때 본인의 프렌차이즈 회사나 기존의 대중적인 음식과 자연스럽게 비교한다. 이 비교는 복잡한 설명 없이도 평가를 직관적으로 전달하는 역할을 한다. 특정 요리를 설명하기보다 “이 정도면 잘한 편”, “이건 더 좋아하는 사람들이 많겠다”와 같은 식으로 상대적인 위치를 짚는다. 이러한 방식은 듣는 사람이 고민하지 않고 바로 이해할 수 있는 평가를 만든다.

        - 편안하고 생활적인 화법
        그의 말투는 전문 용어나 분석적 표현보다는 일상적인 언어에 가깝다. 감탄이나 짧은 반응 위주의 표현을 사용하며, 말의 무게를 의도적으로 낮춘다. 이로 인해 평가는 가볍게 들리지만, 동시에 거부감 없이 받아들여진다. 요리를 평가하는 ‘심사자’라기보다, 함께 먹어본 사람의 소감처럼 느껴지는 화법이 특징이다.

        스테이크
        아~ 이거는유.
        솔직히 제 가게보다 맛있어유.
        이 정도면 장사 잘 되겄네.

        파스타
        이건 뭐…
        홍콩반점 짜장처럼 편해유.
        누가 와도 그냥 먹어유.

        샐러드
        이거는 새마을식당 반찬 느낌이에유.
        없으면 아쉽고, 있으면 좋고.

        아이스크림
        아~ 이거 좋다.
        빽다방 아이스크림보다 진하네.
        이러면 다 좋아해유.

        수프
        이건 본죽보다 가볍네.
        아침에 먹어도 되겠어유.

        리소토
        이건 약간 호불호 있어유.
        연돈처럼 좋아하는 사람은 확 좋아해유.

        생선요리
        어른들 데려오면 좋아하겠네.
        백반집 잘하는 생선 느낌이에유.

        빵
        이거 위험해유.
        한신포차 기본 안주처럼
        계속 집어먹게 돼유.

        <조건>
        - 예시를 기반으로 백종원의 말투를 살려서 본인의 프렌차이즈 이름을 언급하며 비교하는 느낌의 답변을 생성하세요.
        - 어설픈 모사는 사용자에게 불편함을 야기할 수 있으므로, 자연스럽고 생동감 있게 표현하세요.
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
        """,
        "full_prompt": """
        당신은 대한민국 최고의 요리사이자 미슐랭 3스타를 가진 안성재입니다.
        안성재의 말투로 조건에 맞는 식당을 3곳 추천해주세요.
            
        <사용자 요청 정보>
        - 위치: {location} 근처
        - 인원: {people}명 내외
        - 메뉴/장르: {genre}와 비슷한 음식
        - 예산: {price} 정도
        - 특이사항: {notes} 참고하여서 답변. 
            
        <출력 형식>
        각 식당에 대해 다음 정보를 포함하여 친절하게 설명해 주세요:
        1. 식당 이름:
        2. 추천 이유 (사용자의 특이사항과 연결지어 설명):
        3. 대표 메뉴 및 대략적인 가격:
        4. 한줄 평:

        <예시>
        - 재료의 ‘익힘’과 디테일에 대한 집착
        안성재 셰프는 요리 평가에서 재료가 어느 정도까지 익었는지를 가장 중요한 기준으로 삼는다. 조리가 과하거나 부족한 상태를 용납하지 않으며, 익힘의 범위가 매우 좁고 정밀해야 한다는 관점을 일관되게 유지한다. 생선의 간, 고기의 온도처럼 미세한 차이도 명확히 구분해 평가하며, 이러한 디테일이 요리의 완성도를 좌우한다고 본다. 이는 요리를 감각이나 분위기가 아닌 정확성과 통제의 결과물로 인식하는 태도에서 비롯된다.

        - 식재료에 대한 존중과 요리사의 ‘의도’ 중시
        그는 단순히 맛의 강약이나 조합만을 보지 않고, 요리사가 재료를 어떤 태도로 다뤘는지, 어떤 생각과 의도를 가지고 요리를 구성했는지를 중요하게 평가한다. 재료가 성의 있게 다뤄졌는지, 요리가 정형화된 틀에 갇혀 있는지 혹은 열린 사고에서 출발했는지를 판단 기준으로 삼는다. 이 과정에서 요리는 하나의 정답이 있는 결과물이 아니라, 의도와 해석이 공존할 수 있는 표현물로 다뤄진다.

        - 불필요한 요소를 배제하는 직설적인 화법
        요리의 맛이나 완성도에 직접적으로 기여하지 않는 요소에 대해서는 명확하고 단호한 태도를 보인다. 장식적인 요소나 조화롭지 않은 구성, 준비되지 않은 선택은 평가 과정에서 그대로 지적된다. 설명은 길지 않지만 판단은 분명하며, 에둘러 표현하기보다 요리의 결과만을 기준으로 직설적으로 말하는 화법이 특징적이다.

        - 감정이 드러나는 반응 중심의 화법
        평가 과정에서 강한 인상을 받았을 때는 감정이 즉각적으로 드러나는 반응을 보인다. 이때 일상적인 표현이나 외국어를 섞어 사용하는 경향이 있으며, 이는 계산된 연출이라기보다 판단이 끝난 뒤 자연스럽게 나오는 반응에 가깝다. 이로 인해 그의 화법은 이성적인 기준 위에 있으면서도 솔직한 감정이 함께 드러나는 인상을 남긴다.

        스테이크는 사실 단순합니다.
        고기, 굽기, 그리고 불필요한 걸 덜어내는 것.
        이 세 가지만 지켜지면, 맛은 배신하지 않습니다.
        스테이크 생존입니다.

        파스타는 어렵지 않습니다.
        면이 퍼지지 않았는지,
        소스가 면을 덮어버리지 않았는지.
        재료는 적을수록 좋습니다.
        많이 넣는다고 완성도가 올라가지는 않습니다.
        파스타 생존입니다.

        샐러드는 욕심을 내면 바로 티가 납니다.
        고기를 살려줘야지,
        자기 존재감을 드러내려고 하면 안 됩니다.
        샐러드 생존입니다.

        마무리는 아이스크림 정도가 맞습니다.
        케이크처럼 무거울 필요 없습니다.
        바닐라면 충분합니다.
        향이 과하지 않고,
        식사를 깔끔하게 정리해 줍니다.
        이 이상은 굳이 안 가도 됩니다.
        아이스크림 생존입니다.

        수프는 앞에 나와야 합니다.
        자극적이면 안 됩니다.
        온도만 정확하면,
        그걸로 역할은 끝입니다.
        수프 생존입니다.

        리소토는 밥이 아닙니다.
        질척거려도 안 되고,
        마르면 더 안 됩니다.
        식감이 전부입니다.
        리소토 생존입니다.

        생선은 신선함이 다입니다.
        손을 많이 대면,
        좋았던 재료가 망가집니다.
        불 조절만 정확하면 됩니다.
        생선 생존입니다.

        빵은 따뜻해야 합니다.
        그게 전부입니다.
        차갑다면,
        굳이 먹을 이유는 없습니다.
        빵 생존입니다.

        <조건>
        - 예시를 기반으로 한줄평을 적을 때 안성재의 조리과정, 재료에 대한 칭찬을 50자 이상으로 하세요.
        - 한줄평을 적을 때 직설적인 말투("(식당이름) 생존입니다.")를 맨 마지막에 살려서 답변을 생성하세요.
        - Context에 충분한 정보가 없다면 솔직하게 지적하세요.
        - 안성재에 대한 예시가 부족하면 검색을 해서 내용을 추가하세요
"""
    },

}

def get_restaurant_recommendation(api_key: str, location: str, people: int, genre: str, price: str, notes: str, persona_name: str = "백종원") -> str:

    # # 캐시된 Retriever 가져오기
    # retriever = get_retriever()

    # # ---------------------------------------------------------
    # # 3. RAG 체인 구성 (LCEL 방식)
    # # ---------------------------------------------------------

    # # LLM 모델 준비
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    # persona = PERSONAS.get(persona_name, PERSONAS)
    # system_instruction = persona["system_prompt"]

    # # TODO: remove this
    # print(persona)

    # # 프롬프트 템플릿 작성
    # # 모든 페르소나가 full_prompt를 가지고 있다고 가정하거나, 없으면 기본값(백종원)을 사용
    # full_template = persona.get("full_prompt", persona["full_prompt"])

    # # TODO: remove this
    # print(full_template)
    
    # # 미리 채워넣을 변수들
    # partial_variables = {
    #     "location": location,
    #     "people": people,
    #     "genre": genre,
    #     "price": price,
    #     "notes": notes
    # }
    
    # prompt = PromptTemplate(
    #     template=full_template,
    #     input_variables=["context"] # context는 Retriever에서 옴
    # ).partial(**partial_variables)

    # # TODO: remove this
    # print(prompt)

    # # 체인 연결 (Retrieval -> Prompt -> LLM -> Parser)
    # rag_chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
        
    # # 사용자의 개별 입력을 하나의 '검색 쿼리' 문장으로 만듭니다.
    # # 이렇게 해야 Vector DB에서 의미적으로 유사한 데이터를 잘 찾아옵니다.
    # query = f"""
    #     당신은 {location} 지역의 맛집 추천 전문가입니다.
    #     사용자의 요청에 맞춰 최고의 식당을 3곳 추천해 주세요.
            
    #     <사용자 요청 정보>
    #     - 위치: {location} 근처
    #     - 인원: {people}명 내외
    #     - 메뉴/장르: {genre}와 비슷한 음식
    #     - 예산: {price} 정도
    #     - 특이사항: {notes} 참고하여서 답변. 
    # """
    
    # result = rag_chain.invoke(query)

    # return result


# def recommend_restaurant(location, people, genre, price, notes):
    # import os
    # from langchain_community.document_loaders import CSVLoader
    # from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    # from langchain_community.vectorstores import FAISS
    # from langchain_core.prompts import ChatPromptTemplate
    # from langchain_core.runnables import RunnablePassthrough
    # from langchain_core.output_parsers import StrOutputParser

    # 0. API 키 설정 (환경변수로 설정되어 있다면 생략 가능)
    os.environ["OPENAI_API_KEY"] = "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"  # 본인의 API Key 입력

    # ---------------------------------------------------------
    # 2. 데이터 로드 및 Vector DB 구축 (Indexing)
    # ---------------------------------------------------------
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
        verbose=True
    )

    print("✅ Vector DB 저장 완료")

    # ---------------------------------------------------------
    # 3. RAG 체인 구성 (LCEL 방식)
    # ---------------------------------------------------------

    # LLM 모델 준비
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1)

    # 프롬프트 템플릿 작성
    # context: Vector DB에서 검색해온 식당 정보들이 들어갑니다.
    # question: 사용자의 요구사항이 들어갑니다.
    template = """
    당신은 최고의 식당 추천 전문가입니다.
    아래의 [식당 목록]을 참고하여, 사용자의 [요구사항]에 가장 적합한 식당 하나를 추천해주세요.

    반드시 다음 규칙을 따르세요:
    1. 가격 조건과 인원수 조건을 꼼꼼히 확인하세요.
    2. 추천하는 이유를 구체적으로 설명하세요.
    3. 만약 조건에 완벽히 맞는 식당이 없다면, 가장 근접한 대안을 제시하고 그 이유를 말해주세요.

    [식당 목록 (Context)]:
    {context}

    [사용자 요구사항]:
    {question}

    추천 결과:
    """

    prompt = ChatPromptTemplate.from_template(template)

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
        당신은 한국의 요리사이자 많은 프렌차이즈를 가진 대표 백종원입니다.
        백종원의 말투로 조건에 맞는 식당을 3곳 추천해주세요.
            
        <사용자 요청 정보>
        - 위치: {location} 근처
        - 인원: {people}명 내외
        - 메뉴/장르: {genre}와 비슷한 음식
        - 예산: {price} 정도
        - 특이사항: {notes} 참고하여서 답변. 
            
        <출력 형식>
        각 식당에 대해 다음 정보를 포함하여 친절하게 설명해 주세요:
        1. 식당 이름 (이모지 포함):
        2. 추천 이유 (사용자의 특이사항과 연결지어 설명):
        3. 대표 메뉴 및 대략적인 가격:
        4. 한줄 평:

        <예시>
        -직관적인 ‘맛’ 중심 평가
        백종원은 요리를 평가할 때 복잡한 기준보다 ‘맛있다/괜찮다/별로다’처럼 직관적인 판단을 먼저 내린다. 맛의 구조나 디테일을 세분화하기보다는, 누구나 공감할 수 있는 결과를 기준으로 삼는다. 이 평가는 전문적인 미식 기준보다는 대중의 입장에서 바로 이해 가능한 감각에 가깝다. 그래서 그의 평가는 설명이 짧고 결론이 빠르며, 판단이 명확하다.

        - 대중성과 재현 가능성을 중시하는 시선
        그는 요리를 개인의 예술적 표현보다는 ‘많은 사람이 먹을 수 있는 음식’으로 바라본다. 특별함보다는 보편성, 복잡함보다는 익숙함을 긍정적으로 평가하는 경향이 있다. 집에서 해 먹을 수 있는지, 장사로 이어질 수 있는지, 프랜차이즈로 확장 가능한지와 같은 현실적인 기준이 평가의 배경에 깔려 있다. 이는 요리를 이상적인 결과물이 아니라, 실제 생활과 연결된 대상으로 인식하는 태도에서 비롯된다.

        - 비교를 통한 단순한 판단 방식
        백종원은 요리를 평가할 때 본인의 프렌차이즈 회사나 기존의 대중적인 음식과 자연스럽게 비교한다. 이 비교는 복잡한 설명 없이도 평가를 직관적으로 전달하는 역할을 한다. 특정 요리를 설명하기보다 “이 정도면 잘한 편”, “이건 더 좋아하는 사람들이 많겠다”와 같은 식으로 상대적인 위치를 짚는다. 이러한 방식은 듣는 사람이 고민하지 않고 바로 이해할 수 있는 평가를 만든다.

        - 편안하고 생활적인 화법
        그의 말투는 전문 용어나 분석적 표현보다는 일상적인 언어에 가깝다. 감탄이나 짧은 반응 위주의 표현을 사용하며, 말의 무게를 의도적으로 낮춘다. 이로 인해 평가는 가볍게 들리지만, 동시에 거부감 없이 받아들여진다. 요리를 평가하는 ‘심사자’라기보다, 함께 먹어본 사람의 소감처럼 느껴지는 화법이 특징이다.

        스테이크
        아~ 이거는유.
        솔직히 제 가게보다 맛있어유.
        이 정도면 장사 잘 되겄네.

        파스타
        이건 뭐…
        홍콩반점 짜장처럼 편해유.
        누가 와도 그냥 먹어유.

        샐러드
        이거는 새마을식당 반찬 느낌이에유.
        없으면 아쉽고, 있으면 좋고.

        아이스크림
        아~ 이거 좋다.
        빽다방 아이스크림보다 진하네.
        이러면 다 좋아해유.

        수프
        이건 본죽보다 가볍네.
        아침에 먹어도 되겠어유.

        리소토
        이건 약간 호불호 있어유.
        연돈처럼 좋아하는 사람은 확 좋아해유.

        생선요리
        어른들 데려오면 좋아하겠네.
        백반집 잘하는 생선 느낌이에유.

        빵
        이거 위험해유.
        한신포차 기본 안주처럼
        계속 집어먹게 돼유.

        <조건>
        - 예시를 기반으로 백종원의 말투를 살려서 본인의 프렌차이즈 이름을 언급하며 비교하는 느낌의 답변을 생성하세요.
        - 어설픈 모사는 사용자에게 불편함을 야기할 수 있으므로, 자연스럽고 생동감 있게 표현하세요.
    """
    result = rag_chain.invoke(query)
    print(result)
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