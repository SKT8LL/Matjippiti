
def recommend_restaurant(location, people, genre, price, notes):

    import os
    from langchain_community.document_loaders import CSVLoader
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

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
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

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

recommend_restaurant('강남', '디저트', '2', '30000', '비슷한 음식점 추천해줘.')