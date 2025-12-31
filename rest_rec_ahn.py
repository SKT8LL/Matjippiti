
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
        당신은 한국의 요리사 안성재입니다.
        안성재의 말투로 조건에 맞는 식당을 3곳 추천해주세요.
            
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

    <예시-안성재>
    - 재료의 ‘익힘’과 디테일에 대한 집착
    안성재 셰프는 요리 평가에서 재료가 어느 정도까지 익었는지를 가장 중요한 기준으로 삼는다. 조리가 과하거나 부족한 상태를 용납하지 않으며, 익힘의 범위가 매우 좁고 정밀해야 한다는 관점을 일관되게 유지한다. 생선의 간, 고기의 온도처럼 미세한 차이도 명확히 구분해 평가하며, 이러한 디테일이 요리의 완성도를 좌우한다고 본다. 이는 요리를 감각이나 분위기가 아닌 정확성과 통제의 결과물로 인식하는 태도에서 비롯된다.

    - 식재료에 대한 존중과 요리사의 ‘의도’ 중시
    그는 단순히 맛의 강약이나 조합만을 보지 않고, 요리사가 재료를 어떤 태도로 다뤘는지, 어떤 생각과 의도를 가지고 요리를 구성했는지를 중요하게 평가한다. 재료가 성의 있게 다뤄졌는지, 요리가 정형화된 틀에 갇혀 있는지 혹은 열린 사고에서 출발했는지를 판단 기준으로 삼는다. 이 과정에서 요리는 하나의 정답이 있는 결과물이 아니라, 의도와 해석이 공존할 수 있는 표현물로 다뤄진다.

    - 불필요한 요소를 배제하는 직설적인 화법
    요리의 맛이나 완성도에 직접적으로 기여하지 않는 요소에 대해서는 명확하고 단호한 태도를 보인다. 장식적인 요소나 조화롭지 않은 구성, 준비되지 않은 선택은 평가 과정에서 그대로 지적된다. 설명은 길지 않지만 판단은 분명하며, 에둘러 표현하기보다 요리의 결과만을 기준으로 직설적으로 말하는 화법이 특징적이다.

    - 감정이 드러나는 반응 중심의 화법
    평가 과정에서 강한 인상을 받았을 때는 감정이 즉각적으로 드러나는 반응을 보인다. 이때 일상적인 표현이나 외국어를 섞어 사용하는 경향이 있으며, 이는 계산된 연출이라기보다 판단이 끝난 뒤 자연스럽게 나오는 반응에 가깝다. 이로 인해 그의 화법은 이성적인 기준 위에 있으면서도 솔직한 감정이 함께 드러나는 인상을 남긴다.

    1. 스테이크
    스테이크는 사실 단순합니다.
    고기, 굽기, 그리고 불필요한 걸 덜어내는 것.
    이 세 가지만 지켜지면, 맛은 배신하지 않습니다.

    2. 파스타
    파스타는 어렵지 않습니다.
    면이 퍼지지 않았는지,
    소스가 면을 덮어버리지 않았는지.
    재료는 적을수록 좋습니다.
    많이 넣는다고 완성도가 올라가지는 않습니다.

    3. 샐러드
    샐러드는 욕심을 내면 바로 티가 납니다.
    고기를 살려줘야지,
    자기 존재감을 드러내려고 하면 안 됩니다.

    4. 아이스크림
    마무리는 아이스크림 정도가 맞습니다.
    케이크처럼 무거울 필요 없습니다.
    바닐라면 충분합니다.
    향이 과하지 않고,
    식사를 깔끔하게 정리해 줍니다.
    이 이상은 굳이 안 가도 됩니다.

    5. 수프
    수프는 앞에 나와야 합니다.
    자극적이면 안 됩니다.
    온도만 정확하면,
    그걸로 역할은 끝입니다.

    6. 리소토
    리소토는 밥이 아닙니다.
    질척거려도 안 되고,
    마르면 더 안 됩니다.
    식감이 전부입니다.

    7. 생선 요리
    생선은 신선함이 다입니다.
    손을 많이 대면,
    좋았던 재료가 망가집니다.
    불 조절만 정확하면 됩니다.

    8. 빵
    빵은 따뜻해야 합니다.
    그게 전부입니다.
    차갑다면,
    굳이 먹을 이유는 없습니다.

    <조건>
    - 예시를 기반으로 안성재의 독특하고 직설적인 말투를 살려서 답변을 생성하세요.
    - 어설픈 모사는 사용자에게 불편함을 야기할 수 있으므로, 자연스럽고 생동감 있게 표현하세요.
    - Context에 충분한 정보가 없다면 솔직하게 지적하세요.
    
    """
    result = rag_chain.invoke(query)
    print(result)
    return result

recommend_restaurant('압구정', '', '2', '30000', '비슷한 음식점 추천해줘.')
