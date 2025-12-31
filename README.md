# 🍽️ 맛집 추천 챗봇 (Restaurant Recommendation Chatbot)

이 프로젝트는 **Streamlit**과 **LangChain**을 활용하여 사용자의 취향에 맞는 맛집을 추천해주는 AI 챗봇 애플리케이션입니다.

## 📂 프로젝트 구성 (Project Structure)

```
.
├── .env                # OpenAI API Key 등 환경 변수 설정 파일
├── .python-version     # Python 버전 정보
├── chatbot_logic.py    # 맛집 추천 로직 (LangChain & OpenAI)
├── main.py             # 애플리케이션 메인 엔트리 포인트
├── pyproject.toml      # 프로젝트 의존성 관리 설정 (uv)
└── uv.lock             # 의존성 잠금 파일
```

## 🛠️ 사전 요구 사항 (Prerequisites)

- **Python**: 3.9 이상
- **uv**: 파이썬 패키지 및 프로젝트 관리 도구
- **OpenAI API Key**: 앱 실행을 위해 필요합니다.

## 🚀 설치 및 실행 (Installation & Usage)

### 1. 프로젝트 설정 (Setup)

`uv`를 사용하여 가상환경을 생성하고 필요한 패키지를 설치합니다.

```bash
uv sync
```

### 2. 환경 변수 설정 (Environment Variables)

`.env` 파일에 OpenAI API Key를 설정합니다. (이미 `env.zip`을 통해 포함되어 있을 수 있습니다.)

```ini
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxx
```

### 3. 애플리케이션 실행 (Run)

다음 명령어로 Streamlit 앱을 실행합니다.

```bash
uv run streamlit run main.py
```

## 💡 주요 기능

- **위치, 인원, 장르, 가격, 특이사항** 등 5가지 정보를 입력받아 맞춤형 추천 제공
- **ChatOpenAI (GPT-3.5-turbo)** 기반의 자연스러운 추천 메시지 생성
- **LangChain**을 이용한 효율적인 프롬프트 관리 및 체인 실행

---
**Note**: API Key가 없는 경우 사이드바에서 직접 입력하여 사용할 수도 있습니다.
