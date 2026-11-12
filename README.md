# News_Content_Analysis_with_GPT-5.4: News LLM Coding Toolkit (GPT-5.4-mini)

OpenAI API의 **GPT-5.4-mini**를 활용해 **뉴스 기사(텍스트)와 정치 이미지를 자동으로 내용분석(content analysis)** 하는 Jupyter Notebook 모음입니다.
기사 제목·본문이 담긴 CSV를 입력하면, 분석 유형에 맞는 결과(분류값 · 근거 · 신뢰도 등)를 CSV로 저장합니다.

> 이 저장소는 [News_Content_Analysis_with_GPT-4.1](https://github.com/HanJunLeeKR/News_Content_Analysis_with_GPT-4.1)의 **후속 버전**입니다. 파이프라인(비동기 배치 처리, 견고한 JSON 파서, 중간 저장)은 그대로 이어받고, 모델과 API 호출 방식을 GPT-5.4-mini에 맞게 바꾸었습니다. 달라진 점은 아래 「GPT-4.1 버전과의 차이」를 참고하세요.

## 노트북 구성

| 노트북 | 목적 | 결과 컬럼 |
|---|---|---|
| `GPT-5.4-mini_framing_analysis.ipynb` | 뉴스 기사를 5가지 프레임(갈등 / 책임귀인 / 인간적 흥미 / 경제적 결과 / 도덕성; Semetko & Valkenburg, 2000) 중 하나로 분류 | `프레임`, `근거`, `신뢰도`, `비용` |
| `GPT-5.4-mini_binary_coding_analysis.ipynb` | 연구자가 정의한 항목에 해당하는지를 예(1) / 아니오(0)로 판단하는 이진 코딩 | `코딩결과`, `근거`, `신뢰도`, `비용` |
| `GPT-5.4-mini_news_summary.ipynb` | 기사를 정확히 3문장으로 요약 (5W1H 기반) | `요약`, `비용` |
| `GPT-5.4-mini_actor_analysis.ipynb` 🆕 | 기사에 등장하는 행위자를 개인 레벨(실명)과 제도 레벨(정당·기관·단체)로 추출 | `개인행위자`, `제도행위자`, `비용` |
| `GPT-5.4-mini_universal_actor_analysis.ipynb` 🆕 | 정치·경제·종교·미디어·국제·스포츠 등 모든 도메인에 적용 가능한 범용 행위자 추출 (활동명 포함, 외국인은 성만 추출, 플랫폼 제외) | `개인행위자`, `제도행위자`, `비용` |
| `GPT-5.4-mini_political_image_analysis.ipynb` 🆕 | 정치인 이미지(URL)를 3차원(이미지 내용 유형 / 감정적 어조 / 후보자 표현 방식)으로 분류하는 Vision 분석 | `이미지유형`, `감정톤`, `후보표현방식`, `근거`, `신뢰도`, `비용` |

🆕 = GPT-4.1 저장소에 없던 노트북. GPT-4.1 저장소의 관점(perspective) 분석 노트북은 이번 저장소에 포함하지 않았습니다.

## GPT-4.1 버전과의 차이

| 항목 | GPT-4.1 저장소 | GPT-5.4-mini 저장소 (이 저장소) |
|---|---|---|
| 모델 | `gpt-4.1-mini` | `gpt-5.4-mini` |
| API | Chat Completions (`chat.completions.create`) | **Responses API** (`responses.create`) |
| 출력 형식 | `response_format={"type": "json_object"}` + 프롬프트 속 JSON 예시 | **JSON Schema 구조화 출력** (`strict: true`). 필드·타입을 스키마로 강제하고, 신뢰도는 1–100 정수, 이진 코딩은 0/1로 제한 |
| 프롬프트 | 한국어 지시문 + JSON 답변 양식 | 영어 지시문 + 스키마 (분석 기준 `ANALYSIS_CRITERIA` 등 연구자가 채우는 부분은 한국어 유지) |
| 추론·샘플링 | `temperature=0.0` 고정 | `reasoning_effort="none"`, `text_verbosity="medium"`. **`temperature`는 기본 미지정(`None`)** |
| 응답 토큰 상한 | `max_tokens=500` | `max_output_tokens=1024` (요약 512, 범용 행위자 4096) |
| 입력 길이 제한 | 제목+본문 최대 4,000자 | 제목+본문 최대 10,000자 |
| 동시 요청 수 | 3 | 2 (`rate_limit_delay=1.0`, `batch_size=5`는 동일) |
| 비용 컬럼 | 기사당 **$0.002 고정 추정치** | 응답의 `usage` 토큰 수 × 단가로 계산 (단가는 설정 셀의 `PRICE_INPUT_PER_1M`, `PRICE_OUTPUT_PER_1M`) |
| API 키 | 문자열 placeholder를 직접 수정 | 환경변수 `OPENAI_API_KEY`를 우선 사용 (없으면 placeholder) |
| 실행 환경 | 일부 노트북 첫 셀에 Colab 전용 `drive.mount` | Colab 마운트 셀 제거, 로컬(VS Code) 기준 |
| 노트북 | 프레이밍 / 관점 / 이진 코딩 / 요약 | 프레이밍 / 이진 코딩 / 요약 / 행위자 / 범용 행위자 / 정치 이미지 |

### 연구에 사용할 때 알아둘 점

- **재현성**: GPT-4.1 버전은 `temperature=0.0`이었지만 이 버전은 기본값이 `None`입니다. 동일한 설정을 원하면 설정 셀에서 `TEMPERATURE = 0.0`으로 지정하세요(`reasoning_effort="none"`일 때만 유효).
- **결과 비교 가능성**: 모델뿐 아니라 프롬프트, 입력 길이 제한(4,000자 → 10,000자), 출력 형식이 함께 바뀌었으므로 GPT-4.1 결과와 이 버전의 결과를 그대로 비교하면 안 됩니다. 연구자의 직접 코딩과 비교한 **신뢰도 검증**(예: Cohen's κ)을 버전별로 따로 수행하세요.
- **프레임 라벨 정규화**: GPT-4.1 프레이밍 노트북에는 모델이 돌려준 프레임 이름을 허용 목록에 맞추는 `normalize_frame`이 있었지만, 이 버전에는 없습니다. 프레임 이름은 프롬프트로만 제한되므로 결과의 `프레임` 값이 5개 범주와 정확히 일치하는지 점검하세요.
- **비용**: `비용` 컬럼은 토큰 수와 설정한 단가로 계산한 **추정치**입니다. 단가는 [OpenAI 모델 문서](https://developers.openai.com/api/docs/models/gpt-5.4-mini) 기준 입력 $0.75 / 출력 $4.50 (100만 토큰당)로 설정되어 있으며, 가격이 바뀌면 수정하세요. 실제 과금액은 OpenAI 대시보드에서 확인하세요. 같은 문서 기준 컨텍스트 윈도우는 400K 토큰입니다.

## 공통 동작 방식

- 기사 **제목 + 본문**을 합쳐 프롬프트에 삽입합니다 (최대 10,000자).
- 기사마다 **독립된 API 호출**을 사용해 이전 기사의 영향을 받지 않습니다.
- JSON 파싱이 실패할 경우를 대비한 다단계 파서(`EnhancedJSONParser`)를 포함합니다.
- 비동기(`asyncio`) 배치 처리 + 진행바(`tqdm`) + **5배치마다 중간 저장**을 지원합니다.
- 분석에 실패한 기사는 `분류 실패`(프레임·이미지), `-1`(이진 코딩), 빈 문자열(행위자), `요약 실패`(요약)로 표시됩니다.

## 입력 데이터 형식

CSV 파일(UTF-8)에 아래 컬럼이 **반드시** 있어야 합니다.

| 노트북 | 필수 컬럼 | 설명 |
|---|---|---|
| 텍스트 분석 5종 | `title`, `text` | 기사 제목, 기사 본문 |
| 정치 이미지 분석 | `image` | 이미지 URL 문자열 |

그 외 컬럼은 결과 CSV에 그대로 유지됩니다. 데이터 파일은 이 저장소에 포함되어 있지 않습니다.

## 사용 방법

### 1. 패키지 설치

```bash
pip install -r requirements.txt
```

(Colab/Jupyter에서는 각 노트북의 설치 셀로도 가능합니다. GPT-5.4-mini와 Responses API를 쓰려면 `openai` 최신 버전이 필요합니다.)

### 2. 설정 입력

각 노트북 **마지막 실행 셀**의 설정 블록을 수정합니다.

```python
MODEL = "gpt-5.4-mini"
API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY_HERE")  # 환경변수 권장
INPUT_CSV_PATH = "Input CSV file path here"
OUTPUT_CSV_PATH = "Output CSV file path here"

REASONING_EFFORT = "none"   # none | low | medium | high | xhigh
TEXT_VERBOSITY = "medium"   # low | medium | high
TEMPERATURE = None          # reasoning_effort="none"일 때만 유효 (재현성이 필요하면 0.0)
```

### 3. 실행

노트북을 위에서부터 순서대로 실행하면 결과 CSV가 `OUTPUT_CSV_PATH`에 저장됩니다.

## 연구에 맞게 수정하기

- **프레임 / 판단 기준**: 각 노트북의 `ALLOWED_FRAMES`, `ANALYSIS_CRITERIA`, `CODING_SCHEME`과 `PROMPT_TEMPLATE`을 연구 주제에 맞게 수정하세요.
- **이진 코딩 항목**: 이진 코딩 노트북의 `ANALYSIS_CRITERIA`와 `PROMPT_TEMPLATE`에는 `[코딩 대상 항목]` 같은 **자리표시자(placeholder)** 가 들어 있습니다. 실행 전에 본인의 연구 대상과 판단 기준으로 반드시 교체하세요.
- **행위자 분석**: 행위자 노트북의 `ANALYSIS_CRITERIA`에 든 인물·기관 이름은 **예시**입니다. 연구 맥락에 맞게 바꾸세요.
- **추론 강도**: 단순 분류·추출·요약에는 `none`이 적합합니다. 판단이 어려운 코딩은 `low` 이상을 시험해 보되, 그때는 `temperature`를 지정할 수 없고 비용이 늘어납니다.

## 보안 및 데이터 관련 주의

- **API 키를 코드에 직접 입력한 상태로 커밋/푸시하지 마세요.** 환경변수 `OPENAI_API_KEY` 사용을 권장합니다.
- 뉴스 기사 원문 CSV와 결과 CSV에는 저작권이 있는 텍스트가 포함될 수 있으므로 저장소에 올리지 않는 것을 권장합니다 (`.gitignore`에 `*.csv` 포함).
- LLM의 분류 결과는 연구 목적으로 사용할 때 **연구자의 직접 코딩과의 신뢰도 검증**(예: Cohen's κ)을 거치는 것이 좋습니다.

## 참고문헌 (노트북에 명시된 것)

- Semetko, H. A., & Valkenburg, P. M. (2000). Framing European politics: A content analysis of press and television news. *Journal of Communication, 50*(2), 93–109.
- 오나영·노지현·안순태 (2025). 세대 주거문제에서 사회적 재난으로: 전세사기, 미디어 담론과 댓글. 〈한국언론학보〉, 69권 5호, 5–42.
- Ali, S., & Wongsawat, A. (2024). A Multi-Stage Prompt Framework for High-Quality News Summarization with Large Language Models. (뉴스 요약 프롬프트 참고)
- 정치 이미지 코딩 스킴: Grabe & Bucy (2009), Brader (2006), Schill (2012) (노트북 설명 기준)

> 위 참고문헌은 사용자의 노트북에 적힌 내용을 옮긴 것이며, 이번 정리 과정에서 서지정보를 별도로 검증하지는 않았습니다.
