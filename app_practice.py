# app.py 실행을 위한 기본 도구 설치
import os
from dotenv import load_dotenv

import streamlit
import time
import requests # 외부 서버에 HTTP 요청하는 모듈
import tempfile # 임시 파일(temp file) 또는 임시 폴더(temp directory) 를 만들어 사용하는 모듈
import uuid # 고유한 ID(Unique ID) 식별자를 생성하는 모듈 (주로 세션 ID, 임시 Token 등에 사용)
import asyncio # 동시성(Asynchronous) 처리를 구현하는 모듈 (I/O 작업(네트워크 요청, 파일 읽기/쓰기 등)을 비동기적으로 실행하여 프로그램이 멈추지 않도록 함)
import pypdf
import semantic_kernel # Semantic Kernel 관련 라이브러리 추가

from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.text import text_chunker # text_chunker : 긴 텍스트를 작은 의미 단위(청크, chunk)로 분할하는 모듈
from semantic_kernel.functions.kernel_arguments import KernelArguments 
# Semantic Kernel 을 통해 Prompt 를 만들때, Prompt 형식을 구조화하여 전달하는 컨테이너
# KernelArguments 매개변수 : 사용자 입력(input) + 관련 문서/컨텍스트(context)

# .env 파일에서 환경 변수 로드
load_dotenv()

# Azure OpenAI 환경 변수 읽기
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_VERSION = os.getenv("AZURE_OPENAI_VERSION")
AZURE_OPENAI_MODEL = os.getenv("AZURE_OPENAI_MODEL")
AZURE_OPENAI_TEMPERATURE = float(os.getenv("AZURE_OPENAI_TEMPERATURE", 0.5))

# AI Search 환경 변수 읽기
AZURE_SEARCH_SERVICE_ENDPOINT = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_KEY_FIELD_NAME = os.getenv("AZURE_SEARCH_KEY_FIELD_NAME", "id")
AZURE_SEARCH_CONTENT_FIELD_NAME = os.getenv("AZURE_SEARCH_CONTENT_FIELD_NAME", "content")


# [1] Semantic Kernel 모델 선언 후
kernel = semantic_kernel.Kernel()

# [1-1] AzureChatCompletion() 서비스를 Semantic Kernel 에 추가
kernel.add_service(
    AzureChatCompletion(
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_VERSION,
        deployment_name=AZURE_OPENAI_MODEL,
    )
)


# [2] RAG 구현

# [2-1] 기존의 Azure AI Search 인덱스 및 문서를 모두 삭제 & 초기화하는 함수 정의
def clear_azure_search_index():
    search_url = f"{AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs?api-version=2024-07-01&search=*&$select={AZURE_SEARCH_KEY_FIELD_NAME}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}
    
    try: # 오류가 발생할 수 있는 코드 블록 지정
        response = requests.get(search_url, headers=headers)
        response.raise_for_status() 
            # HTTP 응답 코드 확인 (HTTP 응답 Error 발생 시, "HTTPError" 라는 이름으로 예외를 발생시킴)
            # 즉 요청이 실패하면 에러를 반환하는 안전장치
        results = response.json() 
            # HTTP 응답 body 를 JSON 형식으로 처리 => 파이썬 딕셔너리 {} 또는 리스트 [] 형태로 반환
        
        doc_keys = [doc[AZURE_SEARCH_KEY_FIELD_NAME] for doc in results.get("value", [])]
            # results.get("value", []) => "results" 내용 중, "value" 키에 해당하는 값을 가져오고, "value" 키가 없으면, 빈 리스트 [] 를 반환 (에러 방지)
            # for doc in results.get("value", [])] => "results.get("value", [])" 값을 반복/순회하여 "doc" 변수에 할당
            # doc[AZURE_SEARCH_KEY_FIELD_NAME] => 각 문서(doc)에서 특정 키[AZURE_SEARCH_KEY_FIELD_NAME] 값만 추출 => doc_keys = [] 리스트에 담음

        if not doc_keys:  # 아직 doc_keys 변수에 아무런 값이 없을 경우
            print("[LOG] 인덱스가 이미 비어있습니다. 초기화를 건너뜁니다.")
            return True  # 함수가 정상적으로 종료되었음을 해당 함수 호출자(Caller)에게 알림

        docs_to_delete = [{"@search.action": "delete", AZURE_SEARCH_KEY_FIELD_NAME: key} for key in doc_keys]
            # [... for key in doc_keys] => doc_keys 리스트에 있는 각 key 값을 반복하여 리스트 [] 에 담음
            # 그 다음 {"@search.action": "delete", AZURE_SEARCH_KEY_FIELD_NAME: key} 여러개가 반복되어, docs_to_delete = [] 리스트에 담김
 
        index_url = f"{AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/index?api-version=2024-07-01"
        payload = {"value": docs_to_delete}
        
        response = requests.post(index_url, headers=headers, json=payload)
        response.raise_for_status()
            # HTTP 응답 코드 확인 (HTTP 응답 Error 발생 시, "HTTPError" 라는 이름으로 예외를 발생시킴)
            # 즉 요청이 실패하면 에러를 반환하는 안전장치
        
        print(f"[LOG] {len(doc_keys)}개의 문서를 성공적으로 삭제하여 인덱스를 초기화했습니다.")
        return True  # 함수가 정상적으로 종료되었음을 호출자에게 알림

    except requests.RequestException as e: # 위의 오류가 발생할 수 있는 코드 블록에서 (try 블록에서) 오류가 발생했을 때 실행/대응할 코드 정의
        # 위의 try 블록 안에서 HTTP 요청을 수행하다가 requests 관련 오류가 발생하면, 해당 블록이 (except ...) 실행됨
        # 발생한 예외 객체를 e 라는 변수에 저장
        # requests.RequestException 모듈 => HTTP requests 관련 모든 오류를 한번에 잡아주는 상위 모듈

        print(f"[ERROR] Azure AI Search 인덱스 초기화 실패: {e}")
        return False  # 요청 실패를 호출자에게 알림 (이후 로직에서 실패 처리를 할 수 있도록 신호 전달)


# [2-2] Text 를 Load -> Split 처리하는 함수 정의
def load_and_split_pdf(file):
    """업로드된 PDF 파일에서 텍스트를 추출하고 Semantic Kernel의 TextChunker를 사용하여 분할합니다."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file: # with : 컨텍스트 관리자 => 임시파일과 같은 리소스를 자동으로 열고 닫는 작업을 수행
        # tempfile.NamedTemporaryFile() => 이름 있는 임시 파일을 생성
        # delete=False: 파일을 닫아도 지우지 않도록 설정 (tmp_file 변수에 할당된 경로값을 좀 더 사용한 뒤, 아래 "finally" 구문 단계에서 직접 삭제할 예정)
        # as tmp_file: 생성된 파일 객체인 "tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").__enter__()"" 값을 "tmp_file" 이라는 변수에 할당
        
        # with : 컨텍스트 관리자 => 임시파일과 같은 리소스를 자동으로 열고 닫는 작업을 수행 (내부 동작 : __enter__() 실행 => with 아래 코드 실행 => __exit__() 실행)
        # with "A" as "B" 구문에서, "A" 는 __enter__() 메소드와 __exit__() 메소드를 가진 객체이어야 함
        # with "A" as "B" 구문에서, "B" 는 A.__enter__() 의 반환값임
        # __enter__() => 컨텍스트가 "시작"될 때 실행할 동작을 정의 (특별 메소드)
        # __exit__() => 컨텍스트가 "끝날" 때(정상 종료 or 예외 발생) 실행할 동작을 정의 (특별 메소드)

        tmp_file.write(file.getvalue())
            # file.getvalue(): 업로드된 파일 객체 안에 있는 내용을 바이트 단위로 꺼냄
            # tmp_file.write(): 임시 파일에 바이트/문자열 기록
            # 결과적으로 업로드된 파일 내용을 임시 PDF 파일로 복사하여 저장함
        
        tmp_file_path = tmp_file.name
            # tmp_file.name: 위에서 만든 임시 파일의 "경로 값" 추출
            # 해당 "경로 값"을, "tmp_file_path" 변수에 할당

    try: # 오류가 발생할 수 있는 코드 블록 지정
         # 예외 처리 시작
         # try 블록에서 예외가 나더라도, 뒤의 finally 블록을 정의하면, 해당 finally 블록의 내용이 항상 실행됨 (자원 정리 목적)

        reader = pypdf.PdfReader(tmp_file_path)
        # pypdf.PdfReader(): tmp_file_path 경로의 PDF 파일을 열어, 해당 파일 내용을 "reader" 라는 변수에 할당
                
        full_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        # 4가지 구문 실행 순서 : 
            # (1) for page in reader.pages => 페이지 순회
            # (2) if page.extract_text() => 텍스트 존재 여부 확인
            # (3) page.extract_text() => 텍스트가 존재하면, 해당 페이지의 텍스트 추출
            # (4) "".join(...) => 추출된 모든 텍스트를 공백없이("") 하나로 결합
        
        # reader.pages => "reader" 라는 변수에 담긴 파일 내용의 각 페이지 목록에 접근 (PDF 문서의 모든 페이지 객체 목록을, 내부적으로 반복 형식으로 받아옴)
        # 각각의 페이지 객체는 "page" 라는 변수에 하나씩 반복/순회하여 할당
        # if page.extract_text() => page.extract_text() 실행 결과가 빈 문자열이 아닌 경우에만, 
        # page.extract_text() => 각 page 로부터 텍스트 값 추출하는 작업 실행
        # "".join() => .join() 괄호 안의 조건을 통과한 페이지들의 텍스트를 빈 문자열 "" 기준으로 이어 붙여서, 하나의 큰 문자열로 결합하여, full_text 변수에 할당
               
        chunks = text_chunker.split_plaintext_lines(full_text, 1000)
        # text_chunker => Semantic Kernel 에서, 긴 텍스트를 작은 의미 단위(chunk)로 분할하는 모듈
        # .split_plaintext_lines() => 평문을 (Plain Text) Line 단위로 하나씩 쪼개는 모듈
        # .split_plaintext_lines() 의 첫 번째 인자 => 분할할 원본 텍스트 (전체 텍스트)
        # .split_plaintext_lines() 의 두 번째 인자 => 청크의 최대 길이 (단위 : Character 또는 Token)
        # 원본 텍스트를 분할한 청크들은, 리스트에 [] 담아 "chunks" 라는 변수로 할당함

    finally: # 위의 try 블록에서 예외가 나던 안나던 상관없이, finally 블록을 정의하면, 아래 finally 블록의 내용이 항상 실행됨
        os.remove(tmp_file_path) # os.remove(): 지정한 경로의 파일을 삭제
                                 # 위에서 delete=False 로 만든 임시 파일을 여기서 직접 정리함

    return chunks # 위에서 언급한 "chunks" 라는 청크 리스트 [] 값을 반환함
    
    # with => 컨텍스트 관리자 기반으로, 임시파일 등 리소스를 <열고> <닫는> 작업을 <자동>으로 수행함 (임시파일 등 리소스를 관리할 목적으로 사용함)
    # try => 예외 처리 블록 (코드 실행 중 오류/예외가 발생할 수 있는 영역을 지정함)
    #        주로 뒤에 except 블록이 따라오는데, except 블록을 이용하여 try 블록에서의 오류에 대응하는 코드를 사용하면, 오류를 잘 처리할 수 있음
    # finally => 무조건 실행되는 블록 (위의 try 블록에서 정상 실행 또는 예외 발생 상관없이, finally 블록을 통해 임시파일 등 리소스 정리 작업(clean-up) 수행)
    #            주로 임시파일 닫기, 연결 종료, 메모리 해제 등 강제 리소스 정리 목적으로 사용


# [2-3] Split 된 텍스트 조각들을 Azure AI Search 에 <업로드> 하여 <인덱싱> 처리하는 함수 정의
def index_documents_to_azure_search(docs: list[str]): # docs : 함수 호출 시 외부에서 전달되는 인수(argument) 
                                                      # list[str] : 문자열(string)들이 포함된 리스트 형식을 기대한다는 의미
                                                      # 예: ["doc1", "doc2", "doc3"]

    """분할된 텍스트 청크 목록을 Azure AI Search에 인덱싱합니다."""

    if docs:  # 외부에서 전달된 "docs" 에 값이 존재할 경우, 아래 print() 코드 실행
        print("\n--- [디버깅 LOG] 인덱싱될 첫 번째 문서 청크 ---")
        print(docs[0])
        print("------------------------------------------------\n")

    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}
    url = f"{AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/index?api-version=2024-07-01"

    documents_to_upload = [
        {"@search.action": "upload", AZURE_SEARCH_KEY_FIELD_NAME: str(uuid.uuid4()), AZURE_SEARCH_CONTENT_FIELD_NAME: doc} for doc in docs
        ]
        # "@search.action": "upload" => Azure AI Search 의 API 에서 사용되는 특별 키 ("upload" 값 : 문서를 Index 용으로 업로드하겠다는 의미)
        # AZURE_SEARCH_KEY_FIELD_NAME => 검색 Index 의 Key 이름 | str(uuid.uuid4()) => 중복되지 않는 랜덤 UUID 문자열을 생성해서 고유의 Key 로 사용
        # AZURE_SEARCH_CONTENT_FIELD_NAME => 검색 Index 의 Content 이름 | 업로드된 PDF 문서 안의 문자열 내용이 담긴 "doc" 변수를 값으로 넣음
    
    payload = {"value": documents_to_upload}

    try: # 오류가 발생할 수 있는 코드 블록 지정
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # HTTP 응답 코드 확인 (HTTP 응답 Error 발생 시, "HTTPError" 라는 이름으로 예외를 발생시킴) 
                                    # 즉 요청이 실패하면 에러를 반환하는 안전장치
        print(f"[LOG] {len(docs)}개의 문서 청크를 성공적으로 인덱싱했습니다.")
        return True
    
    except requests.RequestException as e: # 위의 오류가 발생할 수 있는 코드 블록에서 오류가 발생했을 때 실행/대응할 코드 정의
        error_details = ""
        try: # 오류가 발생할 수 있는 코드 블록 지정
            if e.response is not None:
                error_details = e.response.json() # .json() => HTTP 응답 body 를 JSON 형식으로 처리 -> 파이썬 딕셔너리 {} 또는 리스트 [] 형태로 반환
        except Exception: # 위의 오류가 발생할 수 있는 코드 블록에서 오류가 발생했을 때 실행/대응할 코드 정의
            error_details = "Could not parse error response from Azure AI Search."
        
        print(f"[ERROR] Azure AI Search 인덱싱 실패: {e}")
        print(f"[ERROR] 상세 정보: {error_details}")

        return False

    # try => 오류가 발생할 수 있는 코드 블록 지정
    # except => # try 블록 (오류가 발생할 수 있는 코드 블록) 에서 오류가 발생했을 때 실행/대응할 코드 정의
    # 즉, try ... except ... 구문의 용도는 => 프로그램이 중간에 멈추지 않고 예외를 처리할 수 있게 해주는 용도로 사용됨


# [2-4] 사용자의 질문(query)을 바탕으로 Azure AI Search 에서 문서 검색 (Azure AI Search 로 RAG 질의)
def search_azure_ai(query: str) -> str: # query => 함수 호출 시 외부에서 전달되는 매개변수(argument) 
                                        # query: str => 외부에서 전달되는 query 매개변수가 문자열(string) 형식으로 들어오는 것을 기대한다는 의미
                                        # -> str => search_azure_ai() 함수가 반환하는 데이터 타입이 문자열(string)임을 나타냄

    """사용자 질문과 관련된 문서를 Azure AI Search에서 검색합니다."""

    url = f"{AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2024-07-01"
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}    
    payload = {
        "search": query,
        "count": True,
        "top": 5,
        "searchFields": AZURE_SEARCH_CONTENT_FIELD_NAME
    }

    try: # 오류가 발생할 수 있는 코드 블록 지정
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # HTTP 응답 코드 확인 (HTTP 응답 Error 발생 시, "HTTPError" 라는 이름으로 예외를 발생시킴)
                                    # 즉 요청이 실패하면 에러를 반환하는 안전장치
        results = response.json() # HTTP 응답 body 를 JSON 형식으로 처리 -> 파이썬 딕셔너리 {} 또는 리스트 [] 형태로 반환
        
        print("\n--- [디버깅 LOG] Azure AI Search 검색 결과 ---")
        print(results)
        print("---------------------------------------------\n")

    except requests.RequestException as e: # 위의 오류가 발생할 수 있는 코드 블록에서 오류가 발생했을 때 실행/대응할 코드 정의
        print(f"[ERROR] Azure AI Search 요청 실패: {e}")
        return "검색 서비스 호출에 실패했습니다. CLI 콘솔의 에러 로그를 확인해주세요."

    return "\n\n".join([doc.get(AZURE_SEARCH_CONTENT_FIELD_NAME, "") for doc in results.get("value", [])])
        # results.get("value", []) => results( {} 또는 [] ) 에서 "value" 키에 해당하는 값 가져오기 ("value" 라는 key가 없으면 빈 리스트 [] 반환)
        # for doc in results.get("value", []) => results.get("value", []) 결과 내용을, 하나씩 "doc" 으로 반복/순회하여 가져옴
        # doc.get(AZURE_SEARCH_CONTENT_FIELD_NAME, "") => "doc" 에서, "AZURE_SEARCH_CONTENT_FIELD_NAME" 키에 해당하는 문서 내용(content) 가져오기
        #                                                 만약 "AZURE_SEARCH_CONTENT_FIELD_NAME" 키가 없으면, 빈 문자열 "" 반환
        # .join() 함수 괄호 안에 포함된 리스트 [] 내부의 문자열들을 "\n\n" (두 줄 개행) 으로 구분하여 이어붙임


# [2-5] RAG 의 System Prompt 템플릿 정의
qa_system_prompt = """
당신은 질문에 답변하는 AI 어시턴트입니다.
제공된 컨텍스트 정보를 사용하여 질문에 답변해 주세요.
만약 컨텍스트에서 답을 찾을 수 없다면, "정보를 찾을 수 없습니다."라고 솔직하게 말해주세요.
답변은 항상 한국어로, 친절한 존댓말로 작성해야 합니다.

---
[컨텍스트]:
{{$context}}
---

[질문]:
{{$input}}

[답변]:
"""


# [3] Streamlit (채팅창) 로직 구현

# streamlit.title          =>  채팅창에 Streamlit 타이틀 표시
# streamlit.chat_input     =>  채팅창에서 사용자 Input 받기

# streamlit.session_state  =>  내부적으로 Streamlit 세션 상태 저장/관리
# streamlit.status         =>  채팅창에 상태 메시지(진행 상황) 표시 (with streamlit.status 형태로 사용) (예 : "파일 처리 중...")

# streamlit.chat_message   =>  채팅창에 말풍선 출력
# streamlit.write          =>  채팅창의 말풍선 안에, 실제 내용을 써서 출력

# streamlit.error          =>  채팅창에 에러 메시지 표시


# [3-1] Streamlit 타이틀 정의
streamlit.title("Azure OpenAI 챗봇 서비스 (Semantic Kernel) 💬")

# [3-2] Streamlit 사이드바 (PDF 파일 업로드) 구현 
with streamlit.sidebar: # with : 컨텍스트 관리자 => 임시파일과 같은 리소스를 자동으로 열고 닫는 작업을 수행

    streamlit.header("RAG 기능 활성화")
    uploaded_file = streamlit.file_uploader("PDF 파일을 업로드하면 해당 내용으로 RAG 챗봇이 동작합니다.", type=["pdf"])

    if uploaded_file is not None: # Upload 된 파일이 존재할 경우

        if "last_uploaded_file" not in streamlit.session_state or streamlit.session_state.last_uploaded_file != uploaded_file.name:
        # streamlit.session_state => Streamlit <세션 상태(state)> 를 저장하고 관리하는 객체
        # streamlit.session_state(세션 상태 저장소)에 "last_uploaded_file" 이라는 Key 가 존재하지 않는 경우
        # 또는 streamlit.session_state 에 저장된 .last_uploaded_file 이라는 Key 값과, 현재 새로 업로드된 파일의 경로/이름이 다를 경우

            with streamlit.status("파일 처리 중...", expanded=True) as status: 
            # streamlit.status => Streamlit 세션의 상태를(status) 채팅창 UI 에 생성/표시
            # "파일 처리 중..." => Streamlit 상태 창에 표시될 메시지
            # expanded=True => 상태 창을 기본적으로 펼쳐진 상태로 보여줌 (즉, 접혀있지 않고, 안의 로그들이 바로 보이도록 설정)

            # streamlit.session_state => Streamlit 세션 상태 저장/관리
            # with streamlit.status => 채팅창에 상태 메시지/진행 상황 표시

                status.write("기존 인덱스 데이터를 초기화합니다...") # PDF 파일 업로드 후, 1번째 메시지 추가/표시

                if clear_azure_search_index(): 
                # 위의 def 함수가(Azure Search 인덱스를 초기화) 성공적으로 끝났을 때만, 아래 PDF 분석 작업을 이어서 수행

                    status.write("PDF 파일을 분석하고 있습니다...") # PDF 파일 업로드 후, 2번째 메시지 추가/표시

                    split_docs = load_and_split_pdf(uploaded_file)
                    status.write(f"파일을 {len(split_docs)}개의 청크로 분할했습니다.") # PDF 파일 업로드 후, 3번째 메시지 추가/표시
                    status.write("분석된 내용을 Azure AI Search에 저장하고 있습니다...") # PDF 파일 업로드 후, 4번째 메시지 추가/표시

                    if index_documents_to_azure_search(split_docs):
                    # 위의 def 함수가(Azure AI Search 에 업로드하여 인덱싱 처리) 성공적으로 끝났을 때만, 아래 작업을 이어서 수행

                        status.update(label="파일 처리 완료!", state="complete", expanded=False)
                        # 위의 4가지 메시지가 없어지고, 5번째 메시지 표시 (streamlit.status.update 의 기능)
                        # label : 업로드창에 표시되는 <텍스트> 지정
                        # state : 업로드창에 표시되는 <아이콘> 지정
                        # expanded=Flase : 상태창을 접힌 상태(collapsed)로 표시 (로그 출력 영역을 닫아둠)

                        # streamlit.status.write => 상태 내용(로그) <추가>
                        # streamlit.status.update => 상태 내용(로그) <변경>
                        
                        streamlit.session_state.rag_enabled = True
                        # streamlit.session_state.rag_enabled = True 이전 :
                        # ==> {'messages': [{'role': 'assistant', 'content': '무엇을 도와드릴까요? PDF를 업로드하면 문서 기반 답변이 가능합니다.'}]}
                        # streamlit.session_state.rag_enabled = True 이후 : 
                        # ==> {'rag_enabled': True, 'messages': [{'role': 'assistant', 'content': '무엇을 도와드릴까요? PDF를 업로드하면 문서 기반 답변이 가능합니다.'}]}

                        # 즉, 원래의 streamlit.session_state 딕셔너리 {} 에서, "rag_enabled" 라는 key 를 새로 만들고, 그 값으로 "True" 값을 저장
                        # 즉, RAG 기능이 활성화되었다는 상태를 기록하기 위해 사용함

                        streamlit.session_state.last_uploaded_file = uploaded_file.name
                        # streamlit.session_state.last_uploaded_file 이전 : 
                        # ==> {'messages': [{'role': 'assistant', 'content': '무엇을 도와드릴까요? PDF를 업로드하면 문서 기반 답변이 가능합니다.'}], 'rag_enabled': True}
                        # streamlit.session_state.last_uploaded_file 이후 : 
                        # ==> {'messages': [{'role': 'assistant', 'content': '무엇을 도와드릴까요? PDF를 업로드하면 문서 기반 답변이 가능합니다.'}], 'rag_enabled': True, 'last_uploaded_file': 'pltr.pdf'}

                        # 즉, 원래의 streamlit.session_state 딕셔너리 {} 에서, last_uploaded_file 라는 Key 를 만들고, 그 값으로 "uploaded_file_name.pdf" 변수의 값을 저장
                        # 즉, 마지막으로 업로드된 PDF 의 경로/파일명을 세션에 기억시키기 위해 사용함

                        if "messages" in streamlit.session_state: # streamlit.session_state 객체 안에 "messages"라는 key 가 존재하면, 아래 코드 실행
                            streamlit.session_state.messages.append(
                                {"role": "assistant", "content": f"✅ **{uploaded_file.name}** 파일의 내용을 성공적으로 학습했습니다."}
                            )
                            # streamlit.session_state 에 저장된 .messages 라는 리스트 [] 안에, {"role": "assistant", "content": "..." } 요소 추가 

                        streamlit.rerun()

                    else:
                    # 위의 def 함수가(Azure AI Search 에 업로드하여 인덱싱 처리) 실패했을 때, 아래 작업을 수행
                        status.update(label="파일 처리 실패", state="error", expanded=True)
                        # label : 업로드창에 표시되는 <텍스트> 지정
                        # state : 업로드창에 표시되는 <아이콘> 지정
                        # expanded=True : 상태창을 펼쳐진 상태(expanded)로 표시 (로그 출력 영역을 오픈함)

                        # streamlit.status.write => 상태 내용(로그) <추가>
                        # streamlit.status.update => 상태 내용(로그) <변경>

                        streamlit.error("파일 내용을 AI Search에 저장하는 데 실패했습니다. CLI 로그를 확인해주세요.")

                else: # 위의 def 함수가(Azure Search 인덱스를 초기화) 실패했을 때, 아래 작업을 수행
                    status.update(label="인덱스 초기화 실패", state="error", expanded=True)
                    # streamlit.status.write => 상태 내용(로그) <추가>
                    # streamlit.status.update => 상태 내용(로그) <변경>

                    streamlit.error("기존 인덱스 데이터를 삭제하는 데 실패했습니다. CLI 로그를 확인해주세요.")


# [3-3] Streamlit 최초 화면 인사말
if "messages" not in streamlit.session_state:  # streamlit.session_state (세션 상태) 라는 딕셔너리 {} 에 "messages" 라는 Key 가 없을 경우, 아래 코드 실행
    streamlit.session_state["messages"] = [{"role": "assistant", "content": "무엇을 도와드릴까요? PDF를 업로드하면 문서 기반 답변이 가능합니다."}]

# streamlit.session_state (딕셔너리 {} 타입)
# ==> {'messages': [{'role': 'assistant', 'content': '무엇을 도와드릴까요? PDF를 업로드하면 문서 기반 답변이 가능합니다.'}]}

# streamlit.session_state.messages (리스트 [] 타입)
# ==> [{'role': 'assistant', 'content': '무엇을 도와드릴까요? PDF를 업로드하면 문서 기반 답변이 가능합니다.'}]


# [3-4] 과거 대화 내역 표시 (채팅 History 표시)
for msg in streamlit.session_state.messages:  
# streamlit.session_state.messages 라는 리스트 [] 에 들어있는 딕셔너리 {} 값을, "msg" 라는 변수에 하나씩 순회하여 담음
# msg ==> {'role': 'assistant', 'content': '무엇을 도와드릴까요? PDF를 업로드하면 문서 기반 답변이 가능합니다.'}

    streamlit.chat_message(msg["role"]).write(msg["content"])
    # .chat_message ==> Streamlit 채팅창에 표시하는 함수
    # msg["role"] ==> 'assistant'
    # msg["content"] ==> '무엇을 도와드릴까요? PDF를 업로드하면 문서 기반 답변이 가능합니다.'


# [3-5] 사용자 입력 처리
if prompt := streamlit.chat_input("질문을 입력해주세요..."):
# .chat_input()로 사용자 입력을 받음 => "prompt" 라는 변수에 저장 => 입력값이 존재할 때만 if 아래 부분을 실행

    streamlit.session_state.messages.append({"role": "user", "content": prompt})
    # streamlit.session_state.messages 라는 리스트 [] 내부에서, {"role": "user", "content": prompt } 라는 딕셔너리 추가
    
    # streamlit.session_state.messages.append({"role": "user", "content": prompt}) 이전의 streamlit.session_state.messages :
    # ==> [{'role': 'assistant', 'content': '...'}, {'role': 'assistant', 'content': '...'}]
    # streamlit.session_state.messages.append({"role": "user", "content": prompt}) 이후의 streamlit.session_state.messages : 
    # ==> [{'role': 'assistant', 'content': '...'}, {'role': 'assistant', 'content': '...'}, {'role': 'user', 'content': '직원수는 몇명인가요'}]

    streamlit.chat_message("user").write(prompt)
    # 위의 {"role": "user", "content": prompt } 내용을, streamlit 채팅창에 말풍선으로 표시

    with streamlit.chat_message("assistant"): 
    # 채팅창에 "assistant" 역할의 말풍선 출력
    # with (컨텍스트 관리자) 구문을 이용하여, 메시지창을 <열고> <닫는> 작업을 자동으로 처리

        with streamlit.spinner("답변을 생성하는 중입니다..."): # .spinner : 사용자 질문에 대해 답변할때, Loading... 아이콘과 함께 나오는 메시지 정의  

            try: # 오류가 발생할 수 있는 코드 블록 지정

                if streamlit.session_state.get("rag_enabled", False):
                # Streamlit 세션에 "rag_enabled" 라는 Key 값이 존재할때, 아래 코드 수행
                # Streamlit 세션에 "rag_enabled" 라는 Key 값이 존재하지 않을때, 기본값 "False" 를 안전하게 반환 (프로그램이 오류를 내서 멈추지 않도록 방지)

                    context = search_azure_ai(prompt) # 사용자 질의가 담긴 "prompt" 를 가지고, Azure AI Search 로 RAG 질의한 다음, 해당 내용을 "context" 변수에 할당
                    arguments = KernelArguments(input=prompt, context=context) # 시멘틱 커널에서, 사용자 Prompt 를 구조화하여 (사용자 prompt + 관련 Context) "argument" 변수에 할당

                    # --- [디버깅 코드 추가] ---
                    print("\n--- [디버깅 LOG] Kernel 호출 직전 데이터 ---")
                    print(f"전달된 Input: {prompt}")
                    print(f"전달된 Context (일부): {context[:500]}...") # 컨텍스트가 너무 길 수 있으므로 앞부분 500자만 출력
                    print("-------------------------------------------\n")
                    # --- [디버깅 코드 끝] ---

                    result = asyncio.run(
                        kernel.invoke_prompt(
                            prompt=qa_system_prompt, 
                            arguments=arguments)
                    )
                    # 동시성 처리 (비동기식 처리) 방식으로, Semantic Kernel 에서의 Prompt 처리를 수행 (매개변수 : 시스템 Prompt + 사용자 Prompt) 
                    # Semantic Kenerl 에서의 Prompt 처리 결과를 "result" 변수에 할당

                    response = str(result)
                    # 위의 "result" 변수 내용을 문자열(string) 형태로 변환하여, "response" 변수에 할당
                
                else: 
                    result = asyncio.run(kernel.invoke_prompt(prompt))
                    # 동시성 처리 (비동기식 처리) 방식으로, Semantic Kernel 에서의 Prompt 처리를 수행 (매개변수 : 시스템 Prompt) 
                    response = str(result)
                    # 위의 "result" 변수 내용을 문자열(string) 형태로 변환하여, "response" 변수에 할당

                streamlit.session_state.messages.append({"role": "assistant", "content": response})
                # streamlit.session_state.messages 라는 리스트 [] 내부에서, {"role": "assistant", "content": response} 요소 추가
                # Before : [{'role': '...', 'content': '...'}, {'role': '...', 'content': '...'}]
                # After  : [{'role': '...', 'content': '...'}, {'role': '...', 'content': '...'}, {"role": "assistant", "content": response}]

                streamlit.write(response)
                # 위에서 만들어진 streamlit 채팅 화면의 말풍선에, 해당 "response" 내용을 추가

                # streamlit.chat_message => 화면에 채팅 말풍선 출력
                # streamlit.write => 화면의 채팅 말풍선 안에, 실제 내용을 써서 출력

            except Exception as e: # 위의 오류가 발생할 수 있는 코드 블록에서 오류가 발생했을 때 실행/대응할 코드 정의

                streamlit.error(f"답변 생성 중 오류가 발생했습니다: {e}") # 에러를 더 상세하게 출력하도록 수정

                print(f"[ERROR] Kernel invocation failed with exception: {e}")

                import traceback # traceback 모듈 => 프로그램 실행 중 발생한 오류의 상세한 경로와 정보를 추적하고 출력
                traceback.print_exc() # 현재 발생한 예외(Exception)의 전체 오류 추적 정보를 CLI 콘솔에 직접 출력


