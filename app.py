# app.py 실행을 위한 기본 도구 설치
import os
import streamlit
import time
import requests
import tempfile
import uuid
import asyncio
import pypdf

# Semantic Kernel 관련 라이브러리 추가
import semantic_kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.text import text_chunker
from semantic_kernel.functions.kernel_arguments import KernelArguments
from dotenv import load_dotenv

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


# [1] Semantic Kernel 모델 선언 후, AzureChatCompletion 서비스를 Semantic Kernel 에 추가
kernel = semantic_kernel.Kernel()

kernel.add_service(
    AzureChatCompletion(
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_VERSION,
        deployment_name=AZURE_OPENAI_MODEL,
    ),
)


# [2] RAG 구현

# [2-1] 기존의 Azure AI Search 인덱스 및 문서를 모두 삭제
def clear_azure_search_index():
    """Azure AI Search 인덱스의 모든 문서를 삭제합니다."""
    print("[LOG] Azure AI Search 인덱스 초기화를 시작합니다...")
    search_url = f"{AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs?api-version=2024-07-01&search=*&$select={AZURE_SEARCH_KEY_FIELD_NAME}"
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}
    
    try:
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()
        results = response.json()
        
        doc_keys = [doc[AZURE_SEARCH_KEY_FIELD_NAME] for doc in results.get("value", [])]

        if not doc_keys:
            print("[LOG] 인덱스가 이미 비어있습니다. 초기화를 건너뜁니다.")
            return True

        documents_to_delete = [
            {"@search.action": "delete", AZURE_SEARCH_KEY_FIELD_NAME: key}
            for key in doc_keys
        ]
        payload = {"value": documents_to_delete}
        
        index_url = f"{AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/index?api-version=2024-07-01"
        response = requests.post(index_url, headers=headers, json=payload)
        response.raise_for_status()
        
        print(f"[LOG] {len(doc_keys)}개의 문서를 성공적으로 삭제하여 인덱스를 초기화했습니다.")
        return True

    except requests.RequestException as e:
        print(f"[ERROR] Azure AI Search 인덱스 초기화 실패: {e}")
        return False


# [2-2] Text 를 Load -> Split
def load_and_split_pdf(file):
    """업로드된 PDF 파일에서 텍스트를 추출하고 Semantic Kernel의 TextChunker를 사용하여 분할합니다."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        reader = pypdf.PdfReader(tmp_file_path)
        # 텍스트가 없는 페이지의 경우 오류를 방지하기 위해 None 체크 추가
        full_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        
        # **수정**: TypeError를 해결하기 위해 'chunk_overlap' 인자를 지원하는
        # 'split_plaintext_sentences' 함수로 변경합니다.
        chunks = text_chunker.split_plaintext_lines(full_text, 1000)
    finally:
        os.remove(tmp_file_path)

    return chunks


# [2-3] Split 된 텍스트 조각들을 Azure AI Search에 업로드
def index_documents_to_azure_search(docs: list[str]):
    """분할된 텍스트 청크 목록을 Azure AI Search에 인덱싱합니다."""
    if docs:
        print("\n--- [디버깅 LOG] 인덱싱될 첫 번째 문서 청크 ---")
        print(docs[0])
        print("------------------------------------------------\n")

    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}
    url = f"{AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/index?api-version=2024-07-01"

    documents_to_upload = [
        {"@search.action": "upload",
         AZURE_SEARCH_KEY_FIELD_NAME: str(uuid.uuid4()),
         AZURE_SEARCH_CONTENT_FIELD_NAME: doc} 
        for doc in docs
    ]
    
    payload = {"value": documents_to_upload}

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"[LOG] {len(docs)}개의 문서 청크를 성공적으로 인덱싱했습니다.")
        return True
    
    except requests.RequestException as e:
        error_details = ""
        try:
            if e.response is not None:
                error_details = e.response.json()
        except Exception:
            error_details = "Could not parse error response from Azure AI Search."
        
        print(f"[ERROR] Azure AI Search 인덱싱 실패: {e}")
        print(f"[ERROR] 상세 정보: {error_details}")
        return False


# [2-4] 사용자의 질문(query)을 바탕으로 Azure AI Search에서 문서 검색
def search_azure_ai(query: str) -> str:
    """사용자 질문과 관련된 문서를 Azure AI Search에서 검색합니다."""
    url = f"{AZURE_SEARCH_SERVICE_ENDPOINT}/indexes/{AZURE_SEARCH_INDEX_NAME}/docs/search?api-version=2024-07-01"
    headers = {"Content-Type": "application/json", "api-key": AZURE_SEARCH_API_KEY}
    
    payload = {
        "search": query,
        "count": True,
        "top": 5,
        "searchFields": AZURE_SEARCH_CONTENT_FIELD_NAME
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        results = response.json()
        
        print("\n--- [디버깅 LOG] Azure AI Search 검색 결과 ---")
        print(results)
        print("---------------------------------------------\n")

    except requests.RequestException as e:
        print(f"[ERROR] Azure AI Search 요청 실패: {e}")
        return "검색 서비스 호출에 실패했습니다. CLI 콘솔의 에러 로그를 확인해주세요."

    return "\n\n".join([doc.get(AZURE_SEARCH_CONTENT_FIELD_NAME, "") for doc in results.get("value", [])])


# [2-5] RAG 프롬프트 템플릿 정의
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

streamlit.title("Azure OpenAI 챗봇 서비스 (Semantic Kernel) 💬")

with streamlit.sidebar:
    streamlit.header("RAG 기능 활성화")

    uploaded_file = streamlit.file_uploader("PDF 파일을 업로드하면 해당 내용으로 RAG 챗봇이 동작합니다.", type=["pdf"])

    if uploaded_file is not None:
        if "last_uploaded_file" not in streamlit.session_state or streamlit.session_state.last_uploaded_file != uploaded_file.name:
            with streamlit.status("파일 처리 중...", expanded=True) as status:
                status.write("기존 인덱스 데이터를 초기화합니다...")
                if clear_azure_search_index():
                    status.write("PDF 파일을 분석하고 있습니다...")
                    split_docs = load_and_split_pdf(uploaded_file)
                    
                    status.write(f"파일을 {len(split_docs)}개의 청크로 분할했습니다.")
                    status.write("분석된 내용을 Azure AI Search에 저장하고 있습니다...")
                    if index_documents_to_azure_search(split_docs):
                        status.update(label="파일 처리 완료!", state="complete", expanded=False)
                        streamlit.session_state.rag_enabled = True
                        streamlit.session_state.last_uploaded_file = uploaded_file.name
                        
                        if "messages" in streamlit.session_state:
                            streamlit.session_state.messages.append(
                                {"role": "assistant", "content": f"✅ **{uploaded_file.name}** 파일의 내용을 성공적으로 학습했습니다.\n\n이제부터 파일 내용과 관련된 질문에 답변할 수 있습니다."}
                            )
                        streamlit.rerun()
                    else:
                        status.update(label="파일 처리 실패", state="error", expanded=True)
                        streamlit.error("파일 내용을 AI Search에 저장하는 데 실패했습니다. CLI 로그를 확인해주세요.")
                else:
                    status.update(label="인덱스 초기화 실패", state="error", expanded=True)
                    streamlit.error("기존 인덱스 데이터를 삭제하는 데 실패했습니다. CLI 로그를 확인해주세요.")


# Streamlit 최초 화면 인사말
if "messages" not in streamlit.session_state:
    streamlit.session_state["messages"] = [{"role": "assistant", "content": "무엇을 도와드릴까요? PDF를 업로드하면 문서 기반 답변이 가능합니다."}]

# 이전 대화 내용 표시
for msg in streamlit.session_state.messages:
    streamlit.chat_message(msg["role"]).write(msg["content"])

# 사용자 입력 처리
if prompt := streamlit.chat_input("질문을 입력해주세요..."):
    streamlit.session_state.messages.append({"role": "user", "content": prompt})
    streamlit.chat_message("user").write(prompt)

    with streamlit.chat_message("assistant"):
        with streamlit.spinner("답변을 생성하는 중입니다..."):
            
            try:
                # RAG 기능이 활성화된 경우
                if streamlit.session_state.get("rag_enabled", False):
                    context = search_azure_ai(prompt)
                    arguments = KernelArguments(input=prompt, context=context)

                    # --- [디버깅 코드 추가] ---
                    print("\n--- [디버깅 LOG] Kernel 호출 직전 데이터 ---")
                    print(f"전달된 Input: {prompt}")
                    # 컨텍스트가 너무 길 수 있으므로 앞부분 500자만 출력
                    print(f"전달된 Context (일부): {context[:500]}...") 
                    print("-------------------------------------------\n")
                    # --- [디버깅 코드 끝] ---

                    result = asyncio.run(
                        kernel.invoke_prompt(
                            prompt=qa_system_prompt, 
                            arguments=arguments
                        )
                    )
                    response = str(result)
                
                # RAG 기능이 비활성화된 경우 (일반 채팅)
                else:
                    result = asyncio.run(kernel.invoke_prompt(prompt))
                    response = str(result)

                streamlit.session_state.messages.append({"role": "assistant", "content": response})
                streamlit.write(response)

            except Exception as e:
                # 에러를 더 상세하게 출력하도록 수정
                streamlit.error(f"답변 생성 중 오류가 발생했습니다: {e}")
                print(f"[ERROR] Kernel invocation failed with exception: {e}")
                # traceback을 출력하면 더 상세한 원인 파악에 도움이 됩니다.
                import traceback
                traceback.print_exc()


