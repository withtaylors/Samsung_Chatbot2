from flask import Flask, request, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

from langchain.schema import Document
from peft import PeftModel, PeftConfig
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFDirectoryLoader

from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import TextStreamer, GenerationConfig

from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.vectorstores import Chroma
import torch
import re
import base64
import os
import unicodedata
import matplotlib.pyplot as plt
import matplotlib.image as img

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.embeddings import HuggingFaceEmbeddings

# HuggingFaceEmbeddings를 사용하여 embeddings 객체 생성
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")

# 나머지 코드는 이전과 동일합니다.

loader = DirectoryLoader("[FINAL] 그래프 전처리", loader_cls=TextLoader)
data=loader.load()

loader_graph = DirectoryLoader("[FINAL] 그래프 전처리", loader_cls=TextLoader)
data_graph=loader_graph.load()

# text split
# chunk_overlap - 겹쳐서 나오는 내용 조절
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=30)
data_split_text = text_splitter.split_documents(data)
text_splitter2 = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=30)
data_split_graph = text_splitter2.split_documents(data_graph)

# embedding
embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask")
db_faiss_text = FAISS.from_documents(data_split_text, embeddings)
db_faiss_graph = FAISS.from_documents(data_split_graph, embeddings)

from rank_bm25 import BM25Okapi

def make_tok(sent):
  return sent.split(" ")

corpus=  [i.page_content for i in data_split_text]
tokenized_corpus = [make_tok(i.page_content) for i in data_split_text]
bm25 = BM25Okapi(tokenized_corpus)

corpus_graph=  [i.page_content for i in data_split_graph]
tokenized_corpus_graph = [make_tok(i.page_content) for i in data_split_graph]
bm25_graph = BM25Okapi(tokenized_corpus_graph)

# search similarity - text
def db_search(query: str, k: int):
    docs = db_faiss_text.similarity_search(query, k)
    return docs

# search similarity - graph
def db_search_graph(query: str, k: int):
    docs = db_faiss_graph.similarity_search(query, k)
    return docs

# BM25Retriever 사용 설정
bm25_retriever = BM25Retriever.from_texts(corpus)
bm25_retriever.k = 6
bm25_retriever_graph = BM25Retriever.from_texts(corpus_graph)
bm25_retriever_graph.k = 6

# FAISS Retriever 설정
faiss_retriever = db_faiss_text.as_retriever(search_kwargs={"k": 10})
faiss_retriever_graph = db_faiss_graph.as_retriever(search_kwargs={"k": 10})

# 앙상블 리트리버 설정 - BM25 가중치 0.2, FAISS 가중치 0.8
ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],
                                       weights=[0.2, 0.8])

# 로컬 llm가져오기
def load_local_llm():
    model_id='mncai/llama2-13b-dpo-v3'
    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0})
    return model

model=load_local_llm()

tokenizer = AutoTokenizer.from_pretrained('mncai/llama2-13b-dpo-v3')
streamer = TextStreamer(tokenizer)

stop_list = ['### ','###','### example:']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids
stop_token_ids = [torch.LongTensor(x).to(0) for x in stop_token_ids]
stop_token_ids


# define custom stopping criteria object
# class StopOnTokens(StoppingCriteria):
#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         for stop_ids in stop_token_ids:
#             if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
#                 return True
#         return False

# stopping_criteria = StoppingCriteriaList([StopOnTokens()])

from konlpy.tag import Okt

#객체 생성

# 파일리스트는 가져옴
file_list=os.listdir('[FINAL] 그래프 png 파일')
file_names = file_list

text=file_list
texts=[]
text2=[]
text3=[]
for i in text:
    i=unicodedata.normalize('NFC',i)
    i=i.replace('.png의','')
    #i=i.replace('_',' ')
    i=i.replace('.PNG의','')
    i=i.replace(' 사본','')

    text2.append(i) #okt.morphs(unicodedata.normalize('NFC',i)))#+okt.verbs(i))

list_of_documents = [
    Document(page_content=name, metadata=dict(mainsource=name.split('_')[0]))
    for name in text2
]

db_faiss_graghs = FAISS.from_documents(list_of_documents, embeddings)

#def 그래프 -> 한번 시도해보기. 질문과 가장 관련된 파일을 가져오기
def graphs(x,docs_input):
    generation_config = GenerationConfig(
        temperature=1,
        top_p=0.8,
        top_k=100,
        max_new_tokens=300,
        early_stopping=True,
        do_sample=True,
    )
    template = f"""### instruction:
        아래 문서는 그래프에 대한 해석이다.
        문서에 주어진 내용만을 이용해서 답하고 문서에서 근거를 찾을 수 없거나 답변하기 모호하면 정보를 찾을 수 없습니다. 라고 답변해줘.

        문서: {docs_input}
        질문 = {x}\n\n### Response:
        output ():
    """

    q = template

    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    result_str = tokenizer.decode(gened[0])

    start_tag = f"[/INST]"
    start_index = result_str.find(start_tag)

    if start_index != -1:
        result_str = result_str[start_index + len(start_tag):].strip()
    return result_str

# 질문과 관련된 비슷한 질문 생성
def make_similar_query(x):
    generation_config = GenerationConfig(
        temperature=1,
        top_p=0.8,
        top_k=100,
        max_new_tokens=60,
        early_stopping=False,
        do_sample=True,
    )
    # 비슷한 문장을 두개 만들어줌. 쿼리가 조금만 달라져도 가지고 오는 문서가 다르기에 - 질문과 관련된 문서 가지고올 확률 높아짐
    template = f"""### instruction:
        질문과 관련된 2개의 다중 검색 질문를 생성해줘.

        질문= {x} ### Response:
        output (주어진 질문과 유사한 질문 2개):
    """

    q = template

    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    result_str = tokenizer.decode(gened[0])

    start_tag = f"[/INST]"
    start_index = result_str.find(start_tag)

    if start_index != -1:
        result_str = result_str[start_index + len(start_tag):].strip()
    return result_str


# find_core 쿼리에 관련된 답변을 1차적으로 생성해 리트리버의 input으로 넣어줌
# 하이드 방법론 - 1차적으로 답변한 것으로 리트리버가 문서 갖고 오도록 유사어와 동의어 들어왔을 때 llm이 인지해서 답변 리트리버가 명확한 답을 가져올 확률이 높다
def find_core(x):
    generation_config = GenerationConfig(
        temperature=1,
        top_p=0.8,
        top_k=100,
        max_new_tokens=300,
        early_stopping=True,
        do_sample=True,
    )

    template = f"""### instruction:
        사실만을 말하는 금융 전문가로서 답해줘.

        질문: {x}\n\n### Response:

    """
    q = template

    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        streamer=streamer,
    )
    result_str = tokenizer.decode(gened[0])

    start_tag = f"[/INST]"
    start_index = result_str.find(start_tag)

    if start_index != -1:
        result_str = result_str[start_index + len(start_tag):].strip()
    return result_str


# 최종 답변을 생성하는 함수
def gen_final(x,docs_input):
    generation_config = GenerationConfig(
        temperature=1,
        top_p=0.8,
        top_k=100,
        max_new_tokens=300,
        early_stopping=True,
        do_sample=True,
        repetition_penalty=1.2,
    )
    # 많이 시도하면서 보완하면서 작성. 답변을 잘 하지 못하는 케이스 파악 후 보완
    template = f"""### instruction:
        너는 사실만을 말하는 금융 전문가야.
        질문이 주어진 문서의 내용과 관련이 있다면, 해당 문서를 바탕으로 구체적이고 명확한 답변을 해줘.
        모든 문서의 내용을 이해하고 고려해서 질문과 관련된 정보를 모두 찾아 답변 해줘.
        만약 질문에 '모두', '전부'와 같은 단어가 포함되어 있다면 해당하는 정보를 문서에서 모두 최대한 찾아서 답변해줘.
        문서에 존재하지 않는 내용은 답하면 안돼.
        질문에 연도가 포함되어 있다면, 그 연도에 해당하는 정확한 데이터를 제공해줘.
        만약 질문에 연도가 명시되지 않았다면, 현재 연도인 2023년의 정보를 사용해줘.
        국내 시장과 관련된 질문이면 '내수'로 이해하고 답변해줘.
        비율이나 성장률, 이익률 등에 대한 질문에는 그에 해당하는 비율 값을 제공해줘.
        대답은 완성된 문장으로 해주고 완성하지 못할것 같으면 이전 문장까지만 만들어줘.
        문서 내용과 관련 없거나 애매한 질문이면 '내용을 찾을 수 없습니다.'라고 답변해줘.
        소속이나 존재를 묻는 질문은 문서에 정확하게 일치하는 내용이 없다면 없다고 답변해줘
        필요한 답변만 하고 불필요한 답변은 생성하면 안돼.
        문서에서 정보를 얻을 수 없다면 임의로 생성하지 말고 반드시 '문서에서 내용을 찾을 수 없습니다.'라고 답변해줘.
        문서들: {docs_input}
        질문: {x}\n\n### Response:
    """

    q = template

    gened = model.generate(
        **tokenizer(
            q,
            return_tensors='pt',
            return_token_type_ids=False
        ).to('cuda'),
        generation_config=generation_config,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        # stopping_criteria=stopping_criteria,
        streamer=streamer,
    )
    
    result_str = tokenizer.decode(gened[0])

    start_tag = f"[/INST]"
    start_index = result_str.find(start_tag)

    if start_index != -1:
        result_str = result_str[start_index + len(start_tag):].strip()
    return result_str

def filter_documents_by_query(documents, query):
    # 쿼리에서 키워드 확인
    query_keywords = [unicodedata.normalize('NFD', keyword) for keyword in ["하이브", "퀄컴", "자동차", "롯데","현대","기아","미국"] if keyword in query]
    query_keywords2 = ["하이브", "퀄컴", "자동차", "롯데","현대","기아","미국"]

    # 먼저 pdf의 종류를 고르고, 아래에 해당되는 키워드가 있는지 살펴서 그래프를 가져옴
    # query_keywords2 = [unicodedata.normalize('NFD', keyword) for keyword in ['매출','재고','영업이익','EPS','eps'] if keyword in query]
    # 해당 키워드를 mainsource로 가지는 문서 필터링
    if len(query_keywords)>0:
      filtered_docs = [doc for doc in documents if (any(item in doc.metadata.get('source', '') for item in query_keywords)) or (any(item in doc.page_content for item in query_keywords + query_keywords2))]
    else:
      filtered_docs= [doc for doc in documents]
    return filtered_docs

from langchain.document_transformers import (
    LongContextReorder,
)


import re

# 딕셔너리 형태
keyword_replacements = {
    'hybe': '하이브',
    'lotte': '롯데',
    'hyundai': '현대',
    'kia': '기아',
    'qualcomm': '퀄컴',
    'bts' : 'BTS'
}

def replace_keywords(query):
    for eng, kor in keyword_replacements.items():
        # re.IGNORECASE를 사용하여 대소문자 구분 없이 검색 및 대체
        query = re.sub(re.compile(eng, re.IGNORECASE), kor, query)
    return query

query=''

@app.route('/')
def index():
    return "Welcome to the API."
    
@app.route('/get_graph_description', methods=['POST'])
def get_graph_description():
    data = request.json
    graph_image_name = data['graphImageName']

    # 그래프 이미지 이름에 기반하여 설명 생성
    # 예: graph_description = create_graph_description(graph_image_name)
    outputs=make_similar_query(graph_image_name)
    
    # 1차적인 답변 생성
    core=find_core(graph_image_name)
    docs = ensemble_retriever.get_relevant_documents(graph_image_name)
    docs2 =ensemble_retriever.get_relevant_documents(core)
    docs3 =ensemble_retriever.get_relevant_documents(outputs.split('\n')[1])
    docs4 =ensemble_retriever.get_relevant_documents(outputs.split('\n')[2].replace('</s>',''))
    docs_all=(docs+docs2+docs3+docs4)
    documents=docs_all

    #예시 사용
    filtered_documents = filter_documents_by_query(documents, graph_image_name)

    #필터링된 문서 정보
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(filtered_documents)
    reordered_docs = [doc.page_content for doc in reordered_docs]
    reordered_docs=set(reordered_docs)

    #Reorded_docs가 최종 쿼리가 된다
    query_final=reordered_docs
    response_text = gen_final(graph_image_name, query_final)
    
    # 문자열을 리스트에 추가 (전체 응답 텍스트를 하나의 요소로)
    text_list = [response_text]
    responses = extract_response(text_list)

    return jsonify({"graphDescription": responses})
    
@app.route('/process_query', methods=['POST'])
def process_query():

    data = request.json
    query = data['query']
    responses = {}
    graph_image_names = {}
    
    # 쿼리를 처리하는 로직
    query = replace_keywords(query)
    # 쿼리에 그래프가 언급되어 있으면 그래프 이미지를 제시하면서 답변한다
    if '그래프' in query:
      
        results_with_scores = db_faiss_graghs.similarity_search(query,6)
        graph_docs=[docssss.page_content for docssss in results_with_scores]
        
        for graph_index,gg in enumerate(graph_docs[0:3]):
            graph_image_names = [str(g) for g in graph_docs[0:3]]
    else :
        outputs=make_similar_query(query)
    
        # 1차적인 답변 생성
        core=find_core(query)
        docs = ensemble_retriever.get_relevant_documents(query)
        #docs = [doc.page_content for doc in docs]
        #core = ','.join(Okt.nouns(query))
        docs2 =ensemble_retriever.get_relevant_documents(core)
        #docs2 = [doc.page_content for doc in docs2]
        docs3 =ensemble_retriever.get_relevant_documents(outputs.split('\n')[1])
        #docs3 = [doc.page_content for doc in docs3]
        docs4 =ensemble_retriever.get_relevant_documents(outputs.split('\n')[2].replace('</s>',''))
        #docs4 = [doc.page_content for doc in docs4]
        docs_all=(docs+docs2+docs3+docs4)
        documents=docs_all
    
        #예시 사용
        filtered_documents = filter_documents_by_query(documents, query)
    
        #필터링된 문서 정보
        #filtered_documents_info = [(doc.page_content, doc.metadata) for doc in filtered_documents]
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(filtered_documents)
        reordered_docs = [doc.page_content for doc in reordered_docs]
        reordered_docs=set(reordered_docs)
    
        #Reorded_docs가 최종 쿼리가 된다
        query_final=reordered_docs
        response_text = gen_final(query, query_final)
        
        # 문자열을 리스트에 추가 (전체 응답 텍스트를 하나의 요소로)
        text_list = [response_text]
        responses = extract_response(text_list)

    return jsonify({"response": responses, "graphImages": graph_image_names})
    

def extract_response(text_list):
    extracted_responses = []
    for text in text_list:
        #print(f"Original text: {text}")  # 디버깅을 위한 출력
        parts = text.split('### Response:')
        #print(f"Split parts: {parts}")  # split 결과 확인
        if len(parts) > 1:
            response = parts[1].strip()
            response = response.replace('</s>', '').strip()
            extracted_responses.append(response)
    return extracted_responses
    
def handle_query(query, query_final):
    return gen_final(query, query_final)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=3000)