from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from pyserini.index import LuceneIndexReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import json
import subprocess
import platform
import torch
from torch.nn.functional import softmax

k = 100

query = 'oj simpson'

climate_searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-climate-fever.multifield')
climate_reader = LuceneIndexReader.from_prebuilt_index('beir-v1.0.0-climate-fever.multifield')

covid_searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-trec-covid.multifield')
covid_reader = LuceneIndexReader.from_prebuilt_index('beir-v1.0.0-trec-covid.multifield')

fever_searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-fever.multifield')
fever_reader = LuceneIndexReader.from_prebuilt_index('beir-v1.0.0-fever.multifield')

news_searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-trec-news.multifield')
news_reader = LuceneIndexReader.from_prebuilt_index('beir-v1.0.0-trec-news.multifield')

scifact_searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-scifact.multifield')
scifact_reader = LuceneIndexReader.from_prebuilt_index('beir-v1.0.0-scifact.multifield')

embedding_model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
query_embedding = embedding_model.encode(query).reshape(1, -1)

climate_rankings = climate_searcher.search(query, k=k)
covid_rankings = covid_searcher.search(query, k=k)
fever_rankings = fever_searcher.search(query, k=k)
news_rankings = news_searcher.search(query, k=k)
scifact_rankings = scifact_searcher.search(query, k=k)

climate_docids = [doc.docid for doc in climate_rankings]
covid_docids = [doc.docid for doc in covid_rankings]
fever_docids = [doc.docid for doc in fever_rankings]
news_docids = [doc.docid for doc in news_rankings]
scifact_docids = [doc.docid for doc in scifact_rankings]

print(len(climate_rankings))
print(len(covid_rankings))
print(len(fever_rankings))
print(len(news_rankings))
print(len(scifact_rankings))

climate_texts = [json.loads(climate_reader.doc(doc.docid).raw())['text'] for doc in climate_rankings]
covid_texts = [json.loads(covid_reader.doc(doc.docid).raw())['text'] for doc in covid_rankings]
fever_texts = [json.loads(fever_reader.doc(doc.docid).raw())['text'] for doc in fever_rankings]
news_texts = [json.loads(news_reader.doc(doc.docid).raw())['text'] for doc in news_rankings]
scifact_texts = [json.loads(scifact_reader.doc(doc.docid).raw())['text'] for doc in scifact_rankings]

climate_vectors = embedding_model.encode(climate_texts, convert_to_numpy=True, convert_to_tensor=False)
covid_vectors = embedding_model.encode(covid_texts, convert_to_numpy=True, convert_to_tensor=False)
fever_vectors = embedding_model.encode(fever_texts, convert_to_numpy=True, convert_to_tensor=False)
news_vectors = embedding_model.encode(news_texts, convert_to_numpy=True, convert_to_tensor=False)
scifact_vectors = embedding_model.encode(scifact_texts, convert_to_numpy=True, convert_to_tensor=False)

climate_vectors_list = [np.array(v).tolist() for v in climate_vectors]
covid_vectors_list = [np.array(v).tolist() for v in covid_vectors]
fever_vectors_list = [np.array(v).tolist() for v in fever_vectors]
news_vectors_list = [np.array(v).tolist() for v in news_vectors]
scifact_vectors_list = [np.array(v).tolist() for v in scifact_vectors]

beir_texts = climate_texts + covid_texts + fever_texts + news_texts + scifact_texts
actual_docids = climate_docids + covid_docids + fever_docids + news_docids + scifact_docids
beir_docids = [i for i in range(len(beir_texts))]
beir_vectors_list = climate_vectors_list + covid_vectors_list + fever_vectors_list + news_vectors_list + scifact_vectors_list


with open('./embeddings/beir_embeddings.jsonl', 'w') as f:
    for i in range(len(beir_texts)):
        f.write(json.dumps({'id': beir_docids[i], 'contents': beir_texts[i], 'vector': beir_vectors_list[i]}) + '\n')

if platform.system() == 'Windows':
    subprocess.call("create_hnsw.bat", shell=True)
else:
    subprocess.call("create_hnsw.sh", shell=True)

searcher = FaissSearcher(
    'hnsw_index',
    embedding_model
)

model_checkpoint = "./roberta_fever_mnli"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model_custom = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

hits = searcher.search(query_embedding, k=k)
# texts are stored at beir_texts[int(hits[i].docid)]
# docids are stored at actual_docids[int(hits[i].docid)]
# scores are stored at hits[i].score
# when feeding format is f"query: {query} context: {beir_texts[int(hits[i].docid)]}"
hits_with_standing = []

label_mapping = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}

for hit in hits:
    input_text = f"query: {query} context: {beir_texts[int(hit.docid)]}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    
    # Forward pass
    outputs = model_custom(**inputs)
    
    # Get logits and probabilities
    logits = outputs.logits
    probs = softmax(logits, dim=-1)
    
    # Get predicted class (index of max probability)
    predicted_class = torch.argmax(probs, dim=-1).item()
    
    # Add result to outputs
    hits_with_standing.append({
        'docid': actual_docids[int(hit.docid)],
        'score': hit.score,
        'text': beir_texts[int(hit.docid)],
        'standing': label_mapping[predicted_class]
    })

hits_with_standing = sorted(hits_with_standing, key=lambda x: x['score'], reverse=True)