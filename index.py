from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from pyserini.index import LuceneIndexReader
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import subprocess

k = 100

query = 'Is global warming a hoax?'

beir_searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-climate-fever.multifield')
beir_reader = LuceneIndexReader.from_prebuilt_index('beir-v1.0.0-climate-fever.multifield')
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')
query_embedding = model.encode(query).reshape(1, -1)

initial_rankings = beir_searcher.search(query, k=k)
beir_docids = [doc.docid for doc in initial_rankings]
beir_texts = [json.loads(beir_reader.doc(doc.docid).raw())['text'] for doc in initial_rankings]
beir_vectors = model.encode(beir_texts, convert_to_numpy=True, convert_to_tensor=False)
beir_vectors_list = [np.array(v).tolist() for v in beir_vectors]

with open('./embeddings/beir_embeddings.jsonl', 'w') as f:
    for i in range(k):
        f.write(json.dumps({'id': beir_docids[i], 'contents': beir_texts[i], 'vector': beir_vectors_list[i]}) + '\n')

subprocess.call("create_hnsw.bat", shell=True)

searcher = FaissSearcher(
    'indexes',
    model
)
hits = searcher.search(query_embedding, k=k)
print(hits[0])