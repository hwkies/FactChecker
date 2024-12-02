from pyserini.search.lucene import LuceneSearcher
from pyserini.index import LuceneIndexReader
from sentence_transformers import SentenceTransformer
import numpy as np
import json

query = 'Is global warming a hoax?'

beir_searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-climate-fever.multifield')
beir_reader = LuceneIndexReader.from_prebuilt_index('beir-v1.0.0-climate-fever.multifield')
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')
query_embedding = model.encode(query)

initial_rankings = beir_searcher.search(query, k=100)
print(initial_rankings[0].docid)
beir_docids = [doc.docid for doc in initial_rankings]
beir_texts = [json.loads(beir_reader.doc(doc.docid).raw())['text'] for doc in initial_rankings]
beir_embeddings = model.encode(beir_texts, convert_to_numpy=True, convert_to_tensor=False)

np.save('./embeddings/beir_embeddings.npy', beir_embeddings)
with open('beir_docids.jsonl', 'w') as f:
    for doc_id in beir_docids:
        f.write(json.dumps({'id': doc_id}) + '\n')

