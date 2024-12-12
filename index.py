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

class RetrievalModel:
    def __init__(self, query: str, k=100):
        self.query = query
        self.k = k
        self.climate_searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-climate-fever.multifield')
        self.climate_reader = LuceneIndexReader.from_prebuilt_index('beir-v1.0.0-climate-fever.multifield')

        self.covid_searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-trec-covid.multifield')
        self.covid_reader = LuceneIndexReader.from_prebuilt_index('beir-v1.0.0-trec-covid.multifield')

        self.fever_searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-fever.multifield')
        self.fever_reader = LuceneIndexReader.from_prebuilt_index('beir-v1.0.0-fever.multifield')

        self.news_searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-trec-news.multifield')
        self.news_reader = LuceneIndexReader.from_prebuilt_index('beir-v1.0.0-trec-news.multifield')

        self.scifact_searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-scifact.multifield')
        self.scifact_reader = LuceneIndexReader.from_prebuilt_index('beir-v1.0.0-scifact.multifield')

        self.embedding_model = SentenceTransformer('multi-qa-mpnet-base-cos-v1')
        self.query_embedding = self.embedding_model.encode(query).reshape(1, -1)

    def __initial_rankings(self):
        climate_rankings = self.climate_searcher.search(self.query, k=self.k)
        covid_rankings = self.covid_searcher.search(self.query, k=self.k)
        fever_rankings = self.fever_searcher.search(self.query, k=self.k)
        news_rankings = self.news_searcher.search(self.query, k=self.k)
        scifact_rankings = self.scifact_searcher.search(self.query, k=self.k)

        climate_docids = [doc.docid for doc in climate_rankings]
        covid_docids = [doc.docid for doc in covid_rankings]
        fever_docids = [doc.docid for doc in fever_rankings]
        news_docids = [doc.docid for doc in news_rankings]
        scifact_docids = [doc.docid for doc in scifact_rankings]

        climate_texts = [json.loads(self.climate_reader.doc(doc.docid).raw())['text'] for doc in climate_rankings]
        covid_texts = [json.loads(self.covid_reader.doc(doc.docid).raw())['text'] for doc in covid_rankings]
        fever_texts = [json.loads(self.fever_reader.doc(doc.docid).raw())['text'] for doc in fever_rankings]
        news_texts = [json.loads(self.news_reader.doc(doc.docid).raw())['text'] for doc in news_rankings]
        scifact_texts = [json.loads(self.scifact_reader.doc(doc.docid).raw())['text'] for doc in scifact_rankings]

        climate_docid_text_pairs = list(zip(climate_docids, climate_texts))
        covid_docid_text_pairs = list(zip(covid_docids, covid_texts))
        fever_docid_text_pairs = list(zip(fever_docids, fever_texts))
        news_docid_text_pairs = list(zip(news_docids, news_texts))
        scifact_docid_text_pairs = list(zip(scifact_docids, scifact_texts))

        all_docid_text_pairs = set(climate_docid_text_pairs + covid_docid_text_pairs + fever_docid_text_pairs + news_docid_text_pairs + scifact_docid_text_pairs)

        all_docids, all_texts = zip(*all_docid_text_pairs)

        all_vectors = self.embedding_model.encode(all_texts, convert_to_numpy=True, convert_to_tensor=False)

        all_vectors_list = [np.array(v).tolist() for v in all_vectors]

        # climate_vectors = self.embedding_model.encode(climate_texts, convert_to_numpy=True, convert_to_tensor=False)
        # covid_vectors = self.embedding_model.encode(covid_texts, convert_to_numpy=True, convert_to_tensor=False)
        # fever_vectors = self.embedding_model.encode(fever_texts, convert_to_numpy=True, convert_to_tensor=False)
        # news_vectors = self.embedding_model.encode(news_texts, convert_to_numpy=True, convert_to_tensor=False)
        # scifact_vectors = self.embedding_model.encode(scifact_texts, convert_to_numpy=True, convert_to_tensor=False)

        # climate_vectors_list = [np.array(v).tolist() for v in climate_vectors]
        # covid_vectors_list = [np.array(v).tolist() for v in covid_vectors]
        # fever_vectors_list = [np.array(v).tolist() for v in fever_vectors]
        # news_vectors_list = [np.array(v).tolist() for v in news_vectors]
        # scifact_vectors_list = [np.array(v).tolist() for v in scifact_vectors]

        #beir_texts = climate_texts + covid_texts + fever_texts + news_texts + scifact_texts
        #actual_docids = climate_docids + covid_docids + fever_docids + news_docids + scifact_docids
        #beir_vectors_list = climate_vectors_list + covid_vectors_list + fever_vectors_list + news_vectors_list + scifact_vectors_list
        beir_docids = [i for i in range(len(all_texts))]
        with open('./embeddings/beir_embeddings.jsonl', 'w') as f:
            for i in range(len(all_texts)):
                f.write(json.dumps({'id': beir_docids[i], 'contents': all_texts[i], 'vector': all_vectors_list[i]}) + '\n')
        return all_texts, all_docids, beir_docids, all_vectors_list

    def __create_hnsw_index(self):
        if platform.system() == 'Windows':
            subprocess.call("create_hnsw.bat", shell=True)
        else:
            subprocess.call("create_hnsw.sh", shell=True)

    def __rerank(self, beir_texts: list, actual_docids: list):
        searcher = FaissSearcher('hnsw_index', self.embedding_model)

        model_checkpoint = "./roberta_fever_mnli"
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        model_custom = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)

        # Use GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_custom.to(device)

        hits = searcher.search(self.query_embedding, k=self.k)
        label_mapping = {0: "SUPPORTS", 1: "REFUTES", 2: "NOT ENOUGH INFO"}

        # Batch processing
        batch_size = 32  # Adjust based on your GPU memory
        hits_with_standing = []

        for i in range(0, len(hits), batch_size):
            batch_hits = hits[i:i+batch_size]
            
            # Prepare batch inputs
            input_texts = [
                f"query: {self.query} context: {beir_texts[int(hit.docid)]}" 
                for hit in batch_hits
            ]
            
            # Batch tokenization
            inputs = tokenizer(
                input_texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True
            ).to(device)
            
            # Batch inference
            with torch.no_grad():
                outputs = model_custom(**inputs)
                
                # Batch processing of probabilities and predictions
                logits = outputs.logits
                probs = softmax(logits, dim=-1)
                predicted_classes = torch.argmax(probs, dim=-1)
            
            # Convert batch results
            for j, hit in enumerate(batch_hits):
                hits_with_standing.append({
                    'docid': actual_docids[int(hit.docid)],
                    'score': hit.score,
                    'text': beir_texts[int(hit.docid)],
                    'standing': label_mapping[predicted_classes[j].item()]
                })

        # Sort and return top k results
        return sorted(hits_with_standing, key=lambda x: x['score'], reverse=True)[:self.k]
    
    def __rerank_without_standings(self, beir_texts: list, actual_docids: list):
        searcher = FaissSearcher('hnsw_index', self.embedding_model)
        hits = searcher.search(self.query_embedding, k=self.k)
        return [{'docid': actual_docids[int(hit.docid)], 'score': hit.score, 'text': beir_texts[int(hit.docid)]} for hit in hits]
    
    def retrieve(self):
        beir_texts, actual_docids, beir_docids, beir_vectors_list = self.__initial_rankings()
        self.__create_hnsw_index()
        return self.__rerank(beir_texts, actual_docids)
    
    def retrieve_without_standings(self):
        beir_texts, actual_docids, beir_docids, beir_vectors_list = self.__initial_rankings()
        self.__create_hnsw_index()
        return self.__rerank_without_standings(beir_texts, actual_docids)
