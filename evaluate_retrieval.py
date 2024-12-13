from pyserini.search import get_topics, get_qrels, get_qrels_file
from index import RetrievalModel
import json
import time
import statistics

climate_topics = get_topics('beir-v1.0.0-climate-fever-test')
covid_topics = get_topics('beir-v1.0.0-trec-covid-test')
fever_topics = get_topics('beir-v1.0.0-fever-test')
news_topics = get_topics('beir-v1.0.0-trec-news-test')
scifact_topics = get_topics('beir-v1.0.0-scifact-test')

climate_topics = {k: climate_topics[k] for k in list(climate_topics)[:5]}
covid_topics = {k: covid_topics[k] for k in list(covid_topics)[:10]}
fever_topics = {k: fever_topics[k] for k in list(fever_topics)[:10]}
news_topics = {k: news_topics[k] for k in list(news_topics)[:10]}
scifact_topics = {k: scifact_topics[k] for k in list(scifact_topics)[:10]}

def safe_get_qrels(dataset):
    try:
        return get_qrels(dataset)
    except Exception as e:
        print(f"Error getting qrels for {dataset}: {e}")
        # Below code pulled from pyserini/pyserini/search/__base.py
        # Needed to rewrite to open file with utf-8 because of unrecognized characters
        qrels_file_path = get_qrels_file(dataset)
        qrels = {}
        with open(qrels_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                qid, _, docid, judgement = line.rstrip().split()
                
                if qid.isdigit():
                    qrels_key = int(qid)
                else:
                    qrels_key = qid
                    
                if docid.isdigit():
                    doc_key = int(docid)
                else:
                    doc_key = docid
                    
                if qrels_key in qrels:
                    qrels[qrels_key][doc_key] = judgement
                else:
                    qrels[qrels_key] = {doc_key: judgement}
        return qrels

climate_qrels = safe_get_qrels('beir-v1.0.0-climate-fever-test')
covid_qrels = safe_get_qrels('beir-v1.0.0-trec-covid-test')
fever_qrels = safe_get_qrels('beir-v1.0.0-fever-test')
news_qrels = safe_get_qrels('beir-v1.0.0-trec-news-test')
scifact_qrels = safe_get_qrels('beir-v1.0.0-scifact-test')

def get_average_precision_sum(topics: dict, qrels: dict, k=100):
    averagePrecisionSum = 0
    for queryid, query in topics.items():
        hits = RetrievalModel(query['title']).retrieve_without_standings()[:k]
        precisionSum = 0
        relevantCount = 0
        for i, hit in enumerate(hits):
            if hit['docid'] in qrels[queryid]:
                relevantCount += 1
                precisionSum += relevantCount / (i + 1)
        if relevantCount > 0:
            averagePrecisionSum += precisionSum / relevantCount
        else:
            averagePrecisionSum += 0
    return averagePrecisionSum

def get_recall_at_k(topics: dict, qrels: dict, k=100):
    total_relevant_docs = 0
    retrieved_relevant_docs = 0
    for queryid, query in topics.items():
        total_query_relevant = sum(1 for rel in qrels[queryid].values() if int(rel) > 0)
        total_relevant_docs += total_query_relevant
        hits = RetrievalModel(query['title']).retrieve_without_standings()[:k]
        query_retrieved_relevant = sum(1 for hit in hits if hit['docid'] in qrels[queryid] and int(qrels[queryid][hit['docid']]) > 0)
        retrieved_relevant_docs += query_retrieved_relevant
    recall_at_100 = retrieved_relevant_docs / total_relevant_docs if total_relevant_docs > 0 else 0
    return recall_at_100

def get_f1_score_at_k(topics: dict, qrels: dict, k=100):
    total_f1_score = 0
    num_queries = len(topics)
    for queryid, query in topics.items():
        hits = RetrievalModel(query['title']).retrieve_without_standings()[:k]
        total_relevant = sum(1 for rel in qrels[queryid].values() if int(rel) > 0)
        retrieved_relevant = sum(1 for hit in hits if hit['docid'] in qrels[queryid] and int(qrels[queryid][hit['docid']]) > 0)
        precision = retrieved_relevant / k if k > 0 else 0
        recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        total_f1_score += f1_score
    average_f1_score = total_f1_score / num_queries

    return average_f1_score

def measure_retrieval_time(topics: dict, num_runs=5, k=100):
    all_query_times = []

    for queryid, query in topics.items():
        query_times = []
        for _ in range(num_runs):
            start_time = time.perf_counter()
            hits = RetrievalModel(query['title']).retrieve()[:k]
            end_time = time.perf_counter()
            query_times.append((end_time - start_time) * 1000)
        all_query_times.append(statistics.mean(query_times))
    overall_mean_time = statistics.mean(all_query_times)
    overall_median_time = statistics.median(all_query_times)
    overall_min_time = min(all_query_times)
    overall_max_time = max(all_query_times)
    overall_std_dev = statistics.stdev(all_query_times) if len(all_query_times) > 1 else 0

    return {
        'mean_time': overall_mean_time,
        'median_time': overall_median_time,
        'min_time': overall_min_time,
        'max_time': overall_max_time,
        'std_dev': overall_std_dev
    }

averagePrecisionSum = get_average_precision_sum(climate_topics, climate_qrels) + get_average_precision_sum(covid_topics, covid_qrels) + get_average_precision_sum(fever_topics, fever_qrels) + get_average_precision_sum(news_topics, news_qrels) + get_average_precision_sum(scifact_topics, scifact_qrels)
lenTopics = len(climate_topics) + len(covid_topics) + len(fever_topics) + len(news_topics) + len(scifact_topics)
meanAveragePrecision = averagePrecisionSum / lenTopics
averageRecall = get_recall_at_k(climate_topics, climate_qrels) + get_recall_at_k(covid_topics, covid_qrels) + get_recall_at_k(fever_topics, fever_qrels) + get_recall_at_k(news_topics, news_qrels) + get_recall_at_k(scifact_topics, scifact_qrels) / 5
f1Score = get_f1_score_at_k(climate_topics, climate_qrels) + get_f1_score_at_k(covid_topics, covid_qrels) + get_f1_score_at_k(fever_topics, fever_qrels) + get_f1_score_at_k(news_topics, news_qrels) + get_f1_score_at_k(scifact_topics, scifact_qrels) / 5
timeStats = measure_retrieval_time(climate_topics, num_runs=1)
jsonResult = {
    "MAP@100": meanAveragePrecision, 
    "Recall@100": averageRecall, 
    "F1@100": f1Score,
    "time_stats": timeStats
    }
print(jsonResult)

with open(f"eval_retrieval_metrics.json", "w") as f:
    json.dump(jsonResult, f)