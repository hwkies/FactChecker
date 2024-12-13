# FactChecker

## Setup
### Link to Fine-Tuned Models
https://drive.google.com/drive/folders/1mekz1wEQiY3HDQzToQ3a2KX81Tq6eUfU  
Dowload our classifiers and place the files in the parent directory of this project before attempting to run our code. If they are not present, the classifiers will not be found and the code will not work.

### Pip Installations: 
Be sure to use the exact versions as there may be versioning errors otherwise  
pip install pyserini==0.43.0  
pip install numpy==1.26.4  
pip install sentence_transformers==3.3.1  
pip install torch==2.5.1+cu121  
pip install datasets==3.1.0  
pip install transformers==4.46.2  
pip install scikit-learn==1.3.0  

### Install Java 21
If you get an error saying JAVA_HOME cannot be found:  
Add the path to the java jdk to your system environment variables  
https://docs.oracle.com/cd/F74770_01/English/Installing/p6_eppm_install_config/89522.htm  

## Running the project
Example Usage (present in retrieve.ipynb) 
```
from index import RetrievalModel

query = "vaccines don't cause autism"
k = 100
numResults = 15

retrieval_model = RetrievalModel(query, k)
retrieval_model.retrieve()[:numResults]
```
In this example, the user specifies a query, a k-value, and a number of results they want to be returned. The query and k-values are passed as parameters to an instance of our retrieval model. The query is the query to be submitted to the model and the k-value is the number of initial results to pull from each pre-built index.  

numResults determines how many of the final results the user wants to be displayed. This number can only be as high as the k-value otherwise the model might attempt to return more results than it has, causing an error.

## index.py
### RetrievalModel Class:
This is the class responsible for performing our retrieval. It starts by taking in a query and a k-value (k-value is optional and is initialized to 100 if not specified) and initializing five pre-built Lucene indexes and searchers. These searchers will perform our initial retrieval by selecting the top-k documents relevant to the query using BM25. The indexes are a set of various fact-checking and news-based multifield impact indexes from the BEIR (Benchmarking IR) pre-built indexes offered by pyserini. They cover a range of common areas of fact-verification including documents surrounding climate change, COVID-19, news stories, and more. After performing the initial retrieval, the results are cleaned to remove duplicates and then processed into vectors by a pre-trained encoding model from the sentence_transformers package. From here, the embeddings are written into a jsonl file in the embeddings directory where they will be accessed when we create our HNSW index. We then create our HNSW index by calling the bash script create_hnsw. This will use pyserini's built-in index creation tools to transform our embeddings into a ready-to-use HNSW index stored in the hnsw_index directory. With the index made, we can then rerank the initial results using this index. We do this by initializing a FaissSearcher and searching our HNSW index using a vectorized version of our query. Once we have these results, we batch-process the stance of each document and eventually return a list of k results ordered by their score with the form below:
```json
{"docid": docid, "score": document score from FaissSearcher, 
"text": document text, "standing": calculated document stance}
```

### Metrics
The metrics we used for our retrieval model were MAP@100, Average Recall @100, F1 Score @100, along with the mean, median, min, max, and standard deviation of query times. The metrics computed are stored in the eval_retrieval_metrics.json file; however, the metrics calculated are not accurate as we ran into computational difficulties when attempting to evaluate these metrics on our model.
#### evaluate_retrieval.py
This is where we did our retrieval metric calculations. We started by pulling topics and qrels aligning with our pre-built indexes in order to retrieve sample queries along with their relevant documents. Because of computational restrictions, we only ran our metric calculations on a subset of 5 queries from the climate-fever topic, but with more time and compute power, these could easily be run on a greater subset of these test queries.

## Fine-Tuning NLI Classifier
### roberta_mnli.py
This is the file where we do our initial fine-tuning. We start by loading the pre-trained RoBERTa-large model and the multi-nli dataset. We then preprocess the dataset and pass it as training data to RoBERTa large.
### roberta_fever.py
This is where we further fine-tune our classifier. Similar to before, we start by loading our previously fine-tuned model and the nli_fever dataset which can be found in the nli_fever directory. We then preprocess our data and use it to fine-tune our model further.

### Storing Fine-Tuned Models
The models are stored after fine-tuning in the roberta_mnli and roberta_fever_mnli directories from where they can be loaded elsewhere in the project. Because of the size of these model files, we are not able to upload them to GitHub and have instead uploaded them to Google Drive. The link associated with this drive can be found above in the [Setup](#setup) section.

### Metrics
The metrics used for these classifiers are accuracy, precision, recall, and f1-score. The exact numbers for these metrics can be found for both our initial fine-tuned model and the final fine-tuned model in the eval_nli_metrics.json file. 
#### evaluate_nli.py
This is the file where we evaluate both stages of our fine-tuned classifier. We start by loading our fine-tuned models and their respective testing data and preprocessing the data to be passed to the models. We then evaluate the models using the built-in trainer.evaluate() function defined in the transformers package and a home-made compute_metrics function which uses outputs from our model and sklearn metric calculators to define the metrics for our models. The output of this file is stored in the eval_nli_metrics.json file under the keys "metrics_mnli" and "metrics_fever".

## More Info:
For more information on the project please follow the fine-tuned models link in [Setup](#setup) and navigate to the Kiesman_Kong_Final_Report pdf.