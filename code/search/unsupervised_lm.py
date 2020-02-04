
from sentence_transformers import SentenceTransformer
import numpy as np
import scipy

import logging
logger = logging.getLogger('better')
hdlr = logging.FileHandler('better.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

class Unsupervised:
    def __init__(self, config):
        self.config = config
    
    def bert_representation(self, sentences):
        embedder = SentenceTransformer('bert-base-nli-stsb-mean-tokens')
        sentences_embeddings = embedder.encode(sentences)
        return sentences_embeddings        

    def roberta_representation(self, sentences):
        embedder = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')
        sentences_embeddings = embedder.encode(sentences)
        return sentences_embeddings 

    def score(self, example_queries, documents_embeddings, num_docs):
        example_queries_embeddings = self.bert_representation(example_queries)        
        scores = np.zeros(num_docs)
        for example_query, example_query_embedding in zip(example_queries, example_queries_embeddings):
            distances = scipy.spatial.distance.cdist([example_query_embedding], documents_embeddings, "cosine")[0]
            scores+=distances
        return scores    

    def search(self, query_dict, documents, documents_embeddings):
        run_id = query_dict["run_id"]
        id2text = query_dict["queries"]
        logger.info("length of id to query text dictionary is {}".format(len(id2text)))
        result_string = ""
        for qid in id2text:
            rank = 1        
            example_queries = id2text[qid].split("\t")
            logger.info("An example of a query is {}".format(example_queries[0]))
            scores = self.score(example_queries, documents_embeddings, len(documents))
            logger.info("scores have been calculated")
            scores_tuples = [(doc_id+1, score) for doc_id, score in enumerate(scores)] ##adding 1 to document id because our document id starts with 1
            scores_tuples = sorted(scores_tuples,key=lambda x: x[1], reverse=False)
            logger.info("scored tuple length is {}".format(len(scores_tuples)))
            logger.info("scores of top-3 documents are {} {} {}".format(scores_tuples[0], scores_tuples[1], scores_tuples[2]))
            logger.info("the top document is {}".format(documents[scores_tuples[0][0]-1]))
            for (doc_id, doc_score) in scores_tuples:
                result_string+=str(qid) + " Q0 " +  str(doc_id) + " " + str(rank) + " " + str(1/doc_score) + " " + str(run_id) + "\n"
                rank+=1
            logger.info("Finished processing query id {}".format(qid))
        return result_string


