
from sentence_transformers import SentenceTransformer
import numpy as np
import scipy
import logging
from code.util.doc_util import get_event_windows_simple
import spacy
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger('better')
hdlr = logging.FileHandler('better.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr) 
logger.setLevel(logging.INFO)

class UnsupervisedLM:
    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load("en_core_web_md")  # make sure to use larger model!

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


    def reformulate_event_query(self, example_queries, example_triggers):        
        assert(len(example_queries) == len(example_triggers))
        query_chunks = []
        reformulated_queries = []

        for (query, trigger) in zip(example_queries, example_triggers):            
            query_chunks, predicates = get_event_windows_simple(query)
            flag = 0 
            predicates_nlp = [self.nlp(predicate) for predicate in predicates]
            trigger_nlp = self.nlp(trigger)
            scores = [trigger_nlp.similarity(predicate_nlp) for predicate_nlp in predicates_nlp]
            for (query_chunk, predicate) in zip(query_chunks, predicates):
                if trigger in predicate:
                    reformulated_queries.append(query_chunk)
                    flag = 1
                    break
            if flag == 0:
                for query_chunk in query_chunks:
                    if trigger in query_chunk:
                        reformulated_queries.append(query_chunk)
                        flag = 1
                        break   
            if flag == 0:
                index = np.argmax(scores)            
                reformulated_queries.append(query_chunks[index])
                flag = 1
                break
        assert(len(reformulated_queries) == len(example_queries))
        return reformulated_queries

    def reformulate_event_query_trigger_window(self, example_queries, example_triggers):        
        assert(len(example_queries) == len(example_triggers))
        reformulated_queries = []

        for (query, trigger) in zip(example_queries, example_triggers):            
            query_terms = query.split()
            query_terms_nlp = [self.nlp(term) for term in query_terms]
            trigger_nlp = self.nlp(trigger)
            scores = [trigger_nlp.similarity(query_term_nlp) for query_term_nlp in query_terms_nlp]
            index = np.argmax(scores)            
            begin = max(index - 5, 0)
            end = min(index + 5, len(query_terms))
            reformulated_queries.append(" ".join(query_terms[begin:end]))    
        assert(len(reformulated_queries) == len(example_queries))
        return reformulated_queries


    def aggregate_scores(self, scores, document_windows):
        scores_aggregated = []
        for (begin, end) in document_windows:
            score = np.min(scores[begin: end])
            scores_aggregated.append(score)
        return scores_aggregated

    def aggregate_scores_max(self, scores, document_windows):
        scores_aggregated = []
        for (begin, end) in document_windows:
            score = np.max(scores[begin: end])
            scores_aggregated.append(score)
        return scores_aggregated

    def search_combined(self, query_dict, documents, documents_embeddings, document_windows, original_documents):
        run_id = query_dict["run_id"]
        id2text = query_dict["queries"]
        id2triggers = query_dict["triggers"]
        logger.info("length of id to query text dictionary is {}".format(len(id2text)))
        result_string = ""
        for qid in id2text:            
            rank = 1        
            example_queries = id2text[qid].strip().split("\t")
            example_triggers = id2triggers[qid].strip().split("\t")
            #the following line just shortens each of the example queries.
            #example_queries = self.reformulate_event_query(example_queries, example_triggers)
            logger.info("An example of a query is {}".format(example_queries[0]))
            scores = self.score(example_queries, documents_embeddings, len(documents))
            scores = self.aggregate_scores(scores, document_windows)
            logger.info("scores have been calculated")
            scores_tuples = [(doc_id+1, score) for doc_id, score in enumerate(scores)] ##adding 1 to document id because our document id starts with 1
            scores_tuples = sorted(scores_tuples,key=lambda x: x[1], reverse=False)
            logger.info("scored tuple length is {}".format(len(scores_tuples)))
            logger.info("scores of top-3 documents are {} {} {}".format(scores_tuples[0], scores_tuples[1], scores_tuples[2]))
            logger.info("the top document is {}".format(original_documents[scores_tuples[0][0]-1]))
            for (doc_id, doc_score) in scores_tuples:
                result_string+=str(qid) + " Q0 " +  str(doc_id) + " " + str(rank) + " " + str(1/doc_score) + " " + str(run_id) + "\n"
                rank+=1
            logger.info("Finished processing query id {}".format(qid))
        return result_string
        

    def search_combined_query(self, query_dict, documents, documents_embeddings, document_windows, original_documents):        
        run_id = query_dict["run_id"]
        id2text = query_dict["queries"]
        id2triggers = query_dict["triggers"]
        logger.info("length of id to query text dictionary is {}".format(len(id2text)))
        result_string = ""
        for qid in id2text:            
            rank = 1        
            example_queries = id2text[qid].strip().split("\t")
            example_triggers = id2triggers[qid].strip().split("\t")
            #the following line just shortens each of the example queries.
            example_queries = self.reformulate_event_query_trigger_window(example_queries, example_triggers)
            logger.info("An example of a query is {}".format(example_queries[0]))
            scores = self.score(example_queries, documents_embeddings, len(documents))
            scores = self.aggregate_scores(scores, document_windows)            
            logger.info("scores have been calculated")
            scores_tuples = [(doc_id+1, score) for doc_id, score in enumerate(scores)] ##adding 1 to document id because our document id starts with 1
            scores_tuples = sorted(scores_tuples,key=lambda x: x[1], reverse=False)
            logger.info("scored tuple length is {}".format(len(scores_tuples)))
            logger.info("scores of top-3 documents are {} {} {}".format(scores_tuples[0], scores_tuples[1], scores_tuples[2]))
            logger.info("the top document is {}".format(original_documents[scores_tuples[0][0]-1]))
            for (doc_id, doc_score) in scores_tuples:
                result_string+=str(qid) + " Q0 " +  str(doc_id) + " " + str(rank) + " " + str(1/doc_score) + " " + str(run_id) + "\n"
                rank+=1
            logger.info("Finished processing query id {}".format(qid))
        return result_string
