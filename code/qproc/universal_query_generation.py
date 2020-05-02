import numpy as np 
import pandas as pd
import json 
import re
import os
import time
import sklearn.utils
import random 

def query_container(id2text, field, query_type):
    """
    Given a query dictionary generate indri query xml
    :param query_dict: containing seeds
    :param field: text, name etc.
    :return: string that contains the query
    """
    st = ''
    for key in id2text:
        query_string = id2text[key]
        query_string = re.sub(r'[^\w\s]','',query_string)       
        #sentence queries are usually very big and there are many terms. so we are converting them to term set. 
        if query_type == "sentences":
            query_string = " ".join(list(set(query_string.split("\t"))))
        st+= '<query>\n'
        st+= '<number>' + str(key) + '</number>\n'
        st+= '<text>' + query_string + '</text>\n'
        st+= '</query>\n'
    return st

def query_xml_container(query, index_directory, retrieval_approach, run_id):
    st = ''
    st+= '<parameters>\n'
    st+= '<index>' + os.getcwd() + "/" + index_directory + '</index>\n'
    st+= '<count>1000</count>\n'
    st+= '<trecFormat>true</trecFormat>\n'
    st+= '<runID>' + str(run_id) +'</runID>\n'
    st+= '<retModel>' + retrieval_approach + '</retModel>\n'
    st+= query
    
    if retrieval_approach=='ql':
        st += '</parameters>\n'
    else:
        st+= '<fbDocs>100</fbDocs>\n'
        st+= '<fbTerms>10</fbTerms>\n'
        st+= '<fbMu>0.5</fbMu>\n'
        st+= '<fbOrigWeight>0.5</fbOrigWeight>\n'
        st+= '</parameters>\n'
    return st

def sample_queries_for_combined_types(df, num_sentences = 1, randomized = False):
    """[Samples query sentences for a specific event type]    
    Arguments:
        df {[dataframe]} -- [a dataframe containing all the queries (sentence and phrase) along with the event types]
        query_type {[type]} -- [sentence or trigger. Trigger would justify the importance of rationale annotation]    
    Keyword Arguments:
        num_sentences {int} -- [number of example events. each time a different event example is sampled.] (default: {1})    
    Returns:
        [dict] -- [event_type and sentences belonging to that type]
    """
    column_name = "sentence_translation"
    trigger_column_name = "trigger_translation"

    query2count = {}
    query2text = {}
    query2triggers = {} 
    for i, event_type in enumerate(df['Event_Type']):
        if df[column_name][i].strip() == "dummy":
            continue
        event_type.strip()
        if event_type in query2text:
            #if query2count[event_type] < num_sentences:    
            st = query2text[event_type]
            st = re.sub(r'[^\w\s]','',st)   
            st+= df[column_name][i] + "\t"
            st_trigger = query2triggers[event_type]
            st_trigger+= df[trigger_column_name][i] + "\t"
            query2text[event_type] = st
            query2triggers[event_type] = st_trigger
            query2count[event_type]+=1
        else:
            query2text[event_type] = df[column_name][i].strip() + "\t"
            query2triggers[event_type] = df[trigger_column_name][i].strip() + "\t"
            query2count[event_type] = 1

    query2textshuff = {}
    query2triggersshuff = {}

    for qid in query2text:
        te = query2text[qid].split("\t")
        tg = query2triggers[qid].split("\t")
        for i, item in enumerate(te):
            if len(te[i]) == 0 or len(tg[i]) == 0:
                te.pop(i)
                tg.pop(i)
        temp_list = list(zip(te, tg))
        random.shuffle(temp_list)
        te, tg = zip(*temp_list)
        query2textshuff[qid] = "\t".join(te[0:num_sentences])
        query2triggersshuff[qid] = "\t".join(tg[0:num_sentences]) 

    return query2textshuff, query2triggersshuff

def main():
    config = json.load(open("code/config/unsupervised_lm_config.json"))
    data_directory = config["data"]
    src_lang = config["src_lang"]
    trg_lang = config["trg_lang"]
    query2id_file = config["query_to_id_file"]
    translation = config["translation"]
    doc_dir = config["index_dir"] #we are naming it as document directory because we do not build an index for unsupervised lm 
    #query is event type
    query2id = json.load(open(os.path.join(data_directory, src_lang, "queries", query2id_file)))
    df = pd.read_csv(os.path.join(data_directory, src_lang, "queries", translation, src_lang + "_translations.csv"), sep="\t")
    df['sentence_translation'].fillna("dummy", inplace=True)
    df['trigger_translation'].fillna("dummy", inplace=True)
    
    run_id = 1

    for num_examples in range(1,11):
        query2text, query2triggers = sample_queries_for_combined_types(df, num_sentences = num_examples)
        query_dict = {}
        id2text = {}
        id2triggers = {} 

        for key in query2text.keys():
            text = query2text[key] #query2text comes up with dictionary entries like
            id = query2id[key]
            id2text[id] = text.strip()
            trigger = query2triggers[key]
            id2triggers[id] = trigger.strip()
            print(text)
            #print("********************************************************************************")
            #print(str(id) + "\t" + trigger)
            assert(len(text.strip().split("\t")) == len(trigger.strip().split("\t")))
            print("{} \t {}".format(key, text))

        query_dict["run_id"] = run_id
        query_dict["trg_lang"] = trg_lang
        query_dict["doc_dir"] = doc_dir #here we have a document directory instead of index directory because we do not need indexing for 16000 documents.
        query_dict["queries"] = id2text
        query_dict["triggers"] = id2triggers
        print(id2triggers)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        os.makedirs(os.path.join(data_directory, src_lang, "universal_queries"), exist_ok=True)
        query_file = open(os.path.join(data_directory, src_lang, "universal_queries", timestr + "_" + str(num_examples) + "_query.json"), "w")
        json.dump(query_dict, query_file)
        run_id+=1

if __name__ == "__main__":
    main()
