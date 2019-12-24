import numpy as np 
import pandas as pd
import json 
import re
import os

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
            query_string = " ".join(list(set(query_string.split())))
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

def sample_queries_for_types(df, query_type, num_sentences = 1):
    
    column_name = ""
    if query_type == "triggers":
        column_name = "trigger_translation"
    elif query_type == "sentences":
        column_name = "sentence_translation"

    query2count = {}
    query2text = {}
    for i, event_type in enumerate(df['Event_Type']):
        if df[column_name][i].strip() == "dummy":
            continue
        event_type.strip()
        if event_type in query2text:
            if query2count[event_type] <num_sentences:    
                st = query2text[event_type]
                st+= df[column_name][i] + " "
                query2text[event_type] = st
        else:
            query2text[event_type] = df[column_name][i] + " "
            query2count[event_type] = 1

    return query2text



def main():
    config = json.load(open("code/config/basic_config.json"))
    data_directory = config["data"]
    src_lang = config["src_lang"]
    trg_lang = config["trg_lang"]
    approach = config["approach"]
    query_type = config["query_type"] 
    query2id_file = config["query_to_id_file"]
    translation = config["translation"]
    index_dir = config["index_dir"]
    #query is event type
    query2id = json.load(open(os.path.join(data_directory, src_lang, "queries", query2id_file)))
    df = pd.read_csv(os.path.join(data_directory, src_lang, "queries", translation, src_lang + "_translations.csv"), sep="\t")
    df['sentence_translation'].fillna("dummy", inplace=True)
    df['trigger_translation'].fillna("dummy", inplace=True)

    approaches = ["ql", "prf"]
    query_types = ["sentences", "triggers"]

    run_id = 1

    for approach in approaches: 
        for query_type in query_types:
            query2text = sample_queries_for_types(df, query_type, num_sentences = 3)
            id2text = {}

            for key in query2text.keys():   
                text = query2text[key]
                id = query2id[key]
                id2text[id] = text

            print("number of topics {}".format(len(id2text)))
            query = query_container(id2text, "text", query_type)
            indri_query = query_xml_container(query, os.path.join(data_directory, trg_lang, index_dir), approach, run_id)
            indri_query_file = open(os.path.join(data_directory, src_lang, "indri_queries", approach, query_type, "query.xml"), "w")
            indri_query_file.write(indri_query)   
            indri_query_file.close()
            run_id+=1

if __name__ == "__main__":
    main()
