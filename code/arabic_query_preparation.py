import numpy as np 
import pandas as pd
import json 

import re


# def query_container(id2text, field, keyword_query):
#     """
#     Given a query dictionary generate indri query xml
#     :param query_dict: containing seeds
#     :param field: text, name etc.
#     :return: string that contains the query
#     """
#     fielded_query_string = ''
#     st = ''
#     count = 1
#     for query_string in query_strings:
#         query_string_splitted = query_string.split()
#         for i in np.arange(len(query_string_splitted)):
#             fielded_query_string += query_string_splitted[i] + "." + field + ' '

#         st+= '<query>\n'
#         st+= '<number>' + str(count) + '</number>\n'
#         st+= '<text>' + fielded_query_string + keyword_query + '</text>\n'
#         st+= '</query>\n'
#         count+=1
#     return st


def query_container(id2text, field):
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
        query_string = " ".join(list(set(query_string.split())))
        # fielded_query_string = ''
        # query_string_splitted = query_string.split()
        # for i in np.arange(len(query_string_splitted)):
        #     fielded_query_string += query_string_splitted[i] + "." + field + ' '

        st+= '<query>\n'
        st+= '<number>' + str(key) + '</number>\n'
        st+= '<text>' + query_string + '</text>\n'
        st+= '</query>\n'
    return st


def query_xml_container(query, index_directory, retrieval_approach, run_id):
    st = ''
    st+= '<parameters>\n'
    st+= '<index>' + index_directory + '</index>\n'
    st+= '<count>1000</count>\n'
    st+= '<trecFormat>true</trecFormat>\n'
    st+= '<runID>' + str(run_id) +'</runID>\n'
    st+= '<retModel>' + retrieval_approach + '</retModel>\n'
    st+= query
    
    if retrieval_approach=='lm':
        st += '</parameters>\n'
    else:
        st+= '<fbDocs>10</fbDocs>\n'
        st+= '<fbTerms>20</fbTerms>\n'
        st+= '<fbMu>0.5</fbMu>\n'
        st+= '<fbOrigWeight>0.5</fbOrigWeight>\n'
        st+= '</parameters>\n'
    return st

#query is event type
query2id = json.load(open("small_data/ace/query2id.json"))
df = pd.read_csv("small_data/ace/arabic/queries/arabic_query_translated.csv", sep="\t", index_col=False)
df['sentence_translation'].fillna("", inplace=True)
df['trigger_translation'].fillna("", inplace=True)

print(len(df))


query2text = {}
for i, event_type in enumerate(df['Event_Type']):
    event_type.strip()
    if event_type in query2text:
        st = query2text[event_type]
        #print(st)
        #print(st)
        st+= df['trigger_translation'][i] + " "
        #st+= df['sentence_translation'][i] + " "
        query2text[event_type] = st
    else:
        query2text[event_type] = df['trigger_translation'][i] + " "
        #query2text[event_type] = df['sentence_translation'][i] + " "

id2text = {}

for key in query2text.keys():   
    text = query2text[key]
    id = query2id[key]
    id2text[id] = text

print (len(query2id))
print (len(query2text))
print (len(id2text))

approaches = ["lm", "prf"]

run_id = 1
for approach in approaches: 
    query = query_container(id2text, "text")
    indri_query = query_xml_container(query, "small_data/ace/english/index", approach, run_id)
    indri_query_file = open("small_data/ace/arabic/indri_queries/" + approach + "/query.xml", "w") 
    indri_query_file.write(indri_query)   
    run_id+=1