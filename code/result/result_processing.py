import os
import json 
import pandas as pd 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

def convert_result_string(result_string, method, query_type):
    result_string_splitted = result_string.strip().split("\t")
    result = []
    result.append(int(result_string_splitted[0]))
    result.append(float(result_string_splitted[1]))
    result.append(float(result_string_splitted[2]))
    result.append(float(result_string_splitted[3]))
    result.append(float(result_string_splitted[4]))
    result.append(method)
    result.append(query_type)
    return result
    

config = json.load(open("code/config/unsupervised_lm_config.json"))
data_directory = config["data"]
src_lang = "chinese"
trg_lang = "english"
representations = ["", "", "bert"]
approaches = ["ql", "prf", "unsupervised_lm"]
query_types = ["sentences", "triggers", "combined_query"]
df_array = []
    
for tuple in zip(approaches, representations):
    print(tuple)
    approach = tuple[0]
    representation = tuple[1]
    for query_type in query_types:
        
        if approach == "unsupervised_lm" and (representation=="" or query_type=="triggers"):
            continue
        if approach == "ql" and query_type=="combined_query":
            continue
        if approach == "prf" and query_type=="combined_query":
            continue
            
        print(os.path.join(data_directory, trg_lang, "results", "data", src_lang, approach, representation, query_type, "output.res"))
        result_file = open(os.path.join(data_directory, trg_lang, "results", "data", src_lang, approach, representation, query_type, "output.res"))
        result_strings = result_file.readlines()
        data = []
        for result_string in result_strings:
            if representation!="":
                result = convert_result_string(result_string, approach + "_" + representation, query_type)
            else:
                result = convert_result_string(result_string, approach, query_type)
            data.append(result)
        df = pd.DataFrame(data, columns = ["#examples", "p5", "p10", "p20", "mAP", "method", "query_type"])
        df_array.append(df)
        #all_df = pd.concat([all_df, df], ignore_index=True)
        
#all_df = df_array[0]
#for df in df_array[1:]:
#    all_df = all_df.append(df, ignore_index=True)
all_df = pd.concat(df_array)
print(all_df)

sns_plot = sns.relplot(x="#examples", y="p5", col="query_type", row="method", height=3, kind="line", estimator=None, data=all_df)
sns_plot.savefig(os.path.join(data_directory, trg_lang, "results", "figures", "result.pdf"))