import os
import subprocess 
import json 
import time
from trectools import TrecQrel, TrecRun, TrecEval

def eval(qrel_file_path, run_file_path):
    """[summary]
    
    Arguments:
        qrel_file_path {[string]} -- [path of the qrel file usually located at the source language folder]
        run_file_path {[string]} -- [path of the run file usually located at the results folder of a language]
    
    Returns:
        [type] -- [precision@10, precision@20, precision@30, mAP rounded up to four digits]
    """

    r1 = TrecRun(run_file_path)
    qrels = TrecQrel(qrel_file_path)

    te = TrecEval(r1, qrels)
    p5 = te.get_precision(depth=5)     
    p10 = te.get_precision(depth=10)
    p20 = te.get_precision(depth=20)
    map = te.get_map()
    rprec = te.get_rprec()
    run_object = r1.evaluate_run(qrels, per_query=True)
    
    return round(p5, 4), round(p10, 4), round(p20, 4), round(map, 4), round(rprec, 4)


def search(data_directory, src_lang, trg_lang, approach, query_type, num_examples): 
    """[summary]
    constructs the query file path and the run file path and then dumps the result in the run file path
    Arguments:
        data_directory {[string]} -- [description]
        src_lang {[string]} -- [language of the query]
        trg_lang {[string]} -- [language of retrieval corpus]
        approach {[string]} -- [indicates which approach to use for retrieval, for example "ql"]
        query_type {[string]} -- [indicate which part of the query to consider for retrieval, for example "trigger" for event trigger query type]
    """
    os.makedirs(os.path.join(os.getcwd(), data_directory, trg_lang, "runs", src_lang, approach, query_type), exist_ok=True)
    run_file_path = os.path.join(os.getcwd(), data_directory, trg_lang, "runs", src_lang, approach, query_type, str(num_examples) + "_run.xml")
    os.makedirs(os.path.join(os.getcwd(), data_directory, src_lang, "indri_queries", approach, query_type), exist_ok=True)
    query_file_path = os.path.join(os.getcwd(), data_directory, src_lang, "indri_queries", approach, query_type, str(num_examples) + "_query.xml")
    cmd = "IndriRunQuery " + query_file_path + " > " + run_file_path
    print("running the search with {}".format(cmd))
    #returned_value = subprocess.call("module load indri/5.13", shell=True)
    returned_value = subprocess.call(cmd, shell=True)
    print("output of Indri is {}".format(returned_value))
    return run_file_path, query_file_path

def main():
    #would like to put a timestamp on result files 
    desc2runobj = {} 
    run_significance = {} 

    config = json.load(open("code/config/basic_config_ace.json"))
    data_directory = config["data"]
    #indicates the language of query 
    src_lang = config["src_lang"]
    #indicates the language of retrieval
    trg_lang = config["trg_lang"]

    #approach = config["approach"]
    #query_type = config["query_type"] 
    
    qrel_file_path = os.path.join(data_directory, trg_lang, "queries", "qrels." + trg_lang + "_events.txt") 
    print(qrel_file_path)
        
    approaches = ["ql", "prf"]
    query_types = ["sentences", "triggers"]
    
    
    for approach in approaches:
        for query_type in query_types:            
            os.makedirs(os.path.join(data_directory, trg_lang, "results", "data", src_lang, approach, query_type), exist_ok=True)
            result_file = open(os.path.join(data_directory, trg_lang, "results", "data", src_lang, approach, query_type, "output.res"), "w")    
            for num_examples in range(1,31):
                run_file_path, _ = search(data_directory, src_lang, trg_lang, approach, query_type, num_examples)
                #print(run_file_path)
                #print(query_file_path)
                print(str(num_examples) + "\t" + src_lang + "\t" + trg_lang + "\t" + approach + "\t" + query_type)
                p5, p10, p20, mAP, rprec= eval(qrel_file_path, run_file_path)
                print("{}\t{}\t{}\t{}\t{}\t{}\n".format(num_examples, p5, p10, p20, mAP, rprec))                   
                result_file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(num_examples, p5, p10, p20, mAP, rprec))   
                #result_str+= approach + "," + query_type + "," + str(p5) + "," + str(p10) + "," + str(p20) + "," + str(map) + "," + str(rprec) + "\n"
                #desc2runobj[run_description] = run_obj
            result_file.close()

    # for key1 in desc2runobj.keys():
    #     for key2 in desc2runobj.keys():
    #         result_r1 = desc2runobj[key1]
    #         result_r2 = desc2runobj[key2]

    #         p5_pvalue = result_r1.compare_with(result_r2, metric="P_5") 
    #         p10_pvalue = result_r1.compare_with(result_r2, metric="P_10")
    #         p20_pvalue = result_r1.compare_with(result_r2, metric="P_20")
    #         map_pvalue = result_r1.compare_with(result_r2, metric="map") 
    #         rprec_pvalue = result_r1.compare_with(result_r2, metric="rprec") 
    #         run_significance[(key1, key2)] = (p5_pvalue, p10_pvalue, p20_pvalue, map_pvalue, rprec_pvalue )
            

    #result_file.write(result_str)
    #print(result_str)
    #print(run_significance)
    
    
if __name__ == "__main__":
    main()
