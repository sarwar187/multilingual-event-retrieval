from trectools import TrecRun, TrecEval, fusion, TrecQrel
import os 

language = "chinese"

#TODO: automate this 
#def fuse_runs(run_file_dir1, run_file_dir2):
#queries = os.listdir(run_file_dir1)   

def eval(r1, qrels):
    """[summary]
    
    Arguments:
        qrel_file_path {[string]} -- [path of the qrel file usually located at the source language folder]
        run_file_path {[string]} -- [path of the run file usually located at the results folder of a language]
    
    Returns:
        [type] -- [precision@10, precision@20, precision@30, mAP rounded up to four digits]
    """
    
    te = TrecEval(r1, qrels)
    p5 = te.get_precision(depth=5)     
    p10 = te.get_precision(depth=10)
    p20 = te.get_precision(depth=20)
    map = te.get_map()
    rprec = te.get_rprec()
    
    return round(p5, 4), round(p10, 4), round(p20, 4), round(map, 4), round(rprec, 4)

result_file = open("output.res", "w")

for num_examples in range(5, 6):
    r1 = TrecRun("small_data/ace/english/runs/chinese/prf/triggers/" + str(num_examples) + "_run.xml")
    #r2 = TrecRun("small_data/ace/english/runs/chinese/ql/sentences/2_run.xml")
    r2 = TrecRun("/mnt/scratch/smsarwar/better/small_data/ace/english/runs/chinese/unsupervised_lm/bert/combined_query/" + str(num_examples) + ".run")
    qrels = TrecQrel("small_data/ace/english/queries/qrels.english_events.txt")
    # Easy way to create new baselines by fusing existing runs:
    fused_run = fusion.reciprocal_rank_fusion([r1,r2])
    # Save run to disk with all its topics
    fused_run.print_subset("my_fused_run.txt", topics=fused_run.topics())    
    p5, p10, p20, mAP, rprec = eval(fused_run, qrels)
    result_file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(num_examples, p5, p10, p20, mAP, rprec))     

result_file.close()
    
    