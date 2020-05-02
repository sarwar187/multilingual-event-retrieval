from trectools import TrecRun, TrecEval, fusion, TrecQrel
import os 

language = "arabic"

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

#just specify two directories
dir1 = "small_data/ace/english/runs/" + language + "/prf/triggers/"
dir2 = "/mnt/scratch/smsarwar/better/small_data/ace/english/runs/" + language + "/unsupervised_lm/bert/combined_query/"

dir1_files = set()
dir2_files = set()

for name in os.listdir(dir1):
    name_splitted = name.split(".")
    #print(name_splitted[0].split("_"))
    file_id = "_".join(name_splitted[0].split("_")[0:2])
    dir1_files.add(file_id)

for name in os.listdir(dir2):
    name_splitted = name.split(".")
    #print(name_splitted[0].split("_"))
    file_id = name_splitted[0].strip()
    print(file_id)
    dir2_files.add(file_id)

#print(dir1_files)
#dir2_files = set(os.listdir(dir2))
#print(dir2_files)

intersection = dir1_files.intersection(dir2_files)
print(len(intersection))

for f in intersection: 
    name = f.split(".")[0]
    name_splitted = name.split("_")
    timestr = name_splitted[0]
    num_examples = name_splitted[1]
    r1 = TrecRun("small_data/ace/english/runs/" + language + "/prf/triggers/" + f + "_run.xml")
    #r2 = TrecRun("small_data/ace/english/runs/chinese/ql/sentences/2_run.xml")
    r2 = TrecRun("/mnt/scratch/smsarwar/better/small_data/ace/english/runs/" + language + "/unsupervised_lm/bert/combined_query/" + f + ".run")
    qrels = TrecQrel("small_data/ace/english/queries/qrels.english_events.txt")
    # Easy way to create new baselines by fusing existing runs:
    fused_run = fusion.reciprocal_rank_fusion([r1,r2])
    # Save run to disk with all its topics
    fused_run.print_subset("my_fused_run.txt", topics=fused_run.topics())    
    p5, p10, p20, mAP, rprec = eval(fused_run, qrels)
    result_file.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(num_examples, p5, p10, p20, mAP, rprec))     

result_file.close()
    
    