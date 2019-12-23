from trectools import TrecQrel, TrecRun, TrecEval

# A typical evaluation workflow
r1 = TrecRun("lm_result_trigger")
r1.topics()[:5] # Shows the first 5 topics: 601, 602, 603, 604, 605

qrels = TrecQrel("small_data/ace/english/queries/qrels.english_events.txt")

te = TrecEval(r1, qrels)
rbp, residuals = te.get_rbp()           # RBP: 0.474, Residuals: 0.001
p10 = te.get_precision(depth=10)     # P@100: 0.186
p20 = te.get_precision(depth=20)
p30 = te.get_precision(depth=30)
map = te.get_map()
print(round(p10, 3), round(p20, 3), round(p30, 3) , round(map, 2))
# Check if documents retrieved by the system were judged:
cover10 = r1.get_mean_coverage(qrels, topX=10)   # 9.99
cover1000 = r1.get_mean_coverage(qrels, topX=1000) # 481.390 
# On average for system 'input.aplrob03a' participating in robust03, 480 documents out of 1000 were judged.
print("Average number of documents judged among top 10: %.2f, among top 1000: %.2f" % (cover10, cover1000))

# # Loads another run
# r2 = TrecRun("./robust03/runs/input.UIUC03Rd1.gz")

# # Check how many documents, on average, in the top 10 of r1 were retrieved in the top 10 of r2
# r1.check_run_coverage(r2, topX=10) # 3.64

# # Evaluates r1 and r2 using all implemented evaluation metrics
# result_r1 = r1.evaluate_run(qrels, per_query=True) 
# result_r2 = r2.evaluate_run(qrels, per_query=True)

# # Inspect for statistically significant differences between the two runs for  P_10 using two-tailed Student t-test
# pvalue = result_r1.compare_with(result_r2, metric="P_10") # pvalue: 0.0167 