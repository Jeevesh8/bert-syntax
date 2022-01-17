import sys
from collections import *

files= [#("base","results/marvin_results_base.txt"),("large","results/marvin_results_large.txt")]
        (f"multi_bert_{i}", f"results/multiberts-seed_{i}_marvin_{sys.argv[1]}.out") for i in range(25)]

by_model={}
conditions=set()
for title,fname in files:
    lines = open(fname)
    results=defaultdict(Counter)
    by_model[title]=results
    skipped = set()
    for line in lines:
        if line.startswith("Better speed"): continue
        if line.startswith("Found"): 
            print(line)
            continue
        if line.startswith("skipping"):
            skipped.add(line.split()[1])
            next(lines)
            continue
        res,c1,c2,w1,w2,s = line.split(None, 5)
        c1 = c1.replace("inanim","anim")
        conditions.add(c1)
        results[c1][res]+=1

print("skipped:",skipped)

print("condition & seed & score & count \\\\")
for cond in conditions:
    for title, results in by_model.items():
        r = results[cond]
        s = "%.2f" % (r['True']/(r['True']+r['False']))
        print(" & ".join(map(str,[cond, title, s, sum(r.values())])),"\\\\")

    


