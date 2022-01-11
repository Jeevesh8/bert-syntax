import sys
from collections import *

files=[#("base","results/gulordava_results_base.txt"),("large","results/gulordava_results_large.txt")]
       (f"multi_bert_{i}", f"results/multiberts-seed_{i}_gul.out") for i in range(25)]

by_model={}
conditions=set()
nskipped=0
for title,fname in files:
    lines = open(fname)
    results=defaultdict(Counter)
    by_model[title]=results
    skipped = set()
    for line in lines:
        if line.startswith("Better speed"): continue
        if line.startswith("skipping"):
            skipped.add(line.split()[1])
            #next(lines) # no need to skip, skipped in testing
            nskipped += 1
            continue
        assert (line.strip().split()[0] in ['True','False']),line
        res,c1,_ = line.split(None, 2)
        conditions.add(c1)
        conditions.add('all')
        results[c1][res]+=1
        print("adding",res,"to",c1)
        results['all'][res]+=1

print("skipped:",nskipped,len(skipped),skipped)

print("condition & seed & score & count \\\\")
for cond in conditions:
    for title, results in by_model.items():
        r = results[cond]
        if sum(r.values())==0: continue
        s = "%.2f" % (r['True']/(r['True']+r['False']))
        print(" & ".join(map(str,[cond, title, s, sum(r.values())])),"\\\\")
    


