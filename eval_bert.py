# coding=utf-8
from transformers import BertForMaskedLM, BertTokenizer
from multiprocessing import Process
import torch
import os, sys
import csv, random

from functools import partial
print = partial(print, flush=True)

def get_probs_for_words(bert,sent,w1,w2):
    pre,target,post=sent.split('***')
    if 'mask' in target.lower():
        target=['[MASK]']
    else:
        target=tokenizer.tokenize(target)
    tokens=['[CLS]']+tokenizer.tokenize(pre)
    target_idx=len(tokens)
    #print(target_idx)
    tokens+=target+tokenizer.tokenize(post)+['[SEP]']
    input_ids=tokenizer.convert_tokens_to_ids(tokens)
    try:
        word_ids=tokenizer.convert_tokens_to_ids([w1,w2])
    except KeyError:
        print("skipping",w1,w2,"bad wins")
        return None
    tens=torch.LongTensor(input_ids).unsqueeze(0)
    res=bert(tens).logits[0,target_idx]
    #res=torch.nn.functional.softmax(res,-1)
    scores = res[word_ids]
    return [float(x) for x in scores]

from collections import Counter
def load_marvin():
    cc = Counter()
    # note: I edited the LM_Syneval/src/make_templates.py script, and run "python LM_Syneval/src/make_templates.py LM_Syneval/data/templates/ > marvin_linzen_dataset.tsv"
    out = []
    marvin_data = [line for line in open("marvin_linzen_dataset.tsv")]
    random.Random(42).shuffle(marvin_data)
    for line in marvin_data:
        case = line.strip().split("\t")
        cc[case[1]]+=1
        g,ug = case[-2],case[-1]
        g = g.split()
        ug = ug.split()
        assert(len(g)==len(ug)),(g,ug)
        diffs = [i for i,pair in enumerate(zip(g,ug)) if pair[0]!=pair[1]]
        if (len(diffs)!=1):
            #print(diffs)
            #print(g,ug)
            continue    
        assert(len(diffs)==1),diffs
        gv=g[diffs[0]]   # good
        ugv=ug[diffs[0]] # bad
        g[diffs[0]]="***mask***"
        g.append(".")
        out.append((case[0],case[1]," ".join(g),gv,ugv))
    return out

def eval_marvin(bert):
    o = load_marvin()
    print(len(o),file=sys.stderr)
    from collections import defaultdict
    import time
    rc = defaultdict(Counter)
    tc = Counter()
    start = time.time()
    for i,(case,tp,s,g,b) in enumerate(o):
        if i>10000:
            break
        ps = get_probs_for_words(bert,s,g,b)
        if ps is None: ps = [0,1]
        gp = ps[0]
        bp = ps[1]
        print(gp>bp,case,tp,g,b,s)
        if i % 100==0:
            print(i,time.time()-start,file=sys.stderr)
            start=time.time()
            sys.stdout.flush()

def eval_lgd(bert):
    lgd_data = [line for line in open("lgd_dataset.tsv",encoding="utf8")]
    random.Random(42).shuffle(lgd_data)
    for i,line in enumerate(lgd_data):
        if i>10000:
            break
        na,_,masked,good,bad = line.strip().split("\t")
        ps = get_probs_for_words(bert,masked,good,bad)
        if ps is None: continue
        gp = ps[0]
        bp = ps[1]
        print(str(gp>bp),na,good,gp,bad,bp,masked.encode("utf8"),sep=u"\t")
        if i%100 == 0:
            print(i,file=sys.stderr)
            sys.stdout.flush()


def read_gulordava():
    rows = csv.DictReader(open("generated.tab",encoding="utf8"),delimiter="\t")
    data=[]
    for row in rows:
        row2=next(rows)
        assert(row['sent']==row2['sent'])
        assert(row['class']=='correct')
        assert(row2['class']=='wrong')
        sent = row['sent'].lower().split()[:-1] # dump the <eos> token.
        good_form = row['form']
        bad_form  = row2['form']
        sent[int(row['len_prefix'])]="***mask***"
        sent = " ".join(sent)
        data.append((sent,row['n_attr'],good_form,bad_form))
    random.Random(42).shuffle(data)
    return data

def eval_gulordava(bert):
    for i,(masked,natt,good,bad) in enumerate(read_gulordava()):
        if i>10000:
            break
        if good in ["is","are"]:
            print("skipping is/are")
            continue
        ps = get_probs_for_words(bert,masked,good,bad)
        if ps is None: continue
        gp = ps[0]
        bp = ps[1]
        print(str(gp>bp),natt,good,gp,bad,bp,masked.encode("utf8"),sep=u"\t")
        if i%100 == 0:
            print(i,file=sys.stderr)
            sys.stdout.flush()

def main(model_name):
    print("using model:", model_name)
    if "marvin" in sys.argv:
        dataset_name = "marvin"
    elif "gul" in sys.argv:
        dataset_name = "gul"
    else:
        dataset_name = "lgd"
    
    with open(os.path.join("results",
                           model_name.split("/")[-1]+"_"+dataset_name+".out"), "w") as f:
        sys.stdout = f
        bert = BertForMaskedLM.from_pretrained(model_name)
        bert.eval()
        if dataset_name=="marvin":
            eval_marvin(bert)
        elif dataset_name=="gul":
            eval_gulordava(bert)
        else:
            eval_lgd(bert)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
if __name__=="__main__":
    for i in range(0,25):
        model_name = f"google/multiberts-seed_{i}"
        p = Process(target=main, args=(model_name,))
        p.start()
