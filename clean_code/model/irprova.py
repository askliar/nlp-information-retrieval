# TODO: REWRITE *ARGS PART TO GENERALIZE FOR EVERY POSSIBLE CASE

from itertools import product
import math
log2 = lambda x: math.log(x, 2)

relevance = ['N', 'R', 'HR']
single_ranking = list(product(relevance, repeat = 5))
rankings = list(product(single_ranking, repeat = 2))


def k_precision(*args):
    (rankings, k) = args
    p_precisions = list()
    e_precisions = list()
    for p, e in rankings:
        p_topk = [1 if p_i in ['R', 'HR'] else 0 for p_i in p[:k]]
        e_topk = [1 if e_i in ['R', 'HR'] else 0 for e_i in e[:k]]
        p_precisions.append(sum(p_topk)/k)
        e_precisions.append(sum(e_topk)/k)
    return p_precisions, e_precisions

# TODO: IMPLEMENT nDCG@k, ERR
#still random implementation
def rel(a):
    return 1 if a == 'R' else 2 if a == 'HR' else 0

def DCGk(*args):
    (rankings, k) = args
    p_DCGs = list()
    e_DCGs = list()

    # max_ranking = map(sum, map(rel, rankings))
    # print(list(max_ranking))
    for p, e in rankings:
        # adding 1 to r since in the original formula it goes from 1 to k
        p_topk = [(2**rel(p_i) - 1) / log2(2+r) for r, p_i in enumerate(p[:k])]
        e_topk = [(2**rel(e_i) - 1) / log2(2+r) for r, e_i in enumerate(e[:k])]
        p_DCGs.append(sum(p_topk))
        e_DCGs.append(sum(e_topk))
    return p_DCGs, e_DCGs

def nDCGk(*args):
    (rankings, k) = args
    p_DCGs, e_DCGs = DCGk(*args)
    #assuming relevances are ordered from less important to most important, and that we can fit k HR in the same query..?
    #basically, divide by the value of the query with result only hr, the best possible
    normalize_const = rel(relevance[-1])**k
    p_DCGs = [x / normalize_const for x in p_DCGs]
    e_DCGs = [x / normalize_const for x in e_DCGs]
    return p_DCGs, e_DCGs

def delta_measure(eval_method, *args):
    delta = list()
    for p_precision, e_precision in zip(*eval_method(*args)):
        if e_precision > p_precision:
            delta.append(e_precision - p_precision)
    return delta

a = delta_measure(nDCGk, rankings, 5)
print(list(sorted(a, reverse=True)))
print(len(a))