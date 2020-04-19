#!/usr/bin/env python3

import argparse
import numpy as np
from datetime import datetime

parser = argparse.ArgumentParser()

parser.add_argument('-f', type=str, required=True, nargs=1, metavar="filename",
                    help='graph description filename')
parser.add_argument('-g', type=float, required=True, nargs=1, metavar="min-gain",
                    help='minimal gain to move node nbetween communities')
parser.add_argument('-v', required=False, action="store_true",
                    help='should I print communities description?')

args = parser.parse_args()

DEBUG = True

def log(*args):
    print(args) if DEBUG else None

fp = args.f[0]
MIN_GAIN = args.g[0]

N = 0
G = {}
W = {}
k = None
m = 0

CV = {}  # maps community to list of nodes
VC = {}  # maps node to it's community

result_assignment = {}

def append_value(dic, key, to_add):
    if dic.get(key) is None:
        dic[key] = set()
    dic[key].add(to_add)


with open(fp) as f:
    first = True
    for line in f.readlines():
        if line[0] == '%':
            continue
        a = [float(x) for x in line.split()]
        if (a == []):
            continue  # --assert last one
        if first:
            assert(len(a) == 3)
            first = False
            v1max = a[0]
            v2max = a[1]
            assert(v1max == v2max)
            assert(v1max % 1 == 0)
            N = int(v1max)
            assert(a[2] % 1 == 0)
            E = int(a[2])
            k = np.zeros(N + 1)
        else:
            a[0] = int(a[0])
            a[1] = int(a[1])
            assert(a[0] <= N)
            assert(a[1] <= N)
            append_value(G, a[0], a[1])
            append_value(G, a[1], a[0])
            assert(len(a) < 4)
            w = a[2] if len(a) > 2 else 1.0
            W[frozenset([a[0], a[1]])] = w
            k[a[0]] += w
            k[a[1]] += w


def init_communities():
    for node in G.keys():
        result_assignment[node] = {node}
        append_value(CV, node, node)
        VC[node] = node


def compute_e(i, c):
    i_ns = G[i]
    real_ns = [x for x in CV[c] if x in i_ns]
    return sum([W[frozenset([i, n])] for n in real_ns])


def compute_ac(c):
    return sum( [ k[i] for i in CV[c] ] )


def compute_mod():
    s1 = sum([compute_e(node, VC[node]) for node in VC.keys()])
    s2 = sum([compute_ac(c) for c in CV.keys()])
    return s1 / (2 * m) - s2 / (4 * m ** 2)


def init():
    global m
    init_communities()
    m = k.sum(0) / 2


def compute_gain(node, new_community):
    bias = W.get(frozenset([node, node]), 0)
    s1 = compute_e(node, new_community) - compute_e(node, VC[node]) - bias
    s2 = k[node] * (compute_ac(VC[node]) - k[node] - compute_ac(new_community))
    return s1 / m + s2 / (2 * m ** 2)


def move_to_community(node, new_community):
    CV[VC[node]].remove(node)
    CV[new_community].add(node)
    VC[node] = new_community


def algo():
    init()
    while True:
        log("modularity: ", compute_mod(), m)
        changed_sth = False
        for v in VC.keys():
            NEW, GAIN = None, None
            for c in set(map(VC.get, G[v])):
                if v in CV[c]:
                    continue
                if len(CV[c]) == 1 and len(CV[v]) == 1 and c > v:
                    continue  # single-node communities, don't move to upper index
                gain = compute_gain(v, c)
                log("gain: ", gain)
                if gain < MIN_GAIN:
                    continue
                if NEW is None:
                    NEW, GAIN = c, gain
                elif gain == GAIN:
                    NEW = min(NEW, c)
                elif gain > GAIN:
                    NEW, GAIN = c, gain
            if NEW is None:
                continue
            log("przenosze ", v, " do ", NEW)
            move_to_community(v, NEW)
            result_assignment[NEW] = result_assignment[NEW].union(result_assignment[v])
            del result_assignment[v]
            changed_sth = True
        if not changed_sth:
            return
        reinit()


def reinit():
    global G, W, k, CV, VC
    GG = {}
    WW = {}
    kk = np.zeros_like(k)
    for c, vs in CV.items():
        if len(vs) == 0:
            continue
        c_weight = 0
        GG[c] = set()
        for v in vs:
            v_inner_ns = G[v].intersection(vs)  # edges into it's community
            v_outer_ns = G[v] - v_inner_ns  # edges outside community

            inner_weight = sum([W[frozenset([v, x])] for x in v_inner_ns])
            c_weight += inner_weight

            log("outer dla ", v,  ": ", v_outer_ns)
            for out in v_outer_ns:
                # update graph description
                new_out = VC[out]
                append_value(GG, c, new_out)

                # update outer edges, only under some circumstance, to make it only once
                if c > new_out:
                    continue  # only lower community does instructions below
                key = frozenset([c, new_out])
                val = W[frozenset([v, out])]
                kk[new_out] += val
                kk[c] += val
                if WW.get(key) is None:
                    WW[key] = 0
                log("zwiekszam ", key, " o ", val)
                WW[key] += val

        if c_weight == 0:  # TODO ujemne wagi
            continue  # no inner edges
        append_value(GG, c, c)
        WW[frozenset([c, c])] = c_weight / 2
        kk[c] += c_weight / 2

    log(">>> reinit end: k: ", kk)
    log(">>> reinit end: W: ", WW)
    G = GG
    W = WW
    k = kk
    VC = {x: x for x in G}
    CV = {x: {x} for x in G}


if __name__ == "__main__":
    t0 = datetime.now()
    algo()
    t1 = datetime.now()
    exec_time = (t1.microsecond - t0.microsecond) / 1000
    print("EXEC_TIME:", exec_time, exec_time)
    if args.v:
        res = str(len(G))
        for k, v in result_assignment.items():
            res += "\n"
            res += str(k) + " "
            res += " ".join(map(str, v))
        print(res)
