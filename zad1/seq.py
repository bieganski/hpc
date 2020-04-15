#!/usr/bin/env python3

import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-f', type=str, required=True, nargs=1, metavar="filename",
                    help='graph description filename')
parser.add_argument('-g', type=str, required=True, nargs=1, metavar="min-gain",
                    help='minimal gain to move node nbetween communities')
parser.add_argument('-v', required=False, action="store_true",
                    help='should I print communities description?')


args = parser.parse_args()

fp = args.f[0]
v1max, v2max, N = 0, 0, 0
G = {}

W = {}

k = None

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


C = N  # nuber of communities

CV = {}  # maps community to list of nodes
VC = {}  # maps node to it's community


def init_communities():
    for node in range(1, N+1):
        append_value(CV, node, node)
        VC[node] = node

# computes e_{i -> C[i]}
def compute_e(i, c):
    i_ns = G[i]
    real_ns = [x for x in CV[c] if x in i_ns]
    # w razie problemów przyjrzeć się temu `if i != n` niżej
    return sum([W[frozenset([i, n])] for n in real_ns if i != n])

def compute_ac(c):
    return sum( [ k[i] for i in CV[c] ] )

def compute_mod():
    assert(k[0] == 0)
    m = k.sum(0) / 2
    s1 = sum( [compute_e(node, VC[node]) for node in range(1,N+1)] )
    s2 = sum( [compute_ac(c)   for c in range(1, C+1)] )
    return s1 / (2 * m) - s2 / (4 * m ** 2)

m = 0
def init():
    global m
    init_communities()
    m = k.sum(0) / 2

def compute_gain(node, new_community):
    s1 = compute_e(node, new_community) - compute_e(node, VC[node])
    s2 = k[node] * ( compute_ac(VC[node]) - k[node] - compute_ac(new_community) )
    return s1 / m + s2 / (2 * m ** 2)



if __name__ == "__main__":
    init()
    print("m", m)
    print(compute_mod())
    for v in range(1, N + 1):
        for c in range(1, C + 1):
            print(v, " to ", c, " gain: ")
            print(compute_gain(v, c))
