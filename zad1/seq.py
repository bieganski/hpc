#!/usr/bin/env python3

import argparse

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


def append_value(dic, key, to_add):
    if dic.get(key) is None:
        dic[key] = set()
    dic[key].add(to_add)


with open(fp) as f:
    first = True
    for line in f.readlines():
        if line[0] == '%':
            continue
        a = [int(x) for x in line.split()]
        if first:
            assert(len(a) == 3)
            first = False
            v1max = a[0]
            v2max = a[1]
            N = a[2]
        else:
            append_value(G, a[0], a[1])
            append_value(G, a[1], a[0])

print(G)
