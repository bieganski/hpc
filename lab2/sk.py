#!/usr/bin/env python3

import termplotlib as tpl
import numpy

import subprocess

RED_IDX = 0
CRIT_IDX = 1


def algo(steps_range=[10, 100, 1000]):
    speedups1 = []
    speedups2 = []
    for steps in steps_range:
        process = subprocess.Popen(['./pi', str(steps)], stdout=subprocess.PIPE)
        stdout = process.communicate()[0]

        out = list (map(lambda x : float(x), stdout.split()))
        speedups1.append(out[RED_IDX])
        speedups2.append(out[CRIT_IDX])
    return speedups1, speedups2


def draw(ranges, speedups1, speedups2):
    fig = tpl.figure()
    fig.plot(ranges, speedups1, label="REDUCTION", width=50, height=15)
    fig.plot(ranges, speedups2, label="CRITICAL", width=50, height=15)
    fig.show()

range = [10, 100, 1000, 5000]
s1, s2 = algo(range)
draw(range, s1, s2)
