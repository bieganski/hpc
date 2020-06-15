#!/usr/bin/env python3


import time
import subprocess, os
from os.path import join



command = "mpirun -n %d ./body3 %s %s %d %f"

OUT_TMP = "tmp_out.remove"
TEST_DIR = "tests"

TEST_IN = "test_100.txt"

# n, step, delta
TESTS_WEAK = [
    (4,  8,   0.5),
    (8,  16,  0.5),
    (16, 32,  0.5),
    (32, 64,  0.5),
    (64, 128, 0.5)
]

TESTS_STRONG = [
    (4,  8, 0.5),
    (8,  8, 0.5),
    (16, 8, 0.5),
    (32, 8, 0.5),
    (64, 8, 0.5)
]

TIME_STRONG_SEQ = 191 # 3:11

TIMES_WEAK_SEQ = [
    1
]

TIMES_WEAK = []
TIMES_STRONG = []

cmd = lambda n, s, d : command % (n, join(TEST_DIR, TEST_IN), OUT_TMP, s, d)


def run(n,s,d):
    start = time.time()
    os.system(cmd(n,s,d))
    return (time.time() - start)


def all():
    for n,s,d in TESTS_STRONG:
        print(cmd(n,s,d))
        time = run(n,s,d)
        print(time)

if __name__ == "__main__":
    all()
