#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <iomanip>

#include "utils.hpp"
#include "body.hpp"

void shift_right(int rank, MsgBuf* sendBuf, MsgBuf* recvBuf) {
    MPI_Sendrecv((void*) sendBuf, MAX_BUF_SIZE, MPI_CHAR,
                NEXT(rank), NULL_TAG,
                (void*) recvBuf, MAX_BUF_SIZE, MPI_CHAR,
                PREV(rank), NULL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    INIT_BUF(recvBuf);
}


double compute_V(double rij, double rik, double rkj) {
    double three = rij * rik * rkj;
    double rij2 = rij * rij;
    double rik2 = rik * rik;
    double rkj2 = rkj * rkj;
    // printf("TEST 3: %f<>%f\n", power(three, 3), std::pow(three, 3));
    // printf("TEST 5: %f<>%f\n", power(three, 5), std::pow(three, 3));
    return E0 * (1.0 / std::pow(three, 3) + 3.0 * ((-rij2 + rik2 + rkj2) * (rij2 - rik2 + rkj2) * (rij2 + rik2 - rkj2)) 
        / (8.0 * std::pow(three, 5)));
}

// we need dummy _end markers for iteration over enum possible
enum direction {
    x,
    y,
    z,
    direction_end
};

enum particle_num {
    first,
    second,
    third,
    particle_num_end
};

enum sign {
    plus,
    minus,
    sign_end
};

#include <tuple>

// r12 = std::sqrt((p2.x - p1.x) * (p2.x - p1.x) +
//                 (p2.y - p1.y) * (p2.y - p1.y) + 
//                 (p2.z - p1.z) * (p2.z - p1.z)); 
// r01 = std::sqrt((p1.x - p0.x) * (p1.x - p0.x) +
//                 (p1.y - p0.y) * (p1.y - p0.y) + 
//                 (p1.z - p0.z) * (p1.z - p0.z));
// r02 = std::sqrt((p2.x - p0.x) * (p2.x - p0.x) +
//                 (p2.y - p0.y) * (p2.y - p0.y) + 
//                 (p2.z - p0.z) * (p2.z - p0.z));

// these are curded because no argument passing, but more readable
#define CURSED_DISTANCE01  (std::sqrt((p1x - p0x) * (p1x - p0x) + \
                            (p1y - p0y) * (p1y - p0y) + \
                            (p1z - p0z) * (p1z - p0z)))

#define CURSED_DISTANCE02 (std::sqrt((p2x - p0x) * (p2x - p0x) + \
                            (p2y - p0y) * (p2y - p0y) + \
                            (p2z - p0z) * (p2z - p0z)))

#define CURSED_DISTANCE12 (std::sqrt((p2x - p1x) * (p2x - p1x) + \
                        (p2y - p1y) * (p2y - p1y) + \
                        (p2z - p1z) * (p2z - p1z)))

inline double normalize(double coord) {
    if (std::abs(coord) < ZERO_EPS)
        coord = ZERO_EPS;
    return coord;
}

std::tuple<double, double, double> compute_distances(const Pos& p0, const Pos& p1, const Pos& p2, 
        direction d, particle_num which, sign _sign) {
    double r01, r02, r12;
    double p0x, p0y, p0z, p1x, p1y, p1z, p2x, p2y, p2z;
    double sign = (_sign == plus ? 1.0 : -1.0);

    p0x = p0.x;
    p0y = p0.y;
    p0z = p0.z;
    p1x = p1.x;
    p1y = p1.y;
    p1z = p1.z;
    p2x = p2.x;
    p2y = p2.y;
    p2z = p2.z;

    if (which == first) {
        r12 = CURSED_DISTANCE12;
        if (d == x) {
            double step = DERIV_EPS * normalize(p0x);
            p0x += sign * step;
        } else if (d == y) {
            double step = DERIV_EPS * normalize(p0y);
            p0y += sign * step;
        } else if (d == z) {
            double step = DERIV_EPS * normalize(p0z);
            p0z += sign * step;
        } else {
            assert(false);
        }
        r01 = CURSED_DISTANCE01;
        r02 = CURSED_DISTANCE02;
    } else if (which == second) {
        r02 = CURSED_DISTANCE02;
        if (d == x) {
            double step = DERIV_EPS * normalize(p1x);
            p1x += sign * step;
        } else if (d == y) {
            double step = DERIV_EPS * normalize(p1y);
            p1y += sign * step;
        } else if (d == z) {
            double step = DERIV_EPS * normalize(p1z);
            p1z += sign * step;
        } else {
            assert(false);
        }
        r01 = CURSED_DISTANCE01;
        r12 = CURSED_DISTANCE12;
    } else if (which == third) {
        r01 = CURSED_DISTANCE01;
        if (d == x) {
            double step = DERIV_EPS * normalize(p2x);
            p2x += sign * step;
        } else if (d == y) {
            double step = DERIV_EPS * normalize(p2y);
            p2y += sign * step;
        } else if (d == z) {
            double step = DERIV_EPS * normalize(p2z);
            p2z += sign * step;
        } else {
            assert(false);
        }
        r02 = CURSED_DISTANCE02;
        r12 = CURSED_DISTANCE12;
    }
    return std::make_tuple(r01, r02, r12);
}

#define UNPACK(__tup, __r01, __r02, __r12) do { \
    __r01 = std::get<0>(__tup); \
    __r02 = std::get<1>(__tup); \
    __r12 = std::get<2>(__tup); \
} while(0);

double compute_force(const Pos& p0, const Pos& p1, const Pos& p2, direction dir) {
    double res[3], restmp, ret;
    double r01, r02, r12;
    volatile double delta;
    double h, coord;

    // printf("P1 : (%f, %f, %f)\n", p1.x, p1.y, p1.z);

    for (int numInt = first; numInt != particle_num_end; numInt++) {
        for (int signInt = plus; signInt != sign_end; signInt++) {
            sign _sign = static_cast<sign>(signInt);
            particle_num num = static_cast<particle_num>(numInt);

            auto tup = compute_distances(p0, p1, p2, dir, num, _sign);
            UNPACK(tup, r01, r02, r12);
            printf("computing V for %9.9f, %9.9f, %9.9f\n", r01, r02, r12);
            restmp = compute_V(r01, r02, r12);
            res[numInt] += (_sign == plus ? restmp : -restmp);
            printf("res[%d] = +- %9.14f\n", numInt, restmp);
        }
        Pos tmp;
        if (numInt == 0) {
            tmp = p0;
        } else if (numInt == 1) {
            tmp = p1;
        } else if (numInt == 2) {
            tmp = p2;
        }
        if (dir == x) { 
            coord = tmp.x;
        } else if (dir == y) {
            coord = tmp.y;
        } else if (dir == z) {
            coord = tmp.z;
        }
        coord = normalize(coord);
        h = DERIV_EPS * coord;
        delta = (coord + h) - (coord - h);
        res[numInt] /= delta;
        // printf("res[%d] /= %f\n", numInt, delta);
    }

    // ret = - (res[0] + res[1] + res[2]) / MASS;
    ret = res[0] + res[1] + res[2];
    printf("@@ RET: (d=%d): %f\n", dir, ret);

    return ret;
}

void compute_interactions(MsgBuf* b0, MsgBuf* b1, MsgBuf* b2) {
    for (int i0 = 0; i0 < b0->particlesNum; i0++) {
        for (int i1 = 0; i1 < b1->particlesNum; i1++) {
            for (int i2 = 0; i2 < b2->particlesNum; i2++) {
                if (b0->owner == b1->owner && i0 == i1)
                    continue;
                if (b0->owner == b2->owner && i0 == i2)
                    continue;
                if (b1->owner == b2->owner && i1 == i2)
                    continue;
                
                Pos p0, p1, p2;
                p0 = b0->elems[i0].pos;
                p1 = b1->elems[i1].pos;
                p2 = b2->elems[i2].pos;

                double fx = compute_force(p0, p1, p2, x);
                double fy = compute_force(p0, p1, p2, y);
                double fz = compute_force(p0, p1, p2, z);

                printf("FORCES: (%9.9f, %9.9f, %9.9f)\n", fx, fy, fz);
                b0->elems[i0].force.fx += fx;
                b0->elems[i0].force.fy += fy;
                b0->elems[i0].force.fz += fz;

                b1->elems[i1].force.fx += fx;
                b1->elems[i1].force.fy += fy;
                b1->elems[i1].force.fz += fz;

                b2->elems[i2].force.fx += fx;
                b2->elems[i2].force.fy += fy;
                b2->elems[i2].force.fz += fz;
            }
        }
    }
}

void body_algo(int rank, MsgBuf* b1) {
    MsgBuf* tmpBuf;
    MsgBuf* buf[3];
    buf[1] = b1;
    buf[0] = (MsgBuf*) malloc(MAX_BUF_SIZE);
    buf[2] = (MsgBuf*) malloc(MAX_BUF_SIZE);
    
    tmpBuf = (MsgBuf*) malloc(MAX_BUF_SIZE);

    memcpy((void*) buf[0], b1, BUF_SIZE(b1));
    INIT_BUF(buf[0]);
    memcpy((void*) buf[2], b1, BUF_SIZE(b1));
    INIT_BUF(buf[2]);
    
    // std::cerr << "malloc: " << BUF_SIZE(b1) << " bytes allocated\n";

    int i = 0;
    printf("triple: (%d, %d, %d)\n", buf[i]->owner, buf[(i + 1) % 3]->owner, buf[(i + 2) % 3]->owner);
    
    i = 2;

    for (int s = NUM_PROC; s >= 0; s -= 3) {
        for (int step = 0; step < s; step++) {
            if (step != 0 || s != NUM_PROC) {
                shift_right(rank, buf[i], tmpBuf);
                memcpy(buf[i], tmpBuf, BUF_SIZE(tmpBuf));
                printf("triple: (%d, %d, %d)\n", buf[0]->owner, buf[1]->owner, buf[2]->owner);
                compute_interactions(buf[0], buf[1], buf[2]);
            }
        }
        i = (i + 1) % 3;
    }
    if (NUM_PROC % 3 == 0) {
        shift_right(rank, buf[i], tmpBuf);
        memcpy(buf[i], tmpBuf, BUF_SIZE(tmpBuf));
        i = i == 0 ? 2 : i - 1; // prv(i, 3)
        shift_right(rank, buf[i], tmpBuf);
        if (rank / (NUM_PROC / 3) == 0) {
            printf("triple: (%d, %d, %d)\n", buf[0]->owner, buf[1]->owner, buf[2]->owner);
            compute_interactions(buf[0], buf[1], buf[2]);
        }
    }


    // all interactions have been computed, now send results to root node
    
    char *gatherBuf0, *gatherBuf1, *gatherBuf2;


    // TODO TODO MOCNE - dokladnie to sprawdzic na pewno są bugi

    // TODO zla liczba 
    size_t dataSize = MAX_BUF_SIZE * NUM_PROC;
    
    if (rank == ROOT_NODE){
        gatherBuf0 = (char*) malloc(dataSize);

        gatherBuf1 = (char*) malloc(dataSize);
        gatherBuf2 = (char*) malloc(dataSize);
    }

    printf("rank %d: wysylam (%d, %d, %d)\n", rank, buf[0]->owner, buf[1]->owner, buf[2]->owner);
    
    MPI_Gather(buf[0], MAX_BUF_SIZE, MPI_CHAR, gatherBuf0, MAX_BUF_SIZE, MPI_CHAR, ROOT_NODE, MPI_COMM_WORLD);
    MPI_Gather(buf[1], MAX_BUF_SIZE, MPI_CHAR, gatherBuf1, MAX_BUF_SIZE, MPI_CHAR, ROOT_NODE, MPI_COMM_WORLD);
    MPI_Gather(buf[2], MAX_BUF_SIZE, MPI_CHAR, gatherBuf2, MAX_BUF_SIZE, MPI_CHAR, ROOT_NODE, MPI_COMM_WORLD);

    if (rank == ROOT_NODE) {
        printf("UDALO SIE, wypisuje zerowy:\n");

        MPI_Request rs[3 * NUM_PROC];
        int i = 0;

        for (int off = 0; off < NUM_PROC * MAX_BUF_SIZE; off += MAX_BUF_SIZE) {
            MsgBuf *tmp = (MsgBuf*) (gatherBuf0 + off);
            INIT_BUF(tmp);
            // BUF_ISEND(tmp, BUF_SIZE(tmp), tmp->owner, &rs[i]);
            printf("recv owner: %d, num: %d\n", tmp->owner, tmp->particlesNum);
            i++;
        }

        for (int off = 0; off < NUM_PROC * MAX_BUF_SIZE; off += MAX_BUF_SIZE) {
            MsgBuf *tmp = (MsgBuf*) (gatherBuf1 + off);
            INIT_BUF(tmp);
            printf("recv owner: %d, num: %d\n", tmp->owner, tmp->particlesNum);
            i++;
        }

        for (int off = 0; off < NUM_PROC * MAX_BUF_SIZE; off += MAX_BUF_SIZE) {
            MsgBuf *tmp = (MsgBuf*) (gatherBuf2 + off);
            INIT_BUF(tmp);
            printf("recv owner: %d, num: %d\n", tmp->owner, tmp->particlesNum);
            i++;
        }
    }
}
