#include <string.h>
#include <stdlib.h>
#include <iostream>
#include <cmath>
#include <iomanip>

#include "utils.hpp"
#include "body.hpp"


extern MPI_Comm ACTIVE_NODES_WORLD;

extern double DELTA_TIME;

void shift_right(int rank, MsgBuf* sendBuf, MsgBuf* tmpRecvBuf) {
    MPI_Sendrecv((void*) sendBuf, MAX_BUF_SIZE, MPI_CHAR,
                NEXT(rank), NULL_TAG,
                (void*) tmpRecvBuf, MAX_BUF_SIZE, MPI_CHAR,
                PREV(rank), NULL_TAG, ACTIVE_NODES_WORLD, MPI_STATUS_IGNORE);
    memcpy(sendBuf, tmpRecvBuf, BUF_SIZE(tmpRecvBuf));
    memset(tmpRecvBuf, '\0', MAX_BUF_SIZE);
    INIT_BUF(sendBuf);
}

inline double normalize(double coord);

double compute_V(double rij, double rik, double rkj) {
    double three = rij * rik * rkj;
    double rij2 = rij * rij;
    double rik2 = rik * rik;
    double rkj2 = rkj * rkj;
    double res;

    // printf("TEST RIJK2 = %3.12f, %3.12f, %3.12f\n", rij2, rik2, rkj2);
    // printf("TEST 3: %3.12f\n", std::pow(three, 3));
    // printf("TEST 5: %3.12f\n", std::pow(three, 5));
    
    res = 1.0 / std::pow(three, 3);
    res += 3.0 * ((-rij2 + rik2 + rkj2) * (rij2 - rik2 + rkj2) * (rij2 + rik2 - rkj2));
    res /= 8.0 * std::pow(three, 5); 
    res *= E0;

    return res;
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
    if (std::abs(coord) < ZERO_EPS) {
        printf("TODO - SHITTY NORMALIZATION DETECTED \n");
        coord = ZERO_EPS;
    }
        
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
            double step = DERIV_EPS * p0x;
            p0x += sign * step;
        } else if (d == y) {
            double step = DERIV_EPS * p0y;
            p0y += sign * step;
        } else if (d == z) {
            double step = DERIV_EPS * p0z;
            p0z += sign * step;
        } else {
            assert(false);
        }
        r01 = CURSED_DISTANCE01;
        r02 = CURSED_DISTANCE02;
    } else if (which == second) {
        r02 = CURSED_DISTANCE02;
        if (d == x) {
            double step = DERIV_EPS * p1x;
            p1x += sign * step;
        } else if (d == y) {
            double step = DERIV_EPS * p1y;
            p1y += sign * step;
        } else if (d == z) {
            double step = DERIV_EPS * p1z;
            p1z += sign * step;
        } else {
            assert(false);
        }
        r01 = CURSED_DISTANCE01;
        r12 = CURSED_DISTANCE12;
    } else if (which == third) {
        r01 = CURSED_DISTANCE01;
        if (d == x) {
            double step = DERIV_EPS * p2x;
            p2x += sign * step;
        } else if (d == y) {
            double step = DERIV_EPS * p2y;
            p2y += sign * step;
        } else if (d == z) {
            double step = DERIV_EPS * p2z;
            p2z += sign * step;
        } else {
            assert(false);
        }
        r02 = CURSED_DISTANCE02;
        r12 = CURSED_DISTANCE12;
    }
    // if (d == x) {
    //     printf("\ndir X: \n(%3.12f, %3.12f, %3.12f) -> (%3.12f, %3.12f, %3.12f)\n(%3.12f, %3.12f, %3.12f) -> (%3.12f, %3.12f, %3.12f)\n(%3.12f, %3.12f, %3.12f) -> (%3.12f, %3.12f, %3.12f)\n\n", 
    //         p0.x, p0.y, p0.z, p0x, p0y, p0z,
    //         p1.x, p1.y, p1.z, p1x, p1y, p1z,
    //         p2.x, p2.y, p2.z, p2x, p2y, p2z);
    // } else if (d == y) {
    //     printf("\ndir Y: \n(%3.12f, %3.12f, %3.12f) -> (%3.12f, %3.12f, %3.12f)\n(%3.12f, %3.12f, %3.12f) -> (%3.12f, %3.12f, %3.12f)\n(%3.12f, %3.12f, %3.12f) -> (%3.12f, %3.12f, %3.12f)\n\n", 
    //         p0.x, p0.y, p0.z, p0x, p0y, p0z,
    //         p1.x, p1.y, p1.z, p1x, p1y, p1z,
    //         p2.x, p2.y, p2.z, p2x, p2y, p2z);
    // } else {
    //     printf("\ndir Z: \n(%3.12f, %3.12f, %3.12f) -> (%3.12f, %3.12f, %3.12f)\n(%3.12f, %3.12f, %3.12f) -> (%3.12f, %3.12f, %3.12f)\n(%3.12f, %3.12f, %3.12f) -> (%3.12f, %3.12f, %3.12f)\n\n", 
    //         p0.x, p0.y, p0.z, p0x, p0y, p0z,
    //         p1.x, p1.y, p1.z, p1x, p1y, p1z,
    //         p2.x, p2.y, p2.z, p2x, p2y, p2z);
    // }
    return std::make_tuple(r01, r02, r12);
}

#define UNPACK(__tup, __r01, __r02, __r12) do { \
    __r01 = std::get<0>(__tup); \
    __r02 = std::get<1>(__tup); \
    __r12 = std::get<2>(__tup); \
} while(0);

std::tuple<double, double, double> compute_force(const Pos& p0, const Pos& p1, const Pos& p2, direction dir, int rank) {
    double res[3] = {0.0, 0.0, 0.0}, restmp, ret;
    double r01, r02, r12;
    volatile double delta;
    double h, coord;

    printf("@@@@@@@@@@@@@@@@@P0 : (%f, %f, %f)\n", p0.x, p0.y, p0.z);
    printf("@@@@@@@@@@@@@@@@@P1 : (%f, %f, %f)\n", p1.x, p1.y, p1.z);
    printf("@@@@@@@@@@@@@@@@@P2 : (%f, %f, %f)\n", p2.x, p2.y, p2.z);

    // for each particle_num {0, 1, 2}, compute 'plus' V and 'minus' V (for derivative purpose) 
    for (int numInt = first; numInt != particle_num_end; numInt++) {
        for (int signInt = plus; signInt != sign_end; signInt++) {
            sign _sign = static_cast<sign>(signInt);
            particle_num num = static_cast<particle_num>(numInt);

            auto tup = compute_distances(p0, p1, p2, dir, num, _sign);
            UNPACK(tup, r01, r02, r12);
            printf("computing V[numInt=%d] for %9.9f, %9.9f, %9.9f\n", numInt, r01, r02, r12);
            restmp = compute_V(r01, r02, r12);
            res[numInt] += (_sign == plus ? restmp : -restmp);
            printf("V[%d] (for rank %d) = %c %9.14f\n", numInt, rank, _sign == plus ? '+' : '-', restmp);
            if (_sign == minus) {
                ; // printf("@@: UWAGA (rank: %d, dir: %d) res diff: %9.12f\n", rank, dir, res[numInt]);
            }
        }
        // we have computed f(x + h) - f(x - h), in next lines we divide by delta x
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
        printf("@@******* PODZIELONE[%d] (rank %d): /=%f == %3.12f\n", numInt, rank, delta, res[numInt]);
    }

    // ret = - (res[0] + res[1] + res[2]) / MASS;
    // ret = res[0] + res[1] + res[2];
    // printf("ret=res[0] + res[1] + res[2] %2.12f = %2.12f + %2.12f + %2.12f\n", ret, res[0], res[1], res[2]);
    // printf("@@ RET: (d=%d): %f\n", dir, ret);

    return std::make_tuple(res[0], res[1], res[2]);
}

void compute_interactions(MsgBuf* b0, MsgBuf* b1, MsgBuf* b2, int rank) {
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

                assert(p0.x != p1.x && p0.x != p2.x && p1.x != p2.x);
                assert(p0.y != p1.y && p0.y != p2.y && p1.y != p2.y);
                assert(p0.z != p1.z && p0.z != p2.z && p1.z != p2.z);

                auto fx = compute_force(p0, p1, p2, x, rank);
                auto fy = compute_force(p0, p1, p2, y, rank);
                auto fz = compute_force(p0, p1, p2, z, rank);

                // printf("FORCES: (%9.9f, %9.9f, %9.9f)\n", fx, fy, fz); BAD FORMAT (tuple)
                FX(b0, i0) += std::get<0>(fx);
                FY(b0, i0) += std::get<0>(fy);
                FZ(b0, i0) += std::get<0>(fz);

                FX(b1, i1) += std::get<1>(fx);
                FY(b1, i1) += std::get<1>(fy);
                FZ(b1, i1) += std::get<1>(fz);

                FX(b2, i2) += std::get<2>(fx);
                FY(b2, i2) += std::get<0>(fy);
                FZ(b2, i2) += std::get<0>(fz);
            }
        }
    }
}


void compute_acc_maybe_vel(MsgBuf* b1, bool compute_vel) {
    double ax, ay, az;
    for (int i = 0; i < b1->particlesNum; i++) {
        ax = -FX(b1, i) / MASS;
        ay = -FY(b1, i) / MASS;
        az = -FZ(b1, i) / MASS;

        if (compute_vel) {
            double vx, vy, vz;

            ParticleDescr& d = b1->elems[i];

            vx = d.vel.vx + 0.5 * DELTA_TIME * (ACCX(b1, i) + ax);
            vy = d.vel.vy + 0.5 * DELTA_TIME * (ACCY(b1, i) + ay);
            vz = d.vel.vz + 0.5 * DELTA_TIME * (ACCZ(b1, i) + az);

            d.vel.vx = vx;
            d.vel.vy = vy;
            d.vel.vz = vz;

            printf("new VEL: (%3.9f, %3.9f, %3.9f)\n", d.vel.vx, d.vel.vy, d.vel.vz);
        }

        ACCX(b1, i) = ax;
        ACCY(b1, i) = ay;
        ACCZ(b1, i) = az;
        printf("new ACC: (%3.9f, %3.9f, %3.9f)\n", ACCX(b1, i), ACCY(b1, i), ACCZ(b1, i));
    }
}

void update_positions(MsgBuf* b1) {
    double x, y, z;
    for (int i = 0; i < b1->particlesNum; i++) {
        // printf("-- LOL  %d ### force %d: (%9.16f, %9.16f, %9.16f)\n", rank, i, FX(b1, i), FY(b1, i), FZ(b1, i));
        ParticleDescr& d = b1->elems[i];
        // assert(d.pos.x != 0.0 &&  d.pos.y != 0.0 && d.pos.z != 0.0); // TODO wywalić
        // printf("LOL OLD: (%2.14f, %2.14f, %2.14f)\n", d.pos.x, d.pos.y, d.pos.z);
        x = d.pos.x + d.vel.vx * DELTA_TIME + 0.5 * d.acc.ax * DELTA_TIME * DELTA_TIME;
        y = d.pos.y + d.vel.vy * DELTA_TIME + 0.5 * d.acc.ay * DELTA_TIME * DELTA_TIME;
        z = d.pos.z + d.vel.vz * DELTA_TIME + 0.5 * d.acc.az * DELTA_TIME * DELTA_TIME;
        // printf("LOL: (%2.14f, %2.14f, %2.14f)\n", x, y, z);

        // update positions
        d.pos.x = x;
        d.pos.y = y;
        d.pos.z = z;
    }
}


// It uses b1
void move_particles(MsgBuf* b0, MsgBuf* b1, MsgBuf* b2, int rank, bool first_iter) {

    for (int i = 0; i < b1->particlesNum; i++) {
        FX(b1, i) += FX(b0, i);
        FY(b1, i) += FY(b0, i);
        FZ(b1, i) += FZ(b0, i);

        FX(b1, i) += FX(b2, i);
        FY(b1, i) += FY(b2, i);
        FZ(b1, i) += FZ(b2, i);
    }

    if (first_iter) {
        compute_acc_maybe_vel(b1, false); // start velocity was given
        return;
    }

    update_positions(b1);
    compute_acc_maybe_vel(b1, true);
}


// if `first_iter`, then only update a(t + dt) only,
// else update:
// 1. x(t + dt)
// 2. a(t + dt)
// 3. v(t + dt)
// Each iteration updates `b1` param, and frees memory it allocated.
void body_algo(int rank, MsgBuf* b1, bool first_iter) {
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
            }
            printf("triple: (%d, %d, %d)\n", buf[0]->owner, buf[1]->owner, buf[2]->owner);
            compute_interactions(buf[0], buf[1], buf[2], rank);
        }
        i = (i + 1) % 3;
    }
    if (NUM_PROC % 3 == 0) {
        i = i == 0 ? 2 : i - 1; // prv(i, 3)
        shift_right(rank, buf[i], tmpBuf);
        if (rank / (NUM_PROC / 3) == 0) {
            printf("triple: (%d, %d, %d)\n", buf[0]->owner, buf[1]->owner, buf[2]->owner);
            compute_interactions(buf[0], buf[1], buf[2], rank);
        }
    }


    // all interactions have been computed, now send results to root node

    size_t dataSize = 3 * MAX_BUF_SIZE * NUM_PROC;    
    char *gatherBuf = rank == ROOT_NODE ? (char*) malloc(dataSize) : NULL;

    printf("rank %d: wysylam (%d, %d, %d)\n", rank, buf[0]->owner, buf[1]->owner, buf[2]->owner);
    
    MPI_Gather(buf[0], MAX_BUF_SIZE, MPI_CHAR, gatherBuf,                               MAX_BUF_SIZE, MPI_CHAR, ROOT_NODE, ACTIVE_NODES_WORLD);
    MPI_Gather(buf[1], MAX_BUF_SIZE, MPI_CHAR, gatherBuf + NUM_PROC * MAX_BUF_SIZE,     MAX_BUF_SIZE, MPI_CHAR, ROOT_NODE, ACTIVE_NODES_WORLD);
    MPI_Gather(buf[2], MAX_BUF_SIZE, MPI_CHAR, gatherBuf + NUM_PROC * MAX_BUF_SIZE * 2, MAX_BUF_SIZE, MPI_CHAR, ROOT_NODE, ACTIVE_NODES_WORLD);

    if (rank == ROOT_NODE) {
        printf("UDALO SIE, wypisuje zerowy:\n");

        MPI_Request *rs = (MPI_Request *) calloc(3 * NUM_PROC, sizeof(MPI_Request));
        int i = 0, j = 0;

        for (int off = 0; off < dataSize; off += MAX_BUF_SIZE) {
            MsgBuf *tmp = (MsgBuf*) (gatherBuf + off);
            INIT_BUF(tmp);
            if (tmp->owner == ROOT_NODE) {
                memcpy(buf[j], tmp, BUF_SIZE_RANK(ROOT_NODE));
                INIT_BUF(buf[j]);
                j++;
            } else {
                BUF_ISEND(tmp, BUF_SIZE(tmp), tmp->owner, &rs[i]);
                printf("recv owner: %d, num: %d\n", tmp->owner, tmp->particlesNum);
                i++;
            }
        }
        assert(j == 3);
        assert(i == 3 * NUM_PROC - 3);
        MPI_Waitall(i, rs, MPI_STATUS_IGNORE);
        free(rs);
    } else {
        MPI_Request rs[3];

        BUF_IRECV(buf[0], BUF_SIZE_RANK(rank), ROOT_NODE, &rs[0]);
        BUF_IRECV(buf[1], BUF_SIZE_RANK(rank), ROOT_NODE, &rs[1]);
        BUF_IRECV(buf[2], BUF_SIZE_RANK(rank), ROOT_NODE, &rs[2]);

        MPI_Waitall(3, rs, MPI_STATUS_IGNORE);
        INIT_BUF(buf[0]);
        INIT_BUF(buf[1]);
        INIT_BUF(buf[2]);

        printf("ALOHA! UDALO SIE ALL ODEBRAC! %d %d %d\n", buf[0]->owner, buf[1]->owner, buf[2]->owner);
    }

    // here each node got 3 copies of it's buffer
    move_particles(buf[0], buf[1], buf[2], rank, first_iter);

    free(buf[0]);
    free(buf[2]);
    free(tmpBuf);
}

