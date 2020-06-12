#pragma once
#include <cstdint> 

typedef struct Pos {
    double x;
    double y;
    double z;
} Pos;

#define POSX(__buf, __i) (__buf->elems[__i].pos.x)
#define POSY(__buf, __i) (__buf->elems[__i].pos.y)
#define POSZ(__buf, __i) (__buf->elems[__i].pos.z)

#define PRINT_POS(pos) printf("POS: (%f, %f, %f)\n", pos.x, pos.y, pos.z);

typedef struct Vel {
    double vx;
    double vy;
    double vz;
} Vel;

#define VELX(__buf, __i) (__buf->elems[__i].vel.vx)
#define VELY(__buf, __i) (__buf->elems[__i].vel.vy)
#define VELZ(__buf, __i) (__buf->elems[__i].vel.vz)


#define PRINT_VEL(v) printf("VEL: (%f, %f, %f)\n", v.vx, v.vy, v.vz);

typedef struct Acc {
    double ax;
    double ay;
    double az;
} Acc;

#define ACCX(__buf, __i) (__buf->elems[__i].acc.ax)
#define ACCY(__buf, __i) (__buf->elems[__i].acc.ay)
#define ACCZ(__buf, __i) (__buf->elems[__i].acc.az)

#define PRINT_ACC(a) printf("ACC: (%f, %f, %f)\n", a.ax, a.ay, v.az);

typedef struct Force {
    double fx;
    double fy;
    double fz;
} Force;

#define FX(__buf, __i) (__buf->elems[__i].force.fx)
#define FY(__buf, __i) (__buf->elems[__i].force.fy)
#define FZ(__buf, __i) (__buf->elems[__i].force.fz)

typedef struct ParticleDescr {
    Pos pos;
    Vel vel;
    Acc acc;
    Force force;
} ParticleDescr;

typedef struct MsgBuf {
    uint32_t owner;
    uint32_t particlesNum;
    ParticleDescr *elems; // it should point just behind itself (rest of buffer) // it's like custom flexible array member implementation
} MsgBuf;

#define INIT_BUF(__msgBuf) ((__msgBuf)->elems = (ParticleDescr*) ((((char*) (__msgBuf)) + sizeof(MsgBuf)))) // (msgBuf->elems = (ParticleDescr*) ( ((char*) msgBuf)) )

#define BUF_SIZE(__msgBuf) (sizeof(MsgBuf) + ((__msgBuf)->particlesNum) * sizeof(ParticleDescr))

#define BUF_SIZE_RANK(__rank) (sizeof(MsgBuf) + (MAX_PART_IDX(__rank) - MIN_PART_IDX(__rank)) * sizeof(ParticleDescr))

#define MAX_BUF_SIZE (BUF_SIZE_RANK(0))