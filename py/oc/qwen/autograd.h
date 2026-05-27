// autograd.h — minimal high-performance autograd engine (scalar core)
#pragma once

#include <stdint.h>
#include <stddef.h>

typedef int8_t  s8;
typedef uint8_t u8;
typedef int16_t s16;
typedef uint16_t u16;
typedef int32_t s32;
typedef uint32_t u32;
typedef int64_t s64;
typedef uint64_t u64;
typedef float   f32;
typedef double  f64;
typedef _Bool   b8;

#define AG_DTYPE_F32 0
#define AG_DTYPE_F64 1
#define AG_MAX_NODES 65536
#define AG_MAX_TENSORS 131072
#define AG_MAX_INPUTS 4
#define AG_MAX_OPS 64

/* tensor element — value + accumulated gradient */
typedef struct {
    f64   val;
    f64   grad;
    int   dtype;
    int   _opaque;       /* 0=tracked, 1=constant */
} AGScalar;

/* op dispatch function pointers */
typedef void (*ag_forward_fn)(const void *meta, const AGScalar **inputs, int in_count, AGScalar *out);
typedef void (*ag_backward_fn)(const void *meta, AGScalar **inputs, int in_count, const AGScalar *out_grad, AGScalar *out);

/* immutable operation descriptor */
typedef struct {
    u8         op_id;
    uint8_t    _pad;
    int16_t    input_count;
    int32_t    inputs[AG_MAX_INPUTS];
    int32_t    output;
    const void *meta;
    ag_forward_fn  fwd;
    ag_backward_fn bwd;
} AGOpNode;

typedef struct {
    const char *name;
    uint8_t     op_id;
    ag_forward_fn  fwd;
    ag_backward_fn bwd;
} AGOpReg;

typedef struct {
    AGOpReg ops[AG_MAX_OPS];
    int     op_count;
} AGOpRegistry;

/* execution context — all storage is inline (stack/arena-backed) */
typedef struct {
    AGScalar   tensor_pool[AG_MAX_TENSORS];
    AGOpNode   node_pool[AG_MAX_NODES];
    AGOpNode  *nodes_sorted[AG_MAX_NODES];
    int        tensor_count;
    int        node_count;
    int        sorted_count;
    AGOpRegistry registry;
} AGGraphCtx;

/* === public API === */

/* registration */
int ag_register_op(AGGraphCtx *ctx, const char *name, uint8_t op_id,
                   ag_forward_fn fwd, ag_backward_fn bwd);

/* construct scalars — return pool index */
int ag_scalar_create(AGGraphCtx *ctx, f64 val);
int ag_constant(AGGraphCtx *ctx, f64 val);

/* binary ops — return output pool index */
int ag_add(AGGraphCtx *ctx, int a, int b);
int ag_sub(AGGraphCtx *ctx, int a, int b);
int ag_mul(AGGraphCtx *ctx, int a, int b);
int ag_div(AGGraphCtx *ctx, int a, int b);
int ag_pow(AGGraphCtx *ctx, int a, int b);

/* unary ops */
int ag_neg(AGGraphCtx *ctx, int a);
int ag_exp(AGGraphCtx *ctx, int a);
int ag_log(AGGraphCtx *ctx, int a);
int ag_sin(AGGraphCtx *ctx, int a);
int ag_cos(AGGraphCtx *ctx, int a);
int ag_sqrt(AGGraphCtx *ctx, int a);
int ag_abs(AGGraphCtx *ctx, int a);

/* access by index */
f64 ag_get(AGGraphCtx *ctx, int idx);
void ag_set(AGGraphCtx *ctx, int idx, f64 val);

/* execution */
void ag_forward(AGGraphCtx *ctx);
void ag_backward(AGGraphCtx *ctx, int output_idx);
void ag_zero_grad(AGGraphCtx *ctx);

/* graph mgmt */
void ag_graph_reset(AGGraphCtx *ctx);
int  ag_tensor_count(AGGraphCtx *ctx);

/* op ids */
#define AG_OP_ADD      0
#define AG_OP_SUB      1
#define AG_OP_MUL      2
#define AG_OP_DIV      3
#define AG_OP_POW      4
#define AG_OP_NEG      5
#define AG_OP_EXP      6
#define AG_OP_LOG      7
#define AG_OP_SIN      8
#define AG_OP_COS      9
#define AG_OP_SQRT     10
#define AG_OP_ABS      11
