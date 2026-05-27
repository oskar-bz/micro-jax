// autograd.c — scalar core + basic ops, forward & backward
#include "autograd.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

/* === pool allocation === */

static int alloc_tensor(AGGraphCtx *ctx) {
    if (ctx->tensor_count >= AG_MAX_TENSORS) {
        fprintf(stderr, "autograd: tensor pool exhausted (%d)\n", AG_MAX_TENSORS);
        return -1;
    }
    int idx = ctx->tensor_count++;
    ctx->tensor_pool[idx].val = 0.0;
    ctx->tensor_pool[idx].grad = 0.0;
    return idx;
}

static int alloc_node(AGGraphCtx *ctx) {
    if (ctx->node_count >= AG_MAX_NODES) {
        fprintf(stderr, "autograd: node pool exhausted (%d)\n", AG_MAX_NODES);
        return -1;
    }
    return ctx->node_count++;
}

/* === op registry === */

int ag_register_op(AGGraphCtx *ctx, const char *name, uint8_t op_id,
                   ag_forward_fn fwd, ag_backward_fn bwd) {
    if (ctx->registry.op_count >= AG_MAX_OPS) return -1;
    AGOpReg *op = &ctx->registry.ops[ctx->registry.op_count++];
    op->name  = name;
    op->op_id = op_id;
    op->fwd   = fwd;
    op->bwd   = bwd;
    (void)name;
    return 0;
}

/* === scalar construction === */

int ag_scalar_create(AGGraphCtx *ctx, f64 val) {
    int idx = alloc_tensor(ctx);
    if (idx < 0) return -1;
    ctx->tensor_pool[idx].val    = val;
    ctx->tensor_pool[idx].grad   = 0.0;
    ctx->tensor_pool[idx].dtype  = AG_DTYPE_F64;
    ctx->tensor_pool[idx]._opaque = 0;  /* tracked */
    return idx;
}

int ag_constant(AGGraphCtx *ctx, f64 val) {
    int idx = alloc_tensor(ctx);
    if (idx < 0) return -1;
    ctx->tensor_pool[idx].val    = val;
    ctx->tensor_pool[idx].grad   = 0.0;
    ctx->tensor_pool[idx].dtype  = AG_DTYPE_F64;
    ctx->tensor_pool[idx]._opaque = 1;  /* constant */
    return idx;
}

/* === accessors === */

f64 ag_get(AGGraphCtx *ctx, int idx) {
    return ctx->tensor_pool[idx].val;
}

void ag_set(AGGraphCtx *ctx, int idx, f64 val) {
    ctx->tensor_pool[idx].val = val;
}

/* === forward execution === */

void ag_forward(AGGraphCtx *ctx) {
    for (int i = 0; i < ctx->sorted_count; i++) {
        AGOpNode *node = ctx->nodes_sorted[i];
        const AGScalar *inputs[AG_MAX_INPUTS];
        for (int j = 0; j < node->input_count; j++) {
            inputs[j] = &ctx->tensor_pool[node->inputs[j]];
        }
        AGScalar *out = &ctx->tensor_pool[node->output];
        node->fwd(node->meta, inputs, node->input_count, out);
    }
}

/* === backward execution === */

void ag_zero_grad(AGGraphCtx *ctx) {
    for (int i = 0; i < ctx->tensor_count; i++) {
        ctx->tensor_pool[i].grad = 0.0;
    }
}

void ag_backward(AGGraphCtx *ctx, int output_idx) {
    ctx->tensor_pool[output_idx].grad = 1.0;

    for (int i = ctx->sorted_count - 1; i >= 0; i--) {
        AGOpNode *node = ctx->nodes_sorted[i];
        const AGScalar *out_grad_src = &ctx->tensor_pool[node->output];
        if (out_grad_src->grad == 0.0) continue;

        AGScalar *input_tensors[AG_MAX_INPUTS];
        for (int j = 0; j < node->input_count; j++) {
            input_tensors[j] = &ctx->tensor_pool[node->inputs[j]];
        }
        AGScalar *out = &ctx->tensor_pool[node->output];

        node->bwd(node->meta, input_tensors, node->input_count,
                  out_grad_src, out);
    }
}

/* === graph mgmt === */

void ag_graph_reset(AGGraphCtx *ctx) {
    ctx->tensor_count = 0;
    ctx->node_count   = 0;
    ctx->sorted_count = 0;
    ctx->registry.op_count = 0;
    memset(ctx->tensor_pool, 0, sizeof(ctx->tensor_pool));
    memset(ctx->node_pool, 0, sizeof(ctx->node_pool));
}

int ag_tensor_count(AGGraphCtx *ctx) {
    return ctx->tensor_count;
}

/* === topo sort === */
/* Sequential construction guarantees topological order. */

/* === op builders === */

static int build_binary(AGGraphCtx *ctx, uint8_t op_id, int a, int b,
                        ag_forward_fn fwd, ag_backward_fn bwd) {
    int out_idx = alloc_tensor(ctx);
    if (out_idx < 0) return -1;
    int nid = alloc_node(ctx);
    if (nid < 0) return -1;

    AGOpNode *node = &ctx->node_pool[nid];
    node->op_id      = op_id;
    node->input_count = 2;
    node->inputs[0]  = a;
    node->inputs[1]  = b;
    node->output     = out_idx;
    node->meta       = NULL;
    node->fwd        = fwd;
    node->bwd        = bwd;

    ctx->sorted_count = ctx->node_count;
    for (int i = 0; i < ctx->node_count; i++) {
        ctx->nodes_sorted[i] = &ctx->node_pool[i];
    }
    return out_idx;
}

static int build_unary(AGGraphCtx *ctx, uint8_t op_id, int a,
                       ag_forward_fn fwd, ag_backward_fn bwd) {
    int out_idx = alloc_tensor(ctx);
    if (out_idx < 0) return -1;
    int nid = alloc_node(ctx);
    if (nid < 0) return -1;

    AGOpNode *node = &ctx->node_pool[nid];
    node->op_id      = op_id;
    node->input_count = 1;
    node->inputs[0]  = a;
    node->output     = out_idx;
    node->meta       = NULL;
    node->fwd        = fwd;
    node->bwd        = bwd;

    ctx->sorted_count = ctx->node_count;
    for (int i = 0; i < ctx->node_count; i++) {
        ctx->nodes_sorted[i] = &ctx->node_pool[i];
    }
    return out_idx;
}

/* === binary ops API === */

static void fwd_add(const void *, const AGScalar **, int, AGScalar *);
static void bwd_add(const void *, AGScalar **, int, const AGScalar *, AGScalar *);
static void fwd_sub(const void *, const AGScalar **, int, AGScalar *);
static void bwd_sub(const void *, AGScalar **, int, const AGScalar *, AGScalar *);
static void fwd_mul(const void *, const AGScalar **, int, AGScalar *);
static void bwd_mul(const void *, AGScalar **, int, const AGScalar *, AGScalar *);
static void fwd_div(const void *, const AGScalar **, int, AGScalar *);
static void bwd_div(const void *, AGScalar **, int, const AGScalar *, AGScalar *);
static void fwd_pow(const void *, const AGScalar **, int, AGScalar *);
static void bwd_pow(const void *, AGScalar **, int, const AGScalar *, AGScalar *);

int ag_add(AGGraphCtx *ctx, int a, int b) {
    return build_binary(ctx, AG_OP_ADD, a, b, fwd_add, bwd_add);
}
int ag_sub(AGGraphCtx *ctx, int a, int b) {
    return build_binary(ctx, AG_OP_SUB, a, b, fwd_sub, bwd_sub);
}
int ag_mul(AGGraphCtx *ctx, int a, int b) {
    return build_binary(ctx, AG_OP_MUL, a, b, fwd_mul, bwd_mul);
}
int ag_div(AGGraphCtx *ctx, int a, int b) {
    return build_binary(ctx, AG_OP_DIV, a, b, fwd_div, bwd_div);
}
int ag_pow(AGGraphCtx *ctx, int a, int b) {
    return build_binary(ctx, AG_OP_POW, a, b, fwd_pow, bwd_pow);
}

/* === unary ops API === */

static void fwd_neg(const void *, const AGScalar **, int, AGScalar *);
static void bwd_neg(const void *, AGScalar **, int, const AGScalar *, AGScalar *);
static void fwd_exp(const void *, const AGScalar **, int, AGScalar *);
static void bwd_exp(const void *, AGScalar **, int, const AGScalar *, AGScalar *);
static void fwd_log(const void *, const AGScalar **, int, AGScalar *);
static void bwd_log(const void *, AGScalar **, int, const AGScalar *, AGScalar *);
static void fwd_sin(const void *, const AGScalar **, int, AGScalar *);
static void bwd_sin(const void *, AGScalar **, int, const AGScalar *, AGScalar *);
static void fwd_cos(const void *, const AGScalar **, int, AGScalar *);
static void bwd_cos(const void *, AGScalar **, int, const AGScalar *, AGScalar *);
static void fwd_sqrt(const void *, const AGScalar **, int, AGScalar *);
static void bwd_sqrt(const void *, AGScalar **, int, const AGScalar *, AGScalar *);
static void fwd_abs(const void *, const AGScalar **, int, AGScalar *);
static void bwd_abs(const void *, AGScalar **, int, const AGScalar *, AGScalar *);

int ag_neg(AGGraphCtx *ctx, int a) {
    return build_unary(ctx, AG_OP_NEG, a, fwd_neg, bwd_neg);
}
int ag_exp(AGGraphCtx *ctx, int a) {
    return build_unary(ctx, AG_OP_EXP, a, fwd_exp, bwd_exp);
}
int ag_log(AGGraphCtx *ctx, int a) {
    return build_unary(ctx, AG_OP_LOG, a, fwd_log, bwd_log);
}
int ag_sin(AGGraphCtx *ctx, int a) {
    return build_unary(ctx, AG_OP_SIN, a, fwd_sin, bwd_sin);
}
int ag_cos(AGGraphCtx *ctx, int a) {
    return build_unary(ctx, AG_OP_COS, a, fwd_cos, bwd_cos);
}
int ag_sqrt(AGGraphCtx *ctx, int a) {
    return build_unary(ctx, AG_OP_SQRT, a, fwd_sqrt, bwd_sqrt);
}
int ag_abs(AGGraphCtx *ctx, int a) {
    return build_unary(ctx, AG_OP_ABS, a, fwd_abs, bwd_abs);
}

/* ===================================================================
 * FORWARD kernels
 * =================================================================== */

static void fwd_add(const void *meta, const AGScalar **inputs, int in_count, AGScalar *out) {
    (void)meta; (void)in_count;
    out->val = inputs[0]->val + inputs[1]->val;
}

static void fwd_sub(const void *meta, const AGScalar **inputs, int in_count, AGScalar *out) {
    (void)meta; (void)in_count;
    out->val = inputs[0]->val - inputs[1]->val;
}

static void fwd_mul(const void *meta, const AGScalar **inputs, int in_count, AGScalar *out) {
    (void)meta; (void)in_count;
    out->val = inputs[0]->val * inputs[1]->val;
}

static void fwd_div(const void *meta, const AGScalar **inputs, int in_count, AGScalar *out) {
    (void)meta; (void)in_count;
    out->val = inputs[0]->val / inputs[1]->val;
}

static void fwd_pow(const void *meta, const AGScalar **inputs, int in_count, AGScalar *out) {
    (void)meta; (void)in_count;
    out->val = pow(inputs[0]->val, inputs[1]->val);
}

static void fwd_neg(const void *meta, const AGScalar **inputs, int in_count, AGScalar *out) {
    (void)meta; (void)in_count;
    out->val = -inputs[0]->val;
}

static void fwd_exp(const void *meta, const AGScalar **inputs, int in_count, AGScalar *out) {
    (void)meta; (void)in_count;
    out->val = exp(inputs[0]->val);
}

static void fwd_log(const void *meta, const AGScalar **inputs, int in_count, AGScalar *out) {
    (void)meta; (void)in_count;
    out->val = log(inputs[0]->val);
}

static void fwd_sin(const void *meta, const AGScalar **inputs, int in_count, AGScalar *out) {
    (void)meta; (void)in_count;
    out->val = sin(inputs[0]->val);
}

static void fwd_cos(const void *meta, const AGScalar **inputs, int in_count, AGScalar *out) {
    (void)meta; (void)in_count;
    out->val = cos(inputs[0]->val);
}

static void fwd_sqrt(const void *meta, const AGScalar **inputs, int in_count, AGScalar *out) {
    (void)meta; (void)in_count;
    out->val = sqrt(inputs[0]->val);
}

static void fwd_abs(const void *meta, const AGScalar **inputs, int in_count, AGScalar *out) {
    (void)meta; (void)in_count;
    out->val = fabs(inputs[0]->val);
}

/* ===================================================================
 * BACKWARD kernels
 * ===================================================================
 * inputs[] — pointers to input tensors (we read .val for derivatives)
 * out_grad — gradient of the output (source)
 *
 * We accumulate into inputs[i]->grad (+=) to handle diamond DAGs.
 * inputs[i]->_opaque guards against writing to constants.
 * =================================================================== */

static void bwd_add(const void *meta, AGScalar **inputs, int in_count,
                    const AGScalar *out_grad, AGScalar *out) {
    (void)meta; (void)out;
    if (in_count != 2) return;
    f64 g = out_grad->grad;
    if (inputs[0]->_opaque == 0) inputs[0]->grad += g;
    if (inputs[1]->_opaque == 0) inputs[1]->grad += g;
}

static void bwd_sub(const void *meta, AGScalar **inputs, int in_count,
                    const AGScalar *out_grad, AGScalar *out) {
    (void)meta; (void)out;
    if (in_count != 2) return;
    f64 g = out_grad->grad;
    if (inputs[0]->_opaque == 0) inputs[0]->grad += g;
    if (inputs[1]->_opaque == 0) inputs[1]->grad -= g;
}

static void bwd_mul(const void *meta, AGScalar **inputs, int in_count,
                    const AGScalar *out_grad, AGScalar *out) {
    (void)meta; (void)out;
    if (in_count != 2) return;
    f64 g = out_grad->grad;
    if (inputs[0]->_opaque == 0) inputs[0]->grad += g * inputs[1]->val;
    if (inputs[1]->_opaque == 0) inputs[1]->grad += g * inputs[0]->val;
}

static void bwd_div(const void *meta, AGScalar **inputs, int in_count,
                    const AGScalar *out_grad, AGScalar *out) {
    (void)meta; (void)out;
    if (in_count != 2) return;
    f64 g = out_grad->grad;
    f64 a = inputs[0]->val;
    f64 b = inputs[1]->val;
    if (inputs[0]->_opaque == 0) inputs[0]->grad += g / b;
    if (inputs[1]->_opaque == 0) inputs[1]->grad -= g * a / (b * b);
}

static void bwd_pow(const void *meta, AGScalar **inputs, int in_count,
                    const AGScalar *out_grad, AGScalar *out) {
    (void)meta; (void)out;
    if (in_count != 2) return;
    f64 g = out_grad->grad;
    f64 a = inputs[0]->val;
    f64 b = inputs[1]->val;
    if (inputs[0]->_opaque == 0) inputs[0]->grad += g * b * pow(a, b - 1.0);
    if (inputs[1]->_opaque == 0) inputs[1]->grad += g * log(fabs(a) + 1e-30) * pow(a, b);
}

static void bwd_neg(const void *meta, AGScalar **inputs, int in_count,
                    const AGScalar *out_grad, AGScalar *out) {
    (void)meta; (void)in_count; (void)out;
    if (inputs[0]->_opaque == 0) inputs[0]->grad -= out_grad->grad;
}

static void bwd_exp(const void *meta, AGScalar **inputs, int in_count,
                    const AGScalar *out_grad, AGScalar *out) {
    (void)meta; (void)in_count; (void)out;
    if (inputs[0]->_opaque == 0) inputs[0]->grad += out_grad->grad * exp(inputs[0]->val);
}

static void bwd_log(const void *meta, AGScalar **inputs, int in_count,
                    const AGScalar *out_grad, AGScalar *out) {
    (void)meta; (void)in_count; (void)out;
    if (inputs[0]->_opaque == 0) inputs[0]->grad += out_grad->grad / inputs[0]->val;
}

static void bwd_sin(const void *meta, AGScalar **inputs, int in_count,
                    const AGScalar *out_grad, AGScalar *out) {
    (void)meta; (void)in_count; (void)out;
    if (inputs[0]->_opaque == 0) inputs[0]->grad += out_grad->grad * cos(inputs[0]->val);
}

static void bwd_cos(const void *meta, AGScalar **inputs, int in_count,
                    const AGScalar *out_grad, AGScalar *out) {
    (void)meta; (void)in_count; (void)out;
    if (inputs[0]->_opaque == 0) inputs[0]->grad += out_grad->grad * (-sin(inputs[0]->val));
}

static void bwd_sqrt(const void *meta, AGScalar **inputs, int in_count,
                     const AGScalar *out_grad, AGScalar *out) {
    (void)meta; (void)in_count; (void)out;
    if (inputs[0]->_opaque == 0) {
        f64 s = sqrt(inputs[0]->val);
        if (s != 0.0) inputs[0]->grad += out_grad->grad / (2.0 * s);
    }
}

static void bwd_abs(const void *meta, AGScalar **inputs, int in_count,
                    const AGScalar *out_grad, AGScalar *out) {
    (void)meta; (void)in_count; (void)out;
    if (inputs[0]->_opaque == 0) {
        f64 v = inputs[0]->val;
        f64 s = (v > 0) ? 1.0 : ((v < 0) ? -1.0 : 0.0);
        inputs[0]->grad += out_grad->grad * s;
    }
}
