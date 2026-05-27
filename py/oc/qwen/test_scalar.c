// test_scalar.c — Stage 1: scalar ops + gradient verification
#include "autograd.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static int tests_run    = 0;
static int tests_passed = 0;
static int tests_failed = 0;

static void check(const char *desc, int condition) {
    tests_run++;
    if (condition) {
        tests_passed++;
        printf("  PASS: %s\n", desc);
    } else {
        tests_failed++;
        printf("  FAIL: %s\n", desc);
    }
}

static void check_close(const char *desc, f64 got, f64 expected, f64 tol) {
    f64 diff = fabs(got - expected);
    f64 scale = fmax(fabs(expected), 1e-10);
    f64 rel_error = diff / scale;
    tests_run++;
    if (rel_error < tol) {
        tests_passed++;
        printf("  PASS: %s\n", desc);
    } else {
        tests_failed++;
        printf("  FAIL: %s\n", desc);
        printf("         got=%.10f  expected=%.10f  tol=%.1e  rel_err=%.2e\n",
               got, expected, tol, rel_error);
    }
}

static AGGraphCtx g_ctx;

/* ------------------------------------------------------------------ */
/*  Test 1: Forward values for each basic op                          */
/* ------------------------------------------------------------------ */

static void test_forward_values(void) {
    printf("\n--- Forward Values ---\n");
    ag_graph_reset(&g_ctx);

    int x = ag_scalar_create(&g_ctx, 3.0);
    int y = ag_scalar_create(&g_ctx, 2.0);

    int s = ag_add(&g_ctx, x, y);
    check_close("add: 3+2=5", ag_get(&g_ctx, s), 5.0, 1e-12);

    s = ag_sub(&g_ctx, x, y);
    check_close("sub: 3-2=1", ag_get(&g_ctx, s), 1.0, 1e-12);

    s = ag_mul(&g_ctx, x, y);
    check_close("mul: 3*2=6", ag_get(&g_ctx, s), 6.0, 1e-12);

    s = ag_div(&g_ctx, x, y);
    check_close("div: 3/2=1.5", ag_get(&g_ctx, s), 1.5, 1e-12);

    s = ag_pow(&g_ctx, x, y);
    check_close("pow: 3^2=9", ag_get(&g_ctx, s), 9.0, 1e-12);

    int n = ag_neg(&g_ctx, x);
    check_close("neg: -3", ag_get(&g_ctx, n), -3.0, 1e-12);

    s = ag_exp(&g_ctx, y);
    check_close("exp: e^2", ag_get(&g_ctx, s), exp(2.0), 1e-12);

    s = ag_log(&g_ctx, x);
    check_close("log: ln(3)", ag_get(&g_ctx, s), log(3.0), 1e-12);

    s = ag_sin(&g_ctx, x);
    check_close("sin: sin(3)", ag_get(&g_ctx, s), sin(3.0), 1e-12);

    s = ag_cos(&g_ctx, x);
    check_close("cos: cos(3)", ag_get(&g_ctx, s), cos(3.0), 1e-12);

    int c05 = ag_constant(&g_ctx, 0.5);
    int px = ag_pow(&g_ctx, x, c05);
    s = ag_sqrt(&g_ctx, x);
    check_close("sqrt: sqrt(3)", ag_get(&g_ctx, s), sqrt(3.0), 1e-12);

    int neg_x = ag_scalar_create(&g_ctx, -5.0);
    s = ag_abs(&g_ctx, neg_x);
    check_close("abs: abs(-5)=5", ag_get(&g_ctx, s), 5.0, 1e-12);

    s = ag_abs(&g_ctx, x);
    check_close("abs: abs(3)=3", ag_get(&g_ctx, s), 3.0, 1e-12);
}

/* ------------------------------------------------------------------ */
/*  Test 2: Gradients for each basic op (analytical derivatives)      */
/* ------------------------------------------------------------------ */

static void test_gradients_analytical(void) {
    printf("\n--- Analytical Gradients ---\n");

    /* d/dx (x + y) = 1 */
    ag_graph_reset(&g_ctx);
    int x = ag_scalar_create(&g_ctx, 3.0);
    int y = ag_scalar_create(&g_ctx, 2.0);
    int s = ag_add(&g_ctx, x, y);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, s);
    check_close("grad add wrt x", g_ctx.tensor_pool[x].grad, 1.0, 1e-12);
    check_close("grad add wrt y", g_ctx.tensor_pool[y].grad, 1.0, 1e-12);

    /* d/dx (x - y) = 1,  d/dy (x - y) = -1 */
    ag_graph_reset(&g_ctx);
    x = ag_scalar_create(&g_ctx, 3.0);
    y = ag_scalar_create(&g_ctx, 2.0);
    s = ag_sub(&g_ctx, x, y);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, s);
    check_close("grad sub wrt x", g_ctx.tensor_pool[x].grad, 1.0, 1e-12);
    check_close("grad sub wrt y", g_ctx.tensor_pool[y].grad, -1.0, 1e-12);

    /* d/dx (x*y) = y,  d/dy (x*y) = x */
    ag_graph_reset(&g_ctx);
    x = ag_scalar_create(&g_ctx, 3.0);
    y = ag_scalar_create(&g_ctx, 2.0);
    s = ag_mul(&g_ctx, x, y);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, s);
    check_close("grad mul wrt x", g_ctx.tensor_pool[x].grad, 2.0, 1e-12);
    check_close("grad mul wrt y", g_ctx.tensor_pool[y].grad, 3.0, 1e-12);

    /* d/dx (x/y) = 1/y,  d/dy (x/y) = -x/y^2 */
    ag_graph_reset(&g_ctx);
    x = ag_scalar_create(&g_ctx, 6.0);
    y = ag_scalar_create(&g_ctx, 3.0);
    s = ag_div(&g_ctx, x, y);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, s);
    check_close("grad div wrt x", g_ctx.tensor_pool[x].grad, 1.0/3.0, 1e-12);
    check_close("grad div wrt y", g_ctx.tensor_pool[y].grad, -6.0/9.0, 1e-12);

    /* d/dx (x^y) = y*x^(y-1) */
    ag_graph_reset(&g_ctx);
    x = ag_scalar_create(&g_ctx, 2.0);
    y = ag_scalar_create(&g_ctx, 3.0);
    s = ag_pow(&g_ctx, x, y);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, s);
    check_close("grad pow wrt x", g_ctx.tensor_pool[x].grad, 3.0 * pow(2.0, 2.0), 1e-12);
    /* d/dy (x^y) = x^y * ln(x) */
    check_close("grad pow wrt y", g_ctx.tensor_pool[y].grad, pow(2.0, 3.0) * log(2.0), 1e-12);

    /* d/dx (-x) = -1 */
    ag_graph_reset(&g_ctx);
    x = ag_scalar_create(&g_ctx, 5.0);
    s = ag_neg(&g_ctx, x);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, s);
    check_close("grad neg", g_ctx.tensor_pool[x].grad, -1.0, 1e-12);

    /* d/dx (exp(x)) = exp(x) */
    ag_graph_reset(&g_ctx);
    x = ag_scalar_create(&g_ctx, 1.0);
    s = ag_exp(&g_ctx, x);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, s);
    check_close("grad exp", g_ctx.tensor_pool[x].grad, exp(1.0), 1e-12);

    /* d/dx (log(x)) = 1/x */
    ag_graph_reset(&g_ctx);
    x = ag_scalar_create(&g_ctx, 2.0);
    s = ag_log(&g_ctx, x);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, s);
    check_close("grad log", g_ctx.tensor_pool[x].grad, 0.5, 1e-12);

    /* d/dx (sin(x)) = cos(x) */
    ag_graph_reset(&g_ctx);
    x = ag_scalar_create(&g_ctx, 0.5);
    s = ag_sin(&g_ctx, x);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, s);
    check_close("grad sin", g_ctx.tensor_pool[x].grad, cos(0.5), 1e-12);

    /* d/dx (cos(x)) = -sin(x) */
    ag_graph_reset(&g_ctx);
    x = ag_scalar_create(&g_ctx, 0.5);
    s = ag_cos(&g_ctx, x);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, s);
    check_close("grad cos", g_ctx.tensor_pool[x].grad, -sin(0.5), 1e-12);

    /* d/dx (sqrt(x)) = 1/(2*sqrt(x)) */
    ag_graph_reset(&g_ctx);
    x = ag_scalar_create(&g_ctx, 4.0);
    s = ag_sqrt(&g_ctx, x);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, s);
    check_close("grad sqrt", g_ctx.tensor_pool[x].grad, 1.0/(2.0*2.0), 1e-12);

    /* d/dx (abs(x)) = sign(x) */
    ag_graph_reset(&g_ctx);
    x = ag_scalar_create(&g_ctx, 5.0);
    s = ag_abs(&g_ctx, x);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, s);
    check_close("grad abs (pos)", g_ctx.tensor_pool[x].grad, 1.0, 1e-12);

    ag_graph_reset(&g_ctx);
    x = ag_scalar_create(&g_ctx, -5.0);
    s = ag_abs(&g_ctx, x);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, s);
    check_close("grad abs (neg)", g_ctx.tensor_pool[x].grad, -1.0, 1e-12);
}

/* ------------------------------------------------------------------ */
/*  Test 3: Chain rule — f(x) = (x^2 + 1) * exp(x)                    */
/*  df/dx = exp(x) * (2x + x^2 + 1)                                   */
/* ------------------------------------------------------------------ */

static void test_chain_rule(void) {
    printf("\n--- Chain Rule ---\n");

    ag_graph_reset(&g_ctx);
    int x = ag_scalar_create(&g_ctx, 2.0);
    int x2 = ag_mul(&g_ctx, x, x);
    int c  = ag_constant(&g_ctx, 1.0);
    int add1 = ag_add(&g_ctx, x2, c);
    int ex   = ag_exp(&g_ctx, x);
    int out  = ag_mul(&g_ctx, add1, ex);

    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, out);

    f64 x_val = 2.0;
    f64 expected_grad = exp(x_val) * (2.0*x_val + x_val*x_val + 1.0);

    check_close("chain: (x^2+1)*exp(x) grad at x=2",
                g_ctx.tensor_pool[x].grad, expected_grad, 1e-10);
    check_close("chain: forward value",
                ag_get(&g_ctx, out),
                (x_val*x_val + 1.0) * exp(x_val), 1e-12);

    /* f(x) = sin(x^2)  =>  df/dx = cos(x^2) * 2x */
    ag_graph_reset(&g_ctx);
    x = ag_scalar_create(&g_ctx, 1.0);
    x2 = ag_mul(&g_ctx, x, x);
    out = ag_sin(&g_ctx, x2);

    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, out);

    x_val = 1.0;
    expected_grad = cos(x_val * x_val) * 2.0 * x_val;
    check_close("chain: sin(x^2) grad at x=1",
                g_ctx.tensor_pool[x].grad, expected_grad, 1e-10);
}

/* ------------------------------------------------------------------ */
/*  Test 4: Finite difference verification                            */
/* ------------------------------------------------------------------ */

static void test_finite_difference(void) {
    printf("\n--- Finite Difference ---\n");
    f64 h = 1e-6;

    /* f(x,y) = x*y + sin(x) */
    ag_graph_reset(&g_ctx);
    int x = ag_scalar_create(&g_ctx, 1.5);
    int y = ag_scalar_create(&g_ctx, 2.5);
    int prod = ag_mul(&g_ctx, x, y);
    int sinp = ag_sin(&g_ctx, x);
    int out2 = ag_add(&g_ctx, prod, sinp);

    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, out2);

    f64 x0 = 1.5, y0 = 2.5;

    /* numerical grad wrt x */
    ag_set(&g_ctx, x, x0 + h); ag_set(&g_ctx, y, y0);
    ag_forward(&g_ctx); f64 fp = ag_get(&g_ctx, out2);
    ag_set(&g_ctx, x, x0 - h); ag_forward(&g_ctx); f64 fm = ag_get(&g_ctx, out2);
    f64 num_grad_x = (fp - fm) / (2.0 * h);

    /* numerical grad wrt y */
    ag_set(&g_ctx, x, x0); ag_set(&g_ctx, y, y0 + h);
    ag_forward(&g_ctx); fp = ag_get(&g_ctx, out2);
    ag_set(&g_ctx, y, y0 - h); ag_forward(&g_ctx); fm = ag_get(&g_ctx, out2);
    f64 num_grad_y = (fp - fm) / (2.0 * h);

    check_close("fd grad x: x*y+sin(x) at (1.5,2.5)",
                g_ctx.tensor_pool[x].grad, num_grad_x, 1e-4);
    check_close("fd grad y: x*y+sin(x) at (1.5,2.5)",
                g_ctx.tensor_pool[y].grad, num_grad_y, 1e-4);

    /* f(x) = exp(x^2)  =>  df/dx = 2x*exp(x^2) */
    ag_graph_reset(&g_ctx);
    int xi = ag_scalar_create(&g_ctx, 0.7);
    int xi2 = ag_mul(&g_ctx, xi, xi);
    int out3 = ag_exp(&g_ctx, xi2);

    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, out3);

    f64 x1 = 0.7;
    f64 expected = 2.0 * x1 * exp(x1 * x1);

    ag_set(&g_ctx, xi, x1 + h); ag_forward(&g_ctx); fp = ag_get(&g_ctx, out3);
    ag_set(&g_ctx, xi, x1 - h); ag_forward(&g_ctx); fm = ag_get(&g_ctx, out3);
    f64 num_g = (fp - fm) / (2.0 * h);

    check_close("fd grad exp(x^2) at x=0.7",
                g_ctx.tensor_pool[xi].grad, num_g, 1e-4);
    check_close("analytical grad exp(x^2) at x=0.7",
                g_ctx.tensor_pool[xi].grad, expected, 1e-10);
}

/* ------------------------------------------------------------------ */
/*  Test 5: Constant tensors don't accumulate gradients               */
/* ------------------------------------------------------------------ */

static void test_constants(void) {
    printf("\n--- Constants ---\n");
    ag_graph_reset(&g_ctx);
    int x = ag_scalar_create(&g_ctx, 3.0);
    int c = ag_constant(&g_ctx, 5.0);
    int out = ag_mul(&g_ctx, x, c);
    ag_forward(&g_ctx);
    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, out);
    check_close("const grad stays 0", g_ctx.tensor_pool[c].grad, 0.0, 1e-15);
    check_close("var grad = const val", g_ctx.tensor_pool[x].grad, 5.0, 1e-12);
}

/* ------------------------------------------------------------------ */
/*  Test 6: Diamond DAG — gradient accumulation                       */
/*  z = x * y,  out = z + z  (z used twice)                           */
/*  d(out)/dx = 2*y, d(out)/dy = 2*x                                  */
/* ------------------------------------------------------------------ */

static void test_diamond_dag(void) {
    printf("\n--- Diamond DAG (Grad Accumulation) ---\n");
    ag_graph_reset(&g_ctx);
    int x = ag_scalar_create(&g_ctx, 3.0);
    int y = ag_scalar_create(&g_ctx, 4.0);
    int z = ag_mul(&g_ctx, x, y);
    int out = ag_add(&g_ctx, z, z);

    ag_forward(&g_ctx);
    check_close("diamond forward", ag_get(&g_ctx, out), 24.0, 1e-12);

    ag_zero_grad(&g_ctx);
    ag_backward(&g_ctx, out);
    check_close("diamond grad x (should be 2*y=8)",
                g_ctx.tensor_pool[x].grad, 8.0, 1e-12);
    check_close("diamond grad y (should be 2*x=6)",
                g_ctx.tensor_pool[y].grad, 6.0, 1e-12);
}

/* ------------------------------------------------------------------ */
/*  Test 7: Graph reset                                               */
/* ------------------------------------------------------------------ */

static void test_graph_reset(void) {
    printf("\n--- Graph Reset ---\n");
    ag_graph_reset(&g_ctx);
    ag_scalar_create(&g_ctx, 1.0);
    ag_scalar_create(&g_ctx, 2.0);
    check("tensor_count > 0 after creates", ag_tensor_count(&g_ctx) > 0);
    ag_graph_reset(&g_ctx);
    check("tensor_count == 0 after reset", ag_tensor_count(&g_ctx) == 0);
}

/* ------------------------------------------------------------------ */
/*  Test 8: Multiple points chain rule                                */
/*  f(x) = (x^2 + 1) * exp(x) at x = -1, 0, 1, 3, 5                 */
/* ------------------------------------------------------------------ */

static void test_chain_rule_multi(void) {
    printf("\n--- Chain Rule Multi-Point ---\n");
    f64 points[] = {-1.0, 0.0, 1.0, 3.0, 5.0};
    int n = sizeof(points) / sizeof(points[0]);

    for (int p = 0; p < n; p++) {
        f64 xv = points[p];
        ag_graph_reset(&g_ctx);
        int xi = ag_scalar_create(&g_ctx, xv);
        int x2 = ag_mul(&g_ctx, xi, xi);
        int c  = ag_constant(&g_ctx, 1.0);
        int add1 = ag_add(&g_ctx, x2, c);
        int ex   = ag_exp(&g_ctx, xi);
        int out  = ag_mul(&g_ctx, add1, ex);

        ag_forward(&g_ctx);
        ag_zero_grad(&g_ctx);
        ag_backward(&g_ctx, out);

        f64 expected = exp(xv) * (2.0*xv + xv*xv + 1.0);
        char buf[128];
        snprintf(buf, sizeof(buf), "chain mult: x=%.0f", xv);
        check_close(buf, g_ctx.tensor_pool[xi].grad, expected, 1e-10);
    }
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(void) {
    printf("=== Stage 1: Scalar Core + Basic Ops Tests ===\n");

    test_forward_values();
    test_gradients_analytical();
    test_chain_rule();
    test_finite_difference();
    test_constants();
    test_diamond_dag();
    test_graph_reset();
    test_chain_rule_multi();

    printf("\n==============================================\n");
    printf("Results: %d/%d passed, %d failed\n",
           tests_passed, tests_run, tests_failed);
    printf("==============================================\n");

    return tests_failed > 0 ? 1 : 0;
}
