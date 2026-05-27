#pragma once
#include "util.h"

#define TENSOR_MAX_ORDER 8

typedef enum {
  DTYPE_U8,
  DTYPE_U16,
  DTYPE_U32,
  DTYPE_U64,
  DTYPE_S8,
  DTYPE_S16,
  DTYPE_S32,
  DTYPE_S64,
  DTYPE_F32,
  DTYPE_F64,
  DTYPE_BOOL,
  DTYPE_COUNT,
} TensorDType;

typedef u8 TensorFlags;
enum {
  TENSOR_OWNS_DATA = 1 << 0,
  TENSOR_CONTIGUOUS = 1 << 1,
  TENSOR_WRITEABLE = 1 << 2,
};

typedef struct {
  u8 *data;
  u64 shape[TENSOR_MAX_ORDER];
  s64 strides[TENSOR_MAX_ORDER];
  Allocator *alloc;
  TensorDType dtype;
  TensorFlags flags;
  u8 order;
} Tensor;

#define ts_elem(T, t, byte_offset) (*((T *)((t)->data + (byte_offset))))

// Compute byte offset given indices
#define ts_offset1(t, i) ((i) * t->strides[0])
#define ts_offset2(t, i, j) (tensor_offset1(t, i) + (j) * t->strides[1])
#define ts_offset3(t, i, j, k) (tensor_offset2(t, i, j) + (k) * t->strides[2])
#define ts_offset4(t, i, j, k, l)                                              \
  (tensor_offset3(t, i, j, k) + (l) * t->strides[3])

// == shape & stride utilities
// compute row-major strides from a given shape
void ts_strides_from_shape(u64 order, const u64 *shape, u64 elem_size,
                           s64 *strides_out);

// total number of elements in the tensor
u64 ts_numel(const Tensor *t);

// true if all elements are laid out contiguously in memory
bool ts_is_contiguous(const Tensor *t);

// collapse compatible dimensions
// e.g. 3-way tensor of shape [3,4,5] with natural strides -> 1-way tensor with
// 60 elements
u64 ts_collapse_strides(u64 rank, const u64 *shape, const s64 *strides,
                        u64 *out_shape, s64 *out_strides);

// broadcast two shapes, returns false if incompatible
bool ts_broadcast_shapes(u64  order_a, const u64 *shape_a,
                         u64  order_b, const u64 *shape_b,
                         u64 *out_order,     u64 *out_shape);

// Allocate a new tensor; data is uninitialized
// Tensor tensor_alloc(Allocator a, DType dtype, usize rank, const usize *shape);
 
// Common fill constructors
Tensor tensor_zeros(Allocator a, TensorDType dtype, u64 rank, const u64 *shape);
Tensor tensor_ones (Allocator a, TensorDType dtype, u64 rank, const u64 *shape);
Tensor tensor_full (Allocator a, TensorDType dtype, u64 rank, const u64 *shape, f64 val);

// 1D range: [start, stop) with step — like np.arange
Tensor tensor_arange(Allocator a, TensorDType dtype, f64 start, f64 stop, f64 step);

// 1D evenly spaced: like np.linspace
Tensor tensor_linspace(Allocator a, f64 start, f64 stop, u64 n);

// Wrap existing data — does NOT take ownership
Tensor tensor_from_data(void *data, TensorDType dtype, u64 rank,
                         const u64 *shape, const s64 *strides);

// Deep copy
Tensor tensor_clone(Allocator a, const Tensor *src);

void tensor_free(Tensor *t);  // only frees if TENSOR_FLAG_OWNS_DATA]
 
// Transpose last two dims (matrix transpose) — just swap strides
Tensor tensor_T(const Tensor *t);
 
// Arbitrary axis permutation: t.shape [2,3,4] → permute [2,0,1] → [4,2,3]
Tensor tensor_permute(const Tensor *t, const u64 *axes);
 
// Reshape — only works on contiguous tensors without copy; else error or clone
Tensor tensor_reshape(Allocator a, const Tensor *t, u64 rank, const u64 *shape);
 
// Insert/remove size-1 dimensions
Tensor tensor_unsqueeze(const Tensor *t, u64 axis);
Tensor tensor_squeeze(const Tensor *t, u64 axis);  // axis must be size 1
 
// Slice: like t[lo:hi:step] along one axis — adjusts data ptr + stride
Tensor tensor_slice(const Tensor *t, u64 axis, s64 lo, s64 hi, s64 step);
 
// Select a single index along an axis → drops that dimension
Tensor tensor_index(const Tensor *t, u64 axis, s64 idx);

// Flip along axis (negate stride, adjust data ptr)
Tensor tensor_flip(const Tensor *t, u64 axis);
 
// Expand a size-1 dimension without copying (stride = 0 trick)
Tensor tensor_expand(const Tensor *t, u64 axis, u64 new_size);
 
// Return a contiguous copy if not already contiguous; else return view
Tensor tensor_contiguous(Allocator a, const Tensor *t);
 
// Collapse to 1D view
Tensor tensor_flatten(Allocator a, const Tensor *t);
