#include "tensor.h"

typedef struct {
  u64 size, align;
  const char *name;
} TensorDTypeMeta;

static const TensorDTypeMeta DTYPE_META[DTYPE_COUNT] = {
  [DTYPE_U8]    = { sizeof( u8), _Alignof( u8),    "u8"},
  [DTYPE_U16]   = { sizeof(u16), _Alignof(u16),   "u16"},
  [DTYPE_U32]   = { sizeof(u32), _Alignof(u32),   "u32"},
  [DTYPE_U64]   = { sizeof(u64), _Alignof(u64),   "u64"},
  [DTYPE_S8]    = { sizeof( s8), _Alignof( s8),    "s8"},
  [DTYPE_S16]   = { sizeof(s16), _Alignof(s16),   "s16"},
  [DTYPE_S32]   = { sizeof(s32), _Alignof(s32),   "s32"},
  [DTYPE_S64]   = { sizeof(s64), _Alignof(s64),   "s64"},
  [DTYPE_F32]   = { sizeof(f32), _Alignof(f32),   "f32"},
  [DTYPE_F64]   = { sizeof(f64), _Alignof(f64),   "f64"},
  [DTYPE_BOOL]  = { sizeof( b8), _Alignof( b8),    "b8"},
};

#define tensor_dtype_size(d) (DTYPE_META[(d)].size)
#define tensor_dtype_align(d) (DTYPE_META[(d)].align)
#define tensor_dtype_name(d) (DTYPE_META[(d)].name)
