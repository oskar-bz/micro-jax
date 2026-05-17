#pragma once
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define null NULL
#define reinterpret(type, val) (*(type *)&val)
#define ptr_at(ptr, element_size, at) ((ptr) + (element_size) * (at))
#define ptr_advance(ptr, by) (typeof(ptr))((u64)(ptr) + by)
#define min(a, b) (a) < (b) ? (a) : (b)
#define max(a, b) (a) > (b) ? (a) : (b)

#define for_to(i, to) for (s64 i = 0; i < to; ++i)
#define from_to(i, from, to) for (s64 i = from; i < to; ++i)

#define defer_loop(begin, end) for (s32 _i = ((begin), 0); !_i; ++_i, (end))
#define defer(fn) __attribute__(cleanup(fn))

#define assert(c)                                                              \
  while (!(c))                                                                 \
  __builtin_unreachable()

typedef char s8;
typedef unsigned char u8;
typedef short s16;
typedef unsigned short u16;
typedef int s32;
typedef unsigned int u32;
typedef long long s64;
typedef unsigned long long u64;
typedef float f32;
typedef double f64;
typedef bool b8;
typedef s32 b32;

typedef struct Allocator Allocator;
typedef struct Arena Arena;
typedef struct ArenaChunk ArenaChunk;
typedef struct ArenaSectionMarker ArenaSectionMarker;

/* === LOGGING === */
#define log_debug(...) _log_(__FILE__, __LINE__, LOG_DEBUG, __VA_ARGS__)
#define log_info(...) _log_(__FILE__, __LINE__, LOG_INFO, __VA_ARGS__)
#define log_warn(...) _log_(__FILE__, __LINE__, LOG_WARN, __VA_ARGS__)
#define log_error(...) _log_(__FILE__, __LINE__, LOG_ERROR, __VA_ARGS__)
#define log_fatal(...) _log_(__FILE__, __LINE__, LOG_FATAL, __VA_ARGS__)

typedef enum {
  LOG_DEBUG,
  LOG_INFO,
  LOG_WARN,
  LOG_ERROR,
  LOG_FATAL,
} log_level_e;

typedef enum {
  COLOR_GREY = 0,
  COLOR_GREEN,
  COLOR_CYAN,
  COLOR_YELLOW,
  COLOR_RED,
} colors_e;

void log_init();
void log_set_color(colors_e color);
void log_set_bold(void);
void log_reset_bold(void);
void log_set_underline(void);
void log_reset_underline(void);
void log_reset(void);
void log_print_time(void);
void _log_(char *file, int line, log_level_e level, char *fmt, ...);

/* === ALLOCATORS === */
typedef void *(*AllocatorAllocFn)(void *, u64, u64);
typedef void *(*AllocatorReallocFn)(void *, void *, u64, u64, u64);
typedef void (*AllocatorFreeFn)(void *, void *, u64, u64);

struct Allocator {
  AllocatorAllocFn alloc;
  AllocatorReallocFn realloc;
  AllocatorFreeFn free;
  void *ctx;
};

// helper macros
#define mem_alloc(a, T) (T *)(a).alloc((a).ctx, sizeof(T), _Alignof(T))
#define mem_alloc_n(a, T, n) (T *)(a).alloc((a).ctx, sizeof(T) * n, _Alignof(T))
#define mem_free(a, p, T) (a).free((a).ctx, p, sizeof(T), _Alignof(T))

// pre-defined allocators
Allocator allocator_heap(void);            // wraps malloc/free
Allocator allocator_nil(void);             // asserts on allocs
Allocator allocator_log(Allocator *inner); // wraps another, logs calls

// arena allocator
struct ArenaChunk {
  ArenaChunk *prev;
  u8 *data;
  u8 *cur;
  ArenaSectionMarker* last_section;
};

struct ArenaSectionMarker {
  ArenaSectionMarker* prev;
};

struct Arena {
  ArenaChunk *head;
  Allocator backing;
  u64 chunk_size;
};

Allocator allocator_arena(Arena *a); // bump pointer, free is no-op

Arena arena_make(u64 initial_cap, Allocator backing);

// for API compatability
void *arena_alloc(Arena *a, u64 size, u64 align);
void *arena_realloc(Arena *a, void *ptr, u64 new_size, u64 old_size, u64 align);
void arena_free(Arena *a, void *ptr, u64 old_size, u64 old_align);

void arena_push_section(Arena *a);
void arena_pop_section(Arena *a);

void arena_reset(Arena *a);   // resets the arena without freeing memory
void arena_destroy(Arena *a); // frees the arena
