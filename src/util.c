#include "util.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include <stdlib.h>
#include <time.h>

/* === LOGGING === */
static bool use_color = false;

void log_init() {
#ifdef _WIN32
  // activate virtual terminal sequences (for color printing)
  HANDLE hconsole = GetStdHandle(STD_OUTPUT_HANDLE);
  if (hconsole == null) {
    return;
  }
  DWORD cur_mode;
  bool ok = GetConsoleMode(hconsole, &cur_mode);
  if (!ok) {
    printf("ERROR: failed to get console mode!\n");
    return;
  }

  ok = SetConsoleMode(hconsole, cur_mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING |
                                    ENABLE_VIRTUAL_TERMINAL_PROCESSING);
  if (!ok) {
    printf("ERROR: failed to set console mode!");
    return;
  }
#endif
  use_color = true;
}

const char *colors[] = {"\x1b[90m", "\x1b[32m", "\x1b[36m",
                        "\x1b[33m", "\x1b[31m", "\x1b[91m"};

const char *log_levels[] = {
    " DEBUG ", " INFO ", " WARN ", " ERROR ", " FATAL ",
};

void log_set_color(colors_e color) {
  if (!use_color)
    return;
  printf("%s", colors[color]);
}

void log_set_bold() {
  if (!use_color)
    return;
  printf("\x1b[1m");
}

void log_reset_bold() {
  if (!use_color)
    return;
  printf("\x1b[22m");
}

void log_set_underline() {
  if (!use_color)
    return;
  printf("\x1b[4m");
}

void log_reset_underline() {
  if (!use_color)
    return;
  printf("\x1b[24m");
}

void log_reset(void) {
  if (!use_color)
    return;
  printf("\x1b[0m");
}

void log_print_time() {
  char buf[50];
  time_t raw_time;
  time(&raw_time);
  struct tm info;
  localtime_s(&info, &raw_time);
  strftime(buf, 49, "%X", &info);
  printf("%s", buf);
}

void _log_(char *file, int line, log_level_e level, char *fmt, ...) {
  log_set_color(COLOR_GREY);
  log_print_time();

  log_set_color(level + 1);
  log_set_bold();
  printf("%s", log_levels[level]);
  log_reset_bold();

  log_set_color(COLOR_GREY);
  printf("%s:%d ", file, line);
  log_reset();

  va_list args;
  va_start(args, fmt);
  vprintf(fmt, args);
  va_end(args);

  printf("\n");
}

/* === ALLOCATORS === */
// wraps malloc/free
void *allocator_malloc_wrapper(void *ctx, u64 size, u64 align) {
  (void)align;
  (void)ctx;
  return malloc(size);
}
void *allocator_realloc_wrapper(void *ctx, void *ptr, u64 new_size,
                                u64 old_size, u64 align) {
  (void)old_size;
  (void)align;
  (void)ctx;
  return realloc(ptr, new_size);
}

void allocator_free_wrapper(void *ctx, void *ptr, u64 size, u64 align) {
  (void)size;
  (void)align;
  (void)ctx;
  return free(ptr);
}

Allocator allocator_heap(void) {
  return (Allocator){
      .alloc = allocator_malloc_wrapper,
      .realloc = allocator_realloc_wrapper,
      .free = allocator_free_wrapper,
      .ctx = null,
  };
}

// asserts on alloc
void *allocator_nil_alloc(void *ctx, u64 size, u64 align) {
  (void)align;
  (void)ctx;
  (void)size;
  log_fatal("Allocation on nil allocator");
  abort();
}

void *allocator_nil_realloc(void *ctx, void *ptr, u64 new_size, u64 old_size,
                            u64 align) {
  (void)old_size;
  (void)align;
  (void)ctx;
  log_fatal("Reallocation on nil allocator");
  abort();
}

void allocator_nil_free(void *ctx, void *ptr, u64 size, u64 align) {
  (void)size;
  (void)align;
  (void)ctx;
  log_fatal("Free on nil allocator");
  abort();
}

Allocator allocator_nil(void) {
  return (Allocator){.alloc = allocator_nil_alloc,
                     .realloc = allocator_nil_realloc,
                     .free = allocator_nil_free,
                     .ctx = null};
}

// wraps another, logs calls
void *allocator_log_alloc(void *ctx, u64 size, u64 align) {
  log_debug("Allocation of size %llu", size);
  Allocator *a = (Allocator *)ctx;
  return a->alloc(a->ctx, size, align);
}

void *allocator_log_realloc(void *ctx, void *ptr, u64 new_size, u64 old_size,
                            u64 align) {
  log_debug("Reallocation: %llu -> %llu", old_size, new_size);
  Allocator *a = (Allocator *)ctx;
  return a->realloc(a->ctx, ptr, new_size, old_size, align);
}

void allocator_log_free(void *ctx, void *ptr, u64 size, u64 align) {
  log_debug("Free of size %llu", size);
  Allocator *a = (Allocator *)ctx;
  return a->free(a->ctx, ptr, size, align);
}

Allocator allocator_log(Allocator *inner) {
  return (Allocator){.alloc = allocator_log_alloc,
                     .realloc = allocator_log_realloc,
                     .free = allocator_log_free,
                     .ctx = &inner};
}

// Arena allocator
Allocator allocator_arena(Arena *a) {
  return (Allocator){
      .alloc = (AllocatorAllocFn)arena_alloc,
      .realloc = (AllocatorReallocFn)arena_realloc,
      .free = (AllocatorFreeFn)arena_free,
  };
}

static void arena_push_chunk(Arena *a) {
  // allocate a new chunk
  ArenaChunk *prev_chunk = a->head;
  a->head = mem_alloc(a->backing, ArenaChunk);
  a->head->prev = prev_chunk;
  a->head->data = a->backing.alloc(a->backing.ctx, a->chunk_size, 8);
  a->head->cur = a->head->data;
  a->head->last_section = null;
}

static void arena_pop_chunk(Arena* a) {
  ArenaChunk* c = a->head;
  a->head = c->prev;
  a->backing.free(a->backing.ctx, c->data, a->chunk_size, 8);
  mem_free(a->backing, c, ArenaChunk);
}

Arena arena_make(u64 initial_cap, Allocator backing) {
  Arena result;
  result.backing = backing;
  result.chunk_size = initial_cap;
  result.head = null;
  arena_push_chunk(&result);
  return result;
}

void *arena_alloc(Arena *a, u64 size, u64 align) {
  u64 padding = -((u64)a->head->cur) & (align - 1);
  u64 used = ((u64)a->head->cur - (u64)a->head->data);
  u64 available = a->chunk_size - used - padding;

  if (available < 0) {
    arena_push_chunk(a);
  }

  a->head->cur += padding;
  void *p = a->head->cur;
  a->head->cur += size;
  return memset(p, 0, size);
}

void arena_free(Arena *a, void *ptr, u64 old_size, u64 old_align) {
  u64 padding = -((u64)a->head->cur) & (old_align - 1);
  u8 *last_alloc = a->head->cur - old_size - padding;
  if (last_alloc == ptr && last_alloc >= a->head->cur) {
    // free last
    a->head->cur = ptr;
  }
  // else do nothing
}

void *arena_realloc(Arena *a, void *ptr, u64 new_size, u64 old_size,
                    u64 align) {
  // check if it was the last allocation
  u64 padding = -((u64)a->head->cur) & (align - 1);
  u8 *last_alloc = a->head->cur - old_size - padding;

  // check if there is enough space left
  u64 used = ((u64)a->head->cur - (u64)a->head->data);
  u64 available = a->chunk_size - used - padding;

  // if we can extend the last allocation, we need less new mem
  // and can skip the memcpy
  if (last_alloc == ptr && available >= new_size - old_size) {
    return ptr;
  }

  if (available < new_size) {
    // new chunk
    arena_push_chunk(a);
  }

  void *p = arena_alloc(a, new_size, align);
  memcpy_s(p, new_size, ptr, old_size);
  return p;
}

void arena_push_section(Arena *a) {
  ArenaSectionMarker *mark =
      arena_alloc(a, sizeof(ArenaSectionMarker), _Alignof(ArenaSectionMarker));
  mark->prev = a->head->last_section;
  a->head->last_section = mark;
}

void arena_pop_section(Arena *a) {
  if (a->head->last_section) {
    a->head->cur = (u8*)a->head->last_section;
    a->head->last_section = a->head->last_section->prev;
  }
}

// resets the arena without freeing memory
void arena_reset(Arena *a) {
  if (!a->head) return;
  
  while (a->head->prev) {
    arena_pop_chunk(a);
  }
}   

void arena_destroy(Arena *a) {
  while (a->head) {
    arena_pop_chunk(a);
  }
}
