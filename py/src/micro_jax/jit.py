"""`jit` — abstract tracing, three classical optimization passes, and execution.

A jit'd function works in two stages:

1. **Tracing** — on the first call (and again whenever inputs of a new shape
   signature appear), `f` is run with placeholder `Array` inputs. A
   `JitRecorder` registered as the active `AbstractRecorder` intercepts every
   `_emit` call and writes an `AbstractEntry` to its tape. Concrete values
   are never computed during this phase; only shapes flow.

2. **Optimizing & executing** — the recorded `AbstractTape` is run through
   three passes in the order prescribed by the spec (§5.2): dead-code
   elimination, constant folding, common-subexpression elimination. The
   resulting `CompiledProgram` is cached. Subsequent calls with the same
   shape signature skip straight to execution.

The cache keeps the python overhead constant on hot paths even though the
arithmetic still runs through numpy. (We're not generating machine code —
this is "micro" jit. The point is the program transformations, not codegen.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .array import Array, _new_id, ensure_array
from .primitives import (
    AbstractRecorder,
    Primitive,
    pop_abstract_recorder,
    push_abstract_recorder,
)


@dataclass
class AbstractEntry:
    """One operation on an abstract tape — analogous to TapeEntry, but
    without primal values (jit has none to save)."""

    prim: Primitive
    input_ids: tuple[int, ...]
    output_id: int
    output_shape: tuple[int, ...]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompiledProgram:
    """A linear program ready for execution.

    Execution is straightforward: seed a dict with the input arrays' data
    and any baked-in constants, then walk the entries in order.
    """

    input_ids: tuple[int, ...]
    entries: list[AbstractEntry]
    output_id: int
    constants: dict[int, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


class JitRecorder(AbstractRecorder):
    """The hook handed to `_emit` while jit is tracing.

    Anything `_emit` sees with an unknown id is a value that came from
    *outside* the trace (a closure, a default arg) — we snapshot its data
    and treat it as a baked-in constant. Anything else is either an input
    or the output of a previously-recorded entry.
    """

    def __init__(self, input_ids: tuple[int, ...]) -> None:
        self.input_ids = input_ids
        self.known_ids: set[int] = set(input_ids)
        self.constants: dict[int, np.ndarray] = {}
        self.entries: list[AbstractEntry] = []

    def record(
        self,
        prim: Primitive,
        inputs: list[Array],
        params: dict[str, Any],
    ) -> Array:
        # Capture external (closure-time) arrays as constants.
        for inp in inputs:
            if inp.id not in self.known_ids:
                self.constants[inp.id] = np.copy(inp.data)
                self.known_ids.add(inp.id)

        # Shape inference: prefer a primitive's bespoke rule, else run fwd
        # on zero-filled placeholders. The fallback is correct for every
        # built-in because their output shape is purely a function of input
        # shapes and params.
        in_shapes = tuple(inp.shape for inp in inputs)
        if prim.abstract_eval is not None:
            out_shape = tuple(prim.abstract_eval(*in_shapes, **params))
        else:
            dummies = [np.zeros(s, dtype=np.float32) for s in in_shapes]
            out_shape = tuple(prim.fwd(*dummies, **params).shape)

        out = Array.__new__(Array)
        out.data = np.zeros(out_shape, dtype=np.float32)  # placeholder
        out.id = _new_id()
        out.level = 0
        out.batch_dim = None

        self.entries.append(
            AbstractEntry(
                prim=prim,
                input_ids=tuple(inp.id for inp in inputs),
                output_id=out.id,
                output_shape=out_shape,
                params=dict(params),
            )
        )
        self.known_ids.add(out.id)
        return out


def _trace(f: Callable, args: tuple[Array, ...]) -> CompiledProgram:
    """Run `f` once with abstract inputs, returning the recorded program."""
    input_ids = tuple(a.id for a in args)
    recorder = JitRecorder(input_ids)
    push_abstract_recorder(recorder)
    # Shape inference falls back to running `fwd` on zero-filled placeholders
    # (see JitRecorder.record). That can legitimately trigger numpy warnings
    # like "divide by zero in power" for ops such as `reciprocal`. The values
    # never reach the user — silence the noise so the trace is quiet.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        try:
            out = f(*args)
        finally:
            pop_abstract_recorder()

    if not isinstance(out, Array):
        raise TypeError(
            "jit: traced function must return a single Array, "
            f"got {type(out).__name__}"
        )

    # If the function returns one of its inputs unchanged, or returns a
    # value captured from a closure, the output id might not be the result
    # of any entry. The executor handles both cases via the constants and
    # input maps.
    if out.id not in recorder.known_ids:
        recorder.constants[out.id] = np.copy(out.data)

    return CompiledProgram(
        input_ids=input_ids,
        entries=list(recorder.entries),
        output_id=out.id,
        constants=dict(recorder.constants),
    )


# ---------------------------------------------------------------------------
# Optimization passes (§5.2)
# ---------------------------------------------------------------------------


def _dce(program: CompiledProgram) -> CompiledProgram:
    """Remove entries whose output is not reachable from the program output."""
    by_out: dict[int, AbstractEntry] = {e.output_id: e for e in program.entries}
    reachable: set[int] = {program.output_id}
    stack = [program.output_id]
    while stack:
        oid = stack.pop()
        entry = by_out.get(oid)
        if entry is None:
            continue
        for iid in entry.input_ids:
            if iid not in reachable:
                reachable.add(iid)
                stack.append(iid)
    program.entries = [e for e in program.entries if e.output_id in reachable]
    return program


def _params_key(params: dict[str, Any]) -> tuple:
    """Build a stable, hashable key from a primitive's params dict."""
    return tuple(sorted(params.items()))


def _const_fold(program: CompiledProgram) -> CompiledProgram:
    """Evaluate entries whose inputs are all already known constants."""
    constants = dict(program.constants)
    new_entries: list[AbstractEntry] = []
    for entry in program.entries:
        if all(iid in constants for iid in entry.input_ids):
            vals = [constants[iid] for iid in entry.input_ids]
            const_out = entry.prim.fwd(*vals, **entry.params)
            constants[entry.output_id] = np.asarray(const_out, dtype=np.float32)
        else:
            new_entries.append(entry)
    program.entries = new_entries
    program.constants = constants
    return program


def _dedup_constants(program: CompiledProgram) -> CompiledProgram:
    """Collapse byte-identical constants. Const-folding routinely produces
    several constants with the same value (e.g. broadcasts of the same
    scalar); deduping them lets CSE merge their downstream uses too.
    """
    if len(program.constants) < 2:
        return program
    rep_for_sig: dict[tuple, int] = {}
    rewrite: dict[int, int] = {}
    for cid, val in program.constants.items():
        sig = (val.shape, str(val.dtype), val.tobytes())
        rep = rep_for_sig.get(sig)
        if rep is None:
            rep_for_sig[sig] = cid
        else:
            rewrite[cid] = rep
    if not rewrite:
        return program
    program.constants = {
        cid: val for cid, val in program.constants.items() if cid not in rewrite
    }
    new_entries: list[AbstractEntry] = []
    for e in program.entries:
        ids = tuple(rewrite.get(i, i) for i in e.input_ids)
        if ids == e.input_ids:
            new_entries.append(e)
        else:
            new_entries.append(
                AbstractEntry(
                    prim=e.prim,
                    input_ids=ids,
                    output_id=e.output_id,
                    output_shape=e.output_shape,
                    params=e.params,
                )
            )
    program.entries = new_entries
    program.output_id = rewrite.get(program.output_id, program.output_id)
    return program


def _cse(program: CompiledProgram) -> CompiledProgram:
    """Fold duplicate (prim, inputs, params) entries into one."""
    canonical: dict[tuple, int] = {}
    rewrite: dict[int, int] = {}

    def remap(ids: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(rewrite.get(i, i) for i in ids)

    new_entries: list[AbstractEntry] = []
    for entry in program.entries:
        remapped = remap(entry.input_ids)
        key = (entry.prim.name, remapped, _params_key(entry.params))
        existing = canonical.get(key)
        if existing is not None:
            rewrite[entry.output_id] = existing
        else:
            canonical[key] = entry.output_id
            if remapped != entry.input_ids:
                entry = AbstractEntry(
                    prim=entry.prim,
                    input_ids=remapped,
                    output_id=entry.output_id,
                    output_shape=entry.output_shape,
                    params=entry.params,
                )
            new_entries.append(entry)

    program.entries = new_entries
    program.output_id = rewrite.get(program.output_id, program.output_id)
    return program


def _optimize(program: CompiledProgram) -> CompiledProgram:
    # Order: DCE -> const-fold -> dedup-consts -> CSE -> DCE.
    # The spec calls for DCE, const-fold, CSE (§5.2). Deduping constants
    # between fold and CSE is a small extension that pays for itself: it
    # lets CSE collapse downstream ops that only differ in *which* identical
    # constant they reference.
    _dce(program)
    _const_fold(program)
    _dedup_constants(program)
    _cse(program)
    _dce(program)
    return program


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _execute(program: CompiledProgram, inputs: list[Array]) -> Array:
    values: dict[int, np.ndarray] = {
        pid: np.asarray(arr.data, dtype=np.float32)
        for pid, arr in zip(program.input_ids, inputs)
    }
    values.update(program.constants)
    for entry in program.entries:
        in_vals = [values[i] for i in entry.input_ids]
        values[entry.output_id] = np.asarray(
            entry.prim.fwd(*in_vals, **entry.params), dtype=np.float32
        )
    return Array(values[program.output_id])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _shape_signature(args: tuple[Any, ...]) -> tuple:
    sig = []
    for a in args:
        arr = ensure_array(a) if not isinstance(a, Array) else a
        sig.append((arr.ndim, tuple(arr.shape)))
    return tuple(sig)


def jit(f: Callable[..., Array]) -> Callable[..., Array]:
    """Return a function that, on first call per shape signature, traces and
    optimizes `f` and then reuses the compiled program on subsequent calls.

    The trade-off vs. eager is overhead amortization. The first call is more
    expensive (tracing + optimization), every subsequent call with a matching
    shape signature is faster because each python-level _emit/Array
    bookkeeping step is gone.
    """
    cache: dict[tuple, CompiledProgram] = {}

    def wrapped(*args: Any) -> Array:
        sig = _shape_signature(args)
        program = cache.get(sig)
        if program is None:
            # Trace inputs are fresh, shape-correct Arrays. We deliberately
            # do *not* reuse the caller's Arrays here — the traced ids must
            # be unique to this program so the cache works across calls.
            trace_inputs = tuple(
                Array(np.zeros(tuple(ensure_array(a).shape), dtype=np.float32))
                for a in args
            )
            program = _optimize(_trace(f, trace_inputs))
            cache[sig] = program
        concrete = [ensure_array(a) for a in args]
        return _execute(program, concrete)

    # Expose internals — handy for tests and introspection.
    wrapped._cache = cache  # type: ignore[attr-defined]
    return wrapped
