"""Trace tapes and the active trace stack (§4 of the spec).

A `Tape` is just a list of `TapeEntry`s recorded in chronological order. The
order matters: the reverse pass walks the tape *backwards*. Every entry stores
everything the corresponding primitive's `vjp` will need — including a deep
copy of the input values from the forward pass — so the reverse pass never
re-runs forward code.

`TraceStack` is the global stack of currently-active tapes. Transforms push a
fresh tape when they begin and pop it when they finish. `_emit` (see
`primitives.py`) reads this stack to decide whether the current operation
should be recorded, and on which tapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .array import Array
    from .primitives import Primitive


@dataclass
class TapeEntry:
    """One row of a tape: enough state to run the primitive's VJP later."""

    prim: "Primitive"
    input_ids: tuple[int, ...]
    # Deep copies of input values at the moment of the forward pass (§4.2).
    # The reverse pass might run much later — by then the original Arrays may
    # have been overwritten or freed.
    input_vals: tuple[np.ndarray, ...]
    # The batch axis (if any) carried by each input at forward time. We need
    # this so the reverse pass can re-tag the primals when reconstructing
    # them — otherwise VJP operations under `vmap` lose their batch axis.
    input_batch_dims: tuple[int | None, ...]
    output_id: int
    output_shape: tuple[int, ...]
    # Optional bag of primitive-specific parameters (e.g. axis for `sum`).
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tape:
    """A list of `TapeEntry`s recorded while a transform is active."""

    level: int
    entries: list[TapeEntry] = field(default_factory=list)

    def record(
        self,
        prim: "Primitive",
        inputs: list["Array"],
        output: "Array",
        params: dict[str, Any] | None = None,
    ) -> None:
        self.entries.append(
            TapeEntry(
                prim=prim,
                input_ids=tuple(x.id for x in inputs),
                input_vals=tuple(np.copy(x.data) for x in inputs),
                input_batch_dims=tuple(x.batch_dim for x in inputs),
                output_id=output.id,
                output_shape=tuple(output.shape),
                params=dict(params) if params else {},
            )
        )

    def __iter__(self):
        return iter(self.entries)

    def __reversed__(self):
        return reversed(self.entries)


# --- the global trace stack --------------------------------------------------
# A python list used as a stack. Position in the list == trace level.
# Level 0 is reserved for "no active tape — operations run eagerly".
_TRACE_STACK: list[Tape] = []


def push_tape() -> Tape:
    """Begin a new trace level. Returns the freshly pushed tape."""
    # The new tape's level is the position it occupies. With one tape on the
    # stack, level becomes 1 — leaving 0 for plain concrete values.
    new_level = len(_TRACE_STACK) + 1
    tape = Tape(level=new_level)
    _TRACE_STACK.append(tape)
    return tape


def pop_tape() -> Tape:
    """End the most recently pushed trace. Returns the popped tape."""
    return _TRACE_STACK.pop()


def active_tapes() -> list[Tape]:
    """All tapes currently on the stack, outer-most first."""
    return list(_TRACE_STACK)


def current_level() -> int:
    """The level a newly-emitted operation would be tagged with if all inputs
    were concrete. Equals the level of the innermost active tape, or 0."""
    return len(_TRACE_STACK)
