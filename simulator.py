"""Discrete-event model for a tiled producer/communication/consumer pipeline.

The model intentionally has no JAX or backend dependencies.  Each stage owns a
single serial resource, while ``buffer_count`` limits how many tiles may be in
flight.  A tile holds its buffer from the start of production until the end of
consumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Stage = Literal["producer", "communication", "consumer"]


class InfeasibleScheduleError(ValueError):
    """Raised when the requested buffers do not fit in the memory budget."""


@dataclass(frozen=True)
class ScheduleEvent:
    tile: int
    stage: Stage
    resource: Stage
    buffer: int
    start_us: float
    end_us: float

    @property
    def duration_us(self) -> float:
        return self.end_us - self.start_us


@dataclass(frozen=True)
class BufferLifetime:
    tile: int
    buffer: int
    start_us: float
    end_us: float


@dataclass(frozen=True)
class ScheduleResult:
    makespan_us: float
    peak_memory_bytes: int
    events: tuple[ScheduleEvent, ...]
    buffer_lifetimes: tuple[BufferLifetime, ...]

    def events_for(self, tile: int) -> tuple[ScheduleEvent, ...]:
        """Return one tile's events in dependency order."""
        return tuple(event for event in self.events if event.tile == tile)


def _validate_inputs(
    tile_count: int,
    producer_us: float,
    communication_us: float,
    consumer_us: float,
    tile_bytes: int,
    buffer_count: int,
    memory_budget_bytes: int,
) -> None:
    integer_values = {
        "tile_count": tile_count,
        "tile_bytes": tile_bytes,
        "buffer_count": buffer_count,
        "memory_budget_bytes": memory_budget_bytes,
    }
    for name, value in integer_values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")

    if tile_count <= 0:
        raise ValueError("tile_count must be positive")
    if tile_bytes <= 0:
        raise ValueError("tile_bytes must be positive")
    if buffer_count <= 0:
        raise ValueError("buffer_count must be positive")
    if memory_budget_bytes < 0:
        raise ValueError("memory_budget_bytes cannot be negative")

    for name, value in {
        "producer_us": producer_us,
        "communication_us": communication_us,
        "consumer_us": consumer_us,
    }.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number")
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and non-negative")


def simulate_pipeline(
    tile_count: int,
    producer_us: float,
    communication_us: float,
    consumer_us: float,
    tile_bytes: int,
    buffer_count: int,
    memory_budget_bytes: int,
) -> ScheduleResult:
    """Construct the earliest FIFO schedule for a three-stage tiled pipeline.

    The producer, communication stage, and consumer each serialize their own
    work but may overlap with the other stages.  Buffers are reused only after
    the corresponding consumer event completes.

    Raises:
        InfeasibleScheduleError: if the requested buffer allocation exceeds the
            memory budget.
    """
    _validate_inputs(
        tile_count,
        producer_us,
        communication_us,
        consumer_us,
        tile_bytes,
        buffer_count,
        memory_budget_bytes,
    )

    allocated_buffers = min(tile_count, buffer_count)
    peak_memory_bytes = allocated_buffers * tile_bytes
    if peak_memory_bytes > memory_budget_bytes:
        raise InfeasibleScheduleError(
            "pipeline requires "
            f"{peak_memory_bytes} bytes for {allocated_buffers} buffers, but "
            f"the budget is {memory_budget_bytes} bytes"
        )

    producer_available = 0.0
    communication_available = 0.0
    consumer_available = 0.0
    # (time at which the buffer can be reused, stable buffer identifier)
    buffers = [(0.0, buffer) for buffer in range(allocated_buffers)]
    events: list[ScheduleEvent] = []
    lifetimes: list[BufferLifetime] = []

    for tile in range(tile_count):
        buffer_available, buffer = min(buffers)

        producer_start = max(producer_available, buffer_available)
        producer_end = producer_start + producer_us
        communication_start = max(producer_end, communication_available)
        communication_end = communication_start + communication_us
        consumer_start = max(communication_end, consumer_available)
        consumer_end = consumer_start + consumer_us

        events.extend(
            (
                ScheduleEvent(
                    tile, "producer", "producer", buffer,
                    producer_start, producer_end,
                ),
                ScheduleEvent(
                    tile, "communication", "communication", buffer,
                    communication_start, communication_end,
                ),
                ScheduleEvent(
                    tile, "consumer", "consumer", buffer,
                    consumer_start, consumer_end,
                ),
            )
        )
        lifetimes.append(
            BufferLifetime(tile, buffer, producer_start, consumer_end)
        )

        producer_available = producer_end
        communication_available = communication_end
        consumer_available = consumer_end
        buffers[buffer] = (consumer_end, buffer)

    return ScheduleResult(
        makespan_us=consumer_available,
        peak_memory_bytes=peak_memory_bytes,
        events=tuple(events),
        buffer_lifetimes=tuple(lifetimes),
    )
