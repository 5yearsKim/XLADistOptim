from itertools import pairwise

import pytest

from simulator import InfeasibleScheduleError, simulate_pipeline


def test_three_tile_double_buffer_schedule_by_hand() -> None:
    result = simulate_pipeline(
        tile_count=3,
        producer_us=2,
        communication_us=3,
        consumer_us=4,
        tile_bytes=100,
        buffer_count=2,
        memory_budget_bytes=200,
    )

    assert result.makespan_us == 18
    assert result.peak_memory_bytes == 200
    assert [
        [(event.start_us, event.end_us) for event in result.events_for(tile)]
        for tile in range(3)
    ] == [
        [(0, 2), (2, 5), (5, 9)],
        [(2, 4), (5, 8), (9, 13)],
        [(9, 11), (11, 14), (14, 18)],
    ]
    assert [(life.buffer, life.start_us, life.end_us) for life in result.buffer_lifetimes] == [
        (0, 0, 9),
        (1, 2, 13),
        (0, 9, 18),
    ]


def test_single_buffer_serializes_entire_tiles() -> None:
    result = simulate_pipeline(
        tile_count=3,
        producer_us=2,
        communication_us=3,
        consumer_us=4,
        tile_bytes=100,
        buffer_count=1,
        memory_budget_bytes=100,
    )

    assert result.makespan_us == 27
    assert result.peak_memory_bytes == 100
    assert [life.start_us for life in result.buffer_lifetimes] == [0, 9, 18]


def test_stage_resources_are_serialized_and_dependencies_hold() -> None:
    result = simulate_pipeline(5, 1, 4, 2, 32, 3, 96)

    for tile in range(5):
        producer, communication, consumer = result.events_for(tile)
        assert producer.end_us <= communication.start_us
        assert communication.end_us <= consumer.start_us

    for stage in ("producer", "communication", "consumer"):
        stage_events = [event for event in result.events if event.stage == stage]
        for previous, current in pairwise(stage_events):
            assert previous.end_us <= current.start_us


def test_rejects_schedule_over_memory_budget() -> None:
    with pytest.raises(InfeasibleScheduleError, match="requires 200 bytes"):
        simulate_pipeline(3, 2, 3, 4, 100, 2, 199)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("tile_count", 0),
        ("producer_us", -1),
        ("communication_us", float("inf")),
        ("tile_bytes", 0),
        ("buffer_count", 0),
        ("memory_budget_bytes", -1),
    ],
)
def test_rejects_invalid_inputs(argument: str, value: object) -> None:
    arguments = {
        "tile_count": 3,
        "producer_us": 2,
        "communication_us": 3,
        "consumer_us": 4,
        "tile_bytes": 100,
        "buffer_count": 2,
        "memory_budget_bytes": 200,
    }
    arguments[argument] = value

    with pytest.raises((TypeError, ValueError)):
        simulate_pipeline(**arguments)
