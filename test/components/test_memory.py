from rainbowxplus.components import Memory, MemoryElement


def test_memory():
    memory = Memory(4)
    elements = [
        MemoryElement(descriptor=("a", "b"), init_prompt="a", mutated_prompt="b"),
        MemoryElement(
            descriptor=("c", "d"), init_prompt="c", mutated_prompt="d", score=0.3
        ),
        MemoryElement(
            descriptor=("e", "f"), init_prompt="e", mutated_prompt="f", score=0.2
        ),
        MemoryElement(
            descriptor=("g", "h"), init_prompt="g", mutated_prompt="h", score=0.4
        ),
    ]

    for element in elements:
        memory.add(element)

    # Check if memory is sorted by score
    assert len(memory.get()) == 4
    assert memory.get()[0].score == 0.4

    # Check if memory is capped at max_size
    memory.add(
        MemoryElement(
            descriptor=("i", "j"), init_prompt="i", mutated_prompt="j", score=0.5
        )
    )
    assert len(memory.get()) == 4
    assert memory.get()[0].score == 0.5
    assert memory.get()[3].score == 0.2

    # Check if memory is cleared
    memory.clear()
    assert len(memory.get()) == 0
