import random
from typing import Tuple, Optional
from pydantic import BaseModel, Field


class MemoryElement(BaseModel):
    descriptor: Tuple[str, ...]
    init_prompt: str
    mutated_prompt: str
    score: Optional[float] = Field(default=0.0, title="Score of the memory element")


class Memory:
    def __init__(self, max_size=100):
        self.max_size = max_size
        self.memory = []

    def add(self, element: MemoryElement):
        self.memory.append(element)
        self.memory.sort(key=lambda x: x.score, reverse=True)

        # In case the memory is full, and all elements have the same max score
        # shuffle the elements to avoid always removing last element
        if len(self.memory) > self.max_size:
            random.shuffle(self.memory)

        self.memory = self.memory[: self.max_size]

    def get(self):
        return self.memory

    def clear(self):
        self.memory = []

    def retrieve(self, num_elements=3, seed=0):
        """Retrieve random `num_elements` elements from memory."""
        if num_elements > len(self.memory):
            return self.memory

        random.seed(seed)
        return random.sample(self.memory, num_elements)
