from typing import List, Dict
from pathlib import Path
import ast
import json
from rainbowxplus.components.memory import MemoryElement


class ExamplesProcessor:
    """Processes and formats examples for prompt generation."""

    def __init__(
        self,
        config_path: str | Path,
        dimension_names: List[str],
        descriptions: Dict[str, str],
    ):
        """
        Initialize the examples processor.

        Args:
            config_path: Path to the examples JSON file
            dimension_names: List of dimension names for categorization
            descriptions: Dictionary mapping categories to their descriptions
        """
        self.dimension_names = dimension_names
        self.descriptions = descriptions
        self.examples = self._load_examples(config_path)

    def _load_examples(self, config_path: str | Path) -> List[MemoryElement]:
        """
        Load and parse examples from the JSON file.

        Args:
            config_path: Path to the examples JSON file

        Returns:
            List of MemoryElement objects
        """
        try:
            with open(config_path) as f:
                raw_examples = json.load(f)

            # Convert string tuples to actual tuples and create MemoryElement objects
            processed_examples = [
                MemoryElement(
                    descriptor=ast.literal_eval(example["descriptor"]),
                    init_prompt=example["init_prompt"],
                    mutated_prompt=example["mutated_prompt"],
                )
                for example in raw_examples
            ]

            return processed_examples

        except (json.JSONDecodeError, FileNotFoundError) as e:
            raise RuntimeError(f"Failed to load examples from {config_path}: {str(e)}")

    def create_examples_prompt(self) -> str:
        """
        Generate formatted examples string with descriptions.

        Returns:
            Formatted string containing all examples with their descriptions
        """
        return self.format_examples_from_list(
            self.examples, self.dimension_names, self.descriptions
        )

    @staticmethod
    def format_example(
        example: MemoryElement, dimension_names: List[str], descriptions: Dict[str, str]
    ) -> str:
        """
        Format a single MemoryElement object into a structured string.

        Args:
            example: MemoryElement object to format
            dimension_names: List of dimension names for categorization
            descriptions: Dictionary mapping categories to their descriptions

        Returns:
            Formatted string containing the example with its description

        Raises:
            ValueError: If dimension names don't match descriptor length
        """
        if len(example.descriptor) != len(dimension_names):
            raise ValueError(
                f"Dimension names and descriptors do not match: "
                f"Expected {len(dimension_names)} dimensions, got {len(example.descriptor)}"
            )

        example_parts = []

        # Add descriptors with descriptions if available
        for name, descriptor in zip(dimension_names, example.descriptor):
            if descriptor in descriptions and descriptions[descriptor]:
                example_parts.append(
                    f"- {name}: {descriptor} - {descriptions[descriptor]}"
                )
            else:
                example_parts.append(f"- {name}: {descriptor}")
            # example_parts.append(f"- {name}: {descriptor}")

        # Add prompts
        example_parts.extend(
            [
                f"- Input prompt: {example.init_prompt}",
                f"- New prompt: {example.mutated_prompt}\n",
            ]
        )

        return "\n".join(example_parts)

    @staticmethod
    def format_examples_from_list(
        examples: List[MemoryElement],
        dimension_names: List[str],
        descriptions: Dict[str, str],
    ) -> str:
        """
        Format a list of MemoryElement objects into a structured string.

        Args:
            examples: List of MemoryElement objects to format
            dimension_names: List of dimension names for categorization
            descriptions: Dictionary mapping categories to their descriptions

        Returns:
            Formatted string containing all examples with their descriptions

        Raises:
            ValueError: If dimension names don't match descriptor length
        """
        formatted_parts = []

        # Add examples
        for idx, example in enumerate(examples, 1):
            if len(example.descriptor) != len(dimension_names):
                raise ValueError(
                    f"Dimension names and descriptors do not match for example {idx}: "
                    f"Expected {len(dimension_names)} dimensions, got {len(example.descriptor)}"
                )

            example_parts = [f"\nExample {idx}"]
            example_parts.append(
                ExamplesProcessor.format_example(example, dimension_names, descriptions)
            )
            formatted_parts.append("\n".join(example_parts))

        return "\n".join(formatted_parts)
