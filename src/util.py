from typing import Sequence, Generator


def flatten(x: Sequence) -> Generator:
    """Recursively flattens arbitrarily nested sequences"""
    for item in x:
        if isinstance(item, list):
            yield from flatten(item)
        else:
            yield item
