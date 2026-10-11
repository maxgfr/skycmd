"""Tiny System One decision models: typed choice / score / yes-no probabilities in one forward pass."""

import os
from typing import Any, Literal, Mapping, Sequence, Union

__version__: str

Source = Union[str, "os.PathLike[str]", bytes, bytearray]
QuestionType = Literal["choice", "score", "noul"]

class Model:
    """A loaded decision model (GGUF: s1 `modern-bert` or s2 `skycmd-s2`).

    `Model(source)` takes a file path (`str` or `os.PathLike`) or the file content (`bytes`,
    `bytearray`). Calls are thread-safe and release the GIL while the model runs.
    """

    def __init__(self, source: Source) -> None: ...
    def decide(self, request: Union[Mapping[str, Any], str]) -> dict[str, Any]:
        """Answer a System One request (the `POST /v1/systemone` body, a dict or JSON text).

        Returns `{"model", "answers", "usage"}`.
        """
    def decide_json(self, request: str) -> str:
        """Same as `decide` with JSON text in and out."""
    def decide_raw(
        self, qtype: QuestionType, instructions: str, options: Sequence[str], state: str
    ) -> list[float]:
        """Probabilities (after the temperature) of already rendered option texts, in order."""
    @property
    def info(self) -> dict[str, Any]:
        """Model description (architecture sizes, temperatures, ...)."""
    @property
    def name(self) -> str: ...
    @property
    def n_params(self) -> int: ...
    @property
    def path(self) -> Union[str, None]:
        """File the model was loaded from, or None when loaded from bytes."""

def load(source: Source) -> Model:
    """Same as `Model(source)`."""
