from collections.abc import Sequence
from typing import Any

class SessionOptions:
    def __init__(self) -> None: ...

class NodeArg:
    name: str
    shape: list[int]
    type: str

class InferenceSession:
    def __init__(
        self,
        path_or_bytes: str | bytes,
        sess_options: SessionOptions | None = None,
        providers: Sequence[str] | None = None,
    ) -> None: ...
    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, Any],
        run_options: object = None,
    ) -> list[Any]: ...
    def get_inputs(self) -> list[NodeArg]: ...
    def get_outputs(self) -> list[NodeArg]: ...
    def set_providers(self, providers: Sequence[str]) -> None: ...
