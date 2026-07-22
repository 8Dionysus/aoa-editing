"""Small RFC-6902 subset with generated inverse operations."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from aoa_editing.domain.invariants import validate_timeline
from aoa_editing.domain.models import PatchOperation, Timeline


class PatchError(ValueError):
    """Patch cannot be safely applied to the selected base version."""


def _tokens(pointer: str) -> list[str]:
    if not pointer.startswith("/"):
        raise PatchError("JSON pointer must start with '/'")
    if pointer == "/":
        return [""]
    return [part.replace("~1", "/").replace("~0", "~") for part in pointer[1:].split("/")]


def _parent(document: Any, pointer: str) -> tuple[Any, str]:
    tokens = _tokens(pointer)
    if not tokens:
        raise PatchError("root replacement is not supported")
    current = document
    for token in tokens[:-1]:
        if isinstance(current, list):
            try:
                current = current[int(token)]
            except (ValueError, IndexError) as error:
                raise PatchError(f"invalid list path segment {token!r}") from error
        elif isinstance(current, dict):
            if token not in current:
                raise PatchError(f"missing object path segment {token!r}")
            current = current[token]
        else:
            raise PatchError(f"path traverses scalar at {token!r}")
    return current, tokens[-1]


def _get(document: Any, pointer: str) -> Any:
    parent, token = _parent(document, pointer)
    if isinstance(parent, list):
        try:
            return parent[int(token)]
        except (ValueError, IndexError) as error:
            raise PatchError(f"invalid list index {token!r}") from error
    if isinstance(parent, dict):
        if token not in parent:
            raise PatchError(f"missing object key {token!r}")
        return parent[token]
    raise PatchError("path parent is a scalar")


def _add(document: Any, pointer: str, value: Any) -> tuple[PatchOperation, str]:
    parent, token = _parent(document, pointer)
    value = deepcopy(value)
    if isinstance(parent, list):
        if token == "-":
            index = len(parent)
        else:
            try:
                index = int(token)
            except ValueError as error:
                raise PatchError(f"invalid list index {token!r}") from error
            if index < 0 or index > len(parent):
                raise PatchError(f"list insertion index out of range: {index}")
        parent.insert(index, value)
        actual = pointer[:-1] + str(index) if pointer.endswith("-") else pointer
        return PatchOperation(op="remove", path=actual), actual
    if isinstance(parent, dict):
        existed = token in parent
        previous = deepcopy(parent.get(token))
        parent[token] = value
        if existed:
            return PatchOperation(op="replace", path=pointer, value=previous), pointer
        return PatchOperation(op="remove", path=pointer), pointer
    raise PatchError("add path parent is a scalar")


def _remove(document: Any, pointer: str) -> PatchOperation:
    parent, token = _parent(document, pointer)
    if isinstance(parent, list):
        try:
            previous = parent.pop(int(token))
        except (ValueError, IndexError) as error:
            raise PatchError(f"invalid list index {token!r}") from error
    elif isinstance(parent, dict):
        if token not in parent:
            raise PatchError(f"missing object key {token!r}")
        previous = parent.pop(token)
    else:
        raise PatchError("remove path parent is a scalar")
    return PatchOperation(op="add", path=pointer, value=deepcopy(previous))


def _replace(document: Any, pointer: str, value: Any) -> PatchOperation:
    previous = deepcopy(_get(document, pointer))
    parent, token = _parent(document, pointer)
    if isinstance(parent, list):
        parent[int(token)] = deepcopy(value)
    elif isinstance(parent, dict):
        parent[token] = deepcopy(value)
    else:
        raise PatchError("replace path parent is a scalar")
    return PatchOperation(op="replace", path=pointer, value=previous)


def apply_timeline_patch(
    timeline: Timeline, operations: list[PatchOperation]
) -> tuple[Timeline, list[PatchOperation]]:
    """Apply operations transactionally and return a validated timeline plus inverse."""

    document = timeline.model_dump(mode="json")
    inverses: list[PatchOperation] = []
    for operation in operations:
        if operation.op == "test":
            actual = _get(document, operation.path)
            if actual != operation.value:
                raise PatchError(
                    f"test failed at {operation.path}: {actual!r} != {operation.value!r}"
                )
            continue
        if operation.op == "add":
            inverse, _ = _add(document, operation.path, operation.value)
        elif operation.op == "remove":
            inverse = _remove(document, operation.path)
        elif operation.op == "replace":
            inverse = _replace(document, operation.path, operation.value)
        else:  # pragma: no cover - Pydantic constrains this branch
            raise PatchError(f"unsupported patch operation: {operation.op}")
        inverses.insert(0, inverse)
    updated = Timeline.model_validate(document)
    validate_timeline(updated)
    return updated, inverses

