"""Runtime checks of the jaxtyping array annotations, applied to the whole package while
the suite runs.

``conftest.py`` installs jaxtyping's import hook with :func:`check_array_shapes` as the
type checker, so every function and dataclass in ``hologradpy`` verifies the dtype and
shape of each argument and return value that carries an array annotation such as
``Float[Tensor, "n_wavelengths H W"]``. Dimension names are bound per call by
``jaxtyped``, so ``*batch`` on an argument and on the return value must agree.

Annotations that name anything else (plain types, forward references, unions with a
non-array member) are left alone. The package keeps several imports behind
``TYPE_CHECKING``, which a general type checker cannot resolve at call time.
"""

from __future__ import annotations

import functools
import inspect
import types
import typing
from typing import Any, Callable

from jaxtyping import AbstractArray


def _array_types(annotation: Any, namespace: dict[str, Any]) -> tuple[type, ...]:
    """The jaxtyping array types one annotation names.

    Args:
        annotation: The annotation as written, a string under ``from __future__ import
            annotations``.
        namespace: The globals of the annotated function, used to evaluate the string.

    Returns:
        tuple[type, ...]: The array types to check a value against, empty when the
        annotation names no array type or mixes array types with other types.
    """
    if isinstance(annotation, str):
        try:
            annotation = eval(annotation, namespace)
        except Exception:
            return ()
    if isinstance(annotation, (types.UnionType, typing._UnionGenericAlias)):
        members = typing.get_args(annotation)
    else:
        members = (annotation,)
    arrays = tuple(
        member
        for member in members
        if isinstance(member, type) and issubclass(member, AbstractArray)
    )
    others = [
        member
        for member in members
        if member not in arrays and member is not type(None)
    ]
    return arrays if arrays and not others else ()


def _describe(value: Any) -> str:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is None:
        return type(value).__name__
    return f"{type(value).__name__} of shape {tuple(shape)} and dtype {dtype}"


def check_array_shapes(function: Callable) -> Callable:
    """Wrap ``function`` so that its array-annotated arguments and return value are
    checked on every call.

    Args:
        function: The function to wrap. Functions without array annotations are returned
            unchanged.

    Returns:
        Callable: The checked function, or ``function`` itself.
    """
    try:
        signature = inspect.signature(function)
        annotations = dict(function.__annotations__)
    except (TypeError, ValueError, AttributeError):
        return function

    namespace = getattr(function, "__globals__", {})
    parameters = {
        name: _array_types(annotations[name], namespace)
        for name, parameter in signature.parameters.items()
        if name in annotations
        and parameter.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }
    parameters = {name: arrays for name, arrays in parameters.items() if arrays}
    returns = ()
    if "return" in annotations:
        returns = _array_types(annotations["return"], namespace)
    if not parameters and not returns:
        return function

    def _check(label: str, value: Any, arrays: tuple[type, ...]) -> None:
        if value is None or any(isinstance(value, array) for array in arrays):
            return
        wanted = " | ".join(str(array) for array in arrays)
        raise TypeError(
            f"{function.__qualname__}: {label} is {_describe(value)}, "
            f"annotated {wanted}."
        )

    @functools.wraps(function)
    def checked(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = signature.bind(*args, **kwargs).arguments
        except TypeError:
            bound = {}
        for name, arrays in parameters.items():
            if name in bound:
                _check(f"argument '{name}'", bound[name], arrays)
        result = function(*args, **kwargs)
        if returns:
            _check("return value", result, returns)
        return result

    return checked
