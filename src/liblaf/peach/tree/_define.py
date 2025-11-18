import functools
from typing import Any, dataclass_transform

import attrs

from liblaf import grapes

from ._field import array, container, field
from ._register_fieldz import register_fieldz


@dataclass_transform(field_specifiers=(attrs.field, array, container, field))
@grapes.wraps(attrs.define)
def define(cls: type | None = None, /, **kwargs) -> Any:
    if cls is None:
        return functools.partial(define, **kwargs)
    cls = grapes.attrs.define(cls, **kwargs)
    cls = register_fieldz(cls)
    return cls
