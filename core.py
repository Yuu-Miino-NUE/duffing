import json
from typing import IO
import numpy as np


class IterItems:
    def __init__(self, items: list[str]) -> None:
        self.items = items

    def __iter__(self):
        for item in self.items:
            yield item, getattr(self, item)

    def dump(self, fp: IO, **kwargs) -> None:
        kwargs.setdefault("indent", 4)
        kwargs.setdefault("cls", IterItemsJsonEncoder)
        json.dump(self, fp, **kwargs)


class IterItemsJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, IterItems):
            return dict(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
