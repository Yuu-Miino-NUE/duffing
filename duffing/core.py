"""Core module for the package."""

import json
from typing import IO
import numpy


class IterItems:
    """Class to iterate items according to a given list.

    Parameters
    ----------
    items : list[str]
        List of item names.
    """

    def __init__(self, items: list[str]) -> None:
        self.items = items

    def __iter__(self):
        for item in self.items:
            yield item, getattr(self, item)

    def out_strs(self):
        return [f"\t{k}: {v}" for k, v in vars(self).items() if k != "items"]

    def dump(self, fp: IO, also: list[str] = [], **kwargs) -> None:
        """Dump the object to a JSON file.

        Parameters
        ----------
        fp : IO
            File pointer.
        also : list[str], optional
            Additional items to dump.
        kwargs : Any
            Keyword arguments for ``json.dump()``.


        .. note::

            The ``also`` parameter is used to dump additional items to the JSON file. The items must be attributes of the object.

        Examples
        --------

        .. code-block:: python

            import numpy as np
            from system import Parameter
            from fix import fix

            xfix0 = np.array([0.0, 0.0])
            param = Parameter(k=0.2, B=0.1, B0=0.1)
            period = 1

            result = fix(xfix0, param, period)

            with open("result.json", "w") as f:
                result.dump(f, also=["eig", "abs_eig"])

        The above code will dump the result object to a JSON file with the following content:

        .. code-block:: json

            {
                "xfix": [
                    0.23214784788536827,
                    0.07559285853639049
                ],
                "parameters": {
                    "k": 0.2,
                    "B": 0.1,
                    "B0": 0.1
                },
                "period": 1,
                "eig": [
                    "(-0.21783034959956155+0.48698936934703324j)",
                    "(-0.21783034959956155-0.48698936934703324j)"
                ],
                "abs_eig": [
                    0.5334873073126374,
                    0.5334873073126374
                ]
            }

        See Also
        --------
        fix.fix: Fixed or periodic point calculation.
        fix.FixResult: Fixed point result class.
        homoclinic.HomoclinicResult: Homoclinic point result class.
        hbf.HbfResult: Homoclinic bifurcation point result class.
        json.dump: Dump JSON object to a file.
        """

        kwargs.setdefault("indent", 4)
        kwargs.setdefault("cls", _IterItemsJsonEncoder)
        if len(also) > 0:
            _items = self.items
            self.items = self.items + also
        json.dump(self, fp, **kwargs)
        if len(also) > 0:
            self.items = _items


class _IterItemsJsonEncoder(json.JSONEncoder):
    """JSON encoder for IterItems object."""

    def default(self, obj):
        if isinstance(obj, IterItems):
            return dict(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, complex):
            return str(obj)
        return super().default(obj)
