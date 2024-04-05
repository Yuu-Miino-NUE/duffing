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

    def dump(self, fp: IO, **kwargs) -> None:
        """Dump the object to a JSON file.

        Parameters
        ----------
        fp : IO
            File pointer.
        kwargs : Any
            Keyword arguments for json.dump().

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
                result.dump(f)

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
                "period": 1
            }

        See Also
        --------
        fixduffing.fix: Fixed or periodic point calculation.
        fixduffing.fixResult: Fixed point result class.
        homoclinic.HomoclinicResult: Homoclinic point result class.
        hbf.HbfResult: Homoclinic bifurcation point result class.
        json.dump: Dump JSON object to a file.
        """

        kwargs.setdefault("indent", 4)
        kwargs.setdefault("cls", _IterItemsJsonEncoder)
        json.dump(self, fp, **kwargs)


class _IterItemsJsonEncoder(json.JSONEncoder):
    """JSON encoder for IterItems object."""

    def default(self, obj):
        if isinstance(obj, IterItems):
            return dict(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super().default(obj)
