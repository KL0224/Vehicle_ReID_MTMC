#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Optional


class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party
    users' custom modules.
    To create a registry (e.g. a backbone registry):
    .. code-block:: python
        BACKBONE_REGISTRY = Registry('BACKBONE')
    To register an object:
    .. code-block:: python
        @BACKBONE_REGISTRY.register()
        class MyBackbone():
            ...
    Or:
    .. code-block:: python
        BACKBONE_REGISTRY.register(MyBackbone)
    """

    def __init__(self, name: str) -> None:
        """
        Args:
            name (str): the name of this registry
        """
        self._name: str = name
        self._obj_map: Dict[str, object] = {}

    # def _do_register(self, name: str, obj: object) -> None:
    #     assert (
    #             name not in self._obj_map
    #     ), "An object named '{}' was already registered in '{}' registry!".format(
    #         name, self._name
    #     )
    #     self._obj_map[name] = obj

    def _do_register(self, name, func_or_class):
        """
        Register a function or class with optional overwrite
        """
        if name in self._obj_map:
            import warnings
            import inspect

            # Get caller info for debugging
            frame = inspect.currentframe().f_back.f_back
            caller_file = frame.f_code.co_filename
            caller_line = frame.f_lineno

            warnings.warn(
                f"\n⚠️ Registry Conflict Detected!\n"
                f"  Object: '{name}'\n"
                f"  Registry: '{self._name}'\n"
                f"  Location: {caller_file}:{caller_line}\n"
                f"  Action: Overwriting existing registration\n",
                stacklevel=3
            )

        self._obj_map[name] = func_or_class

    def register(self, obj: object = None) -> Optional[object]:
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class: object) -> object:
                name = func_or_class.__name__  # pyre-ignore
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        name = obj.__name__  # pyre-ignore
        self._do_register(name, obj)

    def get(self, name: str) -> object:
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    name, self._name
                )
            )
        return ret
