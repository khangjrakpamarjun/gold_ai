from functools import partial, update_wrapper
from typing import Callable, Dict, List

from kedro.pipeline import Pipeline, pipeline


def partial_wrapper(func: Callable, *args, **kwargs) -> Callable:
    """Enables user to pass in arguments that are not datasets when function is called
    in a Kedro pipeline e.g. a string or int value.
    Args:
        func: Callable node function
     Returns:
        Callable
    """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def namespace_pipelines(
    pipelines: Dict[str, Pipeline], namespace: str, ignore: List[str] = None
):
    if not ignore:
        ignore = []

    return {
        name
        if name in ignore
        else f"{namespace}.{name}": pipeline(
            pipe=p,
            parameters={
                param: param.replace("params:", f"params:{namespace}.{name}.")
                for param in [
                    i
                    for node in p.nodes
                    for i in node.inputs
                    if i.startswith("params:")
                ]
            },
            namespace=f"{namespace}.{name}",
        )
        for name, p in pipelines.items()
    }
