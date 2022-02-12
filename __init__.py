import re
from copy import deepcopy
from functools import partial
from operator import itemgetter
from typing import Union, List, Iterable, Any, Callable
import torch
from torch import nn


def _filter_by_match(match_string: Union[str, List[str]], all_names: Iterable[str]):
    assert isinstance(match_string, (str, list, tuple))
    if isinstance(match_string, str):
        match_string = [match_string]
    patterns = [_match_string_to_regex(raw_pattern) for raw_pattern in match_string]
    filtered_names = [name for name in all_names if any(re.search(pattern, name) for pattern in patterns)]
    return filtered_names


def _normalize_output(mod: nn.Module, outp: Any):
    if isinstance(mod, nn.Embedding):
        return outp,
    return outp


def _match_string_to_regex(match_string: Union[str, List[str]]):
    return r'[^\.]*'.join(map(re.escape, match_string.split('*'))) + '$'


def _inject_new_params(base_model: nn.Module, injected_model: nn.Module, freeze_base: bool, freeze_injected: bool):
    if freeze_base:
        [param.requires_grad_(False) for param in base_model.parameters()]
    if freeze_injected:
        [param.requires_grad_(False) for param in injected_model.parameters()]
    for name, param in injected_model.named_parameters():
        base_model.register_parameter(name=name, param=param)


def inject_hooks(model: nn.Module, intervene_fn: Callable, match_string: Union[str, List[str]],
                 clone: bool = True, clear_prev_hooks: bool = False, pass_idx: bool = False,
                 pass_submodule: bool = False, verbose: bool = False):
    submodules = _filter_by_match(match_string, map(itemgetter(0), model.named_modules()))

    if clone:
        model = deepcopy(model)

    if hasattr(model, 'my_hooks'):
        if clear_prev_hooks:
            [hook.remove() for hook in model.my_hooks]
            model.my_hooks = []
    else:
        model.my_hooks = []

    my_hooks = model.my_hooks
    for i, submodule_path in enumerate(submodules):
        if verbose:
            print(f"injection to submodule: {submodule_path}")
        if pass_idx:
            intervene_fn = partial(intervene_fn, idx=i)
        if pass_submodule:
            intervene_fn = partial(intervene_fn, submodule=submodule_path)

        def _inner_fn(mod, inp, out_, *args, **kwargs):
            out = _normalize_output(mod, out_)
            new_out = intervene_fn(*out, *args, **kwargs)
            if new_out is not None:
                assert isinstance(new_out, torch.Tensor)  # TODO: supports only one output
                first_arg = out[0]
                first_arg *= 0
                first_arg += new_out  # we cannot use set, because it is non-differentiable

        my_hooks.append(model.get_submodule(submodule_path).register_forward_hook(_inner_fn))
    return model


def combine_models(base_model: nn.Module, injected_model: nn.Module,
                   freeze_base: bool = False, freeze_injected : bool = False, **kwargs):
    new_model = inject_hooks(base_model, injected_model.forward, **kwargs)
    _inject_new_params(new_model, injected_model, freeze_base=freeze_base, freeze_injected=freeze_injected)
    return new_model


def extract_intermediate_layer(model, submodule_path):
    # TODO: find better way (perhaps using torch fx)
    # TODO: probably better to use inject_hooks. For now keep it this way
    class _IntermediateLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self._model = deepcopy(model)
            self._registered = None
            self._model.get_submodule(submodule_path).register_forward_hook(self._register)

        def _register(self, mod, inp, out_):
            self._registered = _normalize_output(mod, out_)

        def forward(self, *args, **kwargs):
            self._model(*args, **kwargs)
            return self._registered

    return _IntermediateLayer()
