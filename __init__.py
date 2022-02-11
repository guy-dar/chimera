import re
from copy import deepcopy
from functools import partial
from typing import List, Union

import torch
from torch import nn


def _normalize_output(mod, outp):
    if isinstance(mod, nn.Embedding):
        return outp,
    return outp


def _match_string_to_regex(match_string):
    return r'[^\.]*'.join(map(re.escape, match_string.split('*'))) + '$'


def _inject_new_params(new_model, injected_model):
    for name, param in injected_model.named_parameters():
        new_model.register_parameter(name=name, param=param)


def inject_hooks(model, intervene_fn, submodules=None, match_string=None, clone=True,
                 clear_prev_hooks=False, pass_idx=False, pass_submodule=False, verbose=False):
    assert (submodules is None) ^ (match_string is None)
    if submodules is None:
        assert isinstance(match_string, (str, list, tuple))
        if isinstance(match_string, str):
            match_string = [match_string]
        patterns = [_match_string_to_regex(raw_pattern) for raw_pattern in match_string]
        submodules = [name for name, _ in model.named_modules()
                      if any(re.search(pattern, name) for pattern in patterns)]

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
    if clone:
        return model


def combine_models(base_model, injected_model, freeze_base=False, freeze_injected=False, **kwargs):
    if freeze_base:
        [param.requires_grad_(False) for param in base_model.parameters()]
    if freeze_injected:
        [param.requires_grad_(False) for param in injected_model.parameters()]
    new_model = inject_hooks(base_model, injected_model.forward, **kwargs)
    _inject_new_params(new_model, injected_model)
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


def override_parameters(base_model: nn.Module, replacement_model: nn.Module, pass_idx: bool = True,
                        match_string: Union[List[str], str, None] = None, clone: bool = True, verbose: bool = False):
    if isinstance(match_string, str):
        match_string = [match_string]
    patterns = [_match_string_to_regex(raw_pattern) for raw_pattern in match_string]
    replaced_params = [name for name, _ in base_model.named_parameters()
                       if any(re.search(pattern, name) for pattern in patterns)]

    if clone:
        base_model = deepcopy(base_model)

    for idx, name in enumerate(replaced_params):
        if verbose:
            print(f"injection to parameter {name}")
        old_param = base_model.get_parameter(name)
        submodule_name, param_base_name = name.rsplit(name, 1)
        submodule = base_model.get_submodule(submodule_name)
        submodule.register_parameter(f'_chimera_old_{param_base_name}', old_param)
        _replace_fn = partial(replacement_model.forward, old_param)
        if pass_idx:
            _replace_fn = partial(_replace_fn, idx=idx)
        assert (param_base_name in submodule.__dict__) and (submodule.__dict__[param_base_name] == old_param)
        submodule.__dict__[param_base_name] = property(_replace_fn)

    _inject_new_params(base_model, replacement_model)
    return base_model
