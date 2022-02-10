from copy import deepcopy
from functools import partial
from torch import nn


def _normalize_output(mod, outp):
    if isinstance(mod, nn.Embedding):
        return outp,
    return outp


def inject_hooks(model, intervene_fn, submodules=(), clone=True,
                 clear_prev_hooks=False, pass_idx=False, pass_submodule=False):
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
        if pass_idx:
            intervene_fn = partial(intervene_fn, idx=i)
        if pass_submodule:
            intervene_fn = partial(intervene_fn, submodule=submodule_path)
        my_hooks.append(model.get_submodule(submodule_path).register_forward_hook(intervene_fn))
    if clone:
        return model


def combine_models(base_model, injected_model, freeze_base=False, freeze_injected=False, **kwargs):
    if freeze_base:
        [param.requires_grad_(False) for param in base_model.parameters()]
    if freeze_injected:
        [param.requires_grad_(False) for param in injected_model.parameters()]
    new_model = inject_hooks(base_model, injected_model.forward, **kwargs)
    for name, param in injected_model.named_parameters():
        new_model.register_parameter(name=name, param=param)
    return new_model
