"""Project-owned glue; third-party model implementations remain external."""

__all__ = ["build_model", "load_checkpoint", "source_record"]


def __getattr__(name):
    # The separate Python 3.8 PINN environment must not import PyTorch merely
    # to access its own submodule. Existing model exports remain available.
    if name in __all__:
        from . import models
        value = getattr(models, name)
        globals()[name] = value
        return value
    raise AttributeError("module %r has no attribute %r" % (__name__, name))
