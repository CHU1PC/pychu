is_simple_core = False

if is_simple_core:
    from pychu.core_simple import using_config  # noqa
    from pychu.core_simple import (Function, Variable, as_variable,  # noqa
                                   no_grad)
else:
    from pychu.core import using_config  # noqa
    from pychu.core import (Function, Parameter,  # type: ignore  # noqa
                            Variable, _setup_variable_operators, as_variable,
                            no_grad)
    from pychu.dataloader import DataLoader  # noqa
    from pychu.layers import Layer  # noqa
    from pychu.models import Model  # noqa
    import pychu.transforms  # noqa
    import pychu.optimizers  # noqa

_setup_variable_operators()
