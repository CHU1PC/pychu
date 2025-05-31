is_simple_core = True

if is_simple_core:
    from pychu.core_simple import using_config  # noqa
    from pychu.core_simple import (Function, Variable, as_variable,  # noqa
                                   no_grad)
