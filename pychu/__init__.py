is_simple_core = False

if is_simple_core:
    from pychu.core_simple import using_config  # noqa
    from pychu.core_simple import (Function, Variable, as_variable,  # noqa
                                   no_grad)
else:
    from pychu.core import using_config  # noqa
    from pychu.core import (Function, Variable,  # type: ignore  # noqa
                            as_variable, no_grad)
