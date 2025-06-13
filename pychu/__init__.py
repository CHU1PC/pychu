is_simple_core = False

if is_simple_core:
    from pychu.core_simple import using_config  # noqa
    from pychu.core_simple import (Function, Variable, as_variable,  # noqa
                                   no_grad)
else:
    from pychu.core import Variable  # type: ignore # noqa
    from pychu.core import Parameter  # type: ignore # noqa
    from pychu.core import Function  # type: ignore # noqa
    from pychu.core import using_config  # type: ignore # noqa
    from pychu.core import no_grad  # type: ignore # noqa
    from pychu.core import as_array  # type: ignore # noqa
    from pychu.core import as_variable  # type: ignore # noqa
    from pychu.core import setup_variable  # type: ignore # noqa
    from pychu.core import Config  # type: ignore # noqa
    from pychu.layers import Layer  # type: ignore # noqa
    from pychu.models import Model  # type: ignore # noqa
    from pychu.datasets import Dataset  # type: ignore # noqa
    from pychu.dataloader import DataLoader  # type: ignore # noqa

    import pychu.datasets  # type: ignore # noqa
    import pychu.dataloader  # type: ignore # noqa
    import pychu.optimizers  # type: ignore # noqa
    import pychu.functions  # type: ignore # noqa
    import pychu.layers  # type: ignore # noqa
    import pychu.utils  # type: ignore # noqa
    import pychu.cuda  # type: ignore # noqa
    import pychu.transforms  # type: ignore # noqa

setup_variable()
