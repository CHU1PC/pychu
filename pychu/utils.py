def _dot_var(v, verbose=False):
    """v: 変数"""
    # 変数用(variable)
    # 変数の色はorangeの丸
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    # verbose=Trueだと変数のdtypeも出力するようにする
    if verbose and v.data is not None:
        if v.name is not None:
            name += ": "
        name += str(v.shape) + " " + str(v.dtype)
    # id()はメモリのアドレスを返す
    return dot_var.format(id(v), name)


def _dot_func(f):
    # 関数用
    # 関数の色はlightblueの四角
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs:
        txt += dot_edge.format(id(f), id(y()))
    return txt


def get_dot_graph(output, verbose=True):
    # 全部をつなげる
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            # funcs.sort(key=lambda x: x.generation)
            seen_set.add(f)

    add_func(output.creator)
    # 一番最後のoutputに対して_dot_varでtxtに追加
    txt += _dot_var(output, verbose)

    # 一番最後のfuncsからinputsを使ってひとつ前の変数を取得&txtに追加
    # これを最後の変数まで繰り返す
    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)
        return 'digraph g{\n' + txt + '}'
