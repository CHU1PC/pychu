import os
import subprocess

import numpy as np


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

    # ここで矢印をつける
    # inputsをoutputsについてつけていく
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


def plot_dot_graph(output, verbose=True, to_file="graph.png"):
    dot_graph = get_dot_graph(output, verbose)

    # dotデータをファイルに保存
    tmp_dir = os.path.join(os.path.expanduser("~"), ".pychu")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, "tmp_graph.dot")

    with open(graph_path, "w") as f:
        f.write(dot_graph)

    # dotコマンドを呼ぶ
    extension = os.path.splitext(to_file)[1][1:]
    cmd = "dot {} -T {} -o {}".format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)


def sum_to(x, shape):
    """指定したshapeに合わせてxをsumする
    Args:
        x: 入力のndarray
        shape: 出力のshape
    Returns:
        xをshapeに合わせてsumしたndarray
    """
    ndim = len(shape)
    lead = x.ndim - ndim
    lead_axis = tuple(range(lead))

    axis = tuple([i + lead for i, sx in enumerate(shape) if sx == 1])
    y = x.sum(lead_axis + axis, keepdims=True)
    if lead > 0:
        y = y.squeeze(lead_axis)
    return y


def logsumexp(x, axis=1):
    m = x.max(axis=axis, keepdims=True)
    y = x - m
    np.exp(y, out=y)
    s = y.sum(axis=axis, keepdims=True)
    np.log(s, out=s)
    m += s
    return m
