"""静态检查：每个名字都必须有来源（局部 / 参数 / 全局 / 内建）。

    python tests/test_static_names.py

为什么需要它：本项目出过一次真实事故——把 `CHESSBOARD_CORNERS` 批量替换成
`board.corners` 时，`calibrate_camera()` 里并没有 `board` 这个变量，于是
**"运行相机标定"直接 NameError**。而 compileall 只查语法、不查名字，
单元测试又恰好没调用那个函数，两个都拦不住。

pyflakes/ruff 能查这类问题，但本工程唯一装了 opencv 的解释器是系统 Python，
为跑测试往里装包不划算。这里用标准库 ast 写一个够用的版本：
按作用域链解析每个 Name 的读取，找不到来源就报错。
"""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGETS = [ROOT / 'src', ROOT / 'tools', ROOT / 'tests']
BUILTINS = set(dir(builtins)) | {'__file__', '__name__', '__doc__', '__package__'}


class ScopeChecker:
    """按作用域链检查名字解析。"""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.problems: list[str] = []

    # ---- 收集一个作用域里"被绑定"的名字

    @staticmethod
    def bound_names(node: ast.AST) -> set[str]:
        names: set[str] = set()

        def add_target(t: ast.AST) -> None:
            if isinstance(t, ast.Name):
                names.add(t.id)
            elif isinstance(t, (ast.Tuple, ast.List)):
                for e in t.elts:
                    add_target(e)
            elif isinstance(t, ast.Starred):
                add_target(t.value)

        for sub in ast.walk(node):
            if isinstance(sub, ast.Assign):
                for t in sub.targets:
                    add_target(t)
            elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                add_target(sub.target)
            elif isinstance(sub, (ast.For, ast.AsyncFor)):
                add_target(sub.target)
            elif isinstance(sub, ast.comprehension):
                add_target(sub.target)
            elif isinstance(sub, ast.withitem) and sub.optional_vars is not None:
                add_target(sub.optional_vars)
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                names.add(sub.name)
            elif isinstance(sub, ast.NamedExpr):
                add_target(sub.target)
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(sub.name)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                for a in sub.names:
                    names.add((a.asname or a.name).split('.')[0])
            elif isinstance(sub, (ast.Global, ast.Nonlocal)):
                names.update(sub.names)
        return names

    @staticmethod
    def params_of(fn) -> set[str]:
        a = fn.args
        out = {p.arg for p in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs)}
        if a.vararg:
            out.add(a.vararg.arg)
        if a.kwarg:
            out.add(a.kwarg.arg)
        return out

    # ---- 遍历

    def visit(self, node: ast.AST, scopes: list[set[str]],
              globals_: set[str]) -> None:
        """scopes[0] 是最内层。"""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = set(scopes[-1]) if False else set()
                inner.add(child.name)
                local = self.params_of(child) | self.bound_names(child)
                self.visit(child, [local] + scopes, globals_)
            elif isinstance(child, ast.ClassDef):
                local = self.bound_names(child)
                self.visit(child, [local] + scopes, globals_)
            elif isinstance(child, ast.Lambda):
                local = self.params_of(child)
                self.visit(child, [local] + scopes, globals_)
            else:
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                    if not self.resolvable(child.id, scopes, globals_):
                        self.problems.append(
                            f'{self.path.name}:{child.lineno}: 名字 {child.id!r} 未定义 '
                            f'（既不是局部/参数，也不是模块全局或内建）')
                self.visit(child, scopes, globals_)

    @staticmethod
    def resolvable(name: str, scopes: list[set[str]], globals_: set[str]) -> bool:
        return any(name in s for s in scopes) or name in globals_ or name in BUILTINS

    def check(self) -> list[str]:
        tree = ast.parse(self.path.read_text(encoding='utf-8'), filename=str(self.path))
        globals_ = self.bound_names(tree)
        self.visit(tree, [set()], globals_)
        return self.problems


def main() -> int:
    files = sorted(p for d in TARGETS if d.is_dir()
                   for p in d.rglob('*.py'))
    problems: list[str] = []
    for path in files:
        problems.extend(ScopeChecker(path).check())

    if problems:
        print(f'发现 {len(problems)} 处未定义的名字：\n')
        for p in problems:
            print('  ', p)
        print('\n这类问题在运行时才炸，且往往只在某一条分支上炸——必须当错误处理。')
        return 1
    print(f'检查 {len(files)} 个文件，未发现未定义的名字。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
