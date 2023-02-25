from __future__ import annotations

"""

Tool to correct parameters in stubs.

"""

import argparse
import ast
import contextlib
import importlib
import inspect
import io
import subprocess
import sys
import textwrap
import types
import typing
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple, Union

import libcst
import tomli
import typeshed_client
from termcolor import colored


def log(*objects: object) -> None:
    print(colored(" ".join(map(str, objects)), "yellow"))


@dataclass
class ChangeParameterNames(libcst.CSTTransformer):
    func_name: str
    is_classmethod: bool
    sig: inspect.Signature
    stub_params: libcst.Parameters
    num_changed: int = 0

    def leave_Parameters(
        self, original_node: libcst.Parameters, updated_node: libcst.Parameters
    ) -> libcst.Parameters:
        all_stub_params = list(self.stub_params.params)
        all_runtime_params = list(self.sig.parameters.values())
        if self.is_classmethod:
            all_runtime_params.insert(0, inspect.Parameter("cls", inspect.Parameter.POSITIONAL_ONLY))

        if len(all_stub_params) != len(all_runtime_params):
            return original_node

        variadic_parameter_kinds = {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }

        if any(param.kind in variadic_parameter_kinds for param in all_runtime_params):
            return original_node

        new_pos_or_kw: list[libcst.Param] = []

        for runtime_param, stub_param in zip(all_runtime_params, all_stub_params):
            runtime_name, stub_name = runtime_param.name, stub_param.name.value
            if (
                (
                    (stub_name == "self" and not self.is_classmethod) 
                    or (stub_name == "cls" and self.is_classmethod)
                )
                and stub_param == all_stub_params[0]
            ):
                new_pos_or_kw.append(stub_param)
            elif runtime_param.kind is inspect.Parameter.POSITIONAL_ONLY:
                new_name = f"__{runtime_name.lstrip('_')}"
                if new_name == stub_name:
                    new_pos_or_kw.append(stub_param)
                else:
                    new_pos_or_kw.append(
                        stub_param.with_changes(
                            name=libcst.Name(new_name)
                        )
                    )
            else:
                if stub_name == runtime_name:
                    new_param = stub_param
                elif (
                    stub_name.startswith("__") 
                    and not stub_name.endswith("__")
                    # There's lots of reasons why a parameter
                    # might be positional-or-keyword at runtime but positional-only in the stub
                    and runtime_param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
                ):
                    if stub_name[2:] == runtime_name:
                        new_param = stub_param
                    else:
                        new_param = stub_param.with_changes(name=libcst.Name(f"__{runtime_name.lstrip('_')}"))
                else:
                    new_param = stub_param.with_changes(name=libcst.Name(runtime_name))
                if runtime_param.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    new_pos_or_kw.append(new_param)
                elif runtime_param.kind is inspect.Parameter.KEYWORD_ONLY:
                    break
                else:
                    assert False, "Shouldn't get here"
        
        if new_pos_or_kw == list(self.stub_params.params):
            return original_node

        if (len(new_pos_or_kw) + len(self.stub_params.kwonly_params)) != len(all_runtime_params):
            return original_node

        self.num_changed += 1

        try:
            return updated_node.with_changes(params=new_pos_or_kw)
        except libcst.CSTValidationError:
            self.num_changed -= 1
            return original_node



def get_end_lineno(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    if sys.version_info >= (3, 8):
        assert hasattr(node, "end_lineno")
        assert node.end_lineno is not None
        return node.end_lineno
    else:
        return max(
            child.lineno
            for child in ast.iter_child_nodes(node)
            if hasattr(child, "lineno")
        )


def correct_parameters_in_func(
    stub_lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    runtime_func: Any,
) -> tuple[int, dict[int, list[str]]]:
    try:
        sig = inspect.signature(runtime_func)
    except Exception:
        return 0, {}
    start_lineno = node.lineno - 1
    if stub_lines[start_lineno - 1].strip() == "@classmethod":
        start_lineno -= 1
    end_lineno = get_end_lineno(node)
    lines = stub_lines[start_lineno : end_lineno]
    indentation = len(lines[0]) - len(lines[0].lstrip())
    cst = libcst.parse_statement(
        textwrap.dedent("".join(line + "\n" for line in lines))
    )
    assert isinstance(cst, libcst.FunctionDef)
    is_classmethod = any(
        (
            isinstance(deco.decorator, libcst.Name)
            and deco.decorator.value == "classmethod"
        ) for deco in cst.decorators
    )
    visitor = ChangeParameterNames(cst.name.value, is_classmethod, sig, cst.params)
    modified = cst.visit(visitor)
    if visitor.num_changed == 0:
        return 0, {}
    assert isinstance(modified, libcst.FunctionDef)
    new_code = textwrap.indent(libcst.Module(body=[modified]).code, " " * indentation)
    output_dict = {start_lineno: new_code.splitlines()}
    for i in range(start_lineno + 1, end_lineno):
        output_dict[i] = []
    return visitor.num_changed, output_dict


def gather_funcs(
    node: typeshed_client.NameInfo,
    name: str,
    fullname: str,
    runtime_parent: type | types.ModuleType,
    blacklisted_objects: frozenset[str],
) -> Iterator[Tuple[Union[ast.FunctionDef, ast.AsyncFunctionDef], Any]]:
    if fullname in blacklisted_objects:
        log(f"Skipping {fullname}: blacklisted object")
        return
    interesting_classes = (
        ast.ClassDef,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        typeshed_client.OverloadedName,
    )
    if not isinstance(node.ast, interesting_classes):
        return
    # special-case some aliases in the typing module
    if isinstance(runtime_parent, type(typing.Mapping)):
        runtime_parent = runtime_parent.__origin__  # type: ignore[attr-defined]
    try:
        try:
            runtime = getattr(runtime_parent, name)
        except AttributeError:
            runtime = inspect.getattr_static(runtime_parent, name)
    # Some getattr() calls raise TypeError, or something even more exotic
    except Exception:
        log("Could not find", fullname, "at runtime")
        return
    if isinstance(node.ast, ast.ClassDef):
        if not node.child_nodes:
            return
        for child_name, child_node in node.child_nodes.items():
            if child_name.startswith("__") and not child_name.endswith("__"):
                unmangled_parent_name = fullname.split(".")[-1]
                maybe_mangled_child_name = (
                    f"_{unmangled_parent_name.lstrip('_')}{child_name}"
                )
            else:
                maybe_mangled_child_name = child_name
            yield from gather_funcs(
                node=child_node,
                name=maybe_mangled_child_name,
                fullname=f"{fullname}.{child_name}",
                runtime_parent=runtime,
                blacklisted_objects=blacklisted_objects,
            )
    elif isinstance(node.ast, typeshed_client.OverloadedName):
        for definition in node.ast.definitions:
            if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)):
                yield definition, runtime
    elif isinstance(node.ast, (ast.FunctionDef, ast.AsyncFunctionDef)):
        yield node.ast, runtime


def change_parameters_in_stub(
    module_name: str,
    context: typeshed_client.finder.SearchContext,
    blacklisted_objects: frozenset[str],
) -> int:
    print(f"Processing {module_name}... ", end="", flush=True)
    path = typeshed_client.get_stub_file(module_name, search_context=context)
    if path is None:
        raise ValueError(f"Could not find stub for {module_name}")
    try:
        # Redirect stdout when importing modules to avoid noisy output from modules like `this`
        with contextlib.redirect_stdout(io.StringIO()):
            runtime_module = importlib.import_module(module_name)
    except KeyboardInterrupt:
        raise
    # `importlib.import_module("multiprocessing.popen_fork")` crashes with AttributeError on Windows
    # Trying to import serial.__main__ for typeshed's pyserial package will raise SystemExit
    except BaseException as e:
        log(f'Could not import {module_name}: {type(e).__name__}: "{e}"')
        return 0
    stub_names = typeshed_client.get_stub_names(module_name, search_context=context)
    if stub_names is None:
        raise ValueError(f"Could not find stub for {module_name}")
    stub_lines = path.read_text().splitlines()
    # pyanalyze doesn't let you use dict[] here
    replacement_lines: Dict[int, List[str]] = {}
    total_num_changed = 0
    for name, info in stub_names.items():
        funcs = gather_funcs(
            node=info,
            name=name,
            fullname=f"{module_name}.{name}",
            runtime_parent=runtime_module,
            blacklisted_objects=blacklisted_objects,
        )

        for func, runtime_func in funcs:
            num_changed, new_lines = correct_parameters_in_func(
                stub_lines, func, runtime_func
            )
            replacement_lines.update(new_lines)
            total_num_changed += num_changed
    with path.open("w") as f:
        for i, line in enumerate(stub_lines):
            if i in replacement_lines:
                for new_line in replacement_lines[i]:
                    f.write(new_line + "\n")
            else:
                f.write(line + "\n")
    print(f"changed {total_num_changed} functions")
    return total_num_changed


def is_relative_to(left: Path, right: Path) -> bool:
    """Return True if the path is relative to another path or False.

    Redundant with Path.is_relative_to in 3.9+.

    """
    try:
        left.relative_to(right)
        return True
    except ValueError:
        return False


def install_typeshed_packages(typeshed_paths: Sequence[Path]) -> None:
    to_install: List[str] = []
    for path in typeshed_paths:
        metadata_path = path / "METADATA.toml"
        if not metadata_path.exists():
            print(f"{path} does not look like a typeshed package", file=sys.stderr)
            sys.exit(1)
        metadata_bytes = metadata_path.read_text()
        metadata = tomli.loads(metadata_bytes)
        version = metadata["version"]
        to_install.append(f"{path.name}=={version}")
    if to_install:
        command = [sys.executable, "-m", "pip", "install", *to_install]
        print(f"Running install command: {' '.join(command)}")
        subprocess.check_call(command)


# A hardcoded list of stdlib modules to skip
# This is separate to the --blacklists argument on the command line,
# which is for individual functions/methods/variables to skip
#
# `_typeshed` doesn't exist at runtime; no point trying to change parameters
# `antigravity` exists at runtime but it's annoying to have the browser open up every time
STDLIB_MODULE_BLACKLIST = ("_typeshed/*.pyi", "antigravity.pyi")


def load_blacklist(path: Path) -> frozenset[str]:
    with path.open() as f:
        entries = frozenset(line.split("#")[0].strip() for line in f)
    return entries - {""}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--stdlib-path",
        help=(
            "Path to typeshed's stdlib directory. If given, we will change parameters in"
            " stubs in this directory."
        ),
    )
    parser.add_argument(
        "-p",
        "--packages",
        nargs="+",
        help=(
            "List of packages to change parameters in. We will change parameters in all stubs in"
            " these directories. The runtime package must be installed."
        ),
        default=(),
    )
    parser.add_argument(
        "-t",
        "--typeshed-packages",
        nargs="+",
        help=(
            "List of typeshed packages to change parameters in. WARNING: We will install the package locally."
        ),
        default=(),
    )
    parser.add_argument(
        "-b",
        "--blacklists",
        nargs="+",
        help=(
            "List of paths pointing to 'blacklist files',"
            " which can be used to specify functions that argcorrector should skip"
            " trying to change parameters in. Note: if the name of a class is included"
            " in a blacklist, the whole class will be skipped."
        ),
        default=(),
    )
    parser.add_argument(
        "-z",
        "--exit-zero",
        action="store_true",
        help="Exit with code 0 even if there were errors.",
    )
    args = parser.parse_args()

    stdlib_path = Path(args.stdlib_path) if args.stdlib_path else None
    if stdlib_path is not None:
        if not (stdlib_path.is_dir() and (stdlib_path / "VERSIONS").is_file()):
            parser.error(f'"{stdlib_path}" does not point to a valid stdlib directory')

    typeshed_paths = [Path(p) for p in args.typeshed_packages]
    install_typeshed_packages(typeshed_paths)
    package_paths = [Path(p) for p in args.packages] + typeshed_paths
    stdlib_blacklist_path = Path(__file__).parent / "stdlib-blacklist.txt"
    assert stdlib_blacklist_path.exists() and stdlib_blacklist_path.is_file()
    blacklist_paths = [Path(p) for p in args.blacklists] + [stdlib_blacklist_path]

    combined_blacklist = frozenset(
        chain.from_iterable(load_blacklist(path) for path in blacklist_paths)
    )
    context = typeshed_client.finder.get_search_context(
        typeshed=stdlib_path, search_path=package_paths, version=sys.version_info[:2]
    )
    total_changed = 0
    for module, path in typeshed_client.get_all_stub_files(context):
        if stdlib_path is not None and is_relative_to(path, stdlib_path):
            if any(
                path.relative_to(stdlib_path).match(pattern)
                for pattern in STDLIB_MODULE_BLACKLIST
            ):
                log(f"Skipping {module}: blacklisted module")
                continue
            else:
                num_changed = change_parameters_in_stub(
                    module, context, combined_blacklist
                )
                total_changed += num_changed
        elif any(is_relative_to(path, p) for p in package_paths):
            num_changed = change_parameters_in_stub(module, context, combined_blacklist)
            total_changed += num_changed
    m = f"\n--- Changed {total_changed} functions"
    print(colored(m, "green"))
    sys.exit(0)


if __name__ == "__main__":
    main()
