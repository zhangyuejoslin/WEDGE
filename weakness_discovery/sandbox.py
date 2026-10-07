"""Step 5 safety: execute LLM-proposed feature code in a restricted namespace.

Exposes only `math` and side-effect-free builtins; every function call is wrapped
in a SIGALRM timeout (LLMs occasionally write loops that never terminate).
"""
import math
import signal

def _safe_import(name, *args, **kwargs):
    """LLMs write `import math` even when told math is pre-provided; allow exactly that."""
    if name == "math":
        return math
    raise ImportError(f"import of {name!r} not allowed in the sandbox")


SAFE_BUILTINS = {k: __builtins__[k] if isinstance(__builtins__, dict) else getattr(__builtins__, k)
                 for k in [
                     "abs", "all", "any", "bool", "dict", "enumerate", "float", "int",
                     "len", "list", "max", "min", "range", "round", "sorted", "str",
                     "sum", "tuple", "zip", "isinstance", "set", "map", "filter",
                     "ValueError", "TypeError", "KeyError", "ZeroDivisionError", "Exception",
                 ]}
SAFE_BUILTINS["__import__"] = _safe_import


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def extract_functions(code_text):
    """Exec code in the sandbox; return {name: fn} for feature_* callables.
    Raises on any error (caller records the candidate as rejected)."""
    ns = {"math": math, "__builtins__": SAFE_BUILTINS}
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(5)
    try:
        exec(compile(code_text, "<candidate>", "exec"), ns)
    finally:
        signal.alarm(0)
    return {k: v for k, v in ns.items() if k.startswith("feature_") and callable(v)}


def call_with_timeout(fn, arg, seconds=1):
    """Call fn(arg) with a hard timeout. Returns (ok, value_or_error_str)."""
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(seconds)
    try:
        v = fn(arg)
        return True, v
    except Timeout:
        return False, "timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
