"""Parser + applier for the ```edits``` block of the weight-space-programming protocol.

The block is a tiny assignment language over the MLP tensors:

    W1[540:600:6, 0] = [0, -0.02, ...]   # strided slice assignment
    W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]
    b1[0:4] = -1                         # scalar fill
    W3[:, 2] *= 2                        # in-place scale / shift

Only the last fenced ```edits block is read; unparsable statements are skipped.
"""
import ast
import re

import numpy as np

SHAPES = {
    "W1": (600, 32),
    "b1": (32,),
    "W2": (32, 32),
    "b2": (32,),
    "W3": (32, 6),
    "b3": (6,),
}
KEYS = {  # npz keys used by the reference harness
    "W1": "actor/dense1/kernel",
    "b1": "actor/dense1/bias",
    "W2": "actor/dense2/kernel",
    "b2": "actor/dense2/bias",
    "W3": "actor/logits/kernel",
    "b3": "actor/logits/bias",
}

BLOCK_RE = re.compile(r"```edits[ \t]*\n(.*?)```", re.S)
STMT_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(\[[^\]]*\])\s*(\*=|\+=|-=|=)\s*(.+?)\s*$", re.S
)


def zero_weights():
    return {k: np.zeros(s, dtype=np.float32) for k, s in SHAPES.items()}


def extract_block(text):
    blocks = BLOCK_RE.findall(text)
    return blocks[-1] if blocks else None


def _strip_comment(line):
    # no string literals are expected in this language, so '#' always starts a comment
    return line.split("#", 1)[0]


def split_statements(block):
    """Join continuation lines until brackets balance."""
    stmts, buf, depth = [], "", 0
    for raw in block.splitlines():
        line = _strip_comment(raw).rstrip()
        if not line.strip() and not buf:
            continue
        buf = (buf + " " + line.strip()).strip() if buf else line.strip()
        depth += line.count("[") - line.count("]")
        depth += line.count("(") - line.count(")")
        if depth <= 0 and buf:
            stmts.append(buf)
            buf, depth = "", 0
    if buf:
        stmts.append(buf)
    return stmts


def _eval_node(node, variables=None):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_node(node.operand, variables)
    if isinstance(node, ast.Slice):
        return slice(
            _eval_node(node.lower, variables) if node.lower else None,
            _eval_node(node.upper, variables) if node.upper else None,
            _eval_node(node.step, variables) if node.step else None,
        )
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(e, variables) for e in node.elts]
    if isinstance(node, ast.Name) and variables and node.id in variables:
        return variables[node.id]
    raise ValueError(f"unsupported expression: {ast.dump(node)}")


def parse_index(expr, variables=None):
    """'[540:600:6, 0]' -> (slice(540,600,6), 0)"""
    node = ast.parse("T" + expr, mode="eval").body
    if not isinstance(node, ast.Subscript):
        raise ValueError("not a subscript")
    key = node.slice
    elts = key.elts if isinstance(key, ast.Tuple) else [key]
    return tuple(_eval_node(e, variables) for e in elts)


def _parse_rhs(rhs, variables=None):
    try:
        return ast.literal_eval(rhs)
    except (ValueError, SyntaxError):
        node = ast.parse(rhs, mode="eval").body
        return _eval_node(node, variables)


def apply_edits(text, weights=None):
    """Apply the last edits block to `weights` (zero weights by default).

    Returns (weights, report).
    """
    w = zero_weights() if weights is None else weights
    block = extract_block(text)
    report = {
        "found_block": block is not None,
        "statements": 0,
        "applied": 0,
        "entries_written": 0,
        "errors": [],
    }
    if block is None:
        return w, report

    variables = {"W1": w["W1"], "b1": w["b1"], "W2": w["W2"],
                 "b2": w["b2"], "W3": w["W3"], "b3": w["b3"]}
    for i, stmt in enumerate(split_statements(block), 1):
        report["statements"] += 1
        m = STMT_RE.match(stmt)
        if not m:
            report["errors"].append({"line": i, "stmt": stmt[:120], "err": "syntax"})
            continue
        name, idx_expr, op, rhs = m.groups()
        try:
            if name not in variables:
                raise KeyError(f"unknown tensor {name}")
            t = variables[name]
            idx = parse_index(idx_expr, variables)
            if op == "=":
                val = _parse_rhs(rhs, variables)
                arr = np.asarray(val, dtype=np.float32)
                n_target = t[idx].size
                if arr.ndim == 0:
                    t[idx] = float(arr)
                else:
                    if arr.size != n_target:
                        raise ValueError(f"size {arr.size} != target {n_target}")
                    t[idx] = arr.reshape(t[idx].shape)
                report["entries_written"] += n_target
            else:
                val = _parse_rhs(rhs, variables)
                arr = np.asarray(val, dtype=np.float32)
                if arr.ndim != 0:
                    raise ValueError("augmented assignment expects a scalar")
                if op == "*=":
                    t[idx] *= float(arr)
                elif op == "+=":
                    t[idx] += float(arr)
                else:
                    t[idx] -= float(arr)
                report["entries_written"] += t[idx].size
            report["applied"] += 1
        except Exception as e:  # keep going: skipped statements still leave the rest valid
            report["errors"].append({"line": i, "stmt": stmt[:120], "err": str(e)})
    return w, report


def nonzero_counts(w):
    return {k: int(np.count_nonzero(v)) for k, v in w.items()}


def save_npz(w, path):
    np.savez(path, **{KEYS[k]: w[k] for k in w})
