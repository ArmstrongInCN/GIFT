"""Remove source text from public provenance without changing numeric state.

Training journals retain their complete source identities. Public JSON and
weight exports use this boundary to keep repository references, revision hashes
and written adaptation reasons while omitting embedded code. This module uses
only the standard library so repository audits and separate runtimes can import
it without loading a numerical framework.
"""
from __future__ import annotations

import copy
import ast
import re
import textwrap


# Exact names avoid confusing provenance hashes, source paths or textual
# adaptation instructions with copied source. A value below one of these keys
# may itself be a mapping/list of source lines; only its text leaves are omitted.
_SOURCE_FIELDS = frozenset({
    "source_code", "source_text", "source_snippet", "source_snippets",
    "patch_code", "patch_source", "patch_text", "code_before", "code_after",
    "before_code", "after_code", "original_code", "modified_code",
    "implementation_code", "executable_patch",
})
_TEXT = (str, bytes, bytearray)
_OMIT = object()
_UNREGISTERED_SOURCE = "unregistered Python source text"


def _python_source_text(value):
    """Recognize complete Python statements conservatively without executing them.

    This catches renamed provenance fields containing definitions, imports,
    control flow or multiple assignments. A single identifier, formula, URL,
    hash or ordinary prose is not source evidence. It cannot prove that an
    arbitrary string is free of every programming language or truncated snippet.
    """
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except UnicodeDecodeError:
            return False
    if not isinstance(value, str):
        return False
    text = textwrap.dedent(value).strip()
    if not text:
        return False
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError, RecursionError):
        return False
    definitions = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)
    controls = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try,
                ast.Raise, ast.Return, ast.Delete, ast.Assert)
    nodes = list(ast.walk(tree))
    return (any(isinstance(node, definitions + controls) for node in nodes)
            or sum(isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) for node in nodes) >= 2)


def _key_name(key):
    """Compare snake-case and camel-case metadata keys without altering them."""
    if not isinstance(key, str):
        return ""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", key).lower().replace("-", "_")


def _path_key(path, key):
    """Represent arbitrary mapping keys in a human-readable field path."""
    if isinstance(key, str) and key.isidentifier():
        return path + "." + key
    return path + "[" + repr(key) + "]"


def _walk(value, *, clean):
    findings = []
    memo = {}

    def visit(item, path, in_patch=False, source_text=False, text_reason=None):
        if source_text and isinstance(item, _TEXT):
            findings.append({"path": path, "reason": text_reason or "explicit source-code field"})
            return _OMIT
        if isinstance(item, _TEXT) and _python_source_text(item):
            findings.append({"path": path, "reason": _UNREGISTERED_SOURCE})
        if not isinstance(item, (dict, list, tuple)):
            # Tensors, ndarrays, scalars and opaque model configuration values
            # remain the very same objects: no cast, allocation or arithmetic.
            return item
        token = (id(item), in_patch, source_text)
        if token in memo:
            return memo[token]
        if isinstance(item, dict):
            output = copy.copy(item) if clean else {}
            output.clear()
            memo[token] = output
            for key, child in item.items():
                name = _key_name(key)
                explicit = name in _SOURCE_FIELDS
                patch_field = in_patch and name in {"before", "after"}
                child_source = source_text or explicit or patch_field
                reason = (text_reason if source_text else
                          "patch source text" if patch_field else "explicit source-code field")
                child_patch = in_patch or name == "patches"
                result = visit(child, _path_key(path, key), child_patch, child_source, reason)
                if result is not _OMIT:
                    output[key] = result
        elif isinstance(item, list):
            output = copy.copy(item) if clean else []
            output.clear()
            memo[token] = output
            for index, child in enumerate(item):
                result = visit(child, f"{path}[{index}]", in_patch, source_text, text_reason)
                if result is not _OMIT:
                    output.append(result)
        else:
            # Provenance tuples are immutable and may include nested mappings.
            # Named tuples retain their named fields when no text was omitted.
            memo[token] = ()
            values = [visit(child, f"{path}[{index}]", in_patch, source_text, text_reason)
                      for index, child in enumerate(item)]
            values = [child for child in values if child is not _OMIT]
            output = (type(item)(*values) if hasattr(item, "_fields") and len(values) == len(item)
                      else tuple(values))
            memo[token] = output
        if hasattr(item, "__dict__"):
            # torch state_dict is an OrderedDict carrying version information in
            # _metadata. Keep this attribute and container ordering intact.
            for name, child in vars(item).items():
                field = _key_name(name)
                explicit = field in _SOURCE_FIELDS
                patch_field = in_patch and field in {"before", "after"}
                reason = (text_reason if source_text else
                          "patch source text" if patch_field else "explicit source-code field")
                result = visit(child, path + ".@" + name, in_patch or field == "patches",
                               source_text or explicit or patch_field, reason)
                if clean and result is not _OMIT:
                    setattr(output, name, result)
                elif clean and name in vars(output):
                    delattr(output, name)
        return output

    result = visit(value, "$")
    return result, findings


def public_metadata(value):
    """Return public containers with embedded code omitted and numeric leaves intact.

    Only text under explicit source-code keys or a ``patches`` entry's
    ``before``/``after`` keys is removed. Ordinary measurements with those names
    are preserved, including numerical before/after values inside patch records.
    Dict/list/tuple containers are copied; input containers are never modified.
    Recognizable Python statements in other fields raise ValueError so callers
    review that field instead of silently discarding undocumented metadata.
    """
    result, findings = _walk(value, clean=True)
    unknown = [item["path"] for item in findings if item["reason"] == _UNREGISTERED_SOURCE]
    if unknown:
        raise ValueError("Public metadata contains Python source under unregistered fields: " + ", ".join(unknown))
    return result


def source_metadata_findings(value):
    """Return JSON-serializable ``path``/``reason`` findings for embedded code.

    Explicit fields and conservative Python statement detection share the same
    rule with the exporter, including state-dict metadata attributes. Renamed
    code fields are reported and rejected by public_metadata. This detects
    common complete Python snippets, not every language or truncated fragment.
    No numerical payload is converted and no source text is executed.
    """
    return _walk(value, clean=False)[1]
