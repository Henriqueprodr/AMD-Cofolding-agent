# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""LaTeX in an answer, rewritten as the plain text the chat can render.

The chat renders Markdown and nothing else, so an answer written as
``$5.2 \\pm 0.3$ \\mu M`` reaches the screen with its dollars and backslashes
showing.  The system prompt asks for plain text, but a model obliges only
most of the time, and a demo cannot afford the times it does not.

Code is left exactly as written: a SMILES or a TOML snippet is not prose, and
a stray backslash in it is meaningful.
"""

from __future__ import annotations

import re

#: Fenced blocks and inline spans, which pass through untouched.
_CODE = re.compile(r"```.*?```|``.*?``|`[^`\n]*`", re.DOTALL)

#: LaTeX commands with a plain-text equivalent.  Applied longest first, so
#: ``\leq`` is not eaten by ``\le``.
_SYMBOLS = {
    r"\pm": "+/-",
    r"\mp": "-/+",
    r"\times": "x",
    r"\cdot": "*",
    r"\approx": "~",
    r"\sim": "~",
    r"\leq": "<=",
    r"\le": "<=",
    r"\geq": ">=",
    r"\ge": ">=",
    r"\neq": "!=",
    r"\ll": "<<",
    r"\gg": ">>",
    r"\rightarrow": "->",
    r"\longrightarrow": "->",
    r"\to": "->",
    r"\leftarrow": "<-",
    r"\Rightarrow": "=>",
    r"\infty": "inf",
    r"\degree": "deg",
    r"\percent": "%",
    r"\mu": "\u00b5",
    r"\angstrom": "\u00c5",
    r"\AA": "\u00c5",
    r"\alpha": "alpha",
    r"\beta": "beta",
    r"\gamma": "gamma",
    r"\Delta": "Delta",
    r"\delta": "delta",
    r"\kappa": "kappa",
    r"\lambda": "lambda",
    r"\sigma": "sigma",
    r"\pi": "pi",
}

#: A unit prefix and the unit it belongs to, e.g. ``\mu M``.
_PREFIX = re.compile(r"(\\mu|\\AA|\\angstrom)\s*([A-Za-z])")

#: Wrappers whose only job is typesetting: the argument is the content.
_WRAPPERS = re.compile(
    r"\\(?:text|textrm|textbf|textit|mathrm|mathbf|mathit|mathsf|operatorname|"
    r"mbox|si|unit)\s*\{([^{}]*)\}"
)

_FRAC = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")

#: ``IC_{50}`` is IC50; ``10^{-9}`` keeps its caret, which reads fine as text.
_SUBSCRIPT = re.compile(r"_\{([^{}]*)\}")
_SUPERSCRIPT = re.compile(r"\^\{([^{}]*)\}")

#: Spacing commands and the delimiters that size brackets.
_NOISE = re.compile(r"\\(?:left|right|displaystyle|,|;|:|!|\s)")

#: Maths delimiters.  ``$...$`` must stay on one line so that a lone dollar
#: (a price, a shell prompt) does not swallow half a paragraph.
_DISPLAY = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE = re.compile(r"\$([^$\n]+?)\$")
_PAREN = re.compile(r"\\\((.+?)\\\)", re.DOTALL)
_BRACKET = re.compile(r"\\\[(.+?)\\\]", re.DOTALL)

#: An escape that only existed to survive LaTeX.
_ESCAPE = re.compile(r"\\([%&#_{}$])")

#: Whatever command is left after all of the above.
_LEFTOVER = re.compile(r"\\([A-Za-z]+)")


def plain(text: str) -> str:
    """*text* with its LaTeX rewritten, code spans left alone."""
    if not text or ("\\" not in text and "$" not in text):
        return text
    out: list[str] = []
    end = 0
    for match in _CODE.finditer(text):
        out.append(_rewrite(text[end : match.start()]))
        out.append(match.group(0))
        end = match.end()
    out.append(_rewrite(text[end:]))
    return "".join(out)


def _rewrite(text: str) -> str:
    if not text:
        return text
    for delimiter in (_DISPLAY, _INLINE, _PAREN, _BRACKET):
        text = delimiter.sub(lambda m: m.group(1).strip(), text)
    # Before \frac, whose braces nest around these: {k_{on}}.
    text = _SUBSCRIPT.sub(r"\1", text)
    text = _SUPERSCRIPT.sub(r"^\1", text)
    text = _FRAC.sub(r"\1/\2", text)
    while True:
        unwrapped = _WRAPPERS.sub(r"\1", text)
        if unwrapped == text:
            break
        text = unwrapped
    # A unit prefix binds to the unit: the space in "\mu M" only ends the
    # command, where the one in "\pm 0.3" is really there.
    text = _PREFIX.sub(lambda m: _SYMBOLS[m.group(1)] + m.group(2), text)
    for command in sorted(_SYMBOLS, key=len, reverse=True):
        text = text.replace(command, _SYMBOLS[command])
    text = _NOISE.sub(" ", text)
    text = _ESCAPE.sub(r"\1", text)
    text = _LEFTOVER.sub(r"\1", text)
    # The substitutions above leave gaps where the delimiters were.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return re.sub(r" ([,.;:)])", r"\1", text)
