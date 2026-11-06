# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The LaTeX a model writes, as the chat should show it."""

from reinvent4_agent.web.plaintext import plain


def test_inline_maths_loses_its_dollars():
    assert plain(r"Affinity $5.2 \pm 0.3$ $\mu M$") == "Affinity 5.2 +/- 0.3 \u00b5M"


def test_subscripts_and_comparisons():
    assert plain(r"IC$_{50}$ $\ge$ 3") == "IC50 >= 3"


def test_display_and_fractions():
    assert plain(r"\[ \frac{k_{on}}{k_{off}} \]") == "kon/koff"


def test_bare_commands_outside_maths():
    assert plain(r"solubility 45 \mu M") == "solubility 45 \u00b5M"


def test_plain_text_is_untouched():
    text = "Nothing special at all here."
    assert plain(text) is text


def test_code_is_left_alone():
    text = "SMILES `CC(=O)O` and ```\n\\pm stays\n```"
    assert plain(text) == text


def test_a_lone_dollar_is_not_maths():
    assert plain("costs $5 to buy") == "costs $5 to buy"
