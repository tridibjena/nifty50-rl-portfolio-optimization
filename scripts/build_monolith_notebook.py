#!/usr/bin/env python3
"""Flatten the whole package into one self-contained Jupyter notebook.

    python3 scripts/build_monolith_notebook.py [--execute]

The result, ``notebooks/00_full_pipeline.ipynb``, carries **every line of source** in one
document: no imports from ``src/``, no local install, nothing outside the notebook except
third-party libraries. Run it top to bottom and it reproduces the published figures.

This exists because a single notebook was asked for. It is generated rather than
hand-written, which keeps two things true: the notebook cannot drift from ``src/`` while
both exist, and regenerating is a command rather than a merge.

**What flattening costs.** Order is no longer enforced by imports -- it is enforced by
cell order, and a notebook lets you run cells in any order you like. The package form
cannot express a dependency cycle; this form can produce one at runtime by executing
cells out of sequence. The test suite does not come across (pytest collects modules, not
notebooks), so the twenty regression tests and the two CI guards -- ``assert_causal`` and
``assert_no_narration_leak`` -- no longer run against this file. They still run against
``src/`` for as long as it is kept.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE = PROJECT_ROOT / "src" / "nifty_rl"
PIPELINE = PROJECT_ROOT / "scripts" / "run_pipeline.py"
OUTPUT = PROJECT_ROOT / "notebooks" / "00_full_pipeline.ipynb"

FUTURE_IMPORT = "from __future__ import annotations"

#: Grouped for the notebook's section headings. Modules not listed still appear, in
#: dependency order, under "Everything else" -- so adding a module cannot silently drop it.
SECTIONS = [
    (
        "Configuration",
        "Every number you might want to change lives here, and nowhere else. The "
        "dataclasses are frozen, so nothing can quietly mutate halfway through a run.",
        ["config"],
    ),
    (
        "Data and features",
        "Prices come from Yahoo and get cached on disk, so the second run is fast and the "
        "hundredth run gives the same answer as the first. Then the usual indicators — "
        "moving averages, RSI, ATR, volatility, a VIX overlay.",
        ["data", "features"],
    ),
    (
        "What a trade actually costs",
        "This is the part most backtests wave away. Indian delivery equity carries STT, "
        "stamp duty, exchange and SEBI fees, and GST on top of the brokerage — and the "
        "whole stack gets folded into the fill price, so no strategy can forget to pay "
        "it. Two backtesters follow: one for hold/don't-hold signals, one for portfolio "
        "weights. They charge identically, which matters, because otherwise any gap "
        "between the agent and the baselines would partly be an artefact of how they "
        "execute rather than what they decide.",
        ["backtest.costs", "backtest.engine", "backtest.weights"],
    ),
    (
        "The strategies",
        "Three kinds. Simple rules that say hold or don't. Classical portfolio "
        "optimisers that emit weights. And overlays that condition either on the market "
        "regime. There is also a deliberately random strategy in here — it is the control, "
        "and it tells you what a coin flip earns over the same stretch. Without it you "
        "have no idea whether a positive return means anything.",
        ["strategies.signals", "strategies.allocators", "strategies.meta"],
    ),
    (
        "Regime detection",
        "The idea is that markets have moods — calm stretches, nervous ones, outright "
        "crises — and that knowing which one you are in might be worth something.\n\n"
        "The hard part is honesty. It is trivially easy to build a regime label that "
        "looks brilliant and is useless, because it was computed with knowledge of how "
        "the period ended. Every detector below is held to one rule: **the label for a "
        "given day may only use that day and the days before it.** Five different "
        "detectors are implemented so they can be checked against each other.",
        ["regimes.base", "regimes.features", "regimes.hmm", "regimes.threshold",
         "regimes.jump", "regimes.changepoint", "regimes.evaluate"],
    ),
    (
        "The reinforcement learning agent",
        "A PPO agent that allocates across the ten stocks and cash. The simulation it "
        "trades in is written in plain NumPy and kept separate from the gymnasium "
        "wrapper, so the execution logic can be reasoned about on its own — that is "
        "where the subtle bugs live.",
        ["envs.panel", "envs.rewards", "envs.core", "envs.multistock", "agents.train"],
    ),
    (
        "Measuring the results",
        "Two different questions, deliberately kept apart. The first file answers *what "
        "happened* — returns, Sharpe, drawdown, how much of the time you were actually "
        "invested. The second answers *how much of it to believe*, which is the harder "
        "and more important one when you have tried this many strategies.",
        ["metrics.performance", "metrics.stats"],
    ),
    (
        "Walk-forward evaluation — the heart of it",
        "If you take one thing from this notebook, take this.\n\n"
        "A single train/test split gives you exactly one out-of-sample window, and "
        "whatever the market happened to do in that window *is* your result. Change the "
        "split and the answer changes with it. That is one draw from a distribution, not "
        "an evaluation.\n\n"
        "So instead: fit on everything up to a point, trade the next six months with the "
        "parameters frozen, roll forward, refit, repeat. Chain all the out-of-sample "
        "blocks into one continuous track record. Everything above — the scaler, the "
        "regime model, the strategy choice, the PPO agent — is refit *inside* each "
        "window, on that window's training data only.",
        ["validation.walkforward"],
    ),
    (
        "Figures and the report",
        "Charts, and the code that writes RESULTS.md. Every figure also saves the table "
        "behind it as a CSV, because a chart whose numbers you can only get by measuring "
        "pixels is not really a result.",
        ["report.theme", "report.figures", "report.narrate", "report.build"],
    ),
]

#: A plain-English line above each module. The docstrings inside go into the *why* in
#: detail; this is the one-sentence version you would say out loud.
MODULE_NOTES = {
    "config": "Every knob, in one place. Note that the end date is pinned — the original "
              "version let it float, so the published numbers quietly changed on every run.",
    "data": "Download and cache. Keeping every price on one adjusted scale matters: mixing "
            "adjusted closes with raw highs used to inject fake spikes into the volatility.",
    "features": "The indicators. RSI and ATR use Wilder's smoothing rather than a plain "
                "moving average — they genuinely differ, and it propagates into every signal.",
    "backtest.costs": "The full Indian charge stack, folded into the price you actually pay.",
    "backtest.engine": "Backtester for hold/don't-hold signals; each stock gets its own cash. "
                       "A stop-loss now blocks re-entry until the signal goes flat — without "
                       "that, stops paid round-trip costs and protected nothing.",
    "backtest.weights": "Backtester for weight schedules. Rebalances on schedule and drifts "
                        "in between, like a real monthly-rebalanced fund would.",
    "strategies.signals": "The rules — moving averages, RSI, breakout — plus the random one "
                          "that acts as the control.",
    "strategies.allocators": "Classical portfolio construction: equal weight, minimum "
                             "variance, maximum Sharpe, risk parity, HRP. These are the fair "
                             "opponents for an agent that emits weights.",
    "strategies.meta": "Overlays that use the regime — dial exposure down when things look "
                       "bad, or switch strategy entirely.",
    "regimes.base": "The contract every detector signs. Fit however you like; predict using "
                    "only the past.",
    "regimes.features": "What the detectors actually look at — volatility, trend, breadth, "
                        "how correlated everything has become. Market-wide, not per-stock.",
    "regimes.hmm": "A hidden Markov model, written out by hand. Not because the library is "
                   "bad, but because its two most obvious methods both read the future, and "
                   "the guarantee needed to be structural rather than a matter of "
                   "remembering which function to call.",
    "regimes.threshold": "The simple version: split on volatility quantiles. If the HMM "
                         "cannot beat this, its extra complexity is not earning anything.",
    "regimes.jump": "Another backend, with an explicit penalty for switching too often.",
    "regimes.changepoint": "Finds breaks by looking at the whole series at once — which "
                           "makes it useless for trading and ideal as a yardstick.",
    "regimes.evaluate": "Is the detector any good? How fast does it react, how long do its "
                        "regimes last, and does 'state 0' still mean the same thing after "
                        "a refit?",
    "envs.panel": "Flattens everything into dense arrays so the simulator does no pandas "
                  "work while stepping.",
    "envs.rewards": "Swappable reward functions. The differential Sharpe one is the "
                    "interesting one — risk aversion falls out of the objective instead of "
                    "being bolted on with four hand-tuned constants.",
    "envs.core": "The simulation itself. Sells settle before buys — when they were "
                 "interleaved, a sale of the ninth stock could not fund a purchase of the "
                 "first, and the agent literally could not execute its own decisions.",
    "envs.multistock": "The gymnasium wrapper. Deliberately thin; all the logic is above.",
    "agents.train": "Trains PPO, keeps the best checkpoint by Sharpe on data it has not "
                    "trained on, and runs several seeds — because a single seed is an "
                    "anecdote, not a result.",
    "metrics.performance": "What happened: returns, Sharpe, Sortino, drawdown, and how much "
                           "of the time the money was actually invested.",
    "metrics.stats": "How much to believe it. Confidence intervals, a correction for having "
                     "tried many strategies, and a test for whether the selection itself is "
                     "just noise.",
    "validation.walkforward": "The outer loop. Everything above runs inside this.",
    "report.theme": "One consistent look for every chart. Colour carries meaning here rather "
                    "than decoration.",
    "report.figures": "Every chart, each one saving the table behind it alongside.",
    "report.narrate": "Plain-English commentary on each regime episode. Presentation only — "
                      "it never feeds back into anything.",
    "report.build": "Assembles RESULTS.md.",
}


def module_name(path: Path) -> str:
    return str(path.relative_to(PACKAGE)).replace(".py", "").replace("/", ".")


def dependency_order() -> list:
    """Topologically sort the package by its own relative imports.

    Computed rather than hard-coded: a hand-maintained list silently rots the first time
    someone adds a module, and the failure mode is a NameError halfway down a notebook.
    """
    paths = [p for p in sorted(PACKAGE.rglob("*.py")) if p.name != "__init__.py"]
    names = {module_name(p) for p in paths}
    deps = collections.defaultdict(set)

    for path in paths:
        name = module_name(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.level:
                target = node.module or ""
                match = next(
                    (c for c in names
                     if c == target or c.endswith("." + target)
                     or c.split(".")[-1] == target.split(".")[-1]),
                    None,
                )
                if match and match != name:
                    deps[name].add(match)

    order, seen = [], set()

    def visit(name, stack=()):
        if name in seen:
            return
        if name in stack:
            raise SystemExit(f"Import cycle: {' -> '.join(stack + (name,))}")
        for dep in sorted(deps[name]):
            visit(dep, stack + (name,))
        seen.add(name)
        order.append(name)

    for name in sorted(names):
        visit(name)
    return order


def strip_imports(source: str, drop_absolute_prefix: str = "") -> str:
    """Remove the future import and every in-package import from a module.

    In one flat namespace those imports are meaningless -- and a relative import would be
    an outright error, since there is no package to be relative to. Only *top-level*
    statements are considered, so a function-local ``import`` stays where it is.
    """
    tree = ast.parse(source)
    lines = source.splitlines()
    top_level = {id(node) for node in tree.body}
    drop, comment_out = set(), {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        is_future = node.module == "__future__"
        is_relative = bool(node.level)
        is_package = bool(drop_absolute_prefix) and (node.module or "").startswith(
            drop_absolute_prefix
        )
        if not (is_future or is_relative or is_package):
            continue

        span = range(node.lineno - 1, (node.end_lineno or node.lineno))
        if id(node) in top_level:
            drop.update(span)
        else:
            # Function-local import. Deleting the line outright would empty its block if
            # it were the only statement, so replace it in place with a comment at the
            # same indentation -- the names are module-level globals once flattened.
            indent = " " * (len(lines[node.lineno - 1]) - len(lines[node.lineno - 1].lstrip()))
            names = ", ".join(alias.name for alias in node.names)
            comment_out[node.lineno - 1] = f"{indent}# flattened: {names} is defined above"
            drop.update(i for i in span if i != node.lineno - 1)

    kept = [
        comment_out.get(i, line)
        for i, line in enumerate(lines)
        if i not in drop
    ]
    return "\n".join(kept).strip("\n")


def split_docstring(source: str):
    """Separate a leading module docstring from the rest. Returns ``(docstring, body)``."""
    tree = ast.parse(source)
    if not tree.body:
        return "", source
    first = tree.body[0]
    is_docstring = (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    )
    if not is_docstring:
        return "", source

    lines = source.splitlines()
    end = first.end_lineno or first.lineno
    return "\n".join(lines[first.lineno - 1:end]), "\n".join(lines[end:]).strip("\n")


def module_cell(name: str) -> str:
    path = PACKAGE / (name.replace(".", "/") + ".py")
    body = strip_imports(path.read_text())

    if name == "config":
        # PROJECT_ROOT is derived from __file__, which does not exist in a notebook.
        # _PROJECT_ROOT is established in the preamble by walking up to pyproject.toml.
        body = body.replace(
            "PROJECT_ROOT = Path(__file__).resolve().parents[2]",
            "PROJECT_ROOT = _PROJECT_ROOT  # notebook: injected by the preamble cell",
        )

    origin = f"nifty_rl/{name.replace('.', '/')}.py"
    note = MODULE_NOTES.get(name, "")
    header = f"# ── {origin} " + "─" * max(4, 84 - len(origin))
    if note:
        header += "\n" + textwrap.fill(note, width=86, initial_indent="# ", subsequent_indent="# ")

    # Keep the module docstring first so it stays a docstring. `from __future__` is only
    # allowed after a docstring and before anything else, so the order is forced:
    # comment, docstring, future import, code. Emitting the future import first would
    # demote the docstring to a stray string literal sitting in the middle of the cell.
    docstring, body = split_docstring(body)
    parts = [header]
    if docstring:
        parts.append(docstring)
    parts.append(FUTURE_IMPORT)
    return "\n".join(parts) + f"\n\n{body}\n"


def strip_main_guard(source: str) -> str:
    """Remove ``if __name__ == "__main__":`` blocks.

    In a notebook ``__name__`` *is* ``"__main__"``, so the guard fires on cell execution.
    Here that meant ``parse_args()`` reading the kernel's own argv and aborting the
    notebook with ``unrecognized arguments: -f /.../kernel.json``. The run is triggered
    explicitly from the final cell instead.
    """
    lines = source.splitlines()
    drop = set()
    for node in ast.parse(source).body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        is_guard = (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and any(getattr(c, "value", None) == "__main__" for c in test.comparators)
        )
        if is_guard:
            drop.update(range(node.lineno - 1, (node.end_lineno or node.lineno)))
    return "\n".join(line for i, line in enumerate(lines) if i not in drop).strip("\n")


def pipeline_cell() -> str:
    source = PIPELINE.read_text()
    body = strip_main_guard(strip_imports(source, drop_absolute_prefix="nifty_rl"))

    # The sys.path dance and the __file__-derived root are both script-only.
    body = body.replace("PROJECT_ROOT = Path(__file__).resolve().parents[1]",
                        "PROJECT_ROOT = _PROJECT_ROOT")
    body = body.replace('sys.path.insert(0, str(PROJECT_ROOT / "src"))', "")
    body = body.replace("#!/usr/bin/env python3\n", "")

    header = "# ── the driver " + "─" * 71
    return (
        f"{header}\n"
        "# Everything above was definitions. This is the part that actually does something:\n"
        "# load the data, fit the regimes, run the walk-forward, work out what survives,\n"
        "# draw the charts, write the report.\n"
        f"{FUTURE_IMPORT}\n\n{body}\n"
    )


PREAMBLE = '''# Where are we? Walk up the folder tree until we find the project root, so this works
# whether you opened the notebook from notebooks/ or from the repository root.
from pathlib import Path

_here = Path.cwd()
_PROJECT_ROOT = next(
    (p for p in [_here, *_here.parents] if (p / "pyproject.toml").exists()),
    _here,
)
print("project root:", _PROJECT_ROOT)
print("results and figures will be written under here.")
'''

FIGURES_SHIM = '''# A small piece of glue. The driver below was written against the package, where it
# says things like figures.equity_curves(...) — with `figures` being a module. Flattened
# into one namespace there is no module to reach through, so we build one here and point
# it at everything defined so far. Saves rewriting a dozen call sites.
import types as _types

figures = _types.ModuleType("figures")
figures.__dict__.update({k: v for k, v in globals().items() if not k.startswith("__")})
'''


def build(execute: bool = False, smoke: bool = False, output: Path = OUTPUT) -> Path:
    import nbformat as nbf

    order = dependency_order()
    placed, cells = set(), []

    def markdown(text):
        cells.append(nbf.v4.new_markdown_cell(text))

    def code(text, label="cell"):
        # Compile every cell as it is emitted. A syntax error introduced by stripping
        # imports would otherwise surface as a failure halfway through a 20-minute run.
        try:
            compile(text, f"<{label}>", "exec")
        except SyntaxError as exc:
            raise SystemExit(f"generated cell {label!r} does not compile: {exc}") from exc
        cells.append(nbf.v4.new_code_cell(text))

    markdown(
        "# Regime-Aware Portfolio Research — NIFTY 50\n\n"
        "Ten large-cap Indian stocks, six years of daily prices, and a question: can a "
        "reinforcement-learning agent allocate between them better than a handful of "
        "well-understood classical methods?\n\n"
        "The short answer is no — and most of the work here goes into making sure that "
        "answer is trustworthy rather than into avoiding it.\n\n"
        "Everything is in this one notebook: configuration, data handling, feature "
        "construction, five regime detectors, the strategies, a realistic Indian cost "
        "model, two backtesters, the RL environment and agent, the statistics, and the "
        "reporting. Nothing is imported from elsewhere in the project.\n\n"
        "**How to run it:** top to bottom, once. The cells are ordered so that everything "
        "is defined before it is used, which means the order is not cosmetic — if you "
        "jump around and hit a `NameError`, restart the kernel and run all.\n\n"
        "The last cell does the actual work and takes about twenty minutes, most of it "
        "training the agent. There is a ninety-second version noted there if you just "
        "want to see it run.\n\n"
        "---\n\n"
        "*A note on where this file comes from: it is generated from the `src/` package by "
        "`scripts/build_monolith_notebook.py`. If you change something here, change it "
        "there too — otherwise the next regeneration will quietly overwrite you.*"
    )
    code(PREAMBLE, "preamble")

    for title, blurb, names in SECTIONS:
        present = [n for n in order if n in names]
        if not present:
            continue
        markdown(f"## {title}\n\n{blurb}")
        for name in present:
            code(module_cell(name), name)
            placed.add(name)
            if name == "report.figures":
                code(FIGURES_SHIM, "figures-shim")

    leftover = [n for n in order if n not in placed]
    if leftover:
        markdown(
            "## Everything else\n\n"
            "Modules that were added after this notebook's section list was written. They "
            "are still in dependency order, so they run correctly — they just have not "
            "been given a home yet."
        )
        for name in leftover:
            code(module_cell(name), name)

    markdown(
        "## Putting it to work\n\n"
        "That is the whole library. What follows is the script that uses it — load the "
        "data, fit the regime models, check they are worth trusting, run every strategy "
        "through the walk-forward, test what survives, and write it all out."
    )
    code(pipeline_cell(), "run_pipeline")

    markdown(
        "## Run\n\n"
        "This is the one that takes twenty minutes. Almost all of it is PPO — eight "
        "windows, three seeds each, trained from scratch every time.\n\n"
        "In a hurry? Swap the last line for `main(parse_args([\"--no-rl\"]))`. It skips "
        "only the agent, finishes in about ninety seconds, and everything else is "
        "identical.\n\n"
        "Output lands in `results/` (the tables), `assets/v2/` (the charts, each with its "
        "numbers alongside), and `RESULTS.md` (the write-up)."
    )
    if smoke:
        code('main(parse_args(["--no-rl"]))  # smoke build: skips PPO', "run")
    else:
        code(
            "# The full run. Go and make a coffee.\n"
            "#   Quick version, skips the agent:  main(parse_args([\"--no-rl\"]))\n"
            "main(parse_args([]))",
            "run",
        )

    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    })

    if execute:
        from nbconvert.preprocessors import ExecutePreprocessor

        print("executing (this runs the full pipeline) ...")
        ExecutePreprocessor(timeout=3600, kernel_name="python3").preprocess(
            notebook, {"metadata": {"path": str(output.parent)}}
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, str(output))
    return output


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true",
                        help="Run the notebook after generating it (executes the full pipeline).")
    parser.add_argument("--smoke", metavar="PATH", default=None,
                        help="Verification build: emit to PATH with PPO skipped and execute it. "
                             "Proves the flattening works in ~90s instead of ~20min.")
    args = parser.parse_args(argv)

    if args.smoke:
        path = build(execute=True, smoke=True, output=Path(args.smoke))
    else:
        path = build(args.execute)
    size_kb = path.stat().st_size / 1024
    n_code = sum(1 for c in json.loads(path.read_text())["cells"] if c["cell_type"] == "code")
    try:
        shown = path.relative_to(PROJECT_ROOT)
    except ValueError:
        shown = path
    print(f"wrote {shown}  ({size_kb:.0f} KB, {n_code} code cells)")


if __name__ == "__main__":
    main()
