"""Сборка «Построчного разбора ml_freq» (PDF) из исходного кода и пояснений.

Код в документ попадает автоматически: каждый модуль пакета ``gtm_detector/ml_freq``,
каждый класс, функция и метод — с номерами строк, как в редакторе. Пояснения и
примеры пишутся вручную в ``explanations/<путь модуля>.md`` (например,
``explanations/data/repair.md``), по разделу на объект::

    ## @module                         — о модуле в целом
    ## BlockShiftRepair  [1a2b3c4d]    — класс
    ## BlockShiftRepair.assess  [...]  — метод

В квадратных скобках — отпечаток кода объекта, для которого пояснение проверено.
Если код изменился, отпечаток не совпадёт и ``--check`` об этом скажет.

Запуск из корня проекта::

    python docs/ml_freq_walkthrough/build.py --check      # чего не хватает, что устарело
    python docs/ml_freq_walkthrough/build.py --stamp data/repair.py:BlockShiftRepair.assess
    python docs/ml_freq_walkthrough/build.py --stamp data/repair.py   # весь модуль проверен
    python docs/ml_freq_walkthrough/build.py              # собрать PDF

PDF печатает Microsoft Edge (или Chrome) в режиме без окна: ничего ставить не нужно.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import html
import io
import keyword
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "gtm_detector" / "ml_freq"
HERE = Path(__file__).resolve().parent
EXPLANATIONS = HERE / "explanations"
INTRO = EXPLANATIONS / "_intro.md"
OUTPUT = PACKAGE / "Построчный_разбор_ml_freq.pdf"
MODULE_KEY = "@module"

# Порядок разделов — по ходу данных: настройки → данные → признаки → модели → оценка →
# командная строка → тесты. Новые модули, которых здесь нет, добавляются в конец сами.
ORDER = [
    "__init__.py",
    "__main__.py",
    "config.py",
    "data/__init__.py",
    "data/models.py",
    "data/readers.py",
    "data/statistics.py",
    "data/repair.py",
    "data/dataset.py",
    "features/__init__.py",
    "features/builder.py",
    "models/__init__.py",
    "models/base.py",
    "models/classic.py",
    "models/transformer.py",
    "evaluation/__init__.py",
    "evaluation/decoding.py",
    "evaluation/metrics.py",
    "evaluation/experiment.py",
    "evaluation/report.py",
    "audit.py",
    "cli.py",
    "tests/conftest.py",
    "tests/test_data_and_features.py",
    "tests/test_models_and_evaluation.py",
    "tests/test_repair.py",
]

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
]


# --------------------------------------------------------------------------- код


@dataclass
class Item:
    """Объект кода, который получает свой раздел: класс, функция или метод."""

    key: str  # «BlockShiftRepair.assess»
    kind: str  # «метод», «класс», …
    lines: list[int]  # номера показываемых строк (с 1)
    signature: str


@dataclass
class Module:
    path: Path
    rel: str  # «data/repair.py»
    source: str
    items: list[Item] = field(default_factory=list)
    module_lines: list[int] = field(default_factory=list)  # код вне классов и функций
    hashes: dict[str, str] = field(default_factory=dict)

    @property
    def explanation_path(self) -> Path:
        return EXPLANATIONS / Path(self.rel).with_suffix(".md")


def _kind(node: ast.AST, in_class: bool) -> str:
    if isinstance(node, ast.ClassDef):
        names = {_decorator_name(d) for d in node.decorator_list}
        return "класс-данные (dataclass)" if "dataclass" in names else "класс"
    names = {_decorator_name(d) for d in node.decorator_list}
    if not in_class:
        return "функция"
    if "property" in names:
        return "свойство"
    if "classmethod" in names:
        return "метод класса"
    if "staticmethod" in names:
        return "статический метод"
    return "метод"


def _decorator_name(node: ast.expr) -> str:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    return node.id if isinstance(node, ast.Name) else ""


def _start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([d.lineno for d in decorators] + [node.lineno])


def _signature(node: ast.AST, source_lines: list[str]) -> str:
    """Заголовок объекта (def/class до двоеточия) одной строкой."""
    first_body = node.body[0].lineno
    header = source_lines[node.lineno - 1 : first_body - 1] or [source_lines[node.lineno - 1]]
    text = " ".join(line.strip() for line in header)
    return re.sub(r"\s+", " ", text).rstrip(":").strip()


def parse_module(path: Path) -> Module:
    rel = path.relative_to(PACKAGE).as_posix()
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    module = Module(path=path, rel=rel, source=source)
    tree = ast.parse(source)
    covered: set[int] = set()

    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        span = set(range(_start(node), node.end_lineno + 1))
        covered |= span
        if isinstance(node, ast.ClassDef):
            methods = [
                n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            method_lines = set()
            for m in methods:
                method_lines |= set(range(_start(m), m.end_lineno + 1))
            body = sorted(span - method_lines)
            module.items.append(Item(node.name, _kind(node, False), body, _signature(node, lines)))
            for m in methods:
                module.items.append(
                    Item(
                        f"{node.name}.{m.name}",
                        _kind(m, True),
                        list(range(_start(m), m.end_lineno + 1)),
                        _signature(m, lines),
                    )
                )
        else:
            module.items.append(
                Item(node.name, _kind(node, False), sorted(span), _signature(node, lines))
            )

    docstring_end = 0
    if (
        tree.body
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(getattr(tree.body[0], "value", None), ast.Constant)
        and isinstance(tree.body[0].value.value, str)
    ):
        docstring_end = tree.body[0].end_lineno
    module.module_lines = [
        i
        for i in range(1, len(lines) + 1)
        if i not in covered and (i > docstring_end or docstring_end == 0)
    ]
    module.module_lines = _trim_blank(module.module_lines, lines)

    for item in module.items:
        text = "\n".join(lines[i - 1] for i in item.lines)
        module.hashes[item.key] = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    text = "\n".join(lines[i - 1] for i in module.module_lines)
    module.hashes[MODULE_KEY] = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return module


def _trim_blank(numbers: list[int], lines: list[str]) -> list[int]:
    """Убрать пустые строки в начале и конце и сжать подряд идущие пустые строки."""
    result: list[int] = []
    for i in numbers:
        blank = not lines[i - 1].strip()
        if blank and (not result or not lines[result[-1] - 1].strip()):
            continue
        result.append(i)
    while result and not lines[result[-1] - 1].strip():
        result.pop()
    return result


def discover() -> list[Module]:
    files = [
        p
        for p in PACKAGE.rglob("*.py")
        if "__pycache__" not in p.parts and p.read_text(encoding="utf-8").strip()
    ]
    rank = {name: i for i, name in enumerate(ORDER)}
    files.sort(key=lambda p: (rank.get(p.relative_to(PACKAGE).as_posix(), len(ORDER)), str(p)))
    return [parse_module(p) for p in files]


# --------------------------------------------------------------------- пояснения

HEADING = re.compile(r"^## (\S+)(?:\s+\[([0-9a-f]{8})\])?\s*$")


@dataclass
class Entry:
    key: str
    stamp: str | None
    text: str


def read_explanations(path: Path) -> dict[str, Entry]:
    if not path.exists():
        return {}
    entries: dict[str, Entry] = {}
    key, stamp, buffer = None, None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING.match(line)
        if match:
            if key is not None:
                entries[key] = Entry(key, stamp, "\n".join(buffer).strip())
            key, stamp, buffer = match.group(1), match.group(2), []
        elif key is not None:
            buffer.append(line)
    if key is not None:
        entries[key] = Entry(key, stamp, "\n".join(buffer).strip())
    return entries


def check(modules: list[Module]) -> list[str]:
    problems = []
    for module in modules:
        entries = read_explanations(module.explanation_path)
        expected = [MODULE_KEY] + [item.key for item in module.items]
        for key in expected:
            entry = entries.get(key)
            where = f"{module.rel} :: {key}"
            if entry is None or not entry.text:
                problems.append(f"нет пояснения      {where}")
            elif entry.stamp != module.hashes[key]:
                problems.append(f"код изменился      {where}")
        for key in entries.keys() - set(expected):
            problems.append(f"объекта нет в коде {module.rel} :: {key}")
    return problems


def stamp(modules: list[Module], targets: list[str]) -> list[str]:
    """Записать текущие отпечатки кода для проверенных пояснений."""
    by_rel = {m.rel: m for m in modules}
    done = []
    for target in targets:
        rel, _, key = target.partition(":")
        module = by_rel.get(rel)
        if module is None:
            raise SystemExit(f"Нет модуля {rel}")
        path = module.explanation_path
        if not path.exists():
            raise SystemExit(f"Нет файла пояснений {path}")
        keys = {key} if key else set(module.hashes)
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            match = HEADING.match(line)
            if match and match.group(1) in keys and match.group(1) in module.hashes:
                lines[i] = f"## {match.group(1)}  [{module.hashes[match.group(1)]}]"
                done.append(f"{rel}:{match.group(1)}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return done


# ------------------------------------------------------------------- разметка


def inline(text: str) -> str:
    parts = text.split("`")
    out = []
    for i, part in enumerate(parts):
        if i % 2:
            out.append(f"<code>{html.escape(part)}</code>")
        else:
            escaped = html.escape(part)
            escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
            out.append(escaped)
    return "".join(out)


def markdown(text: str) -> str:
    """Небольшое подмножество Markdown: абзацы, списки, таблицы, код, цитаты, ###."""
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            block = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                block.append(lines[i])
                i += 1
            i += 1
            code = "\n".join(block)
            body = "\n".join(highlight(code)) if lang in ("python", "py") else html.escape(code)
            out.append(f'<pre class="example">{body}</pre>')
            continue
        if stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i].strip().strip("|").split("|"))
                i += 1
            if len(rows) > 1 and all(set(c.strip()) <= set("-: ") for c in rows[1]):
                head, body_rows = rows[0], rows[2:]
            else:
                head, body_rows = None, rows
            table = ["<table>"]
            if head:
                table.append(
                    "<tr>" + "".join(f"<th>{inline(c.strip())}</th>" for c in head) + "</tr>"
                )
            for row in body_rows:
                table.append(
                    "<tr>" + "".join(f"<td>{inline(c.strip())}</td>" for c in row) + "</tr>"
                )
            out.append("".join(table) + "</table>")
            continue
        if stripped.startswith("### "):
            out.append(f"<h4>{inline(stripped[4:])}</h4>")
            i += 1
            continue
        if stripped.startswith("> "):
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            out.append(f'<div class="note">{inline(" ".join(quote))}</div>')
            continue
        bullet = re.match(r"^([-*]|\d+\.)\s+", stripped)
        if bullet:
            tag = "ol" if bullet.group(1)[0].isdigit() else "ul"
            items: list[str] = []
            while i < len(lines):
                current = lines[i].strip()
                match = re.match(r"^([-*]|\d+\.)\s+(.*)", current)
                if match:
                    items.append(match.group(2))
                elif current and lines[i].startswith((" ", "\t")) and items:
                    items[-1] += " " + current
                else:
                    break
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        paragraph = []
        while (
            i < len(lines)
            and lines[i].strip()
            and not re.match(r"^(```|\||### |> |[-*]\s|\d+\.\s)", lines[i].strip())
        ):
            paragraph.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(paragraph))}</p>")
    return "\n".join(out)


# ---------------------------------------------------------------- подсветка кода


def highlight(source: str) -> list[str]:
    """Строки кода в HTML с подсветкой; при ошибке разбора — просто экранированный текст."""
    lines = source.splitlines()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source + "\n").readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return [html.escape(line) for line in lines]
    result = [""] * len(lines)
    cursor = [(1, 0)]

    def emit(row: int, col_from: int, col_to: int | None, css: str | None) -> None:
        if row > len(lines):
            return
        text = lines[row - 1][col_from:col_to]
        if not text:
            return
        escaped = html.escape(text)
        result[row - 1] += f'<span class="{css}">{escaped}</span>' if css else escaped

    def fill(start: tuple[int, int], end: tuple[int, int], css: str | None) -> None:
        (r0, c0), (r1, c1) = start, end
        if r0 == r1:
            emit(r0, c0, c1, css)
            return
        emit(r0, c0, None, css)
        for r in range(r0 + 1, r1):
            emit(r, 0, None, css)
        emit(r1, 0, c1, css)

    for token in tokens:
        if token.type in (tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL):
            continue
        fill(cursor[0], token.start, None)
        css = None
        if token.type == tokenize.COMMENT:
            css = "cm"
        elif token.type == tokenize.STRING or token.type in _FSTRING_TYPES:
            css = "st"
        elif token.type == tokenize.NUMBER:
            css = "nu"
        elif token.type == tokenize.NAME and keyword.iskeyword(token.string):
            css = "kw"
        elif token.type == tokenize.NAME and token.string in (
            "self",
            "cls",
            "None",
            "True",
            "False",
        ):
            css = "kw"
        fill(token.start, token.end, css)
        cursor[0] = token.end
    return result


_FSTRING_TYPES = {
    getattr(tokenize, name)
    for name in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END")
    if hasattr(tokenize, name)
}


# ------------------------------------------------------------------------- HTML

CSS = """
@page { size: A4; margin: 16mm 14mm 16mm 14mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10pt; color: #1d1d1b;
       line-height: 1.45; }
h1 { font-size: 19pt; margin: 0 0 6pt; color: #12385f; page-break-before: always;
     border-bottom: 2px solid #2a78d6; padding-bottom: 4pt; }
h1.first { page-break-before: avoid; }
h2 { font-size: 13pt; margin: 18pt 0 4pt; color: #12385f; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; color: #444; }
h4 { font-size: 10.5pt; margin: 10pt 0 2pt; }
.kind { display: inline-block; font-size: 8pt; font-weight: 600; color: #2a78d6;
        background: #e8f1fc; border-radius: 3px; padding: 1px 6px; margin-right: 6px;
        vertical-align: middle; }
.sig { font-family: Consolas, "Courier New", monospace; font-size: 8.5pt; color: #555;
       margin: 2pt 0 6pt; word-break: break-word; }
.code { font-family: Consolas, "Courier New", monospace; font-size: 8pt; line-height: 1.35;
        background: #f6f6f3; border: 1px solid #e3e2dc; border-radius: 4px;
        padding: 4pt 6pt; margin: 4pt 0 8pt; }
.ln { display: flex; white-space: pre-wrap; word-break: break-word; }
.ln .n { flex: 0 0 30pt; color: #9a9993; text-align: right; padding-right: 8pt;
         user-select: none; }
.ln .c { flex: 1; }
.gap { color: #9a9993; padding-left: 38pt; }
pre.example { font-family: Consolas, "Courier New", monospace; font-size: 8pt;
              background: #f3f7fc; border-left: 3px solid #2a78d6; padding: 5pt 8pt;
              white-space: pre-wrap; word-break: break-word; margin: 4pt 0 8pt; }
code { font-family: Consolas, "Courier New", monospace; font-size: 8.8pt;
       background: #f1f0ec; padding: 0 2px; border-radius: 2px; }
.kw { color: #7a3eb1; font-weight: 600; } .st { color: #2c7a3a; }
.cm { color: #8a8a82; font-style: italic; }
.nu { color: #b5541a; }
table { border-collapse: collapse; margin: 4pt 0 8pt; font-size: 9pt; }
th, td { border: 1px solid #d9d8d2; padding: 2pt 6pt; text-align: left; vertical-align: top; }
th { background: #eef2f6; }
.note { background: #fff7e6; border-left: 3px solid #e3a21a; padding: 4pt 8pt; margin: 4pt 0 8pt; }
.missing { background: #fdecea; border-left: 3px solid #d64545; padding: 4pt 8pt; color: #8a1f1f; }
.item { page-break-inside: auto; }
.toc a { color: #12385f; text-decoration: none; }
.toc ul { list-style: none; padding-left: 12pt; margin: 2pt 0; }
.toc > ul { padding-left: 0; }
.toc li.mod { margin-top: 5pt; font-weight: 600; }
.toc li.obj { font-weight: normal; font-size: 9pt; }
.title { font-size: 26pt; color: #12385f; margin: 40pt 0 4pt; }
.subtitle { color: #555; margin-bottom: 20pt; }
"""


def anchor(*parts: str) -> str:
    return "a-" + hashlib.sha1("/".join(parts).encode("utf-8")).hexdigest()[:10]


def code_block(module: Module, numbers: list[int], highlighted: list[str]) -> str:
    rows = []
    previous = None
    for n in numbers:
        if previous is not None and n != previous + 1:
            rows.append('<div class="gap">⋯</div>')
        rows.append(
            f'<div class="ln"><span class="n">{n}</span>'
            f'<span class="c">{highlighted[n - 1] or " "}</span></div>'
        )
        previous = n
    return f'<div class="code">{"".join(rows)}</div>'


def explanation_html(entry: Entry | None) -> str:
    if entry is None or not entry.text:
        return '<div class="missing">Пояснение ещё не написано.</div>'
    return markdown(entry.text)


def render(modules: list[Module]) -> str:
    parts = [
        "<!doctype html><html lang='ru'><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>"
    ]
    parts.append("<div class='title'>Построчный разбор пакета ml_freq</div>")
    total = sum(len(m.items) for m in modules)
    parts.append(
        f"<div class='subtitle'>gtm_detector/ml_freq — {len(modules)} модулей, {total} классов, "
        f"функций и методов. Собрано {date.today():%d.%m.%Y} командой "
        f"<code>python docs/ml_freq_walkthrough/build.py</code>.</div>"
    )
    toc = ["<div class='toc'><h3>Содержание</h3><ul>"]
    if INTRO.exists():
        toc.append(f"<li class='mod'><a href='#{anchor('intro')}'>Как устроен пакет</a></li>")
    for module in modules:
        toc.append(f"<li class='mod'><a href='#{anchor(module.rel)}'>{module.rel}</a><ul>")
        for item in module.items:
            toc.append(
                f"<li class='obj'><a href='#{anchor(module.rel, item.key)}'>"
                f"{html.escape(item.key)}</a></li>"
            )
        toc.append("</ul></li>")
    toc.append("</ul></div>")
    parts.extend(toc)

    if INTRO.exists():
        parts.append(f"<h1 id='{anchor('intro')}'>Как устроен пакет</h1>")
        parts.append(markdown(INTRO.read_text(encoding="utf-8")))

    for module in modules:
        entries = read_explanations(module.explanation_path)
        highlighted = highlight(module.source)
        parts.append(f"<h1 id='{anchor(module.rel)}'>{module.rel}</h1>")
        parts.append(explanation_html(entries.get(MODULE_KEY)))
        if module.module_lines:
            parts.append("<h3>Код модуля вне классов и функций</h3>")
            parts.append(code_block(module, module.module_lines, highlighted))
        for item in module.items:
            parts.append(f"<div class='item'><h2 id='{anchor(module.rel, item.key)}'>")
            parts.append(f"<span class='kind'>{item.kind}</span>{html.escape(item.key)}</h2>")
            parts.append(f"<div class='sig'>{html.escape(item.signature)}</div>")
            parts.append(code_block(module, item.lines, highlighted))
            parts.append(explanation_html(entries.get(item.key)))
            parts.append("</div>")
    parts.append("</body></html>")
    return "\n".join(parts)


def browser() -> str:
    custom = os.environ.get("WALKTHROUGH_BROWSER")
    for candidate in [custom, *EDGE_CANDIDATES]:
        if candidate and Path(candidate).exists():
            return candidate
    for name in ("msedge", "chrome", "chromium", "google-chrome"):
        found = shutil.which(name)
        if found:
            return found
    raise SystemExit("Не найден Edge или Chrome; укажите путь в WALKTHROUGH_BROWSER")


def build_pdf(modules: list[Module], output: Path) -> Path:
    work = Path(tempfile.mkdtemp(prefix="walkthrough_"))
    try:
        page = work / "walkthrough.html"
        page.write_text(render(modules), encoding="utf-8")
        pdf = work / "walkthrough.pdf"
        command = [
            browser(),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--user-data-dir={work / 'profile'}",
            f"--print-to-pdf={pdf}",
            page.as_uri(),
        ]
        subprocess.run(command, check=True, timeout=300, capture_output=True)
        if not pdf.exists():
            raise SystemExit("Браузер не создал PDF")
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pdf, output)
        return output
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Построчный разбор ml_freq в PDF.")
    parser.add_argument("--check", action="store_true", help="только проверить пояснения")
    parser.add_argument(
        "--stamp", nargs="+", metavar="МОДУЛЬ[:ОБЪЕКТ]", help="отметить пояснения проверенными"
    )
    parser.add_argument("--html", type=Path, help="сохранить также HTML (для просмотра)")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="куда сохранить PDF")
    args = parser.parse_args(argv)

    modules = discover()
    if args.stamp:
        for line in stamp(modules, args.stamp):
            print(f"проверено: {line}")
        return 0
    problems = check(modules)
    if args.check:
        print("\n".join(problems) if problems else "Все пояснения на месте и актуальны.")
        return 1 if problems else 0
    if args.html:
        args.html.write_text(render(modules), encoding="utf-8")
    path = build_pdf(modules, args.output)
    print(f"PDF: {path}")
    if problems:
        print(f"Внимание: {len(problems)} пояснений нет или устарели (см. --check).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
