"""Stop-хук: после ответа Claude проверяет, изменился ли проект, и просит обновить документацию.

Отпечаток рабочей копии — дерево git всех файлов (с учётом .gitignore), кроме самой
документации: README, PDF-разбора, его пояснений и папки results/. Если отпечаток
отличается от прошлого, хук останавливает завершение ответа и просит основную сессию
вызвать агента docs-maintainer. Отпечаток сохраняется сразу, поэтому повторно хук
сработает только после новых изменений, а правки самого агента его не запускают.

Состояние лежит в .git/docs_sync_state и в репозиторий не попадает.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Что считается документацией: эти файлы меняет сам агент.
EXCLUDE = [
    ":(exclude,glob)**/README.md",
    ":(exclude,glob)**/*.pdf",
    ":(exclude,glob)docs/ml_freq_walkthrough/explanations/**",
    ":(exclude,glob)results/**",
]

REASON = (
    "В проекте есть изменения с прошлого обновления документации. Вызови агента "
    "docs-maintainer (инструмент Agent, subagent_type=docs-maintainer). В задаче кратко "
    "перечисли, что ты изменил в этом ответе и зачем: агент не видит переписку. "
    "Когда агент закончит, покажи пользователю, какие файлы документации он обновил, "
    "и его предложение описания коммита — дословно, в блоке кода. Если изменения "
    "касаются только служебных файлов и документация не затронута, агент так и ответит."
)


def git(*args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", env=env, check=True
    )
    return result.stdout.strip()


def fingerprint(root: Path) -> str:
    """Хэш дерева рабочей копии без документации; индекс пользователя не трогается."""
    git_dir = Path(git("rev-parse", "--absolute-git-dir"))
    work = Path(tempfile.mkdtemp(prefix="docs_sync_"))
    try:
        index = work / "index"
        if (git_dir / "index").exists():
            shutil.copyfile(git_dir / "index", index)  # копия — чтобы не хэшировать всё заново
        env = {**os.environ, "GIT_INDEX_FILE": str(index)}
        subprocess.run(
            ["git", "add", "-A", "--", ".", *EXCLUDE],
            cwd=root,
            env=env,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "rm", "-r", "--cached", "-q", "--ignore-unmatch", "--", *_includes()],
            cwd=root,
            env=env,
            capture_output=True,
            check=False,
        )
        return git("write-tree", env=env)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _includes() -> list[str]:
    """Те же шаблоны, но для удаления из копии индекса уже отслеживаемой документации."""
    return [pattern.replace(":(exclude,glob)", ":(glob)") for pattern in EXCLUDE]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {}
    if payload.get("stop_hook_active"):
        return 0  # уже продолжаем ответ по просьбе хука — не зацикливаемся

    root = Path(os.environ.get("CLAUDE_PROJECT_DIR") or git("rev-parse", "--show-toplevel"))
    os.chdir(root)
    state = Path(git("rev-parse", "--absolute-git-dir")) / "docs_sync_state"
    current = fingerprint(root)
    previous = state.read_text(encoding="utf-8").strip() if state.exists() else ""
    if current == previous:
        return 0
    state.write_text(current, encoding="utf-8")
    # ASCII-экранирование: вывод хука не зависит от кодовой страницы консоли Windows.
    print(json.dumps({"decision": "block", "reason": REASON}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
