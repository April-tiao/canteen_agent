from __future__ import annotations

import os
from pathlib import Path

from dulwich import porcelain
from dulwich.repo import Repo


REMOTE_URL = "https://github.com/April-tiao/canteen_agent.git"
BRANCH = "main"
COMMIT_MESSAGE = b"Initial canteen gateway project"
AUTHOR = b"Codex <codex@example.com>"


SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".venv", "venv", "env"}
SKIP_SUFFIXES = {".pyc", ".log"}


def iter_project_files(root: Path) -> list[str]:
    files: list[str] = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        files.append(str(relative).replace("\\", "/"))
    return sorted(files)


def ensure_repo(root: Path) -> Repo:
    git_dir = root / ".git"
    if git_dir.exists():
        repo = Repo(str(root))
    else:
        repo = porcelain.init(str(root))
    repo.refs.set_symbolic_ref(b"HEAD", f"refs/heads/{BRANCH}".encode())
    return repo


def ensure_remote(root: Path) -> None:
    repo = Repo(str(root))
    config = repo.get_config()
    section = (b"remote", b"origin")
    if config.has_section(section):
        config.set(section, b"url", REMOTE_URL.encode())
    else:
        porcelain.remote_add(str(root), "origin", REMOTE_URL)
        return
    config.write_to_path()


def main() -> None:
    root = Path(__file__).resolve().parent
    repo = ensure_repo(root)
    ensure_remote(root)

    files = iter_project_files(root)
    porcelain.add(str(root), files)

    try:
        commit_id = porcelain.commit(
            str(root),
            message=COMMIT_MESSAGE,
            author=AUTHOR,
            committer=AUTHOR,
        )
        print(f"created commit {commit_id.decode()}")
    except Exception as exc:
        message = str(exc)
        if "nothing to commit" in message.lower() or "no changes" in message.lower():
            head = repo.head().decode()
            print(f"no new commit needed; HEAD={head}")
        else:
            raise

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    push_kwargs = {}
    if token:
        push_kwargs = {"username": b"x-access-token", "password": token.encode()}

    result = porcelain.push(
        str(root),
        REMOTE_URL,
        refspecs=[f"refs/heads/{BRANCH}:refs/heads/{BRANCH}".encode()],
        **push_kwargs,
    )
    print(result)
    print(f"pushed to {REMOTE_URL} branch {BRANCH}")


if __name__ == "__main__":
    main()
