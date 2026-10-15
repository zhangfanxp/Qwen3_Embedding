#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将本脚本所在目录的项目文件安全地提交到指定 GitHub 仓库。

示例：
    python github_upload.py zhangfanxp/Qwen3_Embedding.git
    python github_upload.py zhangfanxp/Qwen3_Embedding.git --dry-run
    python github_upload.py zhangfanxp/Qwen3_Embedding.git -m "更新 Gradio 界面"

需要：已安装 Git；目标仓库已创建；本机 Git 能访问该仓库（推荐 gh auth login）。
不需要任何额外 Python 第三方包。
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# 这些文件/目录无论 .gitignore 如何设置都不会上传。
SKIP_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".idea", ".vscode", ".cache",
    "node_modules", "models", "dist", "build", ".tox", ".uv-cache",
}
SKIP_FILES = {".DS_Store", ".netrc", "credentials.json", "service-account.json"}
SKIP_PATTERNS = ("*.pem", "*.p12", "*.pfx", "*.key", "id_rsa*", "id_ed25519*")
MAX_FILE_BYTES = 95 * 1024 * 1024  # 预留余量，避免触发 GitHub 100 MiB 文件限制


def run_git(*args: str, cwd: Path | None = None, input_bytes: bytes | None = None,
            check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        ["git", *args], cwd=cwd, input=input_bytes,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if check and result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        detail = detail or result.stdout.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Git 命令执行失败：git {' '.join(args)}\n{detail}")
    return result


def decode(output: bytes) -> str:
    return output.decode("utf-8", errors="replace").strip()


def parse_repository(repository: str, use_ssh: bool = False) -> str:
    """支持 owner/repo(.git)、完整 HTTPS 或 SSH GitHub 地址。"""
    repo = repository.strip().rstrip("/")
    if repo.startswith("https://github.com/"):
        slug = repo[len("https://github.com/"):]
    elif repo.startswith("git@github.com:"):
        slug = repo[len("git@github.com:"):]
        use_ssh = True
    else:
        slug = repo

    slug = slug.removesuffix(".git")
    # GitHub 用户名允许连字符；仓库名支持字母、数字、下划线、点与连字符。
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?/[A-Za-z0-9_.-]+", slug):
        raise ValueError("仓库地址格式错误。示例：zhangfanxp/Qwen3_Embedding.git")
    if ".." in slug.split("/")[1]:
        raise ValueError("仓库名称不能包含连续的点。")
    if use_ssh:
        return f"git@github.com:{slug}.git"
    return f"https://github.com/{slug}.git"


def always_skip(relative: Path) -> bool:
    parts = relative.parts
    name = relative.name
    if any(part in SKIP_DIRS for part in parts[:-1]):
        return True
    if name in SKIP_FILES or name in SKIP_DIRS:
        return True
    if (name == ".env" or name.startswith(".env.")) and name != ".env.example":
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in SKIP_PATTERNS)


def list_local_files(root: Path) -> tuple[list[Path], int]:
    included: list[Path] = []
    skipped = 0
    for current, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not (Path(current) / d).is_symlink()]
        for name in files:
            path = Path(current) / name
            rel = path.relative_to(root)
            if always_skip(rel) or path.is_symlink() or not path.is_file():
                skipped += 1
                continue
            if path.stat().st_size > MAX_FILE_BYTES:
                raise RuntimeError(f"文件过大，已停止上传：{rel}（超过95 MiB）。请使用 Git LFS 或加入 .gitignore。")
            included.append(rel)
    return sorted(included), skipped


def ignored_by_git(work: Path, paths: list[Path]) -> set[str]:
    if not paths:
        return set()
    data = b"\0".join(p.as_posix().encode("utf-8") for p in paths) + b"\0"
    result = run_git("check-ignore", "--no-index", "-z", "--stdin", cwd=work,
                     input_bytes=data, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError("检查 .gitignore 失败：" + decode(result.stderr))
    return {p.decode("utf-8") for p in result.stdout.split(b"\0") if p}


def ensure_branch(work: Path, branch: str | None) -> str:
    has_head = run_git("rev-parse", "--verify", "HEAD", cwd=work, check=False).returncode == 0
    if not has_head:
        name = branch or "main"
        run_git("symbolic-ref", "HEAD", f"refs/heads/{name}", cwd=work)
        return name
    current = decode(run_git("branch", "--show-current", cwd=work).stdout)
    if not current:
        raise RuntimeError("目标仓库处于 detached HEAD 状态，无法安全提交。")
    if branch and branch != current:
        local_exists = run_git(
            "show-ref", "--verify", "--quiet", f"refs/heads/{branch}",
            cwd=work, check=False,
        ).returncode == 0
        remote_exists = run_git(
            "show-ref", "--verify", "--quiet", f"refs/remotes/origin/{branch}",
            cwd=work, check=False,
        ).returncode == 0
        if local_exists:
            run_git("checkout", branch, cwd=work)
        elif remote_exists:
            run_git("checkout", "--track", f"origin/{branch}", cwd=work)
        else:
            run_git("checkout", "-b", branch, cwd=work)
        return branch
    return current


def publish(root: Path, remote_url: str, message: str, dry_run: bool = False,
            branch: str | None = None) -> None:
    if not shutil.which("git"):
        raise RuntimeError("找不到 git。请先运行 git --version；Mac 可安装 Xcode Command Line Tools。")
    source_files, skipped = list_local_files(root)
    if not source_files:
        raise RuntimeError("项目目录中没有可上传的文件。")

    print(f"项目位置：{root}")
    print(f"目标仓库：{remote_url}")
    print("正在读取 GitHub 仓库（本地文件和现有远程历史不会被删除）……")
    with tempfile.TemporaryDirectory(prefix="github-upload-") as td:
        work = Path(td) / "repo"
        run_git("clone", "--quiet", remote_url, str(work))
        target_branch = ensure_branch(work, branch)
        remote_branch_exists = run_git(
            "show-ref", "--verify", "--quiet",
            f"refs/remotes/origin/{target_branch}", cwd=work, check=False,
        ).returncode == 0
        print(f"目标分支：{target_branch}")

        # 先覆盖 .gitignore，令过滤规则与本地项目一致。
        source_ignore = root / ".gitignore"
        if source_ignore.is_file():
            shutil.copy2(source_ignore, work / ".gitignore")

        ignored = ignored_by_git(work, source_files)
        copied: list[str] = []
        for relative in source_files:
            name = relative.as_posix()
            if name in ignored:
                skipped += 1
                continue
            src = root / relative
            dst = work / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(name)

        run_git("add", "-A", "--", ".", cwd=work)
        status = run_git("diff", "--cached", "--name-status", cwd=work)
        changes = decode(status.stdout)
        print(f"可上传文件：{len(copied)} 个；跳过：{skipped} 个")
        if not changes:
            if not remote_branch_exists:
                if dry_run:
                    print("[预演模式] 将创建新远程分支；尚未推送。")
                else:
                    run_git("push", "--quiet", "origin", f"HEAD:refs/heads/{target_branch}", cwd=work)
                    print(f"✓ 已创建远程分支：{target_branch}")
            else:
                print("✓ 远程仓库与本地可上传文件已一致，无需提交。")
            return

        print("\n这次需要提交的变更：")
        lines = changes.splitlines()
        for line in lines[:40]:
            print("  " + line)
        if len(lines) > 40:
            print(f"  ……还有 {len(lines)-40} 项")
        if dry_run:
            print("\n[预演模式] 没有提交，也没有推送到 GitHub。")
            return

        identity = run_git("var", "GIT_AUTHOR_IDENT", cwd=work, check=False)
        if identity.returncode != 0:
            raise RuntimeError(
                "尚未设置 Git 提交身份。请先执行：\n"
                '  git config --global user.name "你的名称"\n'
                '  git config --global user.email "你的GitHub邮箱"'
            )

        run_git("commit", "--quiet", "-m", message, cwd=work)
        print("\n正在推送到 GitHub……")
        try:
            run_git("push", "--quiet", "origin", f"HEAD:refs/heads/{target_branch}", cwd=work)
        except RuntimeError as exc:
            raise RuntimeError(
                f"推送失败（本地原始文件未受影响）。原因：\n{exc}\n"
                "如果其他人刚更新了远程分支，可重新执行本脚本；不会强制覆盖远程历史。"
            ) from exc
        if remote_url.startswith("https://github.com/"):
            repo_page = remote_url.removesuffix(".git")
        elif remote_url.startswith("git@github.com:"):
            repo_page = "https://github.com/" + remote_url.split(":", 1)[1].removesuffix(".git")
        else:
            repo_page = remote_url  # 主要供本地测试仓库使用
        print(f"\n✓ 上传成功！仓库：{repo_page}/tree/{target_branch}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="上传本脚本所在目录的项目文件到指定 GitHub 仓库（尊重 .gitignore）。"
    )
    parser.add_argument("repository", help="例如：zhangfanxp/Qwen3_Embedding.git")
    parser.add_argument("-m", "--message", default="Update Qwen3 Embedding project",
                        help="Git 提交信息")
    parser.add_argument("--ssh", action="store_true", help="使用 SSH 地址 git@github.com:owner/repo.git")
    parser.add_argument("--branch", help="目标分支；默认使用仓库当前默认分支，新仓库默认 main")
    parser.add_argument("--dry-run", action="store_true", help="仅预览变更，不提交、不推送")
    args = parser.parse_args()
    try:
        url = parse_repository(args.repository, args.ssh)
        publish(Path(__file__).resolve().parent, url, args.message, args.dry_run, args.branch)
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        print(
            "\n请确认：① GitHub 仓库已创建；② Git 已安装；③ 已通过 gh auth login 登录"
            " 或正确配置 SSH/Git 凭据；④ Git 用户名和邮箱已设置。",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

