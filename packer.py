"""List share files and write received files to disk."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any


def listShareEntries(sourcePath: str) -> tuple[str, Path, List[Dict[str, Any]]]:
    resolved = Path(sourcePath).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path not found: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Not a directory: {resolved}")

    rootName = resolved.name
    entries: List[Dict[str, Any]] = []
    for path in sorted(resolved.rglob("*")):
        relPath = path.relative_to(resolved).as_posix()
        if path.is_dir():
            try:
                empty = next(path.iterdir(), None) is None
            except OSError:
                empty = False
            if empty:
                entries.append(
                    {
                        "kind": "dir",
                        "relPath": relPath,
                        "absPath": path,
                        "byteSize": 0,
                    }
                )
            continue
        if path.is_file():
            entries.append(
                {
                    "kind": "file",
                    "relPath": relPath,
                    "absPath": path,
                    "byteSize": path.stat().st_size,
                }
            )
    return rootName, resolved, entries


def safeDestPath(outputDir: str, rootName: str, relPath: str) -> Path:
    outputPath = Path(outputDir).expanduser().resolve()
    dest = (outputPath / rootName / relPath).resolve()
    root = (outputPath / rootName).resolve()
    if dest != root and root not in dest.parents:
        raise ValueError(f"Unsafe path: {relPath}")
    return dest


def writeReceivedFile(outputDir: str, rootName: str, relPath: str, data: bytes) -> Path:
    dest = safeDestPath(outputDir, rootName, relPath)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def writeReceivedDir(outputDir: str, rootName: str, relPath: str) -> Path:
    dest = safeDestPath(outputDir, rootName, relPath)
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def humanSize(numBytes: int) -> str:
    size = float(numBytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"
