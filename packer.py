"""List share files, pack small files together, and write received files to disk."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any, Dict, List


SKIP_NAMES = {".DS_Store", "Thumbs.db", ".DS_Store.gz"}
# Files this large stay as their own transfer. Smaller ones are grouped.
LARGE_FILE_BYTES = 48 * 1024
PACK_MAX_BYTES = 40 * 1024
PACK_MAX_FILES = 25


def listShareEntries(sourcePath: str) -> tuple[str, Path, List[Dict[str, Any]]]:
    resolved = Path(sourcePath).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path not found: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Not a directory: {resolved}")

    rootName = resolved.name
    entries: List[Dict[str, Any]] = []
    for path in sorted(resolved.rglob("*")):
        if path.name in SKIP_NAMES:
            continue
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
        if path.is_file() and not path.is_symlink():
            entries.append(
                {
                    "kind": "file",
                    "relPath": relPath,
                    "absPath": path,
                    "byteSize": path.stat().st_size,
                }
            )
    return rootName, resolved, entries


def buildTransferJobs(sourcePath: str) -> tuple[str, List[Dict[str, Any]], int]:
    """Group tiny files into packs so 1000 small files are not 1000 ACK handshakes."""
    rootName, _resolved, entries = listShareEntries(sourcePath)
    jobs: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    pendingBytes = 0

    def flushPack() -> None:
        nonlocal pendingBytes
        if not pending:
            return
        jobs.append(
            {
                "kind": "pack",
                "entries": list(pending),
                "byteSize": pendingBytes,
                "relPath": f"{len(pending)} small files",
            }
        )
        pending.clear()
        pendingBytes = 0

    for entry in entries:
        size = int(entry["byteSize"])
        if entry["kind"] == "file" and size >= LARGE_FILE_BYTES:
            flushPack()
            jobs.append(entry)
            continue
        if len(pending) >= PACK_MAX_FILES or pendingBytes + size > PACK_MAX_BYTES:
            flushPack()
        pending.append(entry)
        pendingBytes += size
    flushPack()
    return rootName, jobs, len(entries)


def buildPackBytes(entries: List[Dict[str, Any]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tarHandle:
        for entry in entries:
            relPath = str(entry["relPath"])
            if entry["kind"] == "dir":
                info = tarfile.TarInfo(name=relPath)
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tarHandle.addfile(info)
                continue
            source = Path(entry["absPath"])
            info = tarHandle.gettarinfo(str(source), arcname=relPath)
            with source.open("rb") as handle:
                tarHandle.addfile(info, handle)
    return buffer.getvalue()


def unpackPackBytes(archiveBytes: bytes, outputDir: str, rootName: str) -> Path:
    outputPath = Path(outputDir).expanduser().resolve()
    destRoot = (outputPath / rootName).resolve()
    destRoot.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO(archiveBytes)
    with tarfile.open(fileobj=buffer, mode="r:gz") as tarHandle:
        for member in tarHandle.getmembers():
            memberPath = Path(member.name)
            if memberPath.is_absolute() or ".." in memberPath.parts:
                raise ValueError(f"Unsafe path in pack: {member.name}")
            dest = (destRoot / member.name).resolve()
            if dest != destRoot and destRoot not in dest.parents:
                raise ValueError(f"Unsafe path in pack: {member.name}")
        if hasattr(tarfile, "data_filter"):
            tarHandle.extractall(path=str(destRoot), filter="data")
        else:
            tarHandle.extractall(path=str(destRoot))
    return destRoot


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
