"""Pack and unpack directories as tar.gz archives."""

from __future__ import annotations

import io
import os
import tarfile
from pathlib import Path


def packDirectory(sourcePath: str) -> tuple[bytes, str]:
    resolved = Path(sourcePath).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path not found: {resolved}")
    if not resolved.is_dir():
        raise NotADirectoryError(f"Not a directory: {resolved}")

    rootName = resolved.name
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tarHandle:
        tarHandle.add(str(resolved), arcname=rootName)

    return buffer.getvalue(), rootName


def unpackArchive(archiveBytes: bytes, outputDir: str) -> Path:
    outputPath = Path(outputDir).expanduser().resolve()
    outputPath.mkdir(parents=True, exist_ok=True)

    buffer = io.BytesIO(archiveBytes)
    with tarfile.open(fileobj=buffer, mode="r:gz") as tarHandle:
        members = tarHandle.getmembers()
        for member in members:
            memberPath = Path(member.name)
            if memberPath.is_absolute() or ".." in memberPath.parts:
                raise ValueError(f"Unsafe path in archive: {member.name}")
        if hasattr(tarfile, "data_filter"):
            tarHandle.extractall(path=str(outputPath), filter="data")
        else:
            tarHandle.extractall(path=str(outputPath))

    return outputPath


def humanSize(numBytes: int) -> str:
    size = float(numBytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def estimateFrameCount(archiveBytes: int, chunkSize: int = 320) -> int:
    dataFrames = (archiveBytes + chunkSize - 1) // chunkSize if archiveBytes else 0
    return dataFrames + 1
