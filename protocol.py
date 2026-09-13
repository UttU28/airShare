"""Chunk protocol for QR directory transfers."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from typing import Any, Dict, List, Optional, Tuple

MAGIC = "CS01"
DEFAULT_CHUNK_SIZE = 1200
HEADER_SEQ = 0


def makeTransferId() -> str:
    return secrets.token_hex(4)


def sha256Hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shortChecksum(data: bytes) -> str:
    return sha256Hex(data)[:12]


def packDirectoryArchive(archiveBytes: bytes, transferId: str, rootName: str) -> Tuple[Dict[str, Any], List[bytes]]:
    """Split archive into header meta + data chunks."""
    archiveHash = sha256Hex(archiveBytes)
    dataChunks: List[bytes] = []
    offset = 0
    while offset < len(archiveBytes):
        dataChunks.append(archiveBytes[offset : offset + DEFAULT_CHUNK_SIZE])
        offset += DEFAULT_CHUNK_SIZE

    totalFrames = len(dataChunks) + 1  # +1 header frame
    headerMeta: Dict[str, Any] = {
        "magic": MAGIC,
        "kind": "header",
        "transferId": transferId,
        "seq": HEADER_SEQ,
        "total": totalFrames,
        "rootName": rootName,
        "byteSize": len(archiveBytes),
        "archiveHash": archiveHash,
        "chunkSize": DEFAULT_CHUNK_SIZE,
    }
    return headerMeta, dataChunks


def encodeHeaderFrame(headerMeta: Dict[str, Any]) -> str:
    return json.dumps(headerMeta, separators=(",", ":"))


def encodeDataFrame(
    transferId: str,
    seq: int,
    total: int,
    chunkBytes: bytes,
) -> str:
    payload = {
        "magic": MAGIC,
        "kind": "data",
        "transferId": transferId,
        "seq": seq,
        "total": total,
        "checksum": shortChecksum(chunkBytes),
        "data": base64.b64encode(chunkBytes).decode("ascii"),
    }
    return json.dumps(payload, separators=(",", ":"))


def decodeFrame(rawText: str) -> Optional[Dict[str, Any]]:
    try:
        payload = json.loads(rawText)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("magic") != MAGIC:
        return None
    if payload.get("kind") not in ("header", "data"):
        return None
    if "transferId" not in payload or "seq" not in payload or "total" not in payload:
        return None
    return payload


def validateDataFrame(payload: Dict[str, Any]) -> Optional[bytes]:
    if payload.get("kind") != "data":
        return None
    encoded = payload.get("data")
    checksum = payload.get("checksum")
    if not isinstance(encoded, str) or not isinstance(checksum, str):
        return None
    try:
        chunkBytes = base64.b64decode(encoded, validate=True)
    except Exception:
        return None
    if shortChecksum(chunkBytes) != checksum:
        return None
    return chunkBytes


def rebuildArchive(headerMeta: Dict[str, Any], chunkMap: Dict[int, bytes]) -> bytes:
    total = int(headerMeta["total"])
    expectedDataFrames = total - 1
    missing = [i for i in range(1, total) if i not in chunkMap]
    if missing:
        raise ValueError(f"Missing frames: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    parts = [chunkMap[i] for i in range(1, total)]
    archiveBytes = b"".join(parts)
    if len(archiveBytes) != int(headerMeta["byteSize"]):
        raise ValueError("Reassembled size does not match header")
    if sha256Hex(archiveBytes) != headerMeta["archiveHash"]:
        raise ValueError("Archive hash mismatch")
    if expectedDataFrames != len(parts):
        raise ValueError("Frame count mismatch")
    return archiveBytes
