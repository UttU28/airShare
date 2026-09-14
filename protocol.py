"""Chunk protocol for QR directory transfers."""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
from typing import Any, Dict, List, Optional, Tuple

MAGIC = "CS01"
# Small payloads keep QR version low so modules stay large on screen (easier under glare).
DEFAULT_CHUNK_SIZE = 200
HEADER_SEQ = 0


def makeTransferId() -> str:
    return secrets.token_hex(4)


def sha256Hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def shortChecksum(data: bytes) -> str:
    return sha256Hex(data)[:12]


def packBytePayload(
    payloadBytes: bytes,
    transferId: str,
    extraMeta: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[bytes]]:
    """Split one file (or empty dir marker) into header + data chunks."""
    contentHash = sha256Hex(payloadBytes)
    dataChunks: List[bytes] = []
    offset = 0
    while offset < len(payloadBytes):
        dataChunks.append(payloadBytes[offset : offset + DEFAULT_CHUNK_SIZE])
        offset += DEFAULT_CHUNK_SIZE

    totalFrames = len(dataChunks) + 1
    headerMeta: Dict[str, Any] = {
        "magic": MAGIC,
        "kind": "header",
        "transferId": transferId,
        "seq": HEADER_SEQ,
        "total": totalFrames,
        "byteSize": len(payloadBytes),
        "archiveHash": contentHash,
        "chunkSize": DEFAULT_CHUNK_SIZE,
    }
    if extraMeta:
        headerMeta.update(extraMeta)
    return headerMeta, dataChunks


def encodeAlign(step: int, handshakeId: str, **extra) -> str:
    payload = {
        "magic": MAGIC,
        "kind": "align",
        "step": int(step),
        "handshakeId": handshakeId,
    }
    payload.update(extra)
    return json.dumps(payload, separators=(",", ":"))


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
    kind = payload.get("kind")
    if kind not in ("header", "data", "status", "align", "ackRequest", "done", "sessionDone"):
        return None
    if kind == "align":
        if "step" not in payload or "handshakeId" not in payload:
            return None
        return payload
    if "transferId" not in payload:
        return None
    if kind in ("header", "data") and ("seq" not in payload or "total" not in payload):
        return None
    if kind in ("status", "ackRequest", "done") and "total" not in payload:
        return None
    if kind == "sessionDone" and "sessionId" not in payload:
        return None
    return payload


def encodeAckRequest(transferId: str, total: int, roundIndex: int) -> str:
    payload = {
        "magic": MAGIC,
        "kind": "ackRequest",
        "transferId": transferId,
        "total": total,
        "round": roundIndex,
    }
    return json.dumps(payload, separators=(",", ":"))


def encodeSessionDone(sessionId: str, fileCount: int) -> str:
    payload = {
        "magic": MAGIC,
        "kind": "sessionDone",
        "sessionId": sessionId,
        "transferId": sessionId,
        "total": fileCount,
        "fileCount": fileCount,
    }
    return json.dumps(payload, separators=(",", ":"))


def encodeDone(transferId: str, total: int) -> str:
    payload = {
        "magic": MAGIC,
        "kind": "done",
        "transferId": transferId,
        "total": total,
        "gotCount": total,
    }
    return json.dumps(payload, separators=(",", ":"))


def encodeStatus(
    transferId: str,
    total: int,
    headerGot: bool,
    gotDataSeqs: List[int],
    roundIndex: int,
) -> str:
    mask = bytearray((total + 7) // 8)
    if headerGot and total > 0:
        mask[0] |= 1
    for seq in gotDataSeqs:
        if 1 <= seq < total:
            mask[seq // 8] |= 1 << (seq % 8)
    gotCount = (1 if headerGot else 0) + len(gotDataSeqs)
    payload = {
        "magic": MAGIC,
        "kind": "status",
        "transferId": transferId,
        "total": total,
        "round": roundIndex,
        "gotCount": gotCount,
        "mask": base64.b64encode(bytes(mask)).decode("ascii"),
    }
    return json.dumps(payload, separators=(",", ":"))


def gotSeqsFromStatus(payload: Dict[str, Any]) -> set[int]:
    total = int(payload["total"])
    encoded = payload.get("mask")
    if not isinstance(encoded, str):
        return set()
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception:
        return set()
    got: set[int] = set()
    for seq in range(total):
        byteIndex = seq // 8
        if byteIndex >= len(raw):
            break
        if raw[byteIndex] & (1 << (seq % 8)):
            got.add(seq)
    return got


def missingSeqsFromStatus(payload: Dict[str, Any]) -> List[int]:
    total = int(payload["total"])
    got = gotSeqsFromStatus(payload)
    return [seq for seq in range(total) if seq not in got]


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
