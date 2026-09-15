"""Build and display QR frames for sending a directory."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence

import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_M

from packer import buildPackBytes, buildTransferJobs, humanSize
from protocol import (
    DEFAULT_CHUNK_SIZE,
    decodeFrame,
    encodeAckRequest,
    encodeAlign,
    encodeDataFrame,
    encodeDataPartFrame,
    encodeHeaderFrame,
    encodeSessionDone,
    makeTransferId,
    missingSeqsFromStatus,
    packBytePayload,
    splitBytes,
)


SCREEN_SIZE = {"w": 1920, "h": 1080}
HEADER_HOLD_SECONDS = 0.75
ALIGN_HOLD_SECONDS = 1.6
_CLAHE = None


def setupFullscreen(windowName: str) -> None:
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    probe = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.imshow(windowName, probe)
    cv2.waitKey(30)
    try:
        _x, _y, width, height = cv2.getWindowImageRect(windowName)
        if width >= 400 and height >= 300:
            SCREEN_SIZE["w"] = int(width)
            SCREEN_SIZE["h"] = int(height)
    except Exception:
        pass
    print(f"Sender fullscreen: {SCREEN_SIZE['w']}x{SCREEN_SIZE['h']}")


def screenSize() -> tuple[int, int]:
    return SCREEN_SIZE["w"], SCREEN_SIZE["h"]


def fitToScreen(image: np.ndarray) -> np.ndarray:
    screenW, screenH = screenSize()
    canvas = np.full((screenH, screenW, 3), (18, 16, 14), dtype=np.uint8)
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        return canvas
    scale = min(screenW / float(width), screenH / float(height))
    newW = max(1, int(round(width * scale)))
    newH = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA
    resized = cv2.resize(image, (newW, newH), interpolation=interpolation)
    x0 = (screenW - newW) // 2
    y0 = (screenH - newH) // 2
    canvas[y0 : y0 + newH, x0 : x0 + newW] = resized
    return canvas


def showOnSender(windowName: str, image: np.ndarray) -> None:
    cv2.imshow(windowName, fitToScreen(image))


def buildQrImage(payloadText: str, boxSize: int = 8, border: int = 4) -> np.ndarray:
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_M,
        box_size=boxSize,
        border=border,
    )
    qr.add_data(payloadText)
    qr.make(fit=True)
    pilImage = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return cv2.cvtColor(np.array(pilImage), cv2.COLOR_RGB2BGR)


class QrPrefetcher:
    """Generate QR images just in time, keeping only a small lookahead in RAM."""

    def __init__(self, payloads: Sequence[str], workerCount: int | None = None, ahead: int = 24):
        self.payloads = payloads
        self.ahead = max(4, ahead)
        self.workerCount = max(1, workerCount or (os.cpu_count() or 4))
        self.executor = ThreadPoolExecutor(max_workers=self.workerCount)
        self.jobs: Dict[int, object] = {}

    def warm(self, seqs: Sequence[int]) -> None:
        for seq in seqs:
            if seq < 0 or seq >= len(self.payloads):
                continue
            if seq not in self.jobs:
                self.jobs[seq] = self.executor.submit(buildQrImage, self.payloads[seq])

    def get(self, seq: int) -> np.ndarray:
        self.warm([seq])
        image = self.jobs[seq].result()
        return image

    def drop(self, seq: int) -> None:
        self.jobs.pop(seq, None)

    def prefetchAround(self, seqs: Sequence[int], startIndex: int) -> None:
        window = seqs[startIndex : startIndex + self.ahead]
        self.warm(window)
        keep = set(window)
        for seq in list(self.jobs):
            if seq not in keep:
                self.drop(seq)

    def close(self) -> None:
        self.executor.shutdown(wait=False)
        self.jobs.clear()


def buildPolaroidCard(
    qrImage: np.ndarray,
    captionLines: Sequence[str],
    progressRatio: float,
    paused: bool = False,
    frameSize: tuple[int, int] | None = None,
) -> np.ndarray:
    """Instant-photo style card: clean QR on top, caption + seek bar below."""
    if frameSize is None:
        frameSize = screenSize()
    cardWidth, cardHeight = frameSize
    captionHeight = max(160, int(cardHeight * 0.16))
    sidePad = max(36, int(cardWidth * 0.05))
    topPad = max(28, int(cardHeight * 0.04))
    paper = (246, 242, 232)
    ink = (42, 38, 34)
    muted = (110, 105, 98)
    barTrack = (210, 205, 196)
    barFill = (70, 120, 40) if not paused else (40, 140, 220)

    qrSize = min(cardWidth - sidePad * 2, cardHeight - topPad - captionHeight)
    qrResized = cv2.resize(qrImage, (qrSize, qrSize), interpolation=cv2.INTER_NEAREST)

    card = np.full((cardHeight, cardWidth, 3), paper, dtype=np.uint8)
    qrX = (cardWidth - qrSize) // 2
    card[topPad : topPad + qrSize, qrX : qrX + qrSize] = qrResized

    cv2.rectangle(
        card,
        (qrX - 1, topPad - 1),
        (qrX + qrSize, topPad + qrSize),
        (220, 215, 205),
        1,
        cv2.LINE_AA,
    )

    textY = topPad + qrSize + max(36, int(captionHeight * 0.28))
    titleScale = max(0.8, cardWidth / 1400.0)
    bodyScale = max(0.6, cardWidth / 1800.0)
    for lineIndex, line in enumerate(captionLines):
        color = ink if lineIndex == 0 else muted
        scale = titleScale if lineIndex == 0 else bodyScale
        thickness = 2 if lineIndex == 0 else 1
        cv2.putText(
            card,
            line,
            (sidePad, textY + lineIndex * int(34 * titleScale)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    barX = sidePad
    barY = cardHeight - max(32, int(cardHeight * 0.035))
    barW = cardWidth - sidePad * 2
    barH = max(10, int(cardHeight * 0.012))
    cv2.rectangle(card, (barX, barY), (barX + barW, barY + barH), barTrack, -1, cv2.LINE_AA)
    fillWidth = max(2, int(barW * max(0.0, min(1.0, progressRatio))))
    cv2.rectangle(card, (barX, barY), (barX + fillWidth, barY + barH), barFill, -1, cv2.LINE_AA)
    tickX = barX + fillWidth - 1
    cv2.rectangle(card, (tickX - 1, barY - 4), (tickX + 2, barY + barH + 4), ink, -1, cv2.LINE_AA)

    if paused:
        cv2.putText(
            card,
            "PAUSED",
            (cardWidth - sidePad - 140, barY - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (40, 140, 220),
            2,
            cv2.LINE_AA,
        )

    return card


def buildEntryFrames(
    entry: dict,
    sessionId: str,
    rootName: str,
    fileIndex: int,
    fileCount: int,
) -> tuple[List[str], dict]:
    transferId = makeTransferId()
    if entry["kind"] == "dir":
        payloadBytes = b""
        relPath = entry["relPath"]
        entryKind = "dir"
    elif entry["kind"] == "pack":
        payloadBytes = buildPackBytes(entry["entries"])
        relPath = f"_pack_{fileIndex:04d}.tgz"
        entryKind = "pack"
    else:
        payloadBytes = entry["absPath"].read_bytes()
        relPath = entry["relPath"]
        entryKind = "file"

    headerMeta, dataChunks = packBytePayload(
        payloadBytes,
        transferId,
        extraMeta={
            "sessionId": sessionId,
            "rootName": rootName,
            "relPath": relPath,
            "entryKind": entryKind,
            "fileIndex": fileIndex,
            "fileCount": fileCount,
            "packFiles": len(entry["entries"]) if entryKind == "pack" else 1,
        },
    )
    frames: List[str] = [encodeHeaderFrame(headerMeta)]
    total = int(headerMeta["total"])
    for index, chunkBytes in enumerate(dataChunks, start=1):
        frames.append(encodeDataFrame(transferId, index, total, chunkBytes))

    summary = {
        "sessionId": sessionId,
        "transferId": transferId,
        "rootName": rootName,
        "relPath": relPath,
        "entryKind": entryKind,
        "fileIndex": fileIndex,
        "fileCount": fileCount,
        "byteSize": len(payloadBytes),
        "totalFrames": total,
        "humanSize": humanSize(len(payloadBytes)),
        "chunksBySeq": {index + 1: chunk for index, chunk in enumerate(dataChunks)},
    }
    return frames, summary


CALIBRATE_FRAME_COUNT = 10


def buildCalibrateFrames(sessionId: str, realFileCount: int) -> tuple[List[str], dict]:
    """Tiny dummy file: 1 header + 9 data QRs so both cameras can lock a fixed crop."""
    transferId = makeTransferId()
    dataFrameCount = max(1, CALIBRATE_FRAME_COUNT - 1)
    payloadBytes = (b"CALIBRATE-QR-LOCK-" * (dataFrameCount * DEFAULT_CHUNK_SIZE))[
        : dataFrameCount * DEFAULT_CHUNK_SIZE
    ]
    headerMeta, dataChunks = packBytePayload(
        payloadBytes,
        transferId,
        extraMeta={
            "sessionId": sessionId,
            "rootName": "calibrate",
            "relPath": "_calibrate.dat",
            "entryKind": "calibrate",
            "fileIndex": 0,
            "fileCount": realFileCount,
        },
    )
    frames: List[str] = [encodeHeaderFrame(headerMeta)]
    total = int(headerMeta["total"])
    for index, chunkBytes in enumerate(dataChunks, start=1):
        frames.append(encodeDataFrame(transferId, index, total, chunkBytes))
    summary = {
        "sessionId": sessionId,
        "transferId": transferId,
        "rootName": "calibrate",
        "relPath": "_calibrate.dat",
        "entryKind": "calibrate",
        "fileIndex": 0,
        "fileCount": realFileCount,
        "byteSize": len(payloadBytes),
        "totalFrames": total,
        "humanSize": humanSize(len(payloadBytes)),
        "chunksBySeq": {index + 1: chunk for index, chunk in enumerate(dataChunks)},
    }
    return frames, summary


def sendUntilComplete(
    windowName: str,
    frames: List[str],
    summary: dict,
    frameDelay: float,
    cameraIndex: int,
    scanRoi: QrScanRoi | None = None,
    stream: CameraStream | None = None,
    detector=None,
) -> str:
    """Play one file's frames until receiver done or quit. Returns complete|quit."""
    if scanRoi is None:
        scanRoi = QrScanRoi()
    qrPrefetch = QrPrefetcher(frames, ahead=24)
    transferId = summary["transferId"]
    totalFrames = len(frames)
    pending = list(range(totalFrames))
    roundIndex = 1
    previousMissing = None
    sameMissingCount = 0
    splitParts = 1
    baseDelay = frameDelay
    chunksBySeq = summary.get("chunksBySeq") or {}
    fileLabel = f"{summary['fileIndex']}/{summary['fileCount']}  {summary['relPath']}"
    try:
        while pending:
            modeNote = (
                f"  split x{splitParts}" if splitParts > 1 else ""
            )
            print(
                f"\n--- {fileLabel}  round {roundIndex}: "
                f"{len(pending)} pending  interval={frameDelay:.2f}s{modeNote} ---"
            )
            lastSeq = pending[-1]
            playQueue = []
            for seq in pending:
                rawChunk = chunksBySeq.get(seq)
                if seq == 0 or splitParts <= 1 or not rawChunk:
                    playQueue.append((seq, frames[seq], "full"))
                else:
                    pieces = splitBytes(rawChunk, splitParts)
                    for partIndex, piece in enumerate(pieces):
                        playQueue.append(
                            (
                                seq,
                                encodeDataPartFrame(
                                    transferId,
                                    seq,
                                    totalFrames,
                                    partIndex,
                                    len(pieces),
                                    piece,
                                ),
                                f"half-QR {partIndex + 1}/{len(pieces)}",
                            )
                        )

            for playIndex, (seq, payloadText, partLabel) in enumerate(playQueue):
                progressRatio = (playIndex + 1) / len(playQueue)
                captionLines = [
                    f"{summary['rootName']}  {fileLabel}",
                    f"seq {seq}  {playIndex + 1}/{len(playQueue)}  {partLabel}  ·  {transferId}",
                    f"{summary['humanSize']}  ·  playing",
                ]
                if partLabel == "full":
                    upcoming = [
                        item[0]
                        for item in playQueue[playIndex : playIndex + 12]
                        if item[2] == "full"
                    ]
                    qrPrefetch.warm(upcoming)
                    qrImage = qrPrefetch.get(seq)
                else:
                    qrImage = buildQrImage(payloadText)
                card = buildPolaroidCard(
                    qrImage,
                    captionLines=captionLines,
                    progressRatio=progressRatio,
                    paused=False,
                )
                showOnSender(windowName, card)
                hold = frameDelay
                if seq == 0:
                    hold = max(frameDelay, HEADER_HOLD_SECONDS)
                if waitForFrame(hold) == "quit":
                    return "quit"

            ackText = encodeAckRequest(transferId, totalFrames, roundIndex)
            lastCard = buildPolaroidCard(
                buildQrImage(ackText),
                captionLines=[
                    f"ACK REQUEST  {fileLabel}  round {roundIndex}",
                    f"last data seq {lastSeq}  ·  {transferId}",
                    "Receiver: send status now",
                ],
                progressRatio=1.0,
                paused=False,
            )
            showOnSender(windowName, lastCard)
            cv2.waitKey(1)

            while True:
                statusPayload = freezeOnLastUntilStatus(
                    windowName,
                    lastCard,
                    transferId,
                    cameraIndex,
                    scanRoi=scanRoi,
                    stream=stream,
                    detector=detector,
                )
                if statusPayload == "quit":
                    return "quit"
                if statusPayload is None:
                    print("No status received. Repeating ACK.")
                    roundIndex += 1
                    break

                if statusPayload.get("kind") == "done":
                    doneIndex = statusPayload.get("fileIndex")
                    if doneIndex is not None and int(doneIndex) != int(summary["fileIndex"]):
                        print(
                            f"Ignoring DONE for file {doneIndex}, "
                            f"still on {summary['fileIndex']}."
                        )
                        continue
                    print(f"File complete: {summary['relPath']}")
                    return "complete"

                pending = missingSeqsFromStatus(statusPayload)
                print(f"Receiver still missing {len(pending)} frame(s) for this file.")
                if not pending:
                    print("Receiver has all chunks. Waiting for FILE COMPLETE QR...")
                    continue
                if previousMissing is not None and pending == previousMissing:
                    sameMissingCount += 1
                    if sameMissingCount >= 3:
                        splitParts = min(8, max(2, splitParts * 2))
                        frameDelay = baseDelay
                        sameMissingCount = 0
                        print(
                            f"Same missing set 3 times. Interval reset to {baseDelay:.2f}s. "
                            f"Sending stuck chunks as {splitParts} smaller QRs "
                            "(original chunk size restored after this file)."
                        )
                    else:
                        frameDelay += 0.5
                        print(
                            f"Missing set unchanged ({sameMissingCount}/3). "
                            f"Interval +0.5s → {frameDelay:.2f}s"
                        )
                else:
                    if previousMissing is not None and splitParts > 1:
                        print("Missing set changed. Restoring full-size data QRs.")
                    sameMissingCount = 0
                    splitParts = 1
                    frameDelay = baseDelay
                previousMissing = list(pending)
                roundIndex += 1
                break
        return "complete"
    finally:
        qrPrefetch.close()


def _clahe():
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _CLAHE


def _detectOnce(detector, image):
    try:
        rawText, points, _straight = detector.detectAndDecode(image)
    except cv2.error:
        return "", None
    except Exception:
        return "", None
    if not rawText:
        rawText = ""
    return rawText, points


def safeDetectAndDecode(detector, frame):
    """Try color, then gray, then contrast — stuck QRs often decode on the second pass."""
    if frame is None or getattr(frame, "size", 0) == 0:
        return "", None
    rawText, points = _detectOnce(detector, frame)
    if rawText:
        return rawText, points
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    rawText, points = _detectOnce(detector, gray)
    if rawText:
        return rawText, points
    height, width = gray.shape[:2]
    if height * width <= 800 * 800:
        enhanced = _clahe().apply(gray)
        rawText, points = _detectOnce(detector, enhanced)
        if rawText:
            return rawText, points
    return "", points


class QrScanRoi:
    """Full-frame until a calibration pass, then a fixed crop forever."""

    def __init__(self, padRatio: float = 0.28) -> None:
        self.box = None
        self.locked = False
        self.calibrating = False
        self.samples: List[tuple[int, int, int, int]] = []
        self.padRatio = padRatio

    def beginCalibrate(self) -> None:
        self.locked = False
        self.calibrating = True
        self.samples = []
        self.box = None
        print("QR calibration: collecting positions (box stays unset until lock).")

    def captureBox(self):
        if self.locked and self.box is not None:
            return self.box
        return None

    def lock(self) -> bool:
        if not self.samples:
            print("QR calibration: no samples, staying on full frame.")
            self.calibrating = False
            return False
        minX = min(s[0] for s in self.samples)
        minY = min(s[1] for s in self.samples)
        maxX = max(s[2] for s in self.samples)
        maxY = max(s[3] for s in self.samples)
        width = max(1, maxX - minX)
        height = max(1, maxY - minY)
        padX = int(width * self.padRatio)
        padY = int(height * self.padRatio)
        self.box = (
            max(0, minX - padX),
            max(0, minY - padY),
            maxX + padX,
            maxY + padY,
        )
        self.locked = True
        self.calibrating = False
        print(
            f"QR search box LOCKED (fixed): {self.box}  "
            f"from {len(self.samples)} samples. Decode/render use this crop only."
        )
        return True

    def process(self, detector, frame):
        """Decode. When locked, `frame` is already the crop. Points are in view coords."""
        rawText, points = safeDetectAndDecode(detector, frame)
        if self.calibrating and (not self.locked) and rawText and points is not None:
            self._addSample(frame, points)
        return rawText, points

    def _addSample(self, frame, points) -> None:
        pts = np.array(points, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 4:
            return
        minX, minY = pts.min(axis=0)
        maxX, maxY = pts.max(axis=0)
        self.samples.append((int(minX), int(minY), int(maxX), int(maxY)))


def drawQrOutline(display, points) -> None:
    if points is None:
        return
    try:
        pts = np.array(points)
        if pts.size == 0:
            return
        pts = pts.astype(np.int32)
        if pts.ndim == 2:
            pts = pts.reshape((-1, 1, 2))
        elif pts.ndim == 3:
            pass
        else:
            return
        cv2.polylines(display, [pts.reshape((-1, 2))], True, (0, 255, 0), 2)
    except cv2.error:
        return


def openCamera(cameraIndex: int):
    backends = []
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(cv2.CAP_AVFOUNDATION)
    backends.append(cv2.CAP_ANY)
    for attempt in range(3):
        for backend in backends:
            capture = cv2.VideoCapture(cameraIndex, backend)
            if not capture.isOpened():
                capture.release()
                continue
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            ok, _frame = capture.read()
            if ok:
                return capture
            capture.release()
        time.sleep(0.6)
        print(f"Waiting for camera permission... try {attempt + 1}/3")
    return None


class CameraStream:
    """Keep grabbing frames in a background thread so UI waits never freeze the camera."""

    def __init__(self, capture) -> None:
        self.capture = capture
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while self.running:
            ok, frame = self.capture.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self, box=None):
        with self.lock:
            if self.frame is None:
                return None
            if box is None:
                return self.frame.copy()
            height, width = self.frame.shape[:2]
            x0, y0, x1, y1 = box
            x0 = max(0, min(width - 1, int(x0)))
            y0 = max(0, min(height - 1, int(y0)))
            x1 = max(x0 + 1, min(width, int(x1)))
            y1 = max(y0 + 1, min(height, int(y1)))
            return np.ascontiguousarray(self.frame[y0:y1, x0:x1])

    def stop(self) -> None:
        self.running = False
        self.thread.join(timeout=1.5)


def composeQrOverCamera(
    cameraFrame: np.ndarray,
    qrCard: np.ndarray,
    canvasSize: tuple[int, int] | None = None,
) -> np.ndarray:
    """Large QR in the center, live camera picture-in-picture at bottom right."""
    if canvasSize is None:
        screenH, screenW = cameraFrame.shape[:2]
    else:
        screenW, screenH = canvasSize
    display = np.full((screenH, screenW, 3), (28, 26, 24), dtype=np.uint8)

    qrScale = min((screenH * 0.88) / float(qrCard.shape[0]), (screenW * 0.72) / float(qrCard.shape[1]))
    qrW = max(1, int(qrCard.shape[1] * qrScale))
    qrH = max(1, int(qrCard.shape[0] * qrScale))
    qrResized = cv2.resize(qrCard, (qrW, qrH), interpolation=cv2.INTER_NEAREST)
    x0 = (screenW - qrW) // 2
    y0 = max(8, (screenH - qrH) // 2 - int(screenH * 0.04))
    y0 = min(y0, screenH - qrH)
    display[y0 : y0 + qrH, x0 : x0 + qrW] = qrResized

    camH, camW = cameraFrame.shape[:2]
    pipW = max(160, min(int(screenW * 0.24), 420))
    pipH = max(90, int(pipW * camH / float(camW))) if camW else int(pipW * 9 / 16)
    pipH = min(pipH, int(screenH * 0.28))
    pip = cv2.resize(cameraFrame, (pipW, pipH), interpolation=cv2.INTER_AREA)
    margin = 18
    px = screenW - pipW - margin
    py = screenH - pipH - margin
    display[py : py + pipH, px : px + pipW] = pip
    cv2.rectangle(display, (px - 2, py - 2), (px + pipW + 1, py + pipH + 1), (0, 255, 255), 2)
    cv2.putText(
        display,
        "camera",
        (px + 8, py + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return display


def makeAlignCard(step: int, handshakeId: str, caption: str) -> np.ndarray:
    payloadText = encodeAlign(step, handshakeId)
    return buildPolaroidCard(
        buildQrImage(payloadText),
        captionLines=[
            f"ALIGN  step {step} / 3",
            caption,
            handshakeId,
        ],
        progressRatio=step / 3.0,
        paused=False,
    )


def runSenderAlignment(
    windowName: str,
    stream: CameraStream,
    detector,
    scanRoi: QrScanRoi | None = None,
):
    """QR1 from receiver, then QR2 and last QR3 from sender. Receiver stays aimed at sender."""
    print("Alignment: camera only until QR 1. Then sender shows QR 2, then last QR 3.")
    if scanRoi is None:
        scanRoi = QrScanRoi()
    handshakeId = None
    alignCard = None
    phase = "scan1"
    qr1Hits = 0
    lastQr1Id = None
    neededHits = 3
    holdUntil = 0.0
    while True:
            frame = stream.read(scanRoi.captureBox())
            if frame is None:
                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), 27):
                    return "quit"
                continue

            rawText, points = scanRoi.process(detector, frame)
            if phase == "scan1":
                step = None
                hid = None
                if rawText:
                    payload = decodeFrame(rawText)
                    if payload and payload.get("kind") == "align":
                        step = int(payload["step"])
                        hid = str(payload["handshakeId"])
                if step == 1 and hid:
                    if hid == lastQr1Id:
                        qr1Hits += 1
                    else:
                        lastQr1Id = hid
                        qr1Hits = 1
                    if qr1Hits >= neededHits:
                        handshakeId = hid
                        print(f"Got align 1 ({handshakeId}). Generating align 2.")
                        alignCard = makeAlignCard(
                            2,
                            handshakeId,
                            "Sender QR 2 — receiver should scan this",
                        )
                        phase = "show2"
                        holdUntil = time.time() + ALIGN_HOLD_SECONDS
                else:
                    qr1Hits = 0
                    lastQr1Id = None

            elif phase == "show2" and time.time() >= holdUntil:
                print("Showing last handshake QR 3 from sender.")
                alignCard = makeAlignCard(
                    3,
                    handshakeId,
                    "Sender QR 3 LAST — receiver scans this, then data starts",
                )
                phase = "show3"
                holdUntil = time.time() + ALIGN_HOLD_SECONDS

            elif phase == "show3" and time.time() >= holdUntil:
                print("Last sender QR shown. Handshake complete. Starting data.")
                return handshakeId

            if phase == "scan1" or alignCard is None:
                display = cv2.resize(frame, screenSize(), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    display,
                    "Waiting for RECEIVER QR 1  —  no sender QR yet",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if qr1Hits:
                    cv2.putText(
                        display,
                        f"QR 1 lock {qr1Hits}/{neededHits}",
                        (40, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (80, 220, 80),
                        2,
                        cv2.LINE_AA,
                    )
            else:
                annotated = frame.copy()
                drawQrOutline(annotated, points)
                display = composeQrOverCamera(annotated, alignCard, canvasSize=screenSize())
            if phase == "scan1" or alignCard is None:
                drawQrOutline(display, points)
            showOnSender(windowName, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return "quit"
    return handshakeId


def runReceiverAlignment(
    windowName: str,
    stream: CameraStream,
    detector,
    scanRoi: QrScanRoi | None = None,
):
    """Show QR1 until sender QR2 is read, then camera-only until last sender QR3."""
    if scanRoi is None:
        scanRoi = QrScanRoi()
    handshakeId = makeTransferId()
    print(f"Alignment: showing QR 1 ({handshakeId}). Last QR will come from the sender.")
    card = makeAlignCard(1, handshakeId, "Receiver QR 1 — sender should scan this")
    phase = 1
    while True:
        frame = stream.read(scanRoi.captureBox())
        if frame is None:
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                return "quit"
            continue

        rawText, points = scanRoi.process(detector, frame)
        if rawText:
            payload = decodeFrame(rawText)
            if payload and payload.get("kind") == "align":
                step = int(payload["step"])
                hid = str(payload["handshakeId"])
                if hid == handshakeId and phase == 1 and step == 2:
                    print("Got sender QR 2. Camera stays on sender for last QR 3.")
                    phase = 2
                    card = None
                elif hid == handshakeId and phase >= 2 and step == 3:
                    print("Got last sender QR 3. Alignment complete. Ready for data.")
                    return handshakeId

        annotated = frame.copy()
        drawQrOutline(annotated, points)
        if card is not None:
            display = composeQrOverCamera(annotated, card, canvasSize=screenSize())
        else:
            display = cv2.resize(annotated, screenSize(), interpolation=cv2.INTER_AREA)
            cv2.putText(
                display,
                "Look at SENDER  —  waiting for last handshake QR 3",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        showOnSender(windowName, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return "quit"


def waitForFrame(duration: float) -> str:
    """Hold a QR on screen. Returns quit if q/esc is pressed."""
    key = cv2.waitKey(max(1, int(duration * 1000))) & 0xFF
    if key in (ord("q"), 27):
        return "quit"
    return "ok"


def freezeOnLastUntilStatus(
    windowName: str,
    lastCard: np.ndarray,
    transferId: str,
    cameraIndex: int,
    scanRoi: QrScanRoi | None = None,
    stream: CameraStream | None = None,
    detector=None,
):
    """Show last QR and scan for status at the same time. Camera stays open for the session."""
    print("Frozen on last frame. Scanning receiver status...")
    if scanRoi is None:
        scanRoi = QrScanRoi()
    ownsCamera = stream is None
    capture = None
    if stream is None:
        capture = openCamera(cameraIndex)
        if capture is None:
            print(f"Could not open camera {cameraIndex} for status scan.")
            print("Allow Camera for Terminal/Cursor in System Settings > Privacy, then retry.")
            return None
        stream = CameraStream(capture)
        detector = cv2.QRCodeDetector()
    if detector is None:
        detector = cv2.QRCodeDetector()
    startedAt = time.time()
    try:
        while True:
            frame = stream.read(scanRoi.captureBox())
            if frame is None:
                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), 27):
                    return "quit"
                continue

            rawText, points = scanRoi.process(detector, frame)
            annotated = frame.copy()
            drawQrOutline(annotated, points)
            display = composeQrOverCamera(annotated, lastCard, canvasSize=screenSize())
            elapsed = time.time() - startedAt
            cv2.putText(
                display,
                f"scanning receiver status  {elapsed:0.1f}s  id={transferId}",
                (40, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            showOnSender(windowName, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return "quit"

            if not rawText:
                continue
            payload = decodeFrame(rawText)
            if payload is None:
                continue
            if payload.get("kind") not in ("status", "done"):
                continue
            if str(payload.get("transferId")) != transferId:
                continue
            if payload.get("kind") == "done":
                print("Receiver FILE COMPLETE QR matched this transferId.")
            else:
                print(
                    f"Got status: {payload.get('gotCount')}/{payload.get('total')} chunks on receiver"
                )
            return payload
    finally:
        if ownsCamera:
            stream.stop()
            if capture is not None:
                capture.release()


def runSender(
    sourcePath: str,
    frameDelay: float | None = None,
    cameraIndex: int = 0,
) -> None:
    if frameDelay is None:
        from app import DEFAULT_FRAME_DELAY

        frameDelay = DEFAULT_FRAME_DELAY

    print("\nListing files (small files packed together to cut ACK round-trips)...")
    rootName, jobs, fileCount = buildTransferJobs(sourcePath)
    sessionId = makeTransferId()
    jobCount = len(jobs)
    totalBytes = sum(int(job.get("byteSize") or 0) for job in jobs)
    print(f"Session ID  : {sessionId}")
    print(f"Root name   : {rootName}")
    print(f"Files/dirs  : {fileCount}")
    print(f"Transfers   : {jobCount}  (packs + large files)")
    print(f"Total data  : {humanSize(totalBytes)}")

    windowName = "codeShare Sender"
    scanRoi = QrScanRoi()
    capture = openCamera(cameraIndex)
    if capture is None:
        raise RuntimeError(
            f"Could not open camera index {cameraIndex}. "
            "Allow Camera access for Terminal or Cursor in System Settings > Privacy."
        )
    stream = CameraStream(capture)
    detector = cv2.QRCodeDetector()
    try:
        setupFullscreen(windowName)
        aligned = runSenderAlignment(windowName, stream, detector, scanRoi=scanRoi)
        if aligned == "quit":
            print("Sender stopped.")
            return

        print("\nCalibration file: 10 QR frames to lock a FIXED search crop.")
        scanRoi.beginCalibrate()
        calFrames, calSummary = buildCalibrateFrames(sessionId, jobCount)
        calResult = sendUntilComplete(
            windowName,
            calFrames,
            calSummary,
            frameDelay,
            cameraIndex,
            scanRoi=scanRoi,
            stream=stream,
            detector=detector,
        )
        if calResult == "quit":
            print("Sender stopped.")
            return
        scanRoi.lock()

        print(f"\nFrame interval: {frameDelay:.2f}s")
        print("Press [q] to quit.\n")

        for fileIndex, entry in enumerate(jobs, start=1):
            label = entry.get("relPath") or entry.get("kind")
            print(
                f"\n===== Transfer {fileIndex}/{jobCount}: {label} "
                f"({humanSize(entry['byteSize'])}) ====="
            )
            frames, summary = buildEntryFrames(
                entry, sessionId, rootName, fileIndex, jobCount
            )
            result = sendUntilComplete(
                windowName,
                frames,
                summary,
                frameDelay,
                cameraIndex,
                scanRoi=scanRoi,
                stream=stream,
                detector=detector,
            )
            if result == "quit":
                print("Sender stopped.")
                return

        sessionText = encodeSessionDone(sessionId, jobCount)
        doneCard = buildPolaroidCard(
            buildQrImage(sessionText),
            captionLines=[
                "SESSION COMPLETE",
                f"{fileCount} files in {jobCount} transfers  ·  {sessionId}",
                "All files received. Quitting.",
            ],
            progressRatio=1.0,
            paused=False,
        )
        showOnSender(windowName, doneCard)
        cv2.waitKey(1500)
        print("All files sent. Sender stopped.")
    finally:
        stream.stop()
        capture.release()
        cv2.destroyAllWindows()
