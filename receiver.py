"""Receive QR frames from camera and reconstruct a directory."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import time

import cv2
import numpy as np

from packer import humanSize, writeReceivedDir, writeReceivedFile
from protocol import decodeFrame, encodeDone, encodeStatus, rebuildArchive, validateDataFrame
from sender import (
    CameraStream,
    QrScanRoi,
    buildPolaroidCard,
    buildQrImage,
    composeQrOverCamera,
    drawQrOutline,
    drawRoiBox,
    openCamera,
    runReceiverAlignment,
    screenSize,
    setupFullscreen,
    showOnSender,
)


ACK_RETRY_SECONDS = 3.0


def missingSeqList(headerGot: bool, chunkMap: Dict[int, bytes], totalFrames: int) -> list[int]:
    missing = []
    if totalFrames <= 0:
        return missing
    if not headerGot:
        missing.append(0)
    for seq in range(1, totalFrames):
        if seq not in chunkMap:
            missing.append(seq)
    return missing


def drawChunkOverlay(
    image: np.ndarray,
    headerGot: bool,
    chunkMap: Dict[int, bytes],
    totalFrames: Optional[int],
    lastSeenSeq: Optional[int],
) -> np.ndarray:
    """Bottom seek bar: each chunk as got / missing, plus a count caption."""
    height, width = image.shape[:2]
    panelHeight = 92
    panelTop = max(0, height - panelHeight)

    overlay = image.copy()
    cv2.rectangle(overlay, (0, panelTop), (width, height), (18, 16, 14), -1)
    display = cv2.addWeighted(overlay, 0.78, image, 0.22, 0)

    margin = 18
    if not totalFrames:
        caption = "waiting for first QR..."
        cv2.putText(
            display,
            caption,
            (margin, panelTop + 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return display

    dataExpected = max(0, totalFrames - 1)
    dataGot = len(chunkMap)
    missing = missingSeqList(headerGot, chunkMap, totalFrames)
    missingPreview = ", ".join(str(seq) for seq in missing[:12])
    if len(missing) > 12:
        missingPreview += f" +{len(missing) - 12}"

    caption = (
        f"chunks {dataGot + (1 if headerGot else 0)} / {totalFrames}   "
        f"data {dataGot}/{dataExpected}   "
        f"header={'yes' if headerGot else 'no'}"
    )
    missingCaption = f"missing: {missingPreview}" if missing else "missing: none"

    cv2.putText(
        display,
        caption,
        (margin, panelTop + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display,
        missingCaption,
        (margin, panelTop + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (160, 170, 255) if missing else (140, 210, 140),
        1,
        cv2.LINE_AA,
    )

    barX = margin
    barY = height - 22
    barW = max(1, width - margin * 2)
    barH = 12
    cv2.rectangle(display, (barX, barY), (barX + barW, barY + barH), (70, 68, 64), -1, cv2.LINE_AA)

    for seq in range(totalFrames):
        x0 = barX + int(seq * barW / totalFrames)
        x1 = barX + int((seq + 1) * barW / totalFrames)
        if x1 <= x0:
            x1 = x0 + 1

        got = headerGot if seq == 0 else seq in chunkMap
        if got:
            color = (70, 190, 80)
        else:
            color = (70, 70, 200)
        cv2.rectangle(display, (x0, barY), (x1, barY + barH), color, -1)

        if lastSeenSeq is not None and seq == lastSeenSeq:
            cv2.rectangle(display, (x0, barY - 3), (x1, barY + barH + 3), (255, 255, 255), 1, cv2.LINE_AA)

    return display


def scaleFrameToFit(frame: np.ndarray, maxWidth: int = 1400, maxHeight: int = 900) -> np.ndarray:
    """Large preview that keeps the camera's native aspect ratio (no stretch)."""
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame

    scale = min(maxWidth / float(width), maxHeight / float(height))
    newWidth = max(1, int(round(width * scale)))
    newHeight = max(1, int(round(height * scale)))
    if newWidth == width and newHeight == height:
        return frame
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(frame, (newWidth, newHeight), interpolation=interpolation)


def trySaveFile(headerMeta, chunkMap, outputPath: Path) -> bool:
    if headerMeta is None:
        return False
    total = int(headerMeta["total"])
    if len(chunkMap) != total - 1:
        return False
    relPath = str(headerMeta.get("relPath") or headerMeta.get("rootName") or "file")
    rootName = str(headerMeta.get("rootName") or "share")
    entryKind = str(headerMeta.get("entryKind") or "file")
    fileIndex = headerMeta.get("fileIndex")
    fileCount = headerMeta.get("fileCount")
    progress = ""
    if fileIndex and fileCount:
        progress = f" [{fileIndex}/{fileCount}]"

    if entryKind == "dir":
        dest = writeReceivedDir(str(outputPath), rootName, relPath)
        print(f"Saved empty dir{progress}: {dest}")
        return True

    print(f"\nFile complete{progress}. Writing {relPath}...")
    payloadBytes = rebuildArchive(headerMeta, chunkMap)
    dest = writeReceivedFile(str(outputPath), rootName, relPath, payloadBytes)
    print(f"Wrote {humanSize(len(payloadBytes))} → {dest}")
    return True


def buildDoneCard(
    transferId: str,
    knownTotal: int,
    relPath: str = "",
    fileIndex: int | None = None,
) -> np.ndarray:
    doneText = encodeDone(transferId, knownTotal, fileIndex=fileIndex)
    return buildPolaroidCard(
        buildQrImage(doneText),
        captionLines=[
            "FILE COMPLETE",
            f"{relPath}" if relPath else f"all {knownTotal} frames received",
            f"{knownTotal} frames  ·  {transferId}  — sender scan this, then next file",
        ],
        progressRatio=1.0,
        paused=False,
    )


def showDoneAcknowledgment(windowName: str, stream, transferId: str, knownTotal: int) -> None:
    doneCard = buildDoneCard(transferId, knownTotal)
    print("Showing FILE COMPLETE QR for sender. Camera PiP bottom-right.")
    holdUntil = time.time() + 8.0
    while time.time() < holdUntil:
        frame = stream.read()
        if frame is None:
            key = cv2.waitKey(30) & 0xFF
            if key in (ord("q"), 27):
                return
            continue
        display = composeQrOverCamera(frame, doneCard, canvasSize=screenSize())
        showOnSender(windowName, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return


def buildStatusCard(
    transferId: str,
    knownTotal: int,
    headerGot: bool,
    chunkMap: Dict[int, bytes],
    roundIndex: int,
) -> np.ndarray:
    gotSeqs = list(chunkMap.keys())
    statusText = encodeStatus(transferId, knownTotal, headerGot, gotSeqs, roundIndex)
    missing = missingSeqList(headerGot, chunkMap, knownTotal)
    print(f"Status QR ready ({len(missing)} missing). Camera stays live behind it.")
    return buildPolaroidCard(
        buildQrImage(statusText),
        captionLines=[
            f"STATUS REPLY   round {roundIndex}",
            f"got {knownTotal - len(missing)}/{knownTotal}   ·   {transferId}",
            f"missing {len(missing)}  — aim SENDER camera here",
        ],
        progressRatio=(knownTotal - len(missing)) / knownTotal if knownTotal else 1.0,
        paused=False,
    )


def runReceiver(outputDir: str, cameraIndex: int = 0) -> None:
    outputPath = Path(outputDir).expanduser().resolve()
    outputPath.mkdir(parents=True, exist_ok=True)

    capture = openCamera(cameraIndex)
    if capture is None:
        raise RuntimeError(
            f"Could not open camera index {cameraIndex}. "
            "Allow Camera access for Terminal or Cursor in System Settings > Privacy."
        )

    detector = cv2.QRCodeDetector()
    stream = CameraStream(capture)
    scanRoi = QrScanRoi()
    headerMeta: Optional[dict] = None
    transferId: Optional[str] = None
    chunkMap: Dict[int, bytes] = {}
    seenPayloads: set[str] = set()
    knownTotal: Optional[int] = None
    lastSeenSeq: Optional[int] = None
    lastReplyRound: Optional[int] = None
    lastReplyAt = 0.0
    statusCard = None
    fileSaved = False
    doneCard = None
    filesSaved = 0

    windowName = "codeShare Receiver"
    setupFullscreen(windowName)

    camW = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    camH = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("\nReceiver ready. Aim camera at sender QR codes.")
    print(f"Camera mode: {camW}x{camH}")
    print("Start with 3-way align (QR1 here, QR2 then last QR3 on sender).")
    print("Files are saved one at a time as each completes.")
    print("Press [q] to quit.\n")

    aligned = runReceiverAlignment(windowName, stream, detector, scanRoi=scanRoi)
    if aligned == "quit":
        stream.stop()
        capture.release()
        cv2.destroyAllWindows()
        return
    if scanRoi.box:
        print(f"QR search box locked after handshake: {scanRoi.box}")

    try:
        while True:
            frame = stream.read()
            if frame is None:
                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), 27):
                    break
                continue

            rawText, points = scanRoi.detect(detector, frame)
            pendingStatusRound = None

            if rawText:
                payload = decodeFrame(rawText)
                if payload is not None:
                    kind = payload["kind"]
                    if kind == "align":
                        continue
                    if kind == "sessionDone":
                        print(
                            f"Session complete. Saved {filesSaved} file(s). Receiver quitting."
                        )
                        break

                    total = int(payload["total"])
                    frameTransferId = str(payload["transferId"])
                    sameFile = transferId is not None and frameTransferId == transferId
                    nextFileReady = fileSaved and not sameFile

                    if transferId is None or nextFileReady:
                        if nextFileReady:
                            print(
                                f"\nPrevious file saved. Switching to {frameTransferId} "
                                f"({kind})"
                            )
                        else:
                            print(f"\nStarting file transfer {frameTransferId}")
                        transferId = frameTransferId
                        headerMeta = None
                        chunkMap = {}
                        seenPayloads = set()
                        lastSeenSeq = None
                        lastReplyRound = None
                        statusCard = None
                        doneCard = None
                        fileSaved = False

                    knownTotal = total

                    if frameTransferId != transferId:
                        print(
                            f"Still receiving current file; ignoring {kind} "
                            f"from {frameTransferId}"
                        )
                    elif kind == "ackRequest":
                        roundIndex = int(payload.get("round", 0))
                        canRetry = (time.time() - lastReplyAt) > ACK_RETRY_SECONDS
                        if roundIndex != lastReplyRound or canRetry:
                            pendingStatusRound = roundIndex
                            print(f"ACK QR detected (round {roundIndex}). Showing status.")
                    elif kind in ("header", "data") and not fileSaved:
                        seq = int(payload["seq"])
                        lastSeenSeq = seq
                        if rawText not in seenPayloads:
                            seenPayloads.add(rawText)
                            if kind == "header" and headerMeta is None:
                                headerMeta = payload
                                print(
                                    f"Header: {payload.get('relPath')} "
                                    f"{payload.get('fileIndex')}/{payload.get('fileCount')} "
                                    f"size={humanSize(int(payload['byteSize']))} "
                                    f"frames={total}"
                                )
                            elif kind == "data" and seq not in chunkMap:
                                chunkBytes = validateDataFrame(payload)
                                if chunkBytes is not None:
                                    chunkMap[seq] = chunkBytes
                                    print(
                                        f"Got data frame seq={seq}  "
                                        f"({len(chunkMap)}/{total - 1} collected)"
                                    )

            if pendingStatusRound is not None and transferId and knownTotal:
                statusCard = buildStatusCard(
                    transferId,
                    knownTotal,
                    headerMeta is not None,
                    chunkMap,
                    pendingStatusRound,
                )
                lastReplyRound = pendingStatusRound
                lastReplyAt = time.time()

            display = frame.copy()
            drawRoiBox(display, scanRoi.box)
            drawQrOutline(display, points)
            preview = scaleFrameToFit(display)
            preview = drawChunkOverlay(
                preview,
                headerGot=headerMeta is not None,
                chunkMap=chunkMap,
                totalFrames=knownTotal,
                lastSeenSeq=lastSeenSeq,
            )
            if statusCard is not None and not fileSaved:
                preview = composeQrOverCamera(display, statusCard, canvasSize=screenSize())
            elif fileSaved and doneCard is not None:
                preview = composeQrOverCamera(display, doneCard, canvasSize=screenSize())
            showOnSender(windowName, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if not fileSaved and trySaveFile(headerMeta, chunkMap, outputPath):
                fileSaved = True
                filesSaved += 1
                relPath = str(headerMeta.get("relPath") or "")
                fileIndex = headerMeta.get("fileIndex")
                if fileIndex is not None:
                    fileIndex = int(fileIndex)
                doneCard = buildDoneCard(
                    str(transferId),
                    int(headerMeta["total"]),
                    relPath,
                    fileIndex=fileIndex,
                )
                statusCard = None
                print("Showing FILE COMPLETE QR. Waiting for next file or session done.")
    finally:
        stream.stop()
        capture.release()
        cv2.destroyAllWindows()
