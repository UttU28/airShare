"""Build and display QR frames for sending a directory."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_M

from packer import humanSize, packDirectory
from protocol import (
    decodeFrame,
    encodeAckRequest,
    encodeAlign,
    encodeDataFrame,
    encodeHeaderFrame,
    makeTransferId,
    missingSeqsFromStatus,
    packDirectoryArchive,
)


SCREEN_SIZE = {"w": 1920, "h": 1080}


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


def buildTransferFrames(sourcePath: str) -> tuple[List[str], dict]:
    archiveBytes, rootName = packDirectory(sourcePath)
    transferId = makeTransferId()
    headerMeta, dataChunks = packDirectoryArchive(archiveBytes, transferId, rootName)

    frames: List[str] = [encodeHeaderFrame(headerMeta)]
    total = int(headerMeta["total"])
    for index, chunkBytes in enumerate(dataChunks, start=1):
        frames.append(encodeDataFrame(transferId, index, total, chunkBytes))

    summary = {
        "transferId": transferId,
        "rootName": rootName,
        "byteSize": len(archiveBytes),
        "totalFrames": total,
        "humanSize": humanSize(len(archiveBytes)),
    }
    return frames, summary


def saveQrFrames(qrImages: List[np.ndarray], outputDir: str) -> Path:
    outPath = Path(outputDir).expanduser().resolve()
    outPath.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(qrImages):
        filePath = outPath / f"frame_{index:05d}.png"
        cv2.imwrite(str(filePath), image)
    return outPath


def safeDetectAndDecode(detector, frame):
    try:
        rawText, points, _straight = detector.detectAndDecode(frame)
    except cv2.error:
        return "", None
    except Exception:
        return "", None
    if not rawText:
        rawText = ""
    return rawText, points


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

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self) -> None:
        self.running = False
        self.thread.join(timeout=1.5)


def composeQrOverCamera(
    cameraFrame: np.ndarray,
    qrCard: np.ndarray,
    canvasSize: tuple[int, int] | None = None,
) -> np.ndarray:
    """One view: live camera behind a large QR. canvasSize defaults to the camera frame."""
    if canvasSize is None:
        screenH, screenW = cameraFrame.shape[:2]
    else:
        screenW, screenH = canvasSize
    display = cv2.resize(cameraFrame, (screenW, screenH), interpolation=cv2.INTER_AREA)
    display = (display.astype(np.float32) * 0.28).astype(np.uint8)

    qrScale = min((screenH * 0.88) / float(qrCard.shape[0]), (screenW * 0.72) / float(qrCard.shape[1]))
    qrW = max(1, int(qrCard.shape[1] * qrScale))
    qrH = max(1, int(qrCard.shape[0] * qrScale))
    qrResized = cv2.resize(qrCard, (qrW, qrH), interpolation=cv2.INTER_NEAREST)
    x0 = (screenW - qrW) // 2
    y0 = (screenH - qrH) // 2
    display[y0 : y0 + qrH, x0 : x0 + qrW] = qrResized
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


def runSenderAlignment(windowName: str, cameraIndex: int):
    """Scan receiver QR1 first. Only then generate and show QR2. Wait for QR3."""
    print("Alignment: camera only until QR 1 is scanned. Then show QR 2.")
    capture = openCamera(cameraIndex)
    if capture is None:
        raise RuntimeError("Could not open sender camera for alignment.")

    stream = CameraStream(capture)
    detector = cv2.QRCodeDetector()
    handshakeId = None
    alignCard = None
    phase = "scan1"
    qr1Hits = 0
    lastQr1Id = None
    neededHits = 3
    try:
        while True:
            frame = stream.read()
            if frame is None:
                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), 27):
                    return "quit"
                continue

            rawText, points = safeDetectAndDecode(detector, frame)
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
                else:
                    qr1Hits = 0
                    lastQr1Id = None

            elif phase == "show2" and rawText:
                payload = decodeFrame(rawText)
                if payload and payload.get("kind") == "align":
                    step = int(payload["step"])
                    hid = str(payload["handshakeId"])
                    if hid == handshakeId and step == 3:
                        print("Got align 3. Handshake complete.")
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
                display = composeQrOverCamera(frame, alignCard, canvasSize=screenSize())
            drawQrOutline(display, points)
            showOnSender(windowName, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return "quit"
    finally:
        stream.stop()
        capture.release()


def runReceiverAlignment(windowName: str, stream: CameraStream, detector):
    """Show QR1 while scanning for QR2, then show QR3. Camera never pauses."""
    handshakeId = makeTransferId()
    print(f"Alignment: showing QR 1 ({handshakeId}), camera looking for sender QR 2.")
    card = makeAlignCard(1, handshakeId, "Receiver QR 1 — sender should scan this")
    phase = 1
    qr3Until = 0.0
    while True:
        frame = stream.read()
        if frame is None:
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                return "quit"
            continue

        rawText, points = safeDetectAndDecode(detector, frame)
        if rawText:
            payload = decodeFrame(rawText)
            if payload and payload.get("kind") == "align":
                step = int(payload["step"])
                hid = str(payload["handshakeId"])
                if phase == 1 and step == 2 and hid == handshakeId:
                    print("Got align 2. Showing align 3.")
                    card = makeAlignCard(
                        3,
                        handshakeId,
                        "Receiver QR 3 — sender final scan",
                    )
                    phase = 3
                    qr3Until = time.time() + 4.0

        if phase == 3 and time.time() >= qr3Until:
            print("Alignment complete. Ready for data QRs.")
            return handshakeId

        display = composeQrOverCamera(frame, card, canvasSize=screenSize())
        drawQrOutline(display, points)
        showOnSender(windowName, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return "quit"


def waitWithKeys(duration: float, paused: bool) -> tuple[str, bool]:
    """Wait up to duration seconds. Returns (action, paused)."""
    deadline = time.time() + duration
    while True:
        remainingMs = max(1, int((deadline - time.time()) * 1000))
        key = cv2.waitKey(min(remainingMs, 50)) & 0xFF
        if key in (ord("q"), 27):
            return "quit", paused
        if key == ord("n"):
            return "next", paused
        if key == ord("p"):
            return "prev", paused
        if key == ord(" "):
            paused = not paused
            if paused:
                deadline = time.time() + 3600
            else:
                deadline = time.time() + duration
        if not paused and time.time() >= deadline:
            return "next", paused


def scanStatusQr(
    windowName: str,
    transferId: str,
    cameraIndex: int,
    timeoutSeconds: float,
):
    capture = openCamera(cameraIndex)
    if capture is None:
        print(f"Could not open camera {cameraIndex} for status scan.")
        print("Allow Camera for Terminal/Cursor in System Settings > Privacy, then retry.")
        return None

    detector = cv2.QRCodeDetector()
    deadline = time.time() + timeoutSeconds
    print(f"Scanning receiver status QR (camera {cameraIndex})...")
    try:
        while time.time() < deadline:
            ok, frame = capture.read()
            if not ok:
                continue
            rawText, points = safeDetectAndDecode(detector, frame)
            display = frame.copy()
            drawQrOutline(display, points)
            left = max(0.0, deadline - time.time())
            cv2.putText(
                display,
                f"Aim at RECEIVER status QR  {left:0.1f}s  id={transferId}",
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(windowName, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return "quit"

            if not rawText:
                continue
            payload = decodeFrame(rawText)
            if payload is None:
                continue
            if payload.get("kind") != "status":
                continue
            if str(payload.get("transferId")) != transferId:
                continue
            print(
                f"Got status: {payload.get('gotCount')}/{payload.get('total')} chunks on receiver"
            )
            return payload
    finally:
        capture.release()
    return None


def freezeOnLastUntilStatus(
    windowName: str,
    lastCard: np.ndarray,
    transferId: str,
    cameraIndex: int,
    statusHold: float,
    statusScan: float,
):
    """Show last QR and scan for status at the same time. Camera thread never sleeps."""
    print("Frozen on last frame. Camera stays live under the QR overlay until status is read.")
    capture = openCamera(cameraIndex)
    if capture is None:
        print(f"Could not open camera {cameraIndex} for status scan.")
        print("Allow Camera for Terminal/Cursor in System Settings > Privacy, then retry.")
        return None

    stream = CameraStream(capture)
    detector = cv2.QRCodeDetector()
    startedAt = time.time()
    try:
        while True:
            frame = stream.read()
            if frame is None:
                key = cv2.waitKey(10) & 0xFF
                if key in (ord("q"), 27):
                    return "quit"
                continue

            rawText, points = safeDetectAndDecode(detector, frame)
            display = composeQrOverCamera(frame, lastCard, canvasSize=screenSize())
            drawQrOutline(display, points)
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
            if payload.get("kind") != "status":
                continue
            if str(payload.get("transferId")) != transferId:
                continue
            print(
                f"Got status: {payload.get('gotCount')}/{payload.get('total')} chunks on receiver"
            )
            return payload
    finally:
        stream.stop()
        capture.release()


def runSender(
    sourcePath: str,
    frameDelay: float | None = None,
    saveDir: str | None = None,
    cameraIndex: int = 0,
    statusHold: float = 4.0,
    statusScan: float = 20.0,
) -> None:
    if frameDelay is None:
        from app import DEFAULT_FRAME_DELAY

        frameDelay = DEFAULT_FRAME_DELAY
    windowName = "codeShare Sender"
    setupFullscreen(windowName)
    aligned = runSenderAlignment(windowName, cameraIndex)
    if aligned == "quit":
        cv2.destroyAllWindows()
        print("Sender stopped.")
        return

    print("\nPacking directory...")
    frames, summary = buildTransferFrames(sourcePath)
    print(f"Transfer ID : {summary['transferId']}")
    print(f"Root name   : {summary['rootName']}")
    print(f"Archive size: {summary['humanSize']}")
    print(f"QR frames   : {summary['totalFrames']}")

    print("Pre-rendering QR images...")
    qrImages = [buildQrImage(payloadText) for payloadText in frames]

    if saveDir:
        savedPath = saveQrFrames(qrImages, saveDir)
        print(f"Saved frames to: {savedPath}")

    transferId = summary["transferId"]
    totalFrames = len(frames)
    pending = list(range(totalFrames))
    roundIndex = 1
    previousMissing = None

    print(f"\nFrame interval: {frameDelay:.2f}s")
    print("Each pass plays once, then FREEZES on the last QR (no loop).")
    print("Only missing frames are sent in the next pass.")
    print("If missing frames stay the same, interval increases by 0.5s and stays up.")
    print("Controls: [n] next  [p] prev  [space] pause/resume  [q] quit\n")

    try:
        while pending:
            print(f"\n--- Round {roundIndex}: sending {len(pending)} pending frame(s)  interval={frameDelay:.2f}s ---")
            playIndex = 0
            paused = False
            lastSeq = pending[-1]
            while playIndex < len(pending):
                seq = pending[playIndex]
                progressRatio = (playIndex + 1) / len(pending)
                captionLines = [
                    f"{summary['rootName']}   round {roundIndex}",
                    f"seq {seq}   {playIndex + 1}/{len(pending)} pending   ·   {transferId}",
                    f"{summary['humanSize']}   ·   {'paused' if paused else 'playing'}",
                ]
                card = buildPolaroidCard(
                    qrImages[seq],
                    captionLines=captionLines,
                    progressRatio=progressRatio,
                    paused=paused,
                )
                showOnSender(windowName, card)
                action, paused = waitWithKeys(frameDelay, paused)
                if action == "quit":
                    print("Sender stopped.")
                    return
                if action == "prev":
                    playIndex = max(0, playIndex - 1)
                    continue
                playIndex += 1

            ackText = encodeAckRequest(transferId, totalFrames, roundIndex)
            lastCard = buildPolaroidCard(
                buildQrImage(ackText),
                captionLines=[
                    f"ACK REQUEST   round {roundIndex}",
                    f"pass done  last data seq {lastSeq}   ·   {transferId}",
                    "Receiver: this QR always means send status now",
                ],
                progressRatio=1.0,
                paused=False,
            )
            showOnSender(windowName, lastCard)
            cv2.waitKey(1)

            statusPayload = freezeOnLastUntilStatus(
                windowName,
                lastCard,
                transferId,
                cameraIndex,
                statusHold,
                statusScan,
            )
            if statusPayload == "quit":
                print("Sender stopped.")
                return

            pending = missingSeqsFromStatus(statusPayload)
            print(f"Receiver still missing {len(pending)} frame(s).")
            if previousMissing is not None and pending == previousMissing:
                frameDelay += 0.5
                print(
                    f"Missing set unchanged. Interval +0.5s → {frameDelay:.2f}s "
                    "(kept for later rounds)."
                )
            previousMissing = list(pending)
            if not pending:
                print("Receiver has everything. Transfer complete.")
                doneCard = buildPolaroidCard(
                    qrImages[0],
                    captionLines=[
                        "TRANSFER COMPLETE",
                        f"{transferId}",
                        "Receiver should unpack now.",
                    ],
                    progressRatio=1.0,
                    paused=False,
                )
                showOnSender(windowName, doneCard)
                cv2.waitKey(800)
                return

            print("Starting next pass automatically.")
            roundIndex += 1
    finally:
        cv2.destroyAllWindows()
    print("Sender stopped.")
