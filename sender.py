"""Build and display QR frames for sending a directory."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_M

from packer import humanSize, packDirectory
from protocol import (
    encodeDataFrame,
    encodeHeaderFrame,
    makeTransferId,
    packDirectoryArchive,
)


def buildQrImage(payloadText: str, boxSize: int = 6, border: int = 2) -> np.ndarray:
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
) -> np.ndarray:
    """Instant-photo style card: clean QR on top, caption + seek bar below."""
    sidePad = 36
    topPad = 36
    captionHeight = 128
    paper = (246, 242, 232)  # warm off-white (BGR)
    ink = (42, 38, 34)
    muted = (110, 105, 98)
    barTrack = (210, 205, 196)
    barFill = (70, 120, 40) if not paused else (40, 140, 220)

    qrSize = 520
    qrResized = cv2.resize(qrImage, (qrSize, qrSize), interpolation=cv2.INTER_NEAREST)

    cardWidth = qrSize + sidePad * 2
    cardHeight = topPad + qrSize + captionHeight
    card = np.full((cardHeight, cardWidth, 3), paper, dtype=np.uint8)

    card[topPad : topPad + qrSize, sidePad : sidePad + qrSize] = qrResized

    # Soft inner edge around the photo area
    cv2.rectangle(
        card,
        (sidePad - 1, topPad - 1),
        (sidePad + qrSize, topPad + qrSize),
        (220, 215, 205),
        1,
        cv2.LINE_AA,
    )

    textY = topPad + qrSize + 34
    for lineIndex, line in enumerate(captionLines):
        color = ink if lineIndex == 0 else muted
        scale = 0.72 if lineIndex == 0 else 0.55
        thickness = 2 if lineIndex == 0 else 1
        cv2.putText(
            card,
            line,
            (sidePad, textY + lineIndex * 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    # Seek / timeline bar
    barX = sidePad
    barY = cardHeight - 28
    barW = qrSize
    barH = 10
    cv2.rectangle(card, (barX, barY), (barX + barW, barY + barH), barTrack, -1, cv2.LINE_AA)

    fillWidth = max(2, int(barW * max(0.0, min(1.0, progressRatio))))
    cv2.rectangle(card, (barX, barY), (barX + fillWidth, barY + barH), barFill, -1, cv2.LINE_AA)

    # Playhead tick
    tickX = barX + fillWidth - 1
    cv2.rectangle(card, (tickX - 1, barY - 4), (tickX + 2, barY + barH + 4), ink, -1, cv2.LINE_AA)

    if paused:
        cv2.putText(
            card,
            "PAUSED",
            (cardWidth - sidePad - 90, barY - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (40, 140, 220),
            1,
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


def saveQrFrames(frames: List[str], outputDir: str) -> Path:
    outPath = Path(outputDir).expanduser().resolve()
    outPath.mkdir(parents=True, exist_ok=True)
    for index, payloadText in enumerate(frames):
        image = buildQrImage(payloadText)
        filePath = outPath / f"frame_{index:05d}.png"
        cv2.imwrite(str(filePath), image)
    return outPath


def runSender(sourcePath: str, frameDelay: float = 0.8, saveDir: str | None = None) -> None:
    print("\nPacking directory...")
    frames, summary = buildTransferFrames(sourcePath)
    print(f"Transfer ID : {summary['transferId']}")
    print(f"Root name   : {summary['rootName']}")
    print(f"Archive size: {summary['humanSize']}")
    print(f"QR frames   : {summary['totalFrames']}")

    if saveDir:
        savedPath = saveQrFrames(frames, saveDir)
        print(f"Saved frames to: {savedPath}")

    print("\nControls: [n] next  [p] prev  [space] pause/resume  [q] quit")
    print("Point the receiving camera at this window.\n")

    windowName = "codeShare Sender"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 640, 760)

    index = 0
    paused = False
    lastAdvance = time.time()
    totalFrames = len(frames)

    while True:
        qrImage = buildQrImage(frames[index])
        progressRatio = (index + 1) / totalFrames
        captionLines = [
            f"{summary['rootName']}",
            f"frame {index + 1} / {totalFrames}   ·   {summary['transferId']}",
            f"{summary['humanSize']}   ·   {'paused' if paused else 'playing'}",
        ]
        card = buildPolaroidCard(
            qrImage,
            captionLines=captionLines,
            progressRatio=progressRatio,
            paused=paused,
        )

        cv2.imshow(windowName, card)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("n"):
            index = (index + 1) % totalFrames
            lastAdvance = time.time()
        elif key == ord("p"):
            index = (index - 1) % totalFrames
            lastAdvance = time.time()
        elif key == ord(" "):
            paused = not paused
            lastAdvance = time.time()

        if not paused and (time.time() - lastAdvance) >= frameDelay:
            index = (index + 1) % totalFrames
            lastAdvance = time.time()

    cv2.destroyAllWindows()
    print("Sender stopped.")
