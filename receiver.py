"""Receive QR frames from camera and reconstruct a directory."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import cv2

from packer import humanSize, unpackArchive
from protocol import decodeFrame, rebuildArchive, validateDataFrame


def runReceiver(outputDir: str, cameraIndex: int = 0) -> None:
    outputPath = Path(outputDir).expanduser().resolve()
    outputPath.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(cameraIndex)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera index {cameraIndex}")

    detector = cv2.QRCodeDetector()
    headerMeta: Optional[dict] = None
    transferId: Optional[str] = None
    chunkMap: Dict[int, bytes] = {}
    seenPayloads: set[str] = set()

    windowName = "codeShare Receiver"
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

    print("\nReceiver ready. Aim camera at sender QR codes.")
    print("Press [q] to quit.\n")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera read failed.")
                break

            rawText, points, _ = detector.detectAndDecode(frame)
            statusLines = []

            if rawText and rawText not in seenPayloads:
                payload = decodeFrame(rawText)
                if payload is not None:
                    seenPayloads.add(rawText)
                    kind = payload["kind"]
                    seq = int(payload["seq"])
                    total = int(payload["total"])
                    frameTransferId = str(payload["transferId"])

                    if transferId is None:
                        transferId = frameTransferId
                    elif frameTransferId != transferId:
                        statusLines.append("Ignoring different transferId")
                    else:
                        if kind == "header" and headerMeta is None:
                            headerMeta = payload
                            print(
                                f"Header: root={payload.get('rootName')} "
                                f"size={humanSize(int(payload['byteSize']))} "
                                f"frames={total}"
                            )
                        elif kind == "data" and seq not in chunkMap:
                            chunkBytes = validateDataFrame(payload)
                            if chunkBytes is not None:
                                chunkMap[seq] = chunkBytes
                                print(f"Got frame {seq}/{total - 1} data  ({len(chunkMap)}/{total - 1})")

            if points is not None and len(points):
                pts = points.astype(int)
                for pointSet in pts:
                    cv2.polylines(frame, [pointSet], True, (0, 255, 0), 2)

            receivedData = len(chunkMap)
            expectedData = int(headerMeta["total"]) - 1 if headerMeta else "?"
            progress = f"id={transferId or '-'}  data={receivedData}/{expectedData}  header={'yes' if headerMeta else 'no'}"
            cv2.putText(
                frame,
                progress,
                (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            for lineIndex, line in enumerate(statusLines):
                cv2.putText(
                    frame,
                    line,
                    (16, 64 + lineIndex * 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(windowName, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if headerMeta is not None:
                total = int(headerMeta["total"])
                if len(chunkMap) == total - 1:
                    print("\nAll frames received. Rebuilding archive...")
                    archiveBytes = rebuildArchive(headerMeta, chunkMap)
                    dest = unpackArchive(archiveBytes, str(outputPath))
                    print(f"Done. Restored under: {dest / headerMeta['rootName']}")
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
