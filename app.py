#!/usr/bin/env python3
"""codeShare — QR-based directory transfer over camera."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RECEIVE_DIR = PROJECT_ROOT / "received"
DEFAULT_SHARE_DIR = Path("/Users/antonio/Desktop/jellyfin")
DEFAULT_CAMERA_INDEX = 0
DEFAULT_FRAME_DELAY = 0.5


def promptChoice(promptText: str, allowed: set[str]) -> str:
    while True:
        value = input(promptText).strip().lower()
        if value in allowed:
            return value
        print(f"Please choose one of: {', '.join(sorted(allowed))}")


def runSendMode() -> None:
    from sender import runSender

    sourcePath = str(DEFAULT_SHARE_DIR)
    print(f"Sharing directory: {sourcePath}")
    runSender(
        sourcePath,
        frameDelay=DEFAULT_FRAME_DELAY,
        cameraIndex=DEFAULT_CAMERA_INDEX,
    )


def runReceiveMode() -> None:
    from receiver import runReceiver

    outputDir = str(DEFAULT_RECEIVE_DIR)
    print(f"Restoring into: {outputDir}")
    print(f"Camera index: {DEFAULT_CAMERA_INDEX}")
    runReceiver(outputDir, cameraIndex=DEFAULT_CAMERA_INDEX)


def printBanner() -> None:
    print("=" * 48)
    print(" codeShare — QR directory transfer")
    print("=" * 48)
    print("1) Send    — pack a directory into QR frames")
    print("2) Receive — scan QR frames and restore files")
    print("q) Quit")
    print("=" * 48)


def main() -> int:
    printBanner()
    if len(sys.argv) > 1:
        choice = sys.argv[1].strip().lower()
        if choice not in {"1", "2", "q"}:
            print(f"Unknown option: {sys.argv[1]}  (use 1, 2, or q)")
            choice = promptChoice("Select option: ", {"1", "2", "q"})
    else:
        choice = promptChoice("Select option: ", {"1", "2", "q"})

    if choice == "q":
        print("Bye.")
        return 0

    try:
        if choice == "1":
            runSendMode()
        else:
            runReceiveMode()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    except Exception as error:
        print(f"\nError: {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
