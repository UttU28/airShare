#!/usr/bin/env python3
"""codeShare — QR-based directory transfer over camera."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_QR_DIR = PROJECT_ROOT / "qrFrames"
DEFAULT_SHARE_DIR = Path("/Users/antonio/Desktop/homeLabOps")


def promptChoice(promptText: str, allowed: set[str]) -> str:
    while True:
        value = input(promptText).strip().lower()
        if value in allowed:
            return value
        print(f"Please choose one of: {', '.join(sorted(allowed))}")


def promptPath(promptText: str) -> str:
    while True:
        value = input(promptText).strip().strip('"').strip("'")
        if value:
            return value
        print("Path cannot be empty.")


def promptFloat(promptText: str, defaultValue: float) -> float:
    raw = input(f"{promptText} [{defaultValue}]: ").strip()
    if not raw:
        return defaultValue
    try:
        return float(raw)
    except ValueError:
        print("Invalid number, using default.")
        return defaultValue


def promptInt(promptText: str, defaultValue: int) -> int:
    raw = input(f"{promptText} [{defaultValue}]: ").strip()
    if not raw:
        return defaultValue
    try:
        return int(raw)
    except ValueError:
        print("Invalid number, using default.")
        return defaultValue


def runSendMode() -> None:
    from sender import runSender

    sourcePath = str(DEFAULT_SHARE_DIR)
    saveDir = str(DEFAULT_QR_DIR)
    print(f"Sharing directory: {sourcePath}")
    print(f"QR PNGs will be saved to: {saveDir}")

    runSender(sourcePath, frameDelay=0.8, saveDir=saveDir)


def runReceiveMode() -> None:
    from receiver import runReceiver

    outputDir = promptPath("Output folder for restored directory: ")
    cameraIndex = promptInt("Camera index", 0)
    runReceiver(outputDir, cameraIndex=cameraIndex)


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
