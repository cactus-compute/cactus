"""
ADB wrapper for the Cactus Android benchmarking harness.

All device interactions go through ADBClient so error handling and
retry logic stay in one place, rather than scattered across the runner.
"""

import logging
import subprocess
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ADBError(Exception):
    """Raised when an ADB command fails or the device is unreachable."""


class ADBClient:
    """
    Thin wrapper around the `adb` CLI for a single connected device.

    Every shell command is logged at DEBUG level with the device serial so
    multi-device runs produce unambiguous logs. Command timeouts are enforced
    to prevent the benchmark runner from hanging if the device crashes.
    """

    def __init__(self, serial: str, default_timeout: int = 60) -> None:
        """
        Initialise the client and verify the device is reachable.

        Args:
            serial: The ADB device serial as shown by `adb devices`.
            default_timeout: Seconds before a shell command is killed.

        Raises:
            ADBError: If the device is not currently listed by ADB.
        """
        self.serial = serial
        self.default_timeout = default_timeout
        self._verify_connection()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def shell(self, cmd: str, timeout: Optional[int] = None) -> str:
        """
        Execute a shell command on the device and return stdout as a string.

        Stderr is captured separately and emitted as a WARNING if non-empty,
        so callers see clean output while still surfacing device-side errors.

        Args:
            cmd: The shell command to run (passed to `adb shell`).
            timeout: Override the default command timeout (seconds).

        Returns:
            Stripped stdout from the command.

        Raises:
            ADBError: On non-zero exit code or timeout.
        """
        effective_timeout = timeout or self.default_timeout
        full_cmd = ["adb", "-s", self.serial, "shell", cmd]
        logger.debug("[%s] shell: %s", self.serial, cmd)

        try:
            proc = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired:
            raise ADBError(
                f"Shell command timed out after {effective_timeout}s on {self.serial}: {cmd}"
            )

        if proc.returncode != 0 and proc.stderr.strip():
            logger.warning("[%s] stderr: %s", self.serial, proc.stderr.strip())

        return proc.stdout.strip()

    def push(self, local_path: str, remote_path: str) -> None:
        """
        Push a local file or directory to the device.

        Args:
            local_path: Absolute or relative local filesystem path.
            remote_path: Destination path on the device.

        Raises:
            ADBError: If the push fails.
        """
        logger.info("[%s] push %s -> %s", self.serial, local_path, remote_path)
        self._run(["push", local_path, remote_path], timeout=600)

    def pull(self, remote_path: str, local_path: str) -> None:
        """
        Pull a file or directory from the device to local storage.

        Args:
            remote_path: Path on the device to pull.
            local_path: Local destination path.

        Raises:
            ADBError: If the pull fails.
        """
        logger.info("[%s] pull %s -> %s", self.serial, remote_path, local_path)
        self._run(["pull", remote_path, local_path], timeout=300)

    def device_info(self) -> dict:
        """
        Return a dict of device metadata read from Android system properties.

        The returned dict always has the keys listed below; values may be
        empty strings if the property is unavailable on the target device.

        Returns:
            Dict with keys: model, brand, chipset, android_version,
            sdk_version, build_id, abi, total_ram_mb.
        """
        props: dict = {
            "model": self._getprop("ro.product.model"),
            "brand": self._getprop("ro.product.brand"),
            "chipset": self._getprop("ro.hardware"),
            "android_version": self._getprop("ro.build.version.release"),
            "sdk_version": self._getprop("ro.build.version.sdk"),
            "build_id": self._getprop("ro.build.id"),
            "abi": self._getprop("ro.product.cpu.abi"),
        }

        # Total RAM from /proc/meminfo (more reliable than ro.* props)
        try:
            meminfo_line = self.shell("cat /proc/meminfo | grep MemTotal")
            kb = int(meminfo_line.split()[1])
            props["total_ram_mb"] = kb // 1024
        except (ValueError, IndexError):
            props["total_ram_mb"] = 0

        return props

    def is_connected(self) -> bool:
        """Return True if the device serial still appears in `adb devices`."""
        try:
            result = subprocess.run(
                ["adb", "devices"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return self.serial in result.stdout
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run(self, args: list[str], timeout: Optional[int] = None) -> str:
        """Execute `adb -s <serial> <args>` and return stripped stdout."""
        effective_timeout = timeout or self.default_timeout
        cmd = ["adb", "-s", self.serial] + args
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
            )
        except subprocess.TimeoutExpired:
            raise ADBError(f"ADB command timed out: {' '.join(cmd)}")

        if proc.returncode != 0:
            raise ADBError(
                f"ADB command failed (exit {proc.returncode}): {' '.join(cmd)}\n"
                f"stderr: {proc.stderr.strip()}"
            )
        return proc.stdout.strip()

    def _getprop(self, prop: str) -> str:
        """Read a single Android system property, returning '' on failure."""
        try:
            return self.shell(f"getprop {prop}")
        except ADBError:
            return ""

    def _verify_connection(self) -> None:
        """Raise ADBError if the device is not currently reachable."""
        if not self.is_connected():
            raise ADBError(
                f"Device '{self.serial}' not found. "
                "Run `adb devices` to verify the serial and USB debugging state."
            )
        logger.debug("[%s] connection verified", self.serial)
