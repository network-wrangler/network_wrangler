"""Logging utilities for Network Wrangler."""

import logging
import sys
from datetime import datetime
from pathlib import Path

WranglerLogger = logging.getLogger("WranglerLogger")


class SafeFileHandler(logging.FileHandler):
    """FileHandler that safely handles flush errors on Windows.

    On Windows, Python's logging can encounter OSError: [Errno 22] Invalid argument
    when flushing log files due to:
    - Rapid consecutive writes overwhelming Windows file handle buffering
    - Large log files causing buffer flush issues
    - Windows-specific file handle limitations in console/terminal environments

    This handler catches and ignores flush errors while periodically logging
    that they're occurring to alert users without spamming the error output.

    See: https://bugs.python.org/issue13415 and related Windows logging issues
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._flush_error_count = 0
        self._last_reported_error_count = 0

    def flush(self):
        """Flush with error handling for Windows OSError issues.

        Catches OSError during flush operations (common on Windows) and tracks
        how many times this occurs. Reports every 1000th error to alert users
        without being too noisy.
        """
        try:
            super().flush()
        except OSError as e:
            # Windows-specific flush errors - track but don't crash
            self._flush_error_count += 1

            # Log every 1000th error to avoid spam while still alerting users
            if self._flush_error_count % 1000 == 0:
                # Use print to avoid recursive logging issues
                print(
                    f"WARNING: {self._flush_error_count} log flush errors encountered "
                    f"(Windows file handle issue). Logging continues but some debug "
                    f"messages may be delayed or lost. Error: {e}",
                    file=sys.stderr
                )


def setup_logging(
    info_log_filename: Path | None = None,
    debug_log_filename: Path | None = None,
    std_out_level: str = "info",
    file_mode: str = "a",
):
    """Sets up the WranglerLogger w.r.t. the debug file location and if logging to console.

    Called by the test_logging fixture in conftest.py and can be called by the user to setup
    logging for their session. If called multiple times, the logger will be reset.

    Args:
        info_log_filename: the location of the log file that will get created to add the INFO log.
            The INFO Log is terse, just gives the bare minimum of details.
            Defaults to file in cwd() `wrangler_[datetime].log`. To turn off logging to a file,
            use log_filename = None.
        debug_log_filename: the location of the log file that will get created to add the DEBUG log
            The DEBUG log is very noisy, for debugging. Defaults to file in cwd()
            `wrangler_[datetime].log`. To turn off logging to a file, use log_filename = None.
        std_out_level: the level of logging to the console. One of "info", "warning", "debug".
            Defaults to "info" but will be set to ERROR if nothing provided matches.
        file_mode: use 'a' to append, 'w' to write without appending
    """
    # add function variable so that we know if logging has been called
    setup_logging.called = True

    DEFAULT_LOG_PATH = Path(f"wrangler_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.debug.log")
    debug_log_filename = debug_log_filename if debug_log_filename else DEFAULT_LOG_PATH

    # Clear handles if any exist already
    WranglerLogger.handlers = []

    WranglerLogger.setLevel(logging.DEBUG)

    FORMAT = logging.Formatter(
        "%(asctime)-15s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S,"
    )
    default_info_f = f"network_wrangler_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.info.log"
    info_log_filename = info_log_filename or Path.cwd() / default_info_f

    # Use SafeFileHandler instead of FileHandler to handle Windows flush errors
    info_file_handler = SafeFileHandler(Path(info_log_filename), mode=file_mode)
    info_file_handler.setLevel(logging.INFO)
    info_file_handler.setFormatter(FORMAT)
    WranglerLogger.addHandler(info_file_handler)

    # create debug file only when debug_log_filename is provided
    if debug_log_filename:
        # Use SafeFileHandler instead of FileHandler to handle Windows flush errors
        debug_log_handler = SafeFileHandler(Path(debug_log_filename), mode=file_mode)
        debug_log_handler.setLevel(logging.DEBUG)
        debug_log_handler.setFormatter(FORMAT)
        WranglerLogger.addHandler(debug_log_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(FORMAT)
    WranglerLogger.addHandler(console_handler)
    if std_out_level == "debug":
        console_handler.setLevel(logging.DEBUG)
    elif std_out_level == "info":
        console_handler.setLevel(logging.INFO)
    elif std_out_level == "warning":
        console_handler.setLevel(logging.WARNING)
    else:
        console_handler.setLevel(logging.ERROR)
