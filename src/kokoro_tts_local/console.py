"""Console encoding support for the bilingual (Chinese/English) entry points."""

import sys


def enable_utf8_console() -> bool:
    """Make ``sys.stdout``/``sys.stderr`` able to carry non-ASCII output.

    The Mandarin CLI and setup script print Chinese on every line. On a Windows
    console still using a legacy code page (cp1252 by default outside UTF-8
    mode) the very first such ``print`` raises ``UnicodeEncodeError`` and takes
    the process down. Reconfigure those streams to UTF-8 with a replacement
    error handler so unprintable glyphs degrade instead of aborting.

    This is a no-op when the current encoding can already represent the text,
    which is the case for essentially every non-Windows terminal. Returns True
    if any stream was reconfigured.
    """
    probe = "下载"  # Two Han characters, representative of the output.
    changed = False
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        encoding = getattr(stream, "encoding", None)
        if reconfigure is None or encoding is None:
            continue
        try:
            probe.encode(encoding)
        except (LookupError, UnicodeEncodeError):
            try:
                reconfigure(encoding="utf-8", errors="replace")
                changed = True
            except (OSError, ValueError):
                # A stream that refuses reconfiguration (a pipe wrapper, a
                # captured buffer) is left as-is; callers still get output.
                pass
    return changed
