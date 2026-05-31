import os
import sys
import faulthandler


def _crash_log_path():
    """Path to a crash log placed next to the EXE (or the source dir)."""
    try:
        import config
        base = (config.frozen_base_dir() if getattr(sys, "frozen", False)
                else os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        base = os.getcwd()
    return os.path.join(base, "audio2text-crash.log")


def _install_crash_logging():
    """Dump native faults and unhandled exceptions to a file for diagnosis.

    A windowed (no-console) build that hits a native access violation — e.g. in
    ctranslate2/cuDNN or the audio backend — just vanishes with no trace.
    faulthandler writes the C stack; the hooks below capture Python tracebacks.
    """
    try:
        log = open(_crash_log_path(), "a", buffering=1, encoding="utf-8", errors="replace")
    except Exception:
        return
    try:
        faulthandler.enable(log)
    except Exception:
        pass

    import traceback

    def _hook(exc_type, exc, tb):
        try:
            traceback.print_exception(exc_type, exc, tb, file=log)
            log.flush()
        except Exception:
            pass

    sys.excepthook = _hook
    try:
        import threading

        def _thread_hook(args):
            _hook(args.exc_type, args.exc_value, args.exc_traceback)
        threading.excepthook = _thread_hook
    except Exception:
        pass


if __name__ == "__main__":
    _install_crash_logging()
    from ui import Audio2TextApp
    app = Audio2TextApp()
    app.mainloop()
