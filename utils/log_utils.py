import logging
import sys
import warnings

log: logging.Logger = None


def init_log():
    global log
    if log is not None:
        return

    log = logging.getLogger()
    log.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(formatter)
    log.addHandler(sh)


def ignore_warning():
    warnings.filterwarnings("ignore", message=".*The dirpath has changed from.*")
    warnings.filterwarnings("ignore", message=".*does not have many workers which may be a bottleneck.*")
    warnings.filterwarnings("ignore", message=".*You're resuming from a checkpoint that ended before the epoch ended.*")
    warnings.filterwarnings("ignore", message=".*Detected KeyboardInterrupt*")


init_log()
ignore_warning()
