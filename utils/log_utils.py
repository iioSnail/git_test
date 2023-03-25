import logging
import sys

log = None


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


init_log()
