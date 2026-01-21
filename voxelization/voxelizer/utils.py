# mof_voxelizer/utils.py
import logging
import sys

def setup_logger(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logger = logging.getLogger()
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
