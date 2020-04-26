import gc
import time
import logging

class PhaseLogger(object):
    def __init__(self, msg, do_gc=True):
        self.msg = msg
        self.do_gc = do_gc

    def __enter__(self):
        self.tbegin = time.time()
        msg = '%s begins.' % (self.msg)
        logging.info(msg)
        if self.do_gc:
            gc.collect()

    def __exit__(self, exc_type, exc_value, traceback):
        self.tend = time.time()
        msg = '%s ends, time cost=%.2f sec' % (self.msg, self.tend-self.tbegin)
        logging.info(msg)
        if self.do_gc:
            gc.collect()
