import logging
class BasePreProcessor(object):
    def __init__(self):
        pass
    def handleData(self, input_raw_data, is_train=True):
        """
        output file will be Train.txt or Test.txt
        """
        pass

class LIBSVMPreprocessor(BasePreProcessor):
    def __init__(self):
        pass
    def handleData(self, input_raw_data, is_train=True):
        pass

_libsvm_preprocessor = LIBSVMPreprocessor()
_PREPROCESSORS = {
    'LIBSVM' : _libsvm_preprocessor,
}
def select(pre_name):
    if pre_name not in _PREPROCESSORS:
        logging.error("PreprocessorName[%s] was not valid" % pre_name)
        return None
    return _PREPROCESSORS[pre_name]