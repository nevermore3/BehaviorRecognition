import logging
import pandas as pd
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
        train_data = pd.read_csv(input_raw_data[0], header = None, sep = '\s+')
        train_label = pd.read_csv(input_raw_data[1], header = None)

        for row_index, row_data in train_data.iterrows():
            for col_index, value in enumerate(row_data):
                train_data.iloc[row_index, col_index] = str(col_index) + ':' + str(value)

        data = pd.concat([train_label, train_data], axis = 1)
        # save data
        data.to_csv(, header = 0, index = 0, sep = ' ')


_libsvm_preprocessor = LIBSVMPreprocessor()
_PREPROCESSORS = {
    'LIBSVM' : _libsvm_preprocessor,
}
def select(pre_name):
    if pre_name not in _PREPROCESSORS:
        logging.error("PreprocessorName[%s] was not valid" % pre_name)
        return None
    return _PREPROCESSORS[pre_name]