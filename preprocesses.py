import logging
import pandas as pd
class BasePreProcessor(object):
    def __init__(self):
        pass
    def handleData(self, input_raw_data, dest_filepath, is_train=True):
        """
        output file will be `dest_filepath`/Train.txt or `dest_filepath`/Test.txt
        """
        pass

class LIBSVMPreprocessor(BasePreProcessor):
    def __init__(self):
        pass
    def handleData(self, input_raw_data, dest_filepath, is_train=True):
        x_input = input_raw_data+'\\X_train.txt'
        y_input = input_raw_data+'\\Y_train.txt'
        train_data = pd.read_csv(x_input, header = None, sep = '\s+')
        train_label = pd.read_csv(y_input, header = None)

        for row_index, row_data in train_data.iterrows():
            for col_index, value in enumerate(row_data):
                train_data.iloc[row_index, col_index] = str(col_index) + ':' + str(value)

        data = pd.concat([train_label[:100], train_data[:100]], axis = 1)
        print(data.shape)
        # save data
        save_path = dest_filepath + "\\" + ("Train.txt" if is_train else "Test.txt")
        data.to_csv(save_path, header=None, index=None, sep=' ')

    def handleData_v2(self, input_raw_data, dest_filepath, is_train=True):
        x_input = input_raw_data+'\\X_train.txt'
        y_input = input_raw_data+'\\Y_train.txt'
        train_data = pd.read_csv(x_input, header = None, sep = '\s+')
        train_label = pd.read_csv(y_input, header = None)

        train_data = train_data[:100]
        train_label = train_label[:100]

        columns = train_data.columns.tolist()
        def handle_row(x):
            res = ""
            for i in range(len(columns)):
                if i == 0:
                    res = str(i) + ":" + str(x[i])
                else:
                    res += " " + str(i) + ":" + str(x[i])
            return res

        logging.info("Start to preprocess data with Length[%d]" % len(train_data))
        train_data['svm_data'] = train_data.apply(handle_row, aixs=1)

        data = pd.DataFrame({'label' : train_label, 'svm_data': train_data.svm_data})
        print(data.shape)
        # save data
        save_path = dest_filepath + "\\" + ("Train.txt" if is_train else "Test.txt")
        data.to_csv(save_path, header=None, index=None, sep=' ')


_libsvm_preprocessor = LIBSVMPreprocessor()
_PREPROCESSORS = {
    'LIBSVM' : _libsvm_preprocessor,
}
def select(pre_name):
    if pre_name not in _PREPROCESSORS:
        logging.error("PreprocessorName[%s] was not valid" % pre_name)
        return None
    return _PREPROCESSORS[pre_name]