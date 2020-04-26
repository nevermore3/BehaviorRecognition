# -*- coding=utf8 -*-
import logging
import libsvm.svm as svm
import libsvm.svmutil as svmutil

class BaseModel(object):
    def __init__(self):
        pass
    def _get_class_name(self):
        return self.__class__.__name__
    def train(self, input_data_path):
        """
        suppose there're one file naming Train.txt under `input_data_path`
        """
        pass
    def save(self, save_path):
        """
        save model file to save_path + "/" + self.model_name + ".model.timestamp"
        """
        pass
    def evaluate(self, input_data_path):
        """
        suppose there're two files naming Test_X.txt and Test_Y.txt under `input_data_path`
        """
        pass

class LIBSVMModel(BaseModel):
    def __init__(self):
        super(LIBSVMModel, self).__init__()
        self._model = None
        self._params = None
        self._data = None
        self._init = False

    def train(self, input_data_path, params="-t 0 -c 4 -b 1'"):
        self._data = svmutil.svm_read_problem(input_data_path + "/Train.txt")
        self._params = svmutil.svm_parameter(params)
        self._model, acc, mse = svmutil.svm_train(self._data, self._params)
        self._init = True
        logging.info("[%s]: train with Acc[%.4f] and Mse[%.4f]" % (self._get_class_name(), acc, mse))
    
    def save(self, save_path):
        svmutil.save_svm_model(save_path + "/" + self._get_class_name() + ".model", self._model)
        logging.info("[%s]: Save Model Done")
    
    def evaluate(self, input_data_path):
        Y, X = svmutil.svm_read_problem(input_data_path + "/Test.txt")
        p_labels, p_acc, p_vals = svmutil.svm_predict(Y, X, self._model)
        logging.info("[%s]: evaluate with Acc[%.4f]" % (self._get_class_name(), p_acc))


_libsvm_model = LIBSVMModel()
_MODELS = {
    'LIBSVM' : _libsvm_model,
}
def select(model_name):
    if model_name not in _MODELS:
        logging.error("Model[%s] was not valid" % model_name)
        return None
    return _MODELS[model_name]

