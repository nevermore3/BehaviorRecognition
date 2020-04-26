# -*- coding=utf8 -*-
import logging
import libsvm.svm as svm
import libsvm.svmutil as svmutil
import commons

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

    """
    options:
        -s svm_type : set type of SVM (default 0)
            0 -- C-SVC
            1 -- nu-SVC
            2 -- one-class SVM
            3 -- epsilon-SVR
            4 -- nu-SVR
        -t kernel_type : set type of kernel function (default 2)
            0 -- linear: u'*v
            1 -- polynomial: (gamma*u'*v + coef0)^degree
            2 -- radial basis function: exp(-gamma*|u-v|^2)
            3 -- sigmoid: tanh(gamma*u'*v + coef0)
        -d degree : set degree in kernel function (default 3)
        -g gamma : set gamma in kernel function (default 1/num_features)
        -r coef0 : set coef0 in kernel function (default 0)
        -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
        -n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
        -p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)
        -m cachesize : set cache memory size in MB (default 100)
        -e epsilon : set tolerance of termination criterion (default 0.001)
        -h shrinking: whether to use the shrinking heuristics, 0 or 1 (default 1)
        -b probability_estimates: whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
        -wi weight: set the parameter C of class i to weight*C, for C-SVC (default 1)
        The k in the -g option means the number of attributes in the input data.
    """
    def train(self, input_data_path, params="-t 0 -c 4 -b 1'"):
        with commons.PhaseLogger("LIBSVM.train.read_problem"):
            self._data = svmutil.svm_read_problem(input_data_path + "/Train.txt")
        self._params = svmutil.svm_parameter(params)
        with commons.PhaseLogger("LIBSVM.train.svm_train"):
            self._model, acc, mse = svmutil.svm_train(self._data, self._params)
        self._init = True
        logging.info("[%s]: train with Acc[%.4f] and Mse[%.4f]" % (self._get_class_name(), acc, mse))
    
    def save(self, save_path):
        with commons.PhaseLogger("LIBSVM.save"):
            svmutil.save_svm_model(save_path + "/" + self._get_class_name() + ".model", self._model)
        logging.info("[%s]: Save Model Done")
    
    def evaluate(self, input_data_path):
        with commons.PhaseLogger("LIBSVM.evaluate.read_problem"):
            Y, X = svmutil.svm_read_problem(input_data_path + "/Test.txt")
        with commons.PhaseLogger("LIBSVM.evaluate.predict"):
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

