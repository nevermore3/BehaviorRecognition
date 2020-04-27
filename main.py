# -*- coding-utf8 -*-

import os
import sys
sys.path.append(os.path.abspath(__file__))

import models
import argparse
import preprocesses
import commons
import logging
logging.basicConfig(level=logging.DEBUG)
def main(args):
    # preprocess original raw data
    with commons.PhaseLogger("Preprocess TrainSet"):
        preprocessor = preprocesses.select(args.preprocess)
        # 这里v2 先用着测试，到时候名字改回来
        preprocessor.handleData_v2(args.train_input, args.intermediate_data, is_train=True)

    with commons.PhaseLogger("Training Model"):
        # train model
        model = models.select(args.model)
        model.train(args.intermediate_data)
    with commons.PhaseLogger("Saving Model"):
        # save model
        model.save(args.intermediate_data)
    with commons.PhaseLogger("Evaluating Model"):
        # test model
        preprocessor.handleData_v2(args.test_input, args.intermediate_data, is_train=False)
        model.evaluate(args.intermediate_data)



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('CommandLine Parser')
    parser.add_argument('-m', '--model', dest='model', type=str, default="LIBSVM", help="model that was used for train and test, currently only support LIBSVM")
    parser.add_argument('-i', '--train_input', dest='train_input', type=str, default="..\\data", help="")
    parser.add_argument('-o','--test_input', dest='test_input', type=str, default="..\\data", help="")
    parser.add_argument('-t', '--intermediate_data', dest='intermediate_data', type=str, default="..\\data", help="intermediate_data filepath")
    parser.add_argument('-p', '--preprocess', dest='preprocess', type=str, default="LIBSVM", help="preprocessor that were defined inside preprocess.py")
    args = parser.parse_args()
    main(args)
