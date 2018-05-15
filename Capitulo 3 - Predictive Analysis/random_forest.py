import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import cross_validation
from sklearn.metrics import classification_report

# Argument parser
def build_arg_parser():
      parser = argparse.ArgumentParser(description='Classify data using \
                  Ensemble Learning techniques')
      parser.add_argument('--classifier-type', dest='classifier_type',
                  required=True, choices=['rf', 'erf'], help="Type of classifier to use; can be either 'rf' or 'erf'")
      return parser

if __name__=='__main__':
      # Parse the input arguments
      args = build_arg_parser().parse_args()
      classifier_type = args.classifier_type

      # Load input data
      input_file = 'data_random_forests.txt'
      data = np.loadtxt(input_file, delimiter=',')
      X, y = data[:, :-1], data[:, -1]