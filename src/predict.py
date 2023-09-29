import os
import numpy as np
import scipy as sp
import sys
import getopt
from collections import defaultdict
from WNNGIP import WNNGIP


def load_data_from_file(dataset, folder):
    with open(os.path.join("..", folder, dataset+"_admat_dgc.txt"), "r") as inf:
        text = inf.read().rstrip().split('\n')
        int_array = [line.split('\t')[1:] for line in text[1:]]

    with open(os.path.join("..", folder, dataset+"_simmat_dc.txt"), "r") as inf:  # the drug similarity file
        text = inf.read().rstrip().split('\n')
        drug_sim = [line.rstrip().split('\t')[1:] for line in text[1:]]

    with open(os.path.join("..", folder, dataset+"_simmat_dg.txt"), "r") as inf:  # the target similarity file
        text = inf.read().rstrip().split('\n')
        target_sim = [line.rstrip().split('\t')[1:] for line in text[1:]]

    intMat = np.array(int_array, dtype=np.float64).T    # drug-target interaction matrix
    drugMat = np.array(drug_sim, dtype=np.float64)      # drug similarity matrix
    targetMat = np.array(target_sim, dtype=np.float64)  # target similarity matrix
    return intMat, drugMat, targetMat


def cross_validation(intMat, seeds, cv=0, num=10):
    cv_data = defaultdict(list)
    for seed in seeds:
        num_drugs, num_targets = intMat.shape
        prng = np.random.RandomState(seed)
        if cv == 0:
            index = prng.permutation(num_drugs)
        if cv == 1:
            index = prng.permutation(intMat.size)
        step = int(index.size/num)
        for i in range(num):
            if i < num-1:
                ii = index[i*step:(i+1)*step]
            else:
                ii = index[i*step:]
            if cv == 0:
                test_data = np.array([[k, j] for k in ii for j in range(num_targets)], dtype=np.int32)
            elif cv == 1:
                test_data = np.array([[k/num_targets, k % num_targets] for k in ii], dtype=np.int32)
            x, y = test_data[:, 0], test_data[:, 1]
            test_label = intMat[x, y]
            W = np.ones(intMat.shape)
            W[x, y] = 0
            cv_data[seed].append((W, test_data, test_label))
    return cv_data


def train(model, cv_data, intMat, drugMat, targetMat):
    aupr, auc = [], []
    for seed in cv_data.keys():
        for W, test_data, test_label in cv_data[seed]:
            model.fix_model(W, intMat, drugMat, targetMat, seed)
            aupr_val, auc_val = model.evaluation(test_data, test_label)
            aupr.append(aupr_val)
            auc.append(auc_val)
    return np.array(aupr, dtype=np.float64), np.array(auc, dtype=np.float64)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h


def write_metric_vector_to_file(auc_vec, file_name):
    np.savetxt(file_name, auc_vec, fmt='%.6f')





def predict(argv):
    try:
        opts, args = getopt.getopt(argv, "m:d:f:c:s:o:n:p", ["method=", "dataset=", "data-dir=", "cvs=", "specify-arg=", "method-options=", "predict-num=", "output-dir=", ])
    except getopt.GetoptError:
        sys.exit()

    data_dir = os.path.join('data')
    output_dir = os.path.join('output')
    cvs, sp_arg, model_settings, predict_num = 1, 1, [], 0

    seeds = [7771, 8367, 22, 1812, 4659]
    for opt, arg in opts:
        if opt == "--method":
            method = arg
        if opt == "--dataset":
            dataset = arg
        if opt == "--data-dir":
            data_dir = arg
        if opt == "--output-dir":
            output_dir = arg
        if opt == "--cvs":
            cvs = int(arg)
        if opt == "--specify-arg":
            sp_arg = int(arg)
        if opt == "--method-options":
            model_settings = [s.split('=') for s in str(arg).split()]
        if opt == "--predict-num":
            predict_num = int(arg)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    if method == 'nrlmf':
        args = {'c': 5, 'K1': 5, 'K2': 5, 'r': 50, 'lambda_d': 0.125, 'lambda_t': 0.125, 'alpha': 0.25, 'beta': 0.125, 'theta': 0.5, 'max_iter': 100}
    if method == 'netlaprls':
        args = {'gamma_d': 10, 'gamma_t': 10, 'beta_d': 1e-5, 'beta_t': 1e-5}
    if method == 'blmnii':
        args = {'alpha': 0.7, 'gamma': 1.0, 'sigma': 1.0, 'avg': False}
    if method == 'wnngip':
        args = {'T': 0.7, 'sigma': 1.0, 'alpha': 0.8}
    if method == 'kbmf':
        args = {'R': 50}
    if method == 'cmf':
        args = {'K': 50, 'lambda_l': 0.5, 'lambda_d': 0.125, 'lambda_t': 0.125, 'max_iter': 30}

    for key, val in model_settings:
        args[key] = val

    intMat, drugMat, targetMat = load_data_from_file(dataset, os.path.join(data_dir, 'datasets'))

    if predict_num == 0:
        if cvs == 1:  # CV setting CVS1
            X, D, T, cv = intMat, drugMat, targetMat, 1
        if cvs == 2:  # CV setting CVS2
            X, D, T, cv = intMat, drugMat, targetMat, 0
        if cvs == 3:  # CV setting CVS3
            X, D, T, cv = intMat.T, targetMat, drugMat, 0
        cv_data = cross_validation(X, seeds, cv)  # separate data for 5 * 10-fold cross-validation

    if sp_arg == 1 or predict_num > 0:
        if method == 'wnngip':
            model = WNNGIP(T=args['T'], sigma=args['sigma'], alpha=args['alpha'])
        if predict_num == 0:
            aupr_vec, auc_vec = train(model, cv_data, X, D, T)
            aupr_avg, aupr_conf = mean_confidence_interval(aupr_vec)
            auc_avg, auc_conf = mean_confidence_interval(auc_vec)
            write_metric_vector_to_file(auc_vec, os.path.join(output_dir, method+"_auc_cvs"+str(cvs)+"_"+dataset+".txt"))
            write_metric_vector_to_file(aupr_vec, os.path.join(output_dir, method+"_aupr_cvs"+str(cvs)+"_"+dataset+".txt"))
            return auc_avg, aupr_avg
