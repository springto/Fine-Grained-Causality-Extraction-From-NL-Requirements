import optparse
import cPickle as pickle

import optimizer as optimizer
import rntn as nnet
#import rnn as nnet
import tree as tr
import time
import pandas as pd
from datetime import datetime
import os
import sys
import utils
from shutil import copyfile
import logging

log = utils.get_logger()


def run(args=None):
    usage = "usage : %prog [options]"
    parser = optparse.OptionParser(usage = usage)

    parser.add_option("--test", action = "store_true", dest = "test", default = False)

    # Paramsfile includes hyperparameters for training
    parser.add_option('--params_file', dest = "params_file", default = './params/exp_params.json',
                      help = "Path to the file  containing the training settings")
    parser.add_option('--data_dir', dest = "data_dir", default = './trees',
                      help = "Directory containing the trees")

    # Directory containing the model to test
    parser.add_option("--model_directory", dest = "test_dir", type = "string")
    parser.add_option("--data", dest = "data", type = "string", default = "train")

    (opts, args) = parser.parse_args(args)

    results_dir = "./results"
    if opts.test:
        pass
    else:
        results_dir_current_job = os.path.join(results_dir, utils.now_as_str_f())
        while os.path.isdir(results_dir_current_job):  # generate a new timestamp if the current one already exists
            results_dir_current_job = os.path.join(results_dir, utils.now_as_str_f())
        os.makedirs(results_dir_current_job)

    # Load training settings (e.g. hyperparameters)
    params = utils.Params(opts.params_file)

    if opts.test:
        pass
    else:
        # Copy the settings file into the results directory
        copyfile(opts.params_file, os.path.join(results_dir_current_job, os.path.basename(opts.params_file)))

    # Get the logger
    if opts.test:
        log_path = os.path.join(opts.test_dir, 'testing.log')
    else:
        log_path = os.path.join(results_dir_current_job, 'training.log')
    log_level = params.log_level if hasattr(params, 'log_level') else logging.DEBUG
    log = utils.get_logger(log_path, log_level)

    if opts.test:
        log.info("Testing directory: " + opts.test_dir)
        log.info("Dataset used for testing: " + opts.data)
    else:
        log.info("Results directory: " + results_dir_current_job)
        log.info("Minibatch: " + str(params.optimizer_settings['minibatch']))
        log.info("Optimizer: " + params.optimizer)
        log.info("Epsilon: " + str(params.optimizer_settings['epsilon']))
        log.info("Alpha: " + str(params.optimizer_settings['alpha']))
        log.info("Number of samples used: " + str(params.sample_size))

    # Testing
    if opts.test:
        test(opts.test_dir, opts.data)
        return

    log.info("Loading data...")
    # load training data
    trees = tr.loadTrees(sample_size = params.sample_size)
    params.numWords = len(tr.loadWordMap())
    overall_performance = pd.DataFrame()

    rnn = nnet.RNN(params.wvecDim, params.outputDim, params.numWords, params.optimizer_settings['minibatch'])
    rnn.initParams()

    sgd = optimizer.SGD(rnn, alpha = params.optimizer_settings['alpha'],
                        minibatch = params.optimizer_settings['minibatch'],
                        optimizer = params.optimizer, epsilon = params.optimizer_settings['epsilon'])

    best_val_cost = float('inf')
    best_epoch = 0

    for e in range(params.num_epochs):
        start = time.time()
        log.info("Running epoch %d" % e)
        df, updated_model, train_cost, train_acc = sgd.run(trees)
        end = time.time()
        log.info("Time per epoch : %f" % (end - start))
        log.info("Training accuracy : %f" % train_acc)
        # VALIDATION
        val_df, val_cost, val_acc = validate(updated_model, results_dir_current_job)

        if val_cost < best_val_cost:
            # best validation cost we have seen so far
            log.info("Validation score improved, saving model")
            best_val_cost = val_cost
            best_epoch = e
            best_epoch_row = {"epoch": e, "train_cost": train_cost, "val_cost": val_cost, "train_acc": train_acc,
                              "val_acc": val_acc}
            with open(results_dir_current_job + "/checkpoint.bin", 'w') as fid:
                pickle.dump(params, fid)
                pickle.dump(sgd.costt, fid)
                rnn.toFile(fid)

        val_df.to_csv(results_dir_current_job + "/validation_preds_epoch_ " + str(e) + ".csv", header = True, index = False)
        df.to_csv(results_dir_current_job + "/training_preds_epoch_" + str(e) + ".csv", header = True, index = False)

        row = {"epoch": e, "train_cost": train_cost, "val_cost": val_cost, "train_acc": train_acc, "val_acc": val_acc}
        overall_performance = overall_performance.append(row, ignore_index = True)

        # break if no val loss improvement in the last epochs
        if (e - best_epoch) >= params.num_epochs_early_stop:
            log.tinfo("No improvement in the last {num_epochs_early_stop} epochs, stop training.".format(num_epochs_early_stop=params.num_epochs_early_stop))
            break

    overall_performance = overall_performance.append(best_epoch_row, ignore_index = True)
    overall_performance.to_csv(results_dir_current_job + "/train_val_costs.csv", header = True, index = False)
    log.info("Experiment end")


def validate(rnn, results_dir):
    # log.info(rnn.W)
    trees = tr.loadTrees("dev")
    log.info("Validation...")
    cost, correct, total, df = rnn.costAndGrad(trees, test = True)
    log.info("Validation: Cost %f, Correct %d/%d, Acc %f" % (cost, correct, total, correct / float(total)))
    return df, cost, correct / float(total)


def test(model_dir, dataSet):
    trees = tr.loadTrees(dataSet)
    total_df = pd.DataFrame()
    assert model_dir is not None, "Must give model to test"
    with open(model_dir + "/checkpoint.bin", 'r') as fid:
        opts = pickle.load(fid)
        _ = pickle.load(fid)
        rnn = nnet.RNN(opts.wvecDim, opts.outputDim, opts.numWords, opts.optimizer_settings['minibatch'])
        rnn.initParams()
        rnn.fromFile(fid)
    log.info("Testing...")
    cost, correct, total, df = rnn.costAndGrad(trees, test = True)
    total_df = total_df.append(df, ignore_index = True)
    total_df.to_csv(model_dir + "/test_preds.csv", header = True, index = False)
    test_performance = pd.DataFrame()
    row = {"Cost": cost, "Correct": correct, "Total": total, "Accuracy": correct / float(total)}
    test_performance = test_performance.append(row, ignore_index = True)
    test_performance.to_csv(model_dir + "/test_performance.csv", header = True, index = False)
    log.info("Cost %f, Correct %d/%d, Acc %f" % (cost, correct, total, correct / float(total)))


if __name__ == '__main__':
    run()
