# -*- coding: utf-8 -*-

'''

Подбор гиперпараметров  с помощью hyperopt.

Справка по hyperopt:
https://github.com/hyperopt/hyperopt/wiki/FMin
http://fastml.com/optimizing-hyperparams-with-hyperopt/
https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf

'''
import sys

import sklearn
import numpy
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import numpy as np
import os, sys
import shutil
import time
import traceback

import main

import utils.main_log

print = utils.main_log.printL
utils.main_log.init('blur_hyper.log', 'hyper')

# кол-во случайных наборов гиперпараметров
N_HYPEROPT_PROBES = 10

# алгоритм сэмплирования гиперпараметров
HYPEROPT_ALGO = tpe.suggest  # tpe.suggest OR hyperopt.rand.suggest

# ---------------------------------------------------------------------

obj_call_count = 0
cur_best_loss = np.inf

'''
filter_levels=(6,8,10,32), AfterReducerChanels=5, adaptivepool_size=6, dropout_rate=0.3, ExitDropout = 0.55
'''
space = {
    # 'booster': hp.choice( 'booster',  ['dart', 'gbtree'] ),
    # 'gamma': hp.uniform('gamma', 0.0, 5.5),
    # 'gamma': hp.loguniform('gamma', -5.0, 0.0),
    # 'eta': hp.loguniform('eta', -4.6, -2.3),
    # 'colsample_bytree': hp.uniform('colsample_bytree', 0.80, 1.0),
    'L1': hp.randint('L1', 6, 10),
    'L2': hp.randint('L2', 8, 15),
    'L3': hp.randint('L3', 10, 28),
    'L4': hp.randint('L4', 20, 36),
    'extra_cat_lines': hp.randint('extra_cat_lines', 2, 3),
    'AfterReducerChanels': hp.randint('AfterReducerChanels', 3, 5),
    'adaptivepool_size': hp.randint('adaptivepool_size', 6, 8),
    'dropout_rate': hp.uniform('dropout_rate', 0.0, 0.0),
    'ExitDropout': hp.uniform('ExitDropout', 0.0, 0.0),
    'extra_lay': hp.choice( 'extra_lay',  [True, False] ),

}

def objective(space):
    global obj_call_count, cur_best_loss

    try:
        params_str = str(space)
        print('Params: {}'.format(params_str))

        obj_call_count += 1

        print('\nobjective call #{} cur_best_loss={:7.5f}'.format(obj_call_count, cur_best_loss))

        try:
            shutil.rmtree('./classifier')
        except:
            time.sleep(3)
            shutil.rmtree('./classifier', ignore_errors=True)

        # test_loss = main.main()
        test_loss = main.main( filter_levels=(space['L1'], space['L2'], space['L3'], space['L4']) , AfterReducerChanels=space['AfterReducerChanels'], adaptivepool_size=int(space['adaptivepool_size']), dropout_rate=space['dropout_rate'], ExitDropout = space['ExitDropout'], extra_cat_lines=space['extra_cat_lines'] , extra_lay = space['extra_lay'])


        # print(type(space))
        # sorted_params = sorted(space.iteritems(), key=lambda z: z[0])
        # params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])

        # loss = model.best_score
        # print('test_acc={}'.format(test_loss))

        print('loss={:<7.5f} Params:{} \n'.format(test_loss, params_str))

        if test_loss < cur_best_loss:
            cur_best_loss = test_loss
            print('!!!!!!!! NEW BEST LOSS={}'.format(cur_best_loss))

        return {'loss': test_loss, 'status': STATUS_OK}
    except: # (KeyboardInterrupt , ValueError):
        print(f'error with {params_str}')
        print(traceback.format_exc() )
        return {'loss': 1, 'status': hyperopt.STATUS_FAIL}


    # --------------------------------------------------------------------------------


def do_process():
    print('Start hyper opt process.')
    trials = Trials()
    best = hyperopt.fmin(fn=objective,
                         space=space,
                         algo=HYPEROPT_ALGO,
                         max_evals=N_HYPEROPT_PROBES,
                         trials=trials,
                         verbose=1)

    print('-' * 50)
    print('The best params:')
    print(best)
    print('\n\n')

if __name__ == '__main__':
    do_process()
