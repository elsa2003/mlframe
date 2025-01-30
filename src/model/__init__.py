
import polars as pl
from utils.types import Dict, List, ModelType, ParamType, PathType
from hyperopt import tpe, hp, fmin, STATUS_OK, space_eval
import joblib
import gc
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
import os
from .gru import (
    GRUClassifier, 
    prepare_gru
    )

from .lightgbm import (
    LGBMClassifier, 
    prepare_lgbm
    )

#%% training:
# split the training dataset into train + valid * 6
# for each train+valid set
# run the loss, compute the objective function for hyperopt
# select the hyper param
# store the model for last two batch for each hyper params, restore

def pl_compute_sample_weights(tname):
    # prepare sw
    tret = pl.col(tname).abs()
    tret = (tret/ tret.quantile(0.99))* 100
    tret = pl.when(tret > 100).then(tret.sqrt() * 10).otherwise(tret)
    return tret.cast(int) + 1

def compute_sample_weights(y):
    # prepare sw
    tret = np.abs(y)
    tret = (tret/ np.quantile(tret, 0.99))* 100
    tret[tret > 100] = np.sqrt(tret[tret > 100]) * 10
    return tret.astype(int) + 1

def compute_batchsize(train):
    return (2 ** int(np.log2(max(len(train)//40000, 1)))) * 512

def train_classifier(
        training_set: pl.DataFrame,
        target: str,
        features: List[str],
        modelClass: ModelType,
        params: ParamType,
        save_path: PathType,
        config: Dict,
        name: str = '',
    ):
    buff = config['training']['overlap_buff']
    training_set = training_set.with_columns(
        pl.col(i).fill_nan(None).fill_null(0.0).alias(i) for i in features
    )
    training_set = training_set.drop_nulls(target)
    X_ = training_set.select(features).to_numpy()[:-buff]
    y_ = training_set.select(target).to_numpy()[:-buff]
    N = len(X_)
    # split dataset
    if N >= config['training']['min_datasize_thres']:
        ntrain = int(N - config['training']['min_datasize_thres']/2)
        nvalid = int(config['training']['min_datasize_thres'] / 12)
    else:
        ntrain = int(N / 2)
        nvalid = int(N / 12)
    # preprocessing X
    X_scaler = StandardScaler()
    X_scaler.fit(X_[:ntrain])
    joblib.dump(X_scaler, os.path.join(save_path, '_'.join(['X_scaler',name,target])))
    # preprocessing y
    y_transformer = KBinsDiscretizer(n_bins=config['training']['class_size'], encode='ordinal', strategy='quantile')
    y_transformer.fit(y_[:ntrain])
    joblib.dump(y_transformer, os.path.join(save_path, '_'.join(['y_transformer',name,target])))
    hpspace = {hpname: hp.choice(hpname, hplist) for hpname, hplist in config['training']['hyperparameters'].items()}
    def objective(hpcomb):
        params.update(hpcomb) 
        scores = []
        for idvalid in range(1,6):
            train_X = X_[:ntrain+nvalid*idvalid - buff]
            train_y = y_[:ntrain+nvalid*idvalid - buff]
            valid_X = X_[ntrain+nvalid*idvalid:ntrain+nvalid*(idvalid+1)]
            valid_y = y_[ntrain+nvalid*idvalid:ntrain+nvalid*(idvalid+1)]
            # batch_size
            params['batch_size'] = compute_batchsize(train_X)
            model = modelClass(**params) 
            sw = compute_sample_weights(train_y) # should i store the sw?
            trans_train_X = X_scaler.transform(train_X)
            trans_valid_X = X_scaler.transform(valid_X)
            train_y = y_transformer.transform(train_y)
            valid_y = y_transformer.transform(valid_y)
            #
            model.fit(trans_train_X, train_y, validation_data = (trans_valid_X, valid_y), sample_weight = sw)
            score, acc = model.evaluate(trans_valid_X, valid_y)
            scores.append(acc)
            del model
            gc.collect()
        output_loss = -(0.0625*scores[0] + 0.0625 * scores[1] + 0.125*scores[2]+ 0.25*scores[3] + 0.5*scores[4])
        # Hyperopt expects a dictionary with a "loss" key and "status"
        return {
            'loss': output_loss,
            'status': STATUS_OK,
            'score_list': scores
        }
    bestidx = fmin(
        fn=objective,        
        space=hpspace,        
        algo=tpe.suggest,    
        max_evals=config['training']['max_hp_evals'],         # How many trials to run (each trial is one combination)
        #trials=trials
    )
    best = space_eval(hpspace, bestidx)
    params.update(best)
    # train predict1
    idvalid = 4
    train_X = X_[:ntrain+nvalid*idvalid - buff]
    train_y = y_[:ntrain+nvalid*idvalid - buff]
    valid_X = X_[ntrain+nvalid*idvalid:ntrain+nvalid*(idvalid+1)]
    valid_y = y_[ntrain+nvalid*idvalid:ntrain+nvalid*(idvalid+1)]
    params['batch_size'] = compute_batchsize(train_X)
    model1 = modelClass(**params) 
    sw = compute_sample_weights(train_y) # should i store the sw?
    trans_train_X = X_scaler.transform(train_X)
    trans_valid_X = X_scaler.transform(valid_X)
    train_y = y_transformer.transform(train_y)
    valid_y = y_transformer.transform(valid_y)
    model1.fit(trans_train_X, train_y, validation_data = (trans_valid_X, valid_y), sample_weight = sw)    
    model1.save(os.path.join(save_path, '_'.join([name,target,'_1'])))
    # train predict2
    idvalid = 5
    train_X = X_[:ntrain+nvalid*idvalid - buff]
    train_y = y_[:ntrain+nvalid*idvalid - buff]
    valid_X = X_[ntrain+nvalid*idvalid:ntrain+nvalid*(idvalid+1)]
    valid_y = y_[ntrain+nvalid*idvalid:ntrain+nvalid*(idvalid+1)]
    params['batch_size'] = compute_batchsize(train_X)
    model2 = modelClass(**params) 
    sw = compute_sample_weights(train_y) # should i store the sw?
    trans_train_X = X_scaler.transform(train_X)
    trans_valid_X = X_scaler.transform(valid_X)
    train_y = y_transformer.transform(train_y)
    valid_y = y_transformer.transform(valid_y)
    model2.fit(trans_train_X, train_y, validation_data = (trans_valid_X, valid_y), sample_weight = sw)
    model2.save(os.path.join(save_path, '_'.join([name,target,'_2'])))
    

def predict_classifier(
        model_class, 
        test_set: pl.DataFrame,
        target: str,
        features: List[str],
        save_path: PathType,
        name: str = ''
    ) -> pl.DataFrame:
    test_set = test_set.with_columns(
        pl.col(i).fill_nan(None).fill_null(0.0).alias(i) for i in features
    )
    X_ = test_set.select(features).to_numpy()
    # load preprocess
    X_scaler = joblib.load(os.path.join(save_path, '_'.join(['X_scaler',name,target])))
    trans_X = X_scaler.transform(X_)
    #y_transformer = joblib.load(os.path.join(save_path, '_'.join(['y_transformer',name,target])))
    model1 = model_class.load(os.path.join(save_path, '_'.join([name,target,'_1'])))
    model2 = model_class.load(os.path.join(save_path, '_'.join([name,target,'_2'])))
    pred1 = model1.predict(trans_X)
    pred2 = model2.predict(trans_X)
    pred = (pred1 + pred2)/2
    test_set = test_set.with_columns(
            pl.Series(pred[:,0]).alias(f'pred_{target}_neg'),
            pl.Series(pred[:,1]).alias(f'pred_{target}_neu'),
            pl.Series(pred[:,2]).alias(f'pred_{target}_pos'),
        )
    return test_set


