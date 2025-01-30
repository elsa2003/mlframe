import polars as pl
import yaml
import os
import torch
from data_loader import (
    get_hist_spot_trades_data,
    get_hist_perp_trades_data,
    get_hist_perp_tickers_data
)


from features import (
    generate_bars_from_strades,
    generate_bars_from_ftrades,
    generate_fret_from_bars,
    generate_features_from_bar
)

from features import (
    load_autoencoder_checkpoint,
    evaluate_autoencoder,
    predict_features,
    train_autoencoder,
    TimeSeriesDataset,
    AutoEncForecast
)

from model import (
    train_classifier, 
    predict_classifier,
    prepare_lgbm,
    prepare_gru,
    )

from utils.utils import to_datetime
#%% get data
start = '2024-10-01'
end = '2024-11-10'
symbol = 'ENSUSDT'
tspot = get_hist_spot_trades_data(symbol, 'tardis', start, end)
tfut = get_hist_perp_trades_data(symbol, 'tardis', start, end)
oifut = get_hist_perp_tickers_data(symbol, 'tardis', start, end)


#%% feature process
sbar = generate_bars_from_strades(tspot, '15m', 'datetime')
fbar = generate_bars_from_strades(tfut, '15m', 'datetime')

sbar = generate_features_from_bar(sbar, 'datetime')
fbar = generate_features_from_bar(fbar, 'datetime')
fbar = generate_fret_from_bars(fbar, 'datetime', 'close', [1, 4, 32, 96])

data = fbar.join(sbar, on = ['datetime'], suffix = '_s', how = 'inner')

###############################
#%% autoencoder
###############################
features = ['rsj1h_s', 'rsj8h_s', 'tpc1h_s', 'tpc8h_s', 'stpdown1h_s', 'stpdown8h_s', 'tpcskew1h_s', 'tpcskew8h_s']
targets = ['ret_T4', 'ret_T32']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join('features', 'autoencoder_config.yaml'), 'r') as f:
    autoencoder_config = yaml.safe_load(f)

data = data.with_columns([pl.when(pl.col(c).is_infinite()).then(pl.lit(0)).otherwise(pl.col(c)).alias(c) for c in features+targets]).with_columns([pl.col(c).fill_nan(0).fill_null(0).alias(c) for c in features+targets])
""" training and evaluatin
ts = TimeSeriesDataset(data, numerical_cols = features, categorical_cols = [], seq_length = autoencoder_config['training']['seq_len'], batch_size = autoencoder_config['data']['batch_size'], prediction_window = 1)
train_iter, test_iter, nb_features = ts.get_loaders()

featuremodel = AutoEncForecast(autoencoder_config['training'], input_size=nb_features).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(featuremodel.parameters(), lr=float(autoencoder_config['training']['lr']))


train_autoencoder(train_iter, test_iter, featuremodel, criterion, optimizer, autoencoder_config, ts)
ts.dump(autoencoder_config['general']['output_dir'], 'tsdataset.pkl')
# eval
model_path = os.path.join(autoencoder_config['general']['output_dir'], 'checkpoint-1000.ckpt')
featuremodel, _, loss, epoch = load_autoencoder_checkpoint(model_path, featuremodel, device, optimizer)
evaluate_autoencoder(test_iter, loss, featuremodel, autoencoder_config, ts)
"""
#%% transform features
ts = TimeSeriesDataset.load(autoencoder_config['general']['output_dir'], 'tsdataset.pkl')
model_path = os.path.join(autoencoder_config['general']['output_dir'], 'checkpoint-1400.ckpt')
featuremodel = AutoEncForecast(autoencoder_config['training'], input_size=ts.nfeatures).to(device)
featuremodel, _, loss, epoch = load_autoencoder_checkpoint(model_path, featuremodel, device, None)
newdf = predict_features(data, ['datetime'] + features, featuremodel, autoencoder_config, ts, predict_batch = 1000)
##########################

#%%
lgbmodel, params = prepare_gru()
params['n_estimators'] = 10
params['num_leaves'] = 15
params['feature_size'] = len(features)
params['nb_epoch'] = 50
params['learning_rate'] = 1e-2
params['patient'] = 10

#%%
with open(os.path.join('model', 'train_config.yaml'), 'r') as f:
    train_config = yaml.safe_load(f)


train_dtend = to_datetime('2024-11-09 16:00')
test_dtbegin = to_datetime('2024-11-10 00:00')
training_set = data.filter(pl.col('datetime') < train_dtend)
test_set = data.filter(pl.col('datetime') > test_dtbegin)
#
train_classifier(
        training_set = training_set,
        target = 'ret_T4',
        features = features,
        modelClass = lgbmodel,
        params = params,
        save_path = '/root/inovest/mlframework/models/',
        config = train_config,
        name = 'demo'
    )

test_set = predict_classifier(
        model_class = lgbmodel,
        test_set = test_set, 
        target = 'ret_T4',
        features = features,
        save_path =  '/root/inovest/mlframework/models/',
        name = 'demo'
    )
