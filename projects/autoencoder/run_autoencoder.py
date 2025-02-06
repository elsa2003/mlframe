import polars as pl
import yaml
import os
import torch

from projects.autoencoder.autoencoder import *
#%%
data = pl.read_parquet("/Users/charles/Downloads/price.parquet")

universe = ['22939647', '8797768', '93425',]

data = data.pivot(
    values="priceclose",
    index="pricedate",
    columns="companyid"
).sort("pricedate")

#%% get data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join('projects/autoencoder/','autoencoder_config.yaml'), 'r') as f:
    autoencoder_config = yaml.safe_load(f)


#data = data.with_columns([pl.when(pl.col(c).is_infinite()).then(pl.lit(0)).otherwise(pl.col(c)).alias(c) for c in universe]).with_columns([pl.col(c).fill_nan(0).fill_null(0).alias(c) for c in universe])
# training and evaluatin
ts = XSReturnDataset(data, universe = universe, seq_length = autoencoder_config['training']['seq_len'], batch_size = autoencoder_config['data']['batch_size'])
train_iter, test_iter, nb_features = ts.get_loaders()

featuremodel = AutoEncForecast(autoencoder_config['training'], input_size=len(universe)).to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(featuremodel.parameters(), lr=float(autoencoder_config['training']['lr']))


train_autoencoder(train_iter, test_iter, featuremodel, criterion, optimizer, autoencoder_config, ts)
ts.dump(autoencoder_config['general']['output_dir'], 'tsdataset.pkl')
# eval
model_path = os.path.join(autoencoder_config['general']['output_dir'], 'checkpoint-1000.ckpt')
featuremodel, _, loss, epoch = load_autoencoder_checkpoint(featuremodel, device, optimizer)
evaluate_autoencoder(test_iter, loss, featuremodel, autoencoder_config, ts)
#%% transform features
ts = TimeSeriesDataset.load(autoencoder_config['general']['output_dir'], 'tsdataset.pkl')
model_path = os.path.join(autoencoder_config['general']['output_dir'], 'checkpoint-1400.ckpt')
featuremodel = AutoEncForecast(autoencoder_config['training'], input_size=ts.nfeatures).to(device)
featuremodel, _, loss, epoch = load_autoencoder_checkpoint(model_path, featuremodel, device, None)
newdf = predict_features(data, ['datetime'] + features, featuremodel, autoencoder_config, ts, predict_batch = 1000)
##########################
