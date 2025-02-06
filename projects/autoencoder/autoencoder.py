import numpy as np
import torch
from torch import nn
import os
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.nn import functional as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from enum import Enum
from typing import List
import joblib

import pandas as pd
import polars as pl
import pkg_resources
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class XSReturnDataset(object):
    def __init__(self, df_polars: pl.DataFrame, universe: List[str], 
                 seq_length: int, batch_size: int, normalize = True):
        """
        :param df_polars: polars dataframe
        :param universe: name of the universe column name
        :param seq_length: window length to use
        :param batch_size:
        """
        self.data = df_polars.to_pandas()

        self.universe = universe

        self.seq_length = seq_length
        self.batch_size = batch_size
        self.normalize = normalize
        if normalize:
            transformations = [("scaler", StandardScaler(), self.universe)]
            self.preprocessor = ColumnTransformer(transformations, remainder="passthrough")

    def preprocess_data(self):
        """Preprocessing function"""
        X = self.data[self.universe]

        X_train, X_test = train_test_split(X, train_size=0.8, shuffle=False)
        if self.normalize:
            X_train = self.preprocessor.fit_transform(X_train)
            X_test = self.preprocessor.transform(X_test)

        return X_train, X_test, None, None

    def frame_series(self, X, y=None):
        """
        Function used to prepare the data for time series prediction
        :param X: set of features
        :param y: targeted value to predict
        :return: TensorDataset
        """
        nb_obs, nb_features = X.shape
        features, recons, target, y_hist = [], [], [], []

        for i in range(1, nb_obs - self.seq_length - 1):
            features.append(torch.FloatTensor(X[i:i + self.seq_length, :]).unsqueeze(0))

            y_hist.append(torch.FloatTensor(X[i - 1: i + self.seq_length - 1, :]).unsqueeze(0))
            recons.append(
                torch.FloatTensor(X[i -1 + self.seq_length:i + self.seq_length, :]))
            target.append(
                torch.FloatTensor(X[i + self.seq_length:i + self.seq_length + 1, :]))

        features_var = torch.cat(features)
        y_hist_var = torch.cat(y_hist)
        recons_var = torch.cat(recons)
        target_var = torch.cat(target)

        return TensorDataset(features_var, y_hist_var, recons_var, target_var)

    def get_loaders(self):
        """
        Preprocess and frame the dataset

        :return: DataLoaders associated to training and testing data
        """
        X_train, X_test, y_train, y_test = self.preprocess_data()
        nb_features = X_train.shape[1]

        train_dataset = self.frame_series(X_train, y_train)
        test_dataset = self.frame_series(X_test, y_test)

        train_iter = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        test_iter = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        return train_iter, test_iter, nb_features

    def invert_scale(self, predictions):
        """
        Inverts the scale of the predictions
        """
        if self.normalize:
            if isinstance(predictions, torch.Tensor):
                predictions = predictions.numpy()

            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            unscaled = self.preprocessor.named_transformers_["scaler"].inverse_transform(predictions)
            return torch.Tensor(unscaled)
        else:
            return predictions

    def prepare_prediction_data(self, inputs: pd.DataFrame, predict_batch = 1000):
        """"""
        X = inputs[self.features]
        idxs = range(len(X))
        if self.normalize:
            X = self.preprocessor.transform(X)

        nb_obs, nb_features = X.shape
        features, y_hist, raw_idx = [], [], []

        for i in range(1, nb_obs - self.seq_length - 1):
            features.append(torch.FloatTensor(X[i:i + self.seq_length, :]).unsqueeze(0))

            y_hist.append(torch.FloatTensor(X[i - 1: i + self.seq_length - 1, :]).unsqueeze(0))
            raw_idx.append(idxs[i + self.seq_length])
        features_var = torch.cat(features)
        y_hist_var = torch.cat(y_hist)
        raw_idx_var = torch.tensor(raw_idx)

        test_dataset = TensorDataset(features_var, y_hist_var, raw_idx_var)
        if len(raw_idx_var) < predict_batch:
            predict_batch = len(raw_idx_var) 
        test_iter = DataLoader(test_dataset, batch_size=predict_batch, shuffle=False, drop_last=True)
        return test_iter

    def dump(self, dir_path: str, name: str) -> None:
        """
        Save the class instance to disk, excluding self.data.
        """
        state = dict(self.__dict__)
        if 'data' in state:
            del state['data']
        joblib.dump(state, os.path.join(dir_path, name))

    @classmethod
    def load(cls, dir_path: str, name: str):
        """
        Load a TimeSeriesDatasetPolars instance from disk.
        The returned instance will NOT have 'data' populated.
        """
        state = joblib.load(os.path.join(dir_path, name))
        
        # Create a new "empty" instance
        obj = cls.__new__(cls)
        # Update its __dict__ with the unpickled state
        obj.__dict__.update(state)
        return obj

    @property
    def nuniverse(self):
        return len(self.universe)

    @property
    def features(self):
        return self.universe


def init_hidden(x: torch.Tensor, hidden_size: int, num_dir: int = 1, xavier: bool = True):
    """
    Initialize hidden.
    Args:
        x: (torch.Tensor): input tensor
        hidden_size: (int):
        num_dir: (int): number of directions in LSTM
        xavier: (bool): wether or not use xavier initialization
    """
    if xavier:
        return nn.init.xavier_normal_(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)
    return Variable(torch.zeros(num_dir, x.size(0), hidden_size)).to(device)


# encoder
class Encoder(nn.Module):
    def __init__(self, config, input_size: int):
        """
        Initialize the model.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = config['hidden_size_encoder']
        self.seq_len = config['seq_len']
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=config['hidden_size_encoder'])

    def forward(self, input_data: torch.Tensor):
        """
        Run forward computation.

        Args:
            input_data: (torch.Tensor): tensor of input daa
        """
        h_t, c_t = (init_hidden(input_data, self.hidden_size),
                    init_hidden(input_data, self.hidden_size))
        input_encoded = Variable(torch.zeros(input_data.size(0), self.seq_len, self.hidden_size))

        for t in range(self.seq_len):
            _, (h_t, c_t) = self.lstm(input_data[:, t, :].unsqueeze(0), (h_t, c_t))
            input_encoded[:, t, :] = h_t
        return _, input_encoded


class AttnEncoder(nn.Module):
    def __init__(self, config, input_size: int):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AttnEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = config['hidden_size_encoder']
        self.seq_len = config['seq_len']
        self.add_noise = config['denoising']
        self.directions = config['directions']
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=1
        )
        self.attn = nn.Linear(
            in_features=2 * self.hidden_size + self.seq_len,
            out_features=1
        )
        self.softmax = nn.Softmax(dim=1)

    @staticmethod
    def _get_noise(input_data: torch.Tensor, sigma=0.01, p=0.1):
        """
        Get noise.

        Args:
            input_data: (torch.Tensor): tensor of input data
            sigma: (float): variance of the generated noise
            p: (float): probability to add noise
        """
        normal = sigma * torch.randn(input_data.shape)
        mask = np.random.uniform(size=(input_data.shape))
        mask = (mask < p).astype(int)
        noise = normal * torch.tensor(mask)
        return noise

    def forward(self, input_data: torch.Tensor):
        """
        Forward computation.

        Args:
            input_data: (torch.Tensor): tensor of input data
        """
        h_t, c_t = (init_hidden(input_data, self.hidden_size, num_dir=self.directions),
                    init_hidden(input_data, self.hidden_size, num_dir=self.directions))

        attentions, input_encoded = (Variable(torch.zeros(input_data.size(0), self.seq_len, self.input_size)),
                                     Variable(torch.zeros(input_data.size(0), self.seq_len, self.hidden_size)))

        if self.add_noise and self.training:
            input_data += self._get_noise(input_data).to(device)

        for t in range(self.seq_len):
            x = torch.cat((h_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           c_t.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1).to(device)), dim=2).to(
                device)  # bs * input_size * (2 * hidden_dim + seq_len)

            e_t = self.attn(x.view(-1, self.hidden_size * 2 + self.seq_len))  # (bs * input_size) * 1
            a_t = self.softmax(e_t.view(-1, self.input_size)).to(device)  # (bs, input_size)

            weighted_input = torch.mul(a_t, input_data[:, t, :].to(device))  # (bs * input_size)
            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(weighted_input.unsqueeze(0), (h_t, c_t))

            input_encoded[:, t, :] = h_t
            attentions[:, t, :] = a_t

        return attentions, input_encoded


# decoder
class Decoder(nn.Module):
    def __init__(self, config, input_size: int):
        """
        Initialize the network.

        Args:
            config:
        """
        super(Decoder, self).__init__()
        self.seq_len = config['seq_len']
        self.hidden_size = config['hidden_size_decoder']
        self.lstm = nn.LSTM(1, config['hidden_size_decoder'], bidirectional=False)
        self.fc = nn.Linear(config['hidden_size_decoder'], input_size)

    def forward(self, _, y_hist: torch.Tensor):
        """
        Forward pass

        Args:
            _:
            y_hist: (torch.Tensor): shifted target
        """
        h_t, c_t = (init_hidden(y_hist, self.hidden_size),
                    init_hidden(y_hist, self.hidden_size))

        for t in range(self.seq_len):
            inp = y_hist[:, t].unsqueeze(0).unsqueeze(2)
            lstm_out, (h_t, c_t) = self.lstm(inp, (h_t, c_t))
        return self.fc(lstm_out.squeeze(0))


class AttnDecoder(nn.Module):
    def __init__(self, config, input_size: int):
        """
        Initialize the network.

        Args:
            config:
        """
        super(AttnDecoder, self).__init__()
        self.seq_len = config['seq_len']
        self.encoder_hidden_size = config['hidden_size_encoder']
        self.decoder_hidden_size = config['hidden_size_decoder']
        self.out_feats = input_size

        self.attn = nn.Sequential(
            nn.Linear(2 * self.decoder_hidden_size + self.encoder_hidden_size, self.encoder_hidden_size),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_size, 1)
        )
        self.lstm = nn.LSTM(input_size=self.out_feats, hidden_size=self.decoder_hidden_size)
        self.fc = nn.Linear(self.encoder_hidden_size + self.out_feats, self.out_feats)
        self.fc_out = nn.Linear(self.decoder_hidden_size + self.encoder_hidden_size, self.out_feats)
        self.fc.weight.data.normal_()

    def forward(self, input_encoded: torch.Tensor, y_history: torch.Tensor):
        """
        Perform forward computation.

        Args:
            input_encoded: (torch.Tensor): tensor of encoded input
            y_history: (torch.Tensor): shifted target
        """
        h_t, c_t = (
            init_hidden(input_encoded, self.decoder_hidden_size), init_hidden(input_encoded, self.decoder_hidden_size))
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size))

        for t in range(self.seq_len):
            x = torch.cat((h_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           c_t.repeat(self.seq_len, 1, 1).permute(1, 0, 2),
                           input_encoded.to(device)), dim=2)

            x = tf.softmax(
                self.attn(
                    x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                ).view(-1, self.seq_len),
                dim=1)

            context = torch.bmm(x.unsqueeze(1), input_encoded.to(device))[:, 0, :]  # (batch_size, encoder_hidden_size)

            y_tilde = self.fc(torch.cat((context.to(device), y_history[:, t].to(device)),
                                        dim=1))  # (batch_size, out_size)

            self.lstm.flatten_parameters()
            _, (h_t, c_t) = self.lstm(y_tilde.unsqueeze(0), (h_t, c_t))

        return self.fc_out(torch.cat((h_t[0], context.to(device)), dim=1))  # predicting value at t=self.seq_length+1


class AutoEncForecast(nn.Module):
    def __init__(self, config, input_size):
        """
        Initialize the network.

        Args:
            config:
            input_size: (int): size of the input
        """
        super(AutoEncForecast, self).__init__()
        self.encoder = AttnEncoder(config, input_size).to(device) if config['input_att'] else \
            Encoder(config, input_size).to(device)
        self.decoder = AttnDecoder(config, input_size).to(device) if config['temporal_att'] else Decoder(config, input_size).to(device)

    def forward(self, encoder_input: torch.Tensor, y_hist: torch.Tensor, return_attention: bool = False):
        """
        Forward computation. encoder_input_inputs.

        Args:
            encoder_input: (torch.Tensor): tensor of input data
            y_hist: (torch.Tensor): shifted target
            return_attention: (bool): whether to return the attention
        """
        attentions, encoder_output = self.encoder(encoder_input)
        outputs = self.decoder(encoder_output, y_hist.float())

        if return_attention:
            return outputs, attentions
        return outputs

class WeightedDiffLayer(nn.Module):
    def __init__(self, input_size):
        """
        """
        super(WeightedDiffLayer, self).__init__()
        self.fc = nn.Linear(input_size, input_size)

    def forward(self, res, r):
        """
        Forward pass:
         - Compute d = w * (y - y_real)
         - Normalize d (using L2 normalization)
         - Compute loss as element-wise product d_norm * r, then aggregate (here, mean)
        """
        # Compute the weighted difference
        d = self.fc(res)

        # Normalize d along the feature dimensions. 
        d_norm = F.normalize(d, p=1, dim=1)

        # Compute the loss as the element-wise product with r.
        ret = d_norm * r
        sharp = torch.mean(ret) / torch.std(ret)

        return sharp

#%% trainer fit

def train_autoencoder(train_iter, test_iter, model, criterion, optimizer, config, ts):
    """
    Training function.

    Args:
        train_iter: (DataLoader): train data iterator
        test_iter: (DataLoader): test data iterator
        model: model
        criterion: loss to use
        optimizer: optimizer to use
        config:
    """
    tb_writer_train = SummaryWriter(logdir=config['general']['output_dir'], filename_suffix="train")
    tb_writer_test = SummaryWriter(logdir=config['general']['output_dir'], filename_suffix="test")

    if not os.path.exists(config['general']['output_dir']):
        os.makedirs(config['general']['output_dir'])

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['training']['lrs_step_size'], gamma=0.5)

    global_step, logging_loss = 0, 0.0
    train_loss = 0.0
    for epoch in tqdm(range(config['training']['num_epochs']), unit="epoch"):
        for i, batch in tqdm(enumerate(train_iter), total=len(train_iter), unit="batch"):
            model.train()
            optimizer.zero_grad()

            feature, y_hist, recons, target = batch
            output = model(feature.to(device), y_hist.to(device))
            loss = criterion(output.to(device), recons.to(device))
            # sharpe regularization
            sr = predmodel(output.to(device) - recons.to(device), target)
            loss += config['training']['reg_sr'] * sr


            if config['training']['reg1']:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() if "bias" not in name])
                loss += config['training']['reg_factor1'] * torch.norm(params, 1)
            if config['training']['reg2']:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() if "bias" not in name])
                loss += config['training']['reg_factor2'] * torch.norm(params, 2)

            if config['training']['gradient_accumulation_steps'] > 1:
                loss = loss / config['training']['gradient_accumulation_steps']

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['max_grad_norm'])
            train_loss += loss.item()

            if (i + 1) % config['training']['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                global_step += 1

                if global_step % config['general']['logging_steps'] == 0:
                    if config['general']['eval_during_training']:
                        results = evaluate_autoencoder(test_iter, criterion, model, config, ts)
                        for key, val in results.items():
                            tb_writer_test.add_scalar("eval_{}".format(key), val, global_step)

                    tb_writer_train.add_scalar("train_loss", (train_loss - logging_loss) / config['general']['logging_steps'],
                                               global_step)
                    tb_writer_train.add_scalar("lr", scheduler.get_last_lr()[0], global_step)
                    logging_loss = train_loss

            if global_step % config['general']['save_steps'] == 0:
                torch.save({
                    "epoch": epoch + 1,
                    "encoder_state_dict": model.encoder.state_dict(),
                    "decoder_state_dict": model.decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion
                }, "{}/checkpoint-{}.ckpt".format(config['general']['output_dir'], global_step))



# eval trainer

def evaluate_autoencoder(test_iter, criterion, model, predmodel, config, ts):
    """
    Evaluate the model on the given test set.

    Args:
        test_iter: (DataLoader): test dataset iterator
        criterion: loss function
        model: model to use
        config: config
    """
    predictions, targets, attentions = [], [], []
    eval_loss = 0.0

    model.eval()
    for i, batch in tqdm(enumerate(test_iter), total=len(test_iter), desc="Evaluating"):
        with torch.no_grad():
            feature, y_hist, recons, target = batch
            output, att = model(feature.to(device), y_hist.to(device), return_attention=True)

            loss = criterion(output.to(device), recons.to(device)).item()

            sr = predmodel(output.to(device) - recons.to(device), target)
            loss += config['training']['reg_sr'] * sr

            if config['training']['reg1']:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() if 'bias' not in name])
                loss += config['training']['reg_factor1'] * torch.norm(params, 1)
            if config['training']['reg2']:
                params = torch.cat([p.view(-1) for name, p in model.named_parameters() if 'bias' not in name])
                loss += config['training']['reg_factor2'] * torch.norm(params, 2)
            eval_loss += loss

            predictions.append(output.squeeze(1).cpu())
            targets.append(target.squeeze(1).cpu())
            attentions.append(att.cpu())

    predictions, targets = torch.cat(predictions), torch.cat(targets)

    if config['general']['do_eval']:
        preds, targets = ts.invert_scale(predictions), ts.invert_scale(targets)

        plt.figure()
        plt.plot(preds, linewidth=.3)
        plt.plot(targets, linewidth=.3)
        plt.savefig("{}/preds.png".format(config['general']['output_dir']))
        plt.close()

        torch.save(targets, os.path.join(config['general']['output_dir'], "targets.pt"))
        torch.save(predictions, os.path.join(config['general']['output_dir'], "predictions.pt"))
        torch.save(attentions, os.path.join(config['general']['output_dir'], "attentions.pt"))

    results = get_autoencoder_eval_report(eval_loss / len(test_iter), predictions, targets)
    file_eval = os.path.join(config['general']['output_dir'], "eval_results.txt")
    with open(file_eval, "w") as f:
        f.write("********* EVAL REPORT ********\n")
        for key, val in results.items():
            f.write("  %s = %s\n" % (key, str(val)))

    return results


def predict_features(df, index_names, model, config, ts, predict_batch = 1000, prefix = 'tr_'):
    """
    predict the features with the model on the given test set.
    """
    predictions = []
    index_list = []
    eval_loss = 0.0
    test_iter = ts.prepare_prediction_data(df.to_pandas(), predict_batch = predict_batch)
    model.eval()
    for i, batch in tqdm(enumerate(test_iter), total=len(test_iter), desc="Predicting"):
        with torch.no_grad():
            feature, y_hist, idx = batch
            output, att = model(feature.to(device), y_hist.to(device), return_attention=True)
            predictions.append(output.squeeze(1).cpu())
            index_list.append(idx)

    predictions = torch.cat(predictions)
    predictions = ts.invert_scale(predictions)
    index_list = torch.cat(index_list)
    new_data = df[index_names][index_list.tolist()]
    return new_data.with_columns(
        (pl.Series(name = prefix + ts.features[ifeat], values = predictions[:,ifeat].tolist()) for ifeat in range(ts.nfeatures))
    ) 

def get_autoencoder_eval_report(eval_loss: float, predictions: torch.Tensor, targets: torch.Tensor):
    """
    Evaluates the accuracy.

    Args:
        eval_loss: (float): loss vlue
        predictions: (torch.Tensor): tensor of predictions
        targets: (torch.Tensor): tensor of targets
    """
    residuals = np.mean(predictions.numpy() - targets.numpy())
    MSE = F.mse_loss(targets.squeeze(), predictions.squeeze()).item()
    return {"MSE": MSE, "residuals": residuals, "loss": eval_loss}



def load_autoencoder_checkpoint(checkpoint_path, model, device, optimizer = None):
    """Load model state from checkpoint file"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    loss = checkpoint["loss"]
    epoch = checkpoint["epoch"]
    return model, optimizer, loss, epoch