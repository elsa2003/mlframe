import numpy as np
from sklearn.model_selection import train_test_split

import torch
import copy
import joblib
import torch.nn as nn
from torch.autograd import Variable

def prepare_gru():
    standard_params = {
        'max_len': 30, 'embedding_size': 128, 'batch_size': 16, 'patience': 10,
        'feature_size': 100, 'class_size': 3, 'learning_rate': 1e-4, 'weight_decay': 1e-4,
        'batch_size': 32, 'nb_epoch': 10, 'dropout': 0.5, 'epoch': 50, 'random_state':  61}
    return GRUClassifier, standard_params


def batch(tensor, batch_size):
    tensor_list = []
    length = tensor.shape[0]
    i = 0
    while True:
        if (i+1) * batch_size >= length:
            tensor_list.append(tensor[i * batch_size: length])
            return tensor_list
        tensor_list.append(tensor[i * batch_size: (i+1) * batch_size])
        i += 1

def create_sequences(data, seq_len):
    """
    Create zero-padded sequences of shape (T, seq_len, N) from data of shape (T, N).
    Returns
    -------
    X : np.ndarray
        Array of shape (T, seq_len, N).
        X[t] contains the sequence of length seq_len ending at time t, with zero-padding
        on the left if t < seq_len - 1.
    """
    T, N = data.shape
    X = np.zeros((T, seq_len, N), dtype=data.dtype)
    for t in range(T):
        # Start index for slicing the original data
        start_idx = t - seq_len + 1  # inclusive
        end_idx   = t + 1           # exclusive in Python slicing
        if start_idx < 0:
            length_of_data = t + 1
            X[t, seq_len - length_of_data : seq_len, :] = data[0 : end_idx, :]
        else:
            X[t] = data[start_idx : end_idx, :]
    return X

class GRUClassifier(object):

    def __init__(
        self, feature_size, embedding_size = 128, class_size = 3, seq_len = 20,
        learning_rate = 1e-4, dropout = 0.5, l2_lambda = 1e-4, weight_decay = 1e-4, patience = 5,
        batch_size=32, nb_epoch=10, min_delta = 1e-4, random_state = 61, model = None, **kwargs
        ):
        if model is None:
            self.model = GRU(feature_size, embedding_size, class_size, dropout = dropout)
        else:
            self.model = model
        self.use_gpu = torch.cuda.is_available()
        self.compile(optimizerClass=torch.optim.Adam, seq_len = seq_len, lr = learning_rate, weight_decay = weight_decay, l2_lambda = l2_lambda, 
                patience = patience, batch_size = batch_size, nb_epoch = nb_epoch, min_delta = min_delta)
        # random state
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        if self.use_gpu:
            torch.cuda.manual_seed_all(random_state)
        self._params = {'seq_len': seq_len, 'feature_size': feature_size, 'embedding_size': embedding_size, 'class_size': class_size, 'learning_rate': learning_rate, 'weight_decay': weight_decay,
                        'dropout': dropout, 'l2_lambda': l2_lambda, 'patience': patience, 'batch_size': batch_size, 'nb_epoch': nb_epoch, 'min_delta': min_delta}

    def compile(self, optimizerClass, seq_len = 20, lr = 1e-4, weight_decay = 1e-4, patience=5, l2_lambda=1e-4, batch_size=32, nb_epoch=10, min_delta = 1e-4):
        """
        Compile the estimator with an optimizer and loss function.
        patience: number of epochs to wait for improvement in val loss before early stopping.
        """
        self.optimizer = optimizerClass(self.model.parameters(), lr = lr, weight_decay = weight_decay)
        self.loss_f = nn.CrossEntropyLoss(reduction='none')
        self.seq_len = seq_len
        self.patience = patience
        self.l2_lambda = l2_lambda
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.min_delta = min_delta

    def _fit(self, X_list, y_list, w_list=None):
        """
        Train for one epoch on the provided mini-batches.
        """
        self.model.train()  # set to training mode (important for dropout)
        loss_list = []
        acc_list = []
        if w_list is None:
            w_list = [None] * len(X_list)
        for X, y, w in zip(X_list, y_list, w_list):
            X_v = Variable(torch.from_numpy(np.swapaxes(X,0,1)).float())
            y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)

            if self.use_gpu:
                X_v = X_v.cuda()
                y_v = y_v.cuda()
                self.model.cuda()

            self.optimizer.zero_grad()
            # Forward pass
            hidden = self.model.initHidden(X_v.size()[1])
            if self.use_gpu:
                hidden = hidden.cuda()
            y_pred = self.model(X_v, hidden)

            # Base loss
            losses = self.loss_f(y_pred, y_v)
    
            l2_reg = 0.0
            # probably fix only gru parameters for penalty
            for param in self.model.parameters(): 
                l2_reg += torch.sum(param ** 2)

            if w is not None:
                # Convert w_batch to tensor
                w_v = Variable(torch.from_numpy(w).float())
                if self.use_gpu:
                    w_v = w_v.cuda()
                # Multiply each sample's loss by its weight
                losses = losses * w_v  # elementwise multiply

            loss = losses.mean() + self.l2_lambda * l2_reg
            loss.backward()
            self.optimizer.step()

            # Logging
            loss_list.append(loss.item())
            classes = torch.topk(y_pred, 1)[1].data.cpu().numpy().flatten()  # move to CPU for numpy
            acc = self._accuracy(classes, y)
            acc_list.append(acc)

        return sum(loss_list) / len(loss_list), sum(acc_list) / len(acc_list)

    def fit(self, X, y, validation_data=(), sample_weight=None):
        X_seq = create_sequences(X, self.seq_len)
        if len(y.shape) == 2:
            y = y.reshape(-1)
        X_list = batch(X_seq, self.batch_size)
        y_list = batch(y, self.batch_size)

        if sample_weight is not None:
            w_list = batch(sample_weight, self.batch_size)
        else:
            w_list = None

        best_val_loss = float('inf')
        no_improve_count = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_epoch = 0

        for epoch in range(1, self.nb_epoch + 1):
            train_loss, train_acc = self._fit(X_list, y_list, w_list)
            val_log = ""

            if validation_data:
                val_loss, val_acc = self.evaluate(validation_data[0], validation_data[1])
                val_log = f"- val_loss: {val_loss:06.4f} - val_acc: {val_acc:06.4f}"

                # Early Stopping Check
                if best_val_loss - val_loss > self.min_delta:
                    best_val_loss = val_loss
                    no_improve_count = 0
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                else:
                    no_improve_count += 1

                if no_improve_count >= self.patience:
                    print(f"Early stopping triggered after epoch {epoch}.")
                    break

            print(f"Epoch {epoch}/{self.nb_epoch} loss: {train_loss:06.4f} - acc: {train_acc:06.4f} {val_log}")
            self.model.load_state_dict(best_model_wts)

    def evaluate(self, X, y):
        """
        Evaluate the model on provided data in a single pass (no mini-batches).
        """
        self.model.eval()  # set to eval mode (important for dropout)
        if len(y.shape) == 2:
            y = y.reshape(-1)
        y_pred = self._predict(X)

        y_v = Variable(torch.from_numpy(y).long(), requires_grad=False)
        if self.use_gpu:
            y_v = y_v.cuda()
            
        with torch.no_grad():
            losses = self.loss_f(y_pred, y_v)
            loss = losses.mean()
        classes = torch.topk(y_pred, 1)[1].data.cpu().numpy().flatten()
        acc = self._accuracy(classes, y)
        return loss.item(), acc

    def _accuracy(self, y_pred, y):
        return sum(y_pred == y) / y.shape[0]

    def predict(self, X):
        return self._predict(X).data.cpu().numpy()

    def _predict(self, X):
        """
        Single-shot prediction for the entire dataset (no mini-batches).
        """
        self.model.eval()
        X_seq = create_sequences(X, self.seq_len)
        X_v = Variable(torch.from_numpy(np.swapaxes(X_seq,0,1)).float())
        if self.use_gpu:
            X_v = X_v.cuda()

        hidden = self.model.initHidden(X_v.size()[1])
        if self.use_gpu:
            hidden = hidden.cuda()

        y_pred = self.model(X_v, hidden)
        return y_pred        

    def predict_classes(self, X):
        return torch.topk(self._predict(X), 1)[1].data.cpu().numpy().flatten()

    def save(self, path):
        torch.save(self.model, path)
        joblib.dump(self._params, path + '_gruparams')

    @classmethod
    def load(self, path ):
        params = joblib.load(path + '_gruparams')
        model = torch.load(path, weights_only=False)
        return GRUClassifier(params['feature_size'], embedding_size = params['embedding_size'], 
            class_size = params['class_size'], learning_rate = params['learning_rate'], weight_decay = params['weight_decay'], dropout = params['dropout'],
            l2_lambda = params['l2_lambda'], patience = params['patience'], batch_size=params['batch_size'], 
            nb_epoch=params['nb_epoch'], min_delta = params['min_delta'], model = model)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
        """
        A simple GRU model with dropout.
        dropout will be applied after the GRU state if num_layers=1, or use built-in
        dropout if num_layers > 1 for the GRU. Here we manually apply dropout to the output state.
        """
        super(GRU, self).__init__()

        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        """
        input: (seq_len, batch, input_size)
        hidden: (num_layers, batch, hidden_size)
        """
        _, hn = self.gru(input, hidden)
        hn = hn[-1]  # get the last layer's hidden state, shape: (batch, hidden_size)
        hn = hn * torch.sigmoid(hn) # swish
        dropped = self.dropout(hn)  # apply dropout
        out = self.linear(dropped)  # shape: (batch, output_size)
        out = nn.functional.softmax(out, dim=1)
        return out

    def initHidden(self, N):
        return Variable(torch.randn(1, N, self.hidden_size))


def test_model():
    class_size = 3

    ## Fake data
    X = np.random.randn(5000 * class_size, 30, 100)
    y = (X.sum(axis=1)).sum(axis=1)
    y[y> 10] = 20
    y[(y<=10) & (y> -10)] = 10
    y[(y<=-10)] = 0
    y /= 10

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    clf = GRUClassifier(100, 128, class_size)
    clf.fit(X_train, y_train, batch_size=32, nb_epoch=50,
            validation_data=(X_test, y_test))
    score, acc = clf.evaluate(X_test, y_test)
    print('Test score:', score)
    print('Test accuracy:', acc)

    # torch.save(model, 'model.pt')
    # model = torch.load(PATH, weights_only=False)