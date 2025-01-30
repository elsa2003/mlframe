import lightgbm as lgb

def prepare_lgbm():
    standard_params = {'n_estimators': 100, 'boosting_type': 'dart', 'num_leaves': 100, 'random_state': 61, 'reg_lambda': 0.6}
    return LGBMClassifier, standard_params



class LGBMClassifier(lgb.LGBMClassifier):
    """
    A wrapper around LightGBM's LGBMClassifier to easily save and load models.
    """
        
    def fit(self, X, y, validation_data=(), sample_weight=None):
        return super().fit(X, y, eval_set=[validation_data], eval_metric = 'multi_error', callbacks = [lgb.early_stopping(self.patience, min_delta = self.min_delta)])

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        loss = self.score(X, y)
        acc = self._accuracy(y_pred, y)
        return loss, acc

    def _accuracy(self, y_pred, y):
        return sum(y_pred == y) / y.shape[0]

    def save(self, path: str) -> None:
        """
        Saves the model instance to the specified file path.
        """
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        """
        Loads a saved model from the specified file path and
        returns an instance of LGBMClassifier.
        """
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(
                f"Loaded object must be an instance of {cls.__name__}, "
                f"but got {type(model)} instead."
            )
        return model