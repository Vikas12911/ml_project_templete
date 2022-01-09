from sklearn import metrics as skmetrics
import numpy as np
class Loss:
    def __init__(self):
        self.metrics = {
            'mae' : self._mae , 
            'mse' : self._mse ,
            'rmse' : self._rmse ,
            'msle' : self._msle ,
            'rmsle' : self._rmsle , 
            'r2' : self._r2
        }

    def __call__(self , metric , y_true , y_pred):
        if metric not in self.metrics:
            raise Expection("Metric not implemented")
        if metric == 'mae':
            return self._mae(y_true , y_pred)
        if metric == 'mse':
            return self._mse(y_true , y_pred)
        if metric == 'rmse':
            return self._rmse(y_true , y_pred)
        if metric == 'msle':
            return self._msle(y_true , y_pred)
        if metric == 'rmsle':
            return self._rmsle(y_true , y_pred)
        if metric == 'r2':
            return self._r2(y_true , y_pred)

    @staticmethod
    def _mae(y_true , y_pred):
        return skmetrics.mean_absolute_error(y_true, y_pred)

    @staticmethod
    def _msle(y_true , y_pred):
        return skmetrics.mean_squared_log_error(y_true, y_pred)
    
    @staticmethod
    def _mse(y_true , y_pred):
        return skmetrics.mean_squared_error(y_true, y_pred)

    @staticmethod
    def _r2(y_true , y_pred):
        return skmetrics.r2_score(y_true, y_pred)
    
    def _rmse(self , y_true , y_pred):
        return np.sqrt(self._mse(y_true, y_pred))
    
    def _rmsle(self , y_true , y_pred):
        return np.sqrt(self._msle(y_true, y_pred))