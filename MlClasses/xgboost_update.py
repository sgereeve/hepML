import xgboost
import numpy as np

class XGBClassifier(xgboost.XGBClassifier):

    def __init__(self, separation_facet = 0.5,max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True,
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=1, nthread=None, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None, **kwargs):
        # print 'XGBClassifier __init__'
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.silent = silent
        self.objective = objective
        self.booster = booster
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.missing = missing if missing is not None else np.nan
        self.kwargs = kwargs
        self._Booster = None
        self.seed = seed
        self.random_state = random_state
        self.nthread = nthread
        self.n_jobs = n_jobs
        self.separation_facet = separation_facet # Add internal variable for classifier facet

    def predict_class(self, data, output_margin=False, ntree_limit=0):
        """
        Predict with `data`. 
        NOTE: This function is not thread safe.
              For each booster object, predict can only be called from one thread.
              If you want to run prediction using multiple thread, call xgb.copy() to make copies
              of model object and then call predict
        Parameters
        ----------
        data : DMatrix
            The dmatrix storing the input.
        output_margin : bool
            Whether to output the raw untransformed margin value.
        ntree_limit : int
            Limit number of trees in the prediction; defaults to 0 (use all trees).
        Returns
        -------
        prediction : numpy array
        """

        # Same changes were added to seperate signal with probabulity > self.separation_facet
        # print '____updated xgboost method______'
        # print 'predict_class was used!'
        test_dmatrix = xgboost.DMatrix(data, missing=self.missing, nthread=self.n_jobs)
        class_probs = self.get_booster().predict(test_dmatrix,
                                                 output_margin=output_margin,
                                                 ntree_limit=ntree_limit)
        if len(class_probs.shape) > 1:
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            column_indexes = np.repeat(0, class_probs.shape[0])
            column_indexes[class_probs > self.separation_facet] = 1
        #print "~~~~~~~~~~~INSIDE CLASS~~~~~~~~~~~",self.separation_facet
        return self._le.inverse_transform(column_indexes)

