import cython_loops, numpy as np, pandas as pd
from scipy.optimize import minimize
import warnings, os

class FNeigh:
    """
    Factorized Neighborhood Model

    Note
    ----
    This package only supports the full model version, with all possible
    item-item effects and full square matrices rather than smaller latent
    factor triplets.
    It can be run in multi-thread mode when fitting with methodd L-BFGS,
    but it will only use multithreading for calculating the objective and gradient,
    as the L-BFGS algorithm used is itself single threaded, so the speed up is
    not big.

    Parameters
    ----------
    center_ratings : bool
        Whether to subtract the global mean from the ratings
    alpha : float
        Parameter for weighting according to the number of ratings from each user
    lambda_bu : int or float
        Regularization for calculating the fixed user biases
    lambda_bi : int or float
        Regularization for calculating the fixed item biases
    lambda_u : float
        Regularization for the variable user biases (model parameters)
    lambda_i : float
        Regularization for the variable item biases (model parameters)
    lambda_W : float
        Regularization for the item-item effects on rating deviations
    lambda_C : float
        Regularization for the implicit item-item effects
    max_threads : int
        Maximum number of processor threads to use.
        If set to -1, will use the maximum available number of cores in the computer being run.
        If fitting with method SGD, will use only one for fitting, but will
        still use more for prediction.

    Attributes
    ----------
    user_dict : dict
        Mapping from user IDs as passed in the data to position in the model parameters
    item_dict : dict
        Mapping from item IDs as passed in the data to position in the model parameters
    is_fitted : bool
        Indicator telling whether the model has been fit to some data
    W_ : array (n_items**2)
        Model coefficients for rating deviations (flattened - reshape as (N_items, N_items) for further use)
    C_ : array (n_items**2)
        Model coefficients for implicit effects (flattened - reshape as (N_items, N_items) for further use)
    bu_ : array (n_users,)
        Model coefficients for variable user bias
    bi_ : array (n_items,)
        Model coefficients for variable item bias
    bias_user_fixed : array (n_users,)
        Fixed user bias (used to calculate deviations)    
    bias_item_fixed : array (n_items,)
        Fixed item bias (used to calculate deviations)

    References
    ----------
    * Factor in the neighbors: Scalable and accurate collaborative filtering (Koren, Y. 2010)
    """
    def __init__(self, center_ratings=True, alpha=0.5, lambda_bu=10, lambda_bi=25,
                 lambda_u=5e-1, lambda_i=5e-2, lambda_W=5e-3, lambda_C=5e-2, max_threads=-1):
        
        ## checking input
        assert (isinstance(alpha, float) or isinstance(alpha, int))
        assert (isinstance(lambda_bu, float) or isinstance(lambda_bu, int))
        assert (isinstance(lambda_bi, float) or isinstance(lambda_bi, int))
        assert isinstance(lambda_u, float)
        assert isinstance(lambda_i, float)
        assert isinstance(lambda_W, float)
        assert isinstance(lambda_C, float)
        assert isinstance(center_ratings, bool)
        assert isinstance(max_threads, int)
        
        ## remembering parameters
        self.center_ratings = center_ratings
        self.alpha = alpha
        self.lambda_bu = lambda_bu
        self.lambda_bi = lambda_bi
        self.lambda_u = lambda_u
        self.lambda_i = lambda_i
        self.lambda_W = lambda_W
        self.lambda_C = lambda_C
        
        ## data to fill in later
        self.user_dict = None
        self.item_dict = None
        
        self.is_fitted = False
        
        ## model parameters
        self.W_ = None
        self.C_ = None
        self.bu_ = None
        self.bi_ = None
        self.bias_item_fixed = None
        self.bias_user_fixed = None

        ## module was made in linux, assuming np.int64 and int are C_long
        ## in windows, this is not the case, so I'll cast all int variables
        try:
            test_long(np.arange(10))
            self._cast_long = False
        except:
            self._cast_long = True

        if max_threads == -1:
            max_threads = os.cpu_count()
            if max_threads is None:
                max_threads = 1
        if self._cast_long:
            max_threads = cython_loops.cast_py(max_threads)
        self.max_threads = max_threads
    
    def fit(self, ratings_df, method='lbfgs', opts_lbfgs={'maxiter':300},
            epochs=100, step_size=5e-3, decrease_step=True, early_stop=True, random_seed=None,
            save_data=True, verbose=False):
        """
        Fits the model to explicit ratings data

        Note
        ----
        The model allows fitting using L-BFGS-B, which calculates the full gradient
        and objective function in order to update the parameters at each iteration,
        and using SGD, which iterates through the data updating the parameters right after
        processing each individual rating.

        SGD has lower memory requirements and is able to achieve a low error in
        less time, but it likely won't reach the optimal solution and requires tuning
        the step size and other parameters. This  is likely an appropriate alternative for
        large datasets.

        L-BFGS-B requires more memory, and is slower to converge, but it always reaches
        the optimal solution and doesn't require playing with parameters like the step size.
        Recommended for smaller datasets.

        You can store the training data in the model object, which allows for faster
        recomendations for the users in the training data, and for predictions on a test set.
        Otherwise, it can only produce top-N recomended lists, as it doesn't have the user
        biases needed for a full rating prediction.

        Parameters
        ----------
        ratings_df : data frame or array (n_ratings, 3)
            Ratings data, including which users rated which items as how good.
            If an array, will assume that the columns are in this order: Rating, User, Item
            If a data frame, must have columns named 'Rating', 'UserId', 'ItemId'
        method : str
            One of 'lbfgs' or 'sgd'
        opts_lbfgs : dict
            Additional options to pass to L-BFGS-B (see SciPy's minimize documentation)
        epochs : int
            (Only for SGD) Maximum number of epochs
        step_size : float
            (Only for SGD) Step size for parameter updates
        decrease_step : bool
            (Only for SGD) Whether to decrease step sizes after each iteration (see Note)
        early_stop : bool
            (Only for SGD) Whether to stop if the error reduction is too small after an epoch
        random_seed : None or int
            (Only for SGD) Random seed to use when shuffling the data at the beginning of each epoch
        save_data : bool
            Whether to store the training data in the model object.
            You'll need this if you want to make predictions on a test set,
            or if you want to make faster recomendations for users in the training data
        verbose : bool
            Whether to print convergence messages after each iteration/epoch.
            If running on an IPython notebook, for L-BFGS-B it will print on the console
            but not on the notebook (this comes from SciPy's implementation)

        Returns
        -------
        self : obj
            Copy of this same object
        """
        
        ## checking input
        if method == 'SGD':
            method = 'sgd'
        if (method == 'L-BFGS') or (method == 'LBFGS') or (method == 'L-BFGS-B'):
            method = 'lbfgs'
        assert (method == 'sgd') or (method == 'lbfgs')
        if epochs is not None:
            assert isinstance(epochs, int)
            assert epochs>0
        if step_size is not None:
            assert isinstance(step_size, float)
            
        assert isinstance(save_data, bool)
        assert isinstance(verbose, bool)
        
        self._process_data(ratings_df)
        del ratings_df
        self._calculate_initial_params()
        
        if method == 'sgd':
            self._fit_sgd(step_size, epochs,
                          decrease_step, early_stop,
                          random_seed, verbose)
        else:
            self._fit_lbfgs(opts_lbfgs, verbose)
            
        self.save_data = save_data
        if self.save_data:
            pass
        else:
            del self._traindata
            del self._n_rated_by_user
            del self._st_ix_user
            
        self.is_fitted = True
        return self
    
    def _process_data(self, ratings_df):
        # checking input
        if isinstance(ratings_df, np.ndarray):
            assert len(ratings_df.shape) > 1
            assert ratings_df.shape[1] >= 3
            ratings_df = ratings_df.as_matrix()[:,:3]
            ratings_df.columns = ['Rating', 'UserId', 'ItemId']
        
        if ratings_df.__class__.__name__ == 'DataFrame':
            assert ratings_df.shape[0] > 0
            assert 'Rating' in ratings_df.columns.values
            assert 'UserId' in ratings_df.columns.values
            assert 'ItemId' in ratings_df.columns.values
            ratings_df = ratings_df[['Rating', 'UserId', 'ItemId']]
        else:
            raise ValueError("'ratings_df' must be a data frame or a numpy array")
        
        # reindexing users and items
        self.user_dict = dict()
        self.item_dict = dict()
        cnt_user = 0
        cnt_item = 0
        for i in ratings_df.itertuples():
            if i.UserId not in self.user_dict:
                self.user_dict[i.UserId] = cnt_user
                cnt_user += 1
            if i.ItemId not in self.item_dict:
                self.item_dict[i.ItemId] = cnt_item
                cnt_item += 1
        self._item_mapping = pd.DataFrame([(k,v) for k,v in self.item_dict.items()])
        self._item_mapping.rename(columns={0:'origId', 1:'newId'}, inplace=True)
        self._item_mapping.sort_values('newId', inplace=True)
        self._item_mapping = self._item_mapping.origId.values
        
        ratings_df['UserId'] = ratings_df.UserId.map(lambda x: self.user_dict[x])
        ratings_df['ItemId'] = ratings_df.ItemId.map(lambda x: self.item_dict[x])
        ratings_df.sort_values(['UserId','ItemId'], inplace = True)
        ratings_df.reset_index(drop = True, inplace = True)
        
        ratings_df['Rating'] = ratings_df.Rating.astype('float64')
        ratings_df['UserId'] = ratings_df.UserId.astype('int64')
        ratings_df['ItemId'] = ratings_df.ItemId.astype('int64')
        
        self.nusers = cnt_user
        self.nitems = cnt_item
        
        # other data needed for fitting
        self._n_rated_by_user = ratings_df.groupby('UserId')['ItemId'].agg(lambda x: len(x)).as_matrix()
        self._st_ix_user = np.cumsum(self._n_rated_by_user)
        self._st_ix_user = np.r_[[0], self._st_ix_user[:self._st_ix_user.shape[0]-1]]
        
        self._traindata = ratings_df.copy()
        self._nobs = self._traindata.shape[0]

        if self._cast_long:
            self._traindata['UserId'] = cython_loops.cast_np(self._traindata.UserId.values)
            self._traindata['ItemId'] = cython_loops.cast_np(self._traindata.ItemId.values)
            self._n_rated_by_user = cython_loops.cast_np(self._n_rated_by_user)
            self._st_ix_user = cython_loops.cast_np(self._st_ix_user)
            self._nobs = cython_loops.cast_py(self._nobs)
            self.nusers = cython_loops.cast_py(self.nusers)
            self.nitems = cython_loops.cast_py(self.nitems)
            self.user_dict = {k:cython_loops.cast_py(v) for k,v in self.user_dict.items()}
            self.item_dict = {k:cython_loops.cast_py(v) for k,v in self.item_dict.items()}

        return None
    
    def _calculate_initial_params(self):
        if self.center_ratings:
            self.global_mean = self._traindata.Rating.mean()
        else:
            self.global_mean = 0
        self._traindata['Rating_centered']=self._traindata.Rating - self.global_mean
        
        self.bias_item_fixed = self._traindata.groupby('ItemId')['Rating_centered'].agg(['sum','count'])\
            .apply(lambda x: x['sum']/(x['count'] + self.lambda_bi), axis=1)\
            .to_frame().rename(columns={0:'item_bias'})
        self._traindata = pd.merge(self._traindata, self.bias_item_fixed, left_on='ItemId', right_index=True)
        self._traindata.loc[:, 'Rating_centered_item'] = self._traindata.Rating_centered - self._traindata.item_bias
        self.bias_user_fixed = self._traindata.groupby('UserId')['Rating_centered_item'].mean()
        
        self.bias_user_fixed = self.bias_user_fixed.as_matrix().reshape(-1)
        self.bias_item_fixed = self.bias_item_fixed.as_matrix().reshape(-1)
        
        self._traindata.loc[:, 'Rating_centered_item_and_user'] = self._traindata.Rating_centered \
                            -self.bias_item_fixed[self._traindata.ItemId].reshape(-1)\
                            -self.bias_user_fixed[self._traindata.UserId].reshape(-1)
                
        del self._traindata['Rating']
        del self._traindata['Rating_centered_item']
        
        return None
    
    def _fit_sgd(self, step_size, epochs, decrease_step, early_stop, random_seed, verbose):
        self.W_ = np.zeros(self.nitems**2, dtype='float64')
        self.C_ = np.zeros(self.nitems**2, dtype='float64')
        self.bu_ = self.bias_user_fixed.astype('float64').copy()
        self.bi_ = self.bias_item_fixed.astype('float64').copy()
        
        if random_seed is None:
            rnd_seed = 0
            use_seed = 0
        else:
            use_seed = 1
            rnd_seed = random_seed

        if self._cast_long:
            self._traindata['UserId'] = cython_loops.cast_np(self._traindata.UserId.values)
            self._traindata['ItemId'] = cython_loops.cast_np(self._traindata.ItemId.values)
            self._n_rated_by_user = cython_loops.cast_np(self._n_rated_by_user)
            self._st_ix_user = cython_loops.cast_np(self._st_ix_user)
            self._nobs = cython_loops.cast_py(self._nobs)
            self.nusers = cython_loops.cast_py(self.nusers)
            self.nitems = cython_loops.cast_py(self.nitems)
            epochs = cython_loops.cast_py(epochs)
            decrease_step = cython_loops.cast_py(decrease_step)
            early_stop = cython_loops.cast_py(early_stop)
            verbose = cython_loops.cast_py(verbose)
            rnd_seed = cython_loops.cast_py(rnd_seed)
            use_seed = cython_loops.cast_py(use_seed)

        cython_loops.optimize_sgd(
            self._traindata.Rating_centered_item_and_user.values,
            self._traindata.Rating_centered.values,
            self._traindata.UserId.values,
            self._traindata.ItemId.values,
            self._n_rated_by_user,
            self._st_ix_user,
            self._nobs, self.nusers, self.nitems,
            self.W_, self.C_, self.bu_, self.bi_,
            self.alpha, self.lambda_u, self.lambda_i,
            self.lambda_W, self.lambda_C, step_size,
            epochs, decrease_step, early_stop, verbose,
            rnd_seed, use_seed
        )
    
    def _fit_lbfgs(self, opts, verbose):
        x0 = np.r_[np.zeros(self.nitems**2 ,dtype='float64'),
                   np.zeros(self.nitems**2 ,dtype='float64'),
                   self.bias_user_fixed.astype('float64').copy(),
                   self.bias_item_fixed.astype('float64').copy()]
        
        if self._cast_long:
            self._traindata['UserId'] = cython_loops.cast_np(self._traindata.UserId.values)
            self._traindata['ItemId'] = cython_loops.cast_np(self._traindata.ItemId.values)
            self._n_rated_by_user = cython_loops.cast_np(self._n_rated_by_user)
            self._st_ix_user = cython_loops.cast_np(self._st_ix_user)
            self._nobs = cython_loops.cast_py(self._nobs)
            self.nusers = cython_loops.cast_py(self.nusers)
            self.nitems = cython_loops.cast_py(self.nitems)

        fun_args = (
            self._traindata.Rating_centered.values,
            self._traindata.Rating_centered_item_and_user.values,
            self._traindata.UserId.values, self._traindata.ItemId.values,
            self._n_rated_by_user,
            self._st_ix_user,
            self._nobs, self.nusers, self.nitems,
            self.alpha,
            self.lambda_u, self.lambda_i,
            self.lambda_W, self.lambda_C,
            self.max_threads
        )
        
        # diagonals for W and C should be zero
        con = np.repeat(None, 2*x0.shape[0]).reshape((x0.shape[0], 2))
        con[np.arange(self.nitems) * (self.nitems+1), :] = 0
        con[2 * np.arange(self.nitems) * (self.nitems+1), :] = 0
            
        opts_dct = dict()
        if verbose:
            opts_dct['disp'] = True
            opts_dct['iprint'] = 1
        if opts is not None:
            opts_dct.update(opts)

        res = minimize(
             fun = cython_loops.calc_obj,
             x0 = x0,
             args = fun_args,
             method = 'L-BFGS-B',
             jac = cython_loops.calc_gradient,
             bounds = con,
             options = opts_dct
        )
        
        if not res['success']:
            warnings.warn('Optimization did not converge - model might not perform well')
        
        self.W_ = res['x'][:self.nitems**2].copy()
        self.C_ = res['x'][self.nitems**2:2*(self.nitems**2)].copy()
        self.bu_ = res['x'][2*(self.nitems**2):2*(self.nitems**2)+self.nusers].copy()
        self.bi_ = res['x'][2*(self.nitems**2)+self.nusers:].copy()

        del res
        del con
    
    def topN(self, ratings=None, items=None, n=10, uid=None):
        """
        Get Top-N highest-rated predictions according to the model

        Note
        ----
        This recomended list will filter-out items that were already rated.
        If saving the training data on the object, can also make predictions
        for users from the training data without having to provide their
        rating history again.

        Parameters
        ----------
        ratings : array-like
            Ratings used to make recomendations (ignored when passing uid)
        items : array-like
            Items that were rated (ignored when passing uid)
        n : int
            How many items to output
        uid : obj
            User from the training data for which to make recomendations

        Returns
        -------
        top-N : array (n,)
            Items not rated by the user, order from highest to lowest predicted rating
        """
        assert self.is_fitted
        
        ## case 1: predicting for a known user
        if uid is not None:
            assert self.save_data
            assert uid in self.user_dict
            uid = self.user_dict[uid]
            pred = np.zeros(self.nitems, dtype='float64')
            cython_loops.recommend_all_known_user(
                pred,
                self.W_, self.C_, self.bu_, self.bi_,
                self._traindata.Rating_centered_item_and_user.values,
                self._traindata.ItemId.values,
                self._n_rated_by_user,
                self._st_ix_user,
                self.nitems, uid, self.alpha,
                self.max_threads
            )
            
        ## case 2: predicting for an unknown user
        else: 
            if isinstance(ratings, list) or isinstance(ratings, tuple):
                ratings = np.array(ratings)
            if isinstance(items, list) or isinstance(items, tuple):
                items = np.array(items)
                
                
            if isinstance(ratings, np.ndarray) or ratings.__class__.__name__=='Series':
                if ratings.__class__.__name__=='Series':
                    ratings = ratings.values
                if len(ratings.shape) > 1:
                    ratings = ratings.reshape(-1)
                assert ratings.shape[0] > 0
            else:
                raise ValueError("'ratings' and 'items' must be numpy arrays or pandas series")
                
            if isinstance(items, np.ndarray) or items.__class__.__name__=='Series':
                if items.__class__.__name__=='Series':
                    items = items.values
                if len(items.shape) > 1:
                    items = items.reshape(-1)
                assert items.shape[0] > 0
            else:
                raise ValueError("'ratings' and 'items' must be numpy arrays or pandas series")
                
            assert ratings.shape[0] == items.shape[0]
            
            n_thisuser = items.shape[0]
            if self._cast_long:
                n_thisuser = cython_loops.cast_py(n_thisuser)
            iid = np.array([self.item_dict[i] for i in items if i in self.item_dict])
            if iid.shape[0] == 0:
                raise ValueError("None of the items rated were in the training set")
            Rd = ratings - self.global_mean - self.bias_item_fixed[iid]
            ## user bias has to be calculated on-the-fly
            bias_thisuser = Rd.sum()/(n_thisuser + self.lambda_bu)
            Rd = Rd - bias_thisuser
            
            pred = np.zeros(self.nitems, dtype='float64')
            cython_loops.recommend_from_ratings(
                pred,
                self.W_, self.C_, self.bi_,
                Rd,
                iid,
                n_thisuser,
                self.nitems,
                self.alpha,
                self.max_threads
            )
            
        ## sorting predictions from better to worse
        reclist = np.argsort(pred)
        
        ## eliminating already rated items
        chosen = list()
        if uid is None:
            set_rated = set(iid)
        else:
            st = self._st_ix_user[uid]
            end = st + self._n_rated_by_user[uid]
            set_rated = set(self._traindata.ItemId.values[st:end])

        for i in reclist:
            if i not in set_rated:
                chosen.append(i)
            if len(chosen) >= n:
                break
        
        ## outputting items in their original IDs
        return self._item_mapping[np.array(chosen)]
            
    
    def predict(self, uids, iids):
        """
        Predict ratings for a given list of combinations of users and items

        Note
        ----
        Can only make predictions for users and items that were in the training data.
        Required saving the data in the model object (see parameter in the `fit` method)

        Parameters
        ----------
        uids : array-like
            User IDs for which to make prediction of each item
        iids: array-like
            Item IDs for which to make predictions

        Returns
        -------
        pred : array (n_iids,)
            Predicted ratings for the combinations of users and items in the input
        """
        assert self.is_fitted
        assert self.save_data
        
        ## determining whether to predict one or multiple ratings
        if isinstance(uids, list) or isinstance(uids, tuple):
            uids = np.array(uids)
        if isinstance(iids, list) or isinstance(iids, tuple):
            iids = np.array(iids)
            
        if isinstance(uids, np.ndarray) or uids.__class__.__name__=='Series':
            if uids.__class__.__name__=='Series':
                uids = uids.values
            if iids.__class__.__name__=='Series':
                iids = iids.values
                
            if len(uids.shape) > 1:
                uids = uids.reshape(-1)
            if len(iids.shape) > 1:
                iids = iids.reshape(-1)
            assert uids.shape[0] == iids.shape[0]
                
            if uids.shape[0] == 1:
                multiple = False
                uid = uids[0]
                iid = iids[0]
            elif uids.shape[0] > 1:
                multiple = True
            else:
                raise ValueError("Invalid input - pass an array of user and item IDs")
                
        else:
            multiple = False
                
        ## case 1: predicting a list of ratings
        if multiple:
            try:
                uid = pd.Series(uids).map(lambda x: self.user_dict[x]).as_matrix()
            except:
                raise ValueError("Can only predict for users who were in the training data")
            
            try:
                iid = pd.Series(iids).map(lambda x: self.item_dict[x]).as_matrix()
            except:
                raise ValueError("Can only predict for items which were in the training data")

            nuids_pred = uids.shape[0]
            if self._cast_long:
                nuids_pred = cython_loops.cast_py(nuids_pred)
                
            pred = np.zeros(iid.shape[0], dtype='float64')
            cython_loops.predict_test_set(
                self.W_, self.C_, self.bu_, self.bi_,
                self._traindata.Rating_centered_item_and_user.values,
                self._traindata.ItemId.values,
                self._n_rated_by_user,
                self._st_ix_user,
                pred,
                uid, iid,
                nuids_pred, self.nitems, self.alpha,
                self.max_threads
            )
            return pred + self.global_mean
        
        ## case 2: predicting a single rating
        else:
            if uids.__class__.__name__=='ndarray':
                uids = uids[0]
            if iids.__class__.__name__=='ndarray':
                iids = iids[0]
            assert uids in self.user_dict
            assert iids in self.item_dict

            uid = self.user_dict[uids]
            iid = self.item_dict[iids]

            return cython_loops.predict_single_known_user(
                self.W_, self.C_, self.bu_, self.bi_,
                self._traindata.Rating_centered_item_and_user.values,
                self._traindata.ItemId.values,
                self._st_ix_user,
                self._n_rated_by_user,
                self.nitems, uid, iid, self.alpha
            )  + self.global_mean
