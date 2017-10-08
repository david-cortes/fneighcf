import pandas as pd, numpy as np, warnings

class FNeigh:
    """Recommender system based on parameterized Item-Item effects.
    
    Note
    ----
    All users and items are reindexed internally, so you can use non-integer IDs and don't have to worry about enumeration gaps.
    
    The model parameters are estimated using gradient descent (can also use stochastic gradient descent if desired).
    The objective function doesn't standardize the rating prediction error according to the number of ratings,
    thus the regularization parameters need to be set to larger numbers for larger datasets.

    Parameters
    ----------
    use_biases : bool
        Whether the model should use user and item rating biases as parameters.
        If set to False, the ratings are centered by subtracting the mean rating from each user.
        If set to True, the ratings are centered by subtracting the global mean and the user and item bias,
        which become model parameters.
        Setting it to false results in final recommendations for users being more varied.
    norm_nratings : float
        Normalization parameter for the number of ratings per user.
        Setting it to zero means no normalization by number of ratings.
    reg_param_biases : float
        Regularization parameter for the user and item biases in the model. Ignored when use_biases=False.
    reg_param_interactions : float
        Regularization parameter for the item interaction parameters in the model.
    save_ratings : bool
        Whether the model should store the ratings used to construct it.
        Allows for faster computation of recommendations for existing users.
    """
    
    def __init__(self, use_biases=True,norm_nratings=-0.5, reg_param_biases=1e-3, reg_param_interactions=1e-1, save_ratings=True):
        self._use_biases=use_biases
        self._reg_param_biases=reg_param_biases
        self._reg_param_interactions=reg_param_interactions
        self._norm_nratings=norm_nratings
        self._save_ratings=save_ratings
    
    def _create_df(self,ratings):
        self._user_orig_to_int=dict()
        self._item_orig_to_int=dict()
        self._item_int_to_orig=dict()
        
        if type(ratings)==list:
            self._ratings_df=pd.DataFrame(ratings,columns=['UserId','ItemId','Rating'])
        elif type(ratings)==pd.core.frame.DataFrame:
            if ('UserId' not in ratings.columns.values) or ('ItemId' not in ratings.columns.values) or ('Rating' not in ratings.columns.values):
                raise ValueError("Ratings data frame must contain the columns 'UserId','ItemId' and 'Rating'")
            self._ratings_df=ratings[['UserId','ItemId','Rating']].copy()
        else:
            raise ValueError("Ratings must be a list of tuples or pandas data frame")
            
        cnt_users=0
        cnt_items=0
        for i in self._ratings_df.itertuples():
            if i.UserId not in self._user_orig_to_int:
                self._user_orig_to_int[i.UserId]=cnt_users
                cnt_users+=1
            if i.ItemId not in self._item_orig_to_int:
                self._item_orig_to_int[i.ItemId]=cnt_items
                self._item_int_to_orig[cnt_items]=i.ItemId
                cnt_items+=1
                
        self._ratings_df['UserId']=self._ratings_df.UserId.map(lambda x: self._user_orig_to_int[x])
        self._ratings_df['ItemId']=self._ratings_df.ItemId.map(lambda x: self._item_orig_to_int[x])
        self._ratings_df=self._ratings_df.reset_index(drop=True)
        
        self.nusers=cnt_users
        self.nitems=cnt_items
    
    def _initiate_parameters(self,ratings_df):
        nratings_per_user=ratings_df.groupby('UserId')['Rating'].count()
        self._rating_norm_per_user=(nratings_per_user**(self._norm_nratings))
        self._items_rated_per_user=ratings_df.groupby('UserId')['ItemId'].aggregate(lambda x: list(x))
        self._indexes_per_user=ratings_df.groupby('UserId')['ItemId'].aggregate(lambda x: list(x.index))
        self.err_track=list()
        
        self._item_interactions=np.zeros((self.nitems,self.nitems))
        self._item_interactions_passive=np.zeros((self.nitems,self.nitems))
        
        if self._use_biases:
            self._global_bias=ratings_df.Rating.mean()
            self._ratings_df['dev']=self._ratings_df.Rating-self._global_bias
            self._user_bias=self._ratings_df.groupby('UserId')['dev'].mean()
            self._ratings_df=pd.merge(self._ratings_df,self._user_bias.to_frame().rename(columns={'dev':'user_bias'}),left_on='UserId',right_index=True,how='left')
            self._ratings_df['dev']=self._ratings_df.Rating-self._global_bias-self._ratings_df.user_bias
            self._item_bias=self._ratings_df.groupby('ItemId')['dev'].mean()
            del self._ratings_df['user_bias']
        else:
            avg_rating_by_user=self._ratings_df.groupby('UserId')['Rating'].mean().to_frame().rename(columns={'Rating':"AvgUserRating"})
            self._global_bias=0.0
            self._item_bias=pd.Series([0.0]*self.nitems)
            self._user_bias=avg_rating_by_user.AvgUserRating
            self._ratings_df=pd.merge(self._ratings_df,avg_rating_by_user,left_on='UserId',right_index=True,how='left')
            self._ratings_df['dev']=self._ratings_df.Rating-self._ratings_df.AvgUserRating
            
    def _calculate_errors(self,verbose=False,iteration=None):
        if self._use_biases:
            self._ratings_df['dev']=self._ratings_df.Rating-self._ratings_df.apply(lambda x: self._user_bias[x['UserId']]+self._item_bias[x['ItemId']],axis=1)
            if verbose:
                print('Iteration',str(iteration+1))
                print('RMSE after biases only:',np.sqrt(self._ratings_df.dev.map(lambda x: x**2).mean()))

        self._ratings_df['implicit']=self._ratings_df.apply(lambda x: self._rating_norm_per_user[x['UserId']]*np.sum(self._item_interactions_passive[int(x['ItemId']),self._items_rated_per_user[x['UserId']]]),axis=1)
        self._ratings_df['explicit']=self._ratings_df.apply(lambda x: self._rating_norm_per_user[x['UserId']]*np.sum(self._item_interactions[int(x['ItemId']),self._items_rated_per_user[x['UserId']]]*self._ratings_df.dev.loc[self._indexes_per_user[x['UserId']]]),axis=1)
        self._ratings_df['err']=self._ratings_df.dev-self._ratings_df.implicit-self._ratings_df.explicit
        
        self.err_track.append(np.sqrt((self._ratings_df.err**2).mean()))
        if verbose:
            if not self._use_biases:
                print('Iteration',str(iteration+1))
            print('RMSE before update: ',self.err_track[-1])
            print('')
            
    def _update_step_sizes(self,iteration,step_size_biases,step_size_interactions,decrease_step_frac,decrease_step_sqrt):
        if decrease_step_sqrt:
            step_size_biases=step_size_biases/np.sqrt(iteration+2)
            step_size_interactions=step_size_interactions/np.sqrt(iteration+2)
        elif decrease_step_frac:
            step_size_biases*=decrease_step_frac
            step_size_interactions*=decrease_step_frac
            
        return step_size_biases,step_size_interactions


    def _sgd_fit_w_biases(self,maxiter,step_size_biases,step_size_interactions,decrease_step_frac,decrease_step_sqrt):
        for iteration in range(maxiter):
            for i in self._ratings_df.sample(frac=1).itertuples():
                ###### calculating errors
                deviations=(self._ratings_df.Rating[self._indexes_per_user[i.UserId]]-self._user_bias[i.UserId]-self._item_bias[i.ItemId]).values.reshape(-1)
                implicit_eff=self._rating_norm_per_user[i.UserId]*np.sum(self._item_interactions[i.ItemId,self._items_rated_per_user[i.UserId]]*deviations)
                explicit_eff=self._rating_norm_per_user[i.UserId]*np.sum(self._item_interactions_passive[i.ItemId,self._items_rated_per_user[i.UserId]])
                err=i.Rating-self._user_bias[i.UserId]-self._item_bias[i.ItemId]-implicit_eff-explicit_eff

                ###### updating parameters
                self._user_bias+=step_size_biases*(err-self._reg_param_biases*self._user_bias)
                self._item_bias+=step_size_biases*(err-self._reg_param_biases*self._item_bias)
                self._item_interactions[i.ItemId,self._items_rated_per_user[i.UserId]]+=step_size_interactions*(self._rating_norm_per_user[i.UserId]*err*deviations-self._reg_param_interactions*self._item_interactions[i.ItemId,self._items_rated_per_user[i.UserId]])
                self._item_interactions_passive[i.ItemId,self._items_rated_per_user[i.UserId]]+=step_size_interactions*(self._rating_norm_per_user[i.UserId]*err-self._reg_param_interactions*self._item_interactions_passive[i.ItemId,self._items_rated_per_user[i.UserId]])

                self._item_interactions[i.ItemId,i.ItemId]=0
                self._item_interactions_passive[i.ItemId,i.ItemId]=0

        step_size_biases,step_size_interactions=self._update_step_sizes(iteration,step_size_biases,step_size_interactions,decrease_step_frac,decrease_step_sqrt)

    
    def _sgd_fit_wo_biases(self,maxiter,step_size_biases,step_size_interactions,decrease_step_frac,decrease_step_sqrt):
        for iteration in range(maxiter):
            for i in self._ratings_df.sample(frac=1).itertuples():
                ###### calculating errors
                deviations=(self._ratings_df.dev.loc[self._indexes_per_user[i.UserId]]).values.reshape(1,-1)
                implicit_eff=self._rating_norm_per_user[i.UserId]*np.sum(self._item_interactions[i.ItemId,self._items_rated_per_user[i.UserId]]*deviations)
                explicit_eff=self._rating_norm_per_user[i.UserId]*np.sum(self._item_interactions_passive[i.ItemId,self._items_rated_per_user[i.UserId]])
                err=i.dev-implicit_eff-explicit_eff

                ###### updating parameters
                self._item_interactions[i.ItemId,self._items_rated_per_user[i.UserId]]+=step_size_interactions*(self._rating_norm_per_user[i.UserId]*err*deviations-self._reg_param_interactions*self._item_interactions[i.ItemId,self._items_rated_per_user[i.UserId]])
                self._item_interactions_passive[i.ItemId,self._items_rated_per_user[i.UserId]]+=step_size_interactions*(self._rating_norm_per_user[i.UserId]*err-self._reg_param_interactions*self._item_interactions_passive[i.ItemId,self._items_rated_per_user[i.UserId]])

                self._item_interactions[i.ItemId,i.ItemId]=0
                self._item_interactions_passive[i.ItemId,i.ItemId]=0

        step_size_biases,step_size_interactions=self._update_step_sizes(iteration,step_size_biases,step_size_interactions,decrease_step_frac,decrease_step_sqrt)
    
    def _gd_fit(self,maxiter,step_size_biases,step_size_interactions,decrease_step_frac,decrease_step_sqrt,verbose):
        for iteration in range(maxiter):
            
            ####### getting errors and checking stopping criterion
            self._calculate_errors(verbose,iteration)
            if len(self.err_track)>2:
                if self.err_track[-1]>=1.01*self.err_track[-2]:
                    warnings.warn("RMSE increased after gradient descent iteration, stopping optimization. Try decreasing the step sizes and/or adjusting the regularization parameters")
                    break
                if self.err_track[-1]>=.995*self.err_track[-2]:
                    break

            ####### parameters to update
            if self._use_biases:
                update_user_bias=self._ratings_df.groupby('UserId')['err'].sum()
                update_item_bias=self._ratings_df.groupby('ItemId')['err'].sum()
            update_item_interactions=np.zeros_like(self._item_interactions)
            update_item_interactions_passive=np.zeros_like(self._item_interactions_passive)

            ####### looped gradient calculation
            for i in self._ratings_df.itertuples():
                update_item_interactions[i.ItemId,self._items_rated_per_user[i.UserId]]+=self._rating_norm_per_user[i.UserId]*i.err*self._ratings_df.dev.loc[self._indexes_per_user[i.UserId]]
                update_item_interactions_passive[i.ItemId,self._items_rated_per_user[i.UserId]]+=self._rating_norm_per_user[i.UserId]*i.err

            ####### updating parameters
            if self._use_biases:
                self._user_bias+=step_size_biases*(update_user_bias-self._reg_param_biases*self._user_bias)
                self._item_bias+=step_size_biases*(update_item_bias-self._reg_param_biases*self._item_bias)
            self._item_interactions+=step_size_interactions*(update_item_interactions-self._reg_param_interactions*self._item_interactions)
            self._item_interactions_passive+=step_size_interactions*(update_item_interactions_passive-self._reg_param_interactions*self._item_interactions_passive)

            np.fill_diagonal(self._item_interactions,0)
            np.fill_diagonal(self._item_interactions_passive,0)
            step_size_biases,step_size_interactions=self._update_step_sizes(iteration,step_size_biases,step_size_interactions,decrease_step_frac,decrease_step_sqrt)
    
    def fit(self, ratings, maxiter=10, step_size_biases=1e-3, step_size_interactions=1e-2,
            decrease_step_frac=0.9, decrease_step_sqrt=True, use_sgd=False,verbose=False):
        """Fit a model to ratings data.

        Note
        ----
        Step sizes might require a lot of manual tuning depending on the specific data used and the size of the data.
        
        This is a pure python implementation, so fitting the model can get quite slow.
        As a reference point, on a regular desktop computer, it takes about 5 minutes to run one gradient descent update
        on the movielens-100k dataset, and about 1 hour to run one iteration over the movielens-1M.
        
        For larger datasets, it's recommended to switch to stochastic gradient descent instead (use_sgd=True),
        as it will require fewer passes over the data to converge.

        Parameters
        ----------
        ratings : data frame or list of tuples
            Ratings data to build the model.
            If a data frame is passed, it must contain the columns 'UserId','ItemId' and 'Rating'.
            If a list of tuples is passed, tuples must be in the format (UserId,ItemId,Rating)
        maxiter : int
            Maximum number of iterations. Will terminate earlier if the RMSE decreases too little.
            Larger data sets require fewer iterations than smaller ones.
        step_size_biases : float
            Step size for gradient descent updates of model biases.
            Ignored if the model was called with use_biases=False.
            Can also be set to zero to define biases with a heuristic, thus not becoming model parameters.
            This last option might make recommendations better when using more ratings that were not in the training data.
        step_size_interactions : float
            Step size for gradient descent updates of item interaction parameters.
        decrease_step_frac: float
            Fraction by which the step sizes will be reduced for each next iteration.
            Setting it to 1 means constant step sizes.
        decrease_step_sqrt: bool
            Whether to use, at each iteration, a step size equal to value/sqrt(iteration).
            'decrease_step_frac' is ignored when this is set to 'True'.
        use_sgd: bool
            If set to true, the model parameters will be updated immediately after iterating over each row.
            Recommended to adjust step sizes and regularization parameters when trying this approach.
        verbose: bool
            Whether to print the RMSE after each iteration. Not available when using stochastic gradient descent.
        """
        self._create_df(ratings)
        self._initiate_parameters(self._ratings_df)
        
        if self._use_biases:
            self._ratings_df['Rating']-=self._global_bias
        else:
            if verbose and not use_sgd:
                print('RMSE of null model: ',(self._ratings_df.Rating**2).mean())
                print('')
            
        if not use_sgd:
            self._gd_fit(maxiter,step_size_biases,step_size_interactions,decrease_step_frac,decrease_step_sqrt,verbose)
        else:
            if self._use_biases:
                self._sgd_fit_w_biases(maxiter,step_size_biases,step_size_interactions,decrease_step_frac,decrease_step_sqrt)
            else:
                self._sgd_fit_wo_biases(maxiter,step_size_biases,step_size_interactions,decrease_step_frac,decrease_step_sqrt)
            
        if not self._save_ratings:
            del self._indexes_per_user
            del self._items_rated_per_user
            del self._rating_norm_per_user
            del self._ratings_df
        else:
            if self._use_biases:
                self._ratings_df['Rating']+=self._global_bias
    
    def predict(self,user,item):
        """Predict what rating would a user give to an item.

        Note
        ----
        If you saved the ratings into the model and wish to do faster predictions over a list, use the 'predict_list' method.
        Function only available for existing users, and only if the ratings were saved into the model.
        The 'top_n' function with 'scores=True' can be used for faster recommended lists.

        Parameters
        ----------
        user
            User for which to make a prediction.
        item
            Item for which to predict a rating.

        Returns
        -------
        float
            Rating predicted by the model.
        """
        if user in self._user_orig_to_int:
            userId=self._user_orig_to_int[user]
        else:
            raise ValueError('Invalid User')
            
        if item in self._item_orig_to_int:
            itemId=self._item_orig_to_int[item]
        else:
            raise ValueError('Invalid Item')
            
        try:
            temp=self._ratings_df.Rating.loc[0]
        except:
            raise ValueError("'predict' only available when the ratings are saved in the model")
            
        return self._global_bias+self._user_bias[userId]+self._item_bias[itemId]+self._rating_norm_per_user[userId]*np.sum(self._item_interactions[itemId,self._items_rated_per_user[userId]]*self._ratings_df.dev.loc[self._indexes_per_user[userId]])+self._rating_norm_per_user[userId]*np.sum(self._item_interactions_passive[itemId,self._items_rated_per_user[userId]])
    
    def score(self,rating_history,item,user=None,use_item_bias=True):
        """Get the score a user give to an item. Similar to 'predict', but without adding user bias and global bias
        (thus the ranking of items by their predicted score is the same, but the predicted scores are not reflective of actual scores).

        Note
        ----
        If you saved the ratings into the model and wish to do faster predictions over a list, use the 'predict_list' method.
        Function only available for existing users, and only if the ratings were saved into the model.
        The 'top_n' function with 'scores=True' can be used for faster recommended lists.

        Parameters
        ----------
        rating_history: list of tuples, dict, or data frame
            Rating history of the user.
            If a list, must contain tuples in the format (ItemId,Rating).
            If a dict, must contain entries in the format {ItemId:Rating}.
            If a data frame, must contain columns 'ItemId','Rating'.
            If passing an empty list or only 1 element, will recommend the most popular items.
            Ratings for items that were not in the training set will be ignored.
        item
            Item for which to predict a rating.
        user
            User for which to make a prediction.
            If set to None, the user bias will be calculated as mean(Rating-global_bias) from the ratings provided here.
        use_item_bias: bool
            Whether to include the item bias (favoring overall better-rated items) when predicting ratings.
            Recommendations that don't take this into account are more diverse and more customized for each user,
            but overall not as good.

        Returns
        -------
        float
            Score assigned by the model.
        """
        # checking the input
        if user in self._user_orig_to_int:
            userId=self._user_orig_to_int[user]
        else:
            raise ValueError('Invalid User')
            
        if item in self._item_orig_to_int:
            itemId=self._item_orig_to_int[item]
        else:
            raise ValueError('Invalid Item')
        
        # putting the rating history in teh required format
        if type(rating_history)==list:
            if rating_history<2:
                return [self._item_int_to_orig[i] for i in self._item_bias.nlargest(n).index]
            rating_norm=len(rating_history)**self._norm_nratings
            ratings=pd.DataFrame([(i[0],i[1]) for i in rating_history],columns=['ItemId','Rating'])
        elif type(rating_history)==dict:
            ratings=pd.DataFrame.from_dict(rating_history,orient='index').reset_index().rename(columns={'index':'ItemId',0:'Rating'})
            rating_norm=rating_history.shape[0]**self._norm_nratings
        elif type(rating_history)==pd.core.frame.DataFrame:
            ratings=rating_history[['ItemId','Rating']]
            rating_norm=rating_history.shape[0]**self._norm_nratings
        else:
            raise ValueError("Ratings must be a list of tuples or pandas data frame")
        
        # reindexing items
        ratings['ItemId']=ratings.ItemId.map(lambda x: self._item_orig_to_int[x] if x in self._item_orig_to_int else np.nan)
        ratings=ratings.loc[~ratings.ItemId.isnull()].sort_values('ItemId')
        rated_items=list(ratings.ItemId)
        if len(rated_items)<2:
            warnings.warn('Fewer than two of the items rated had parameters in the model.')
            return self._item_bias[self._item_int_to_orig[itemId]]
        
        # calculating the score
        if self._use_biases:
            if user==None:
                user_bias=np.mean(ratings.Rating-self._global_bias)
            else:
                user_bias=self._user_bias[self._user_orig_to_int[user]]
                
            deviations=ratings.apply(lambda x: x['Rating']-self._global_bias-user_bias-self._item_bias[x['ItemId']],axis=1)
            if use_item_bias:
                item_bias=self._item_bias[itemId]
            else:
                item_bias=0
            return item_bias+rating_norm*np.sum(self._item_interactions[itemId,rated_items]*deviations.values.reshape(1,-1))+rating_norm*np.sum(self._item_interactions_passive[itemId,rated_items])
            
        else:
            user_bias=ratings.Rating.mean()
            deviations=ratings['Rating']-user_bias
            return rating_norm*np.sum(self._item_interactions[itemId,rated_items]*deviations.values.reshape(1,-1))+rating_norm*np.sum(self._item_interactions_passive[itemId,rated_items])
            
    
    def predict_list(self,combinations):
        """Predict the ratings for a list of user and item combinations.

        Note
        ----
        Function only available if the ratings were saved into the model.

        Parameters
        ----------
        combinations: data frame or list of tuples
            If a data frame is passed, it must contain the columns 'UserId' and 'ItemId'.
            If a list of tuples is passed, tuples must be in the format (UserId,ItemId).

        Returns
        -------
        list
            Predicted ratings for the input list of combinations of users and items.
        """
        # checking if the model had saved the ratings
        try:
            temp=self._ratings_df.Rating.loc[0]
        except:
            raise ValueError("'predict_list' only available when the ratings are saved in the model")
            
        # putting the input in the necessary form
        if type(combinations)==list:
            df=pd.DataFrame(combinations,columns=['UserId','ItemId'])
            df['UserId']=df.UserId.map(lambda x: self._user_orig_to_int[x])
            df['ItemId']=df.ItemId.map(lambda x: self._item_orig_to_int[x])
        elif type(combinations)==pd.core.frame.DataFrame:
            df=combinations[['UserId','ItemId']]
        else:
            raise ValueError('Input must be a list of tuples or a data frame')
        
        try:
            df['UserId']=df.UserId.map(lambda x: self._user_orig_to_int[x])
            df['ItemId']=df.ItemId.map(lambda x: self._item_orig_to_int[x])
        except:
            raise ValueError('One or more of the users and/or items in the list was not present in the training data.')
        
        # predicting the ratings
        df['PredictedRating']=self._global_bias+df.apply(lambda x: self._user_bias[x['UserId']]+self._item_bias[x['ItemId']],axis=1)
        df['PredictedRating']+=df.apply(lambda x: self._rating_norm_per_user[x['UserId']]*np.sum(self._item_interactions_passive[int(x['ItemId']),self._items_rated_per_user[x['UserId']]]),axis=1)
        df['PredictedRating']+=df.apply(lambda x: self._rating_norm_per_user[x['UserId']]*np.sum(self._item_interactions[int(x['ItemId']),self._items_rated_per_user[x['UserId']]]*self._ratings_df.dev.loc[self._indexes_per_user[x['UserId']]]),axis=1)
        
        return list(df.PredictedRating)
    
    def top_n_saved(self, user, n, use_item_bias=True, scores=False):
        """Get Top-N recommended items for a user, using the saved ratings to speed up computations.

        Note
        ----
        Function only available if the ratings were saved into the model.
        
        Parameters
        ----------
        user
            User for which to get recommended list.
        n: int
            Number of items to recommend.
        use_item_bias: bool
            Whether to include the item bias (favoring overall better-rated items) when predicting ratings.
            Recommendations that don't take this into account are more diverse and more customized for each user,
            but overall not as good.
        scores: bool
            Whether to include model scores into the output, or only IDs

        Returns
        -------
        list
            Top-N recommended items for the user.
            If 'scores=True', list of tuples containing (Item,Score)
        """
        # checking the input
        if user not in self._user_orig_to_int:
            raise ValueError('Invalid User')
        try:
            temp=self._ratings_df.Rating.loc[0]
        except:
            raise ValueError("'top_n_saved' only available when the ratings are saved in the model")
        
        # scoring the items
        userId=self._user_orig_to_int[user]
        if self._use_biases and use_item_bias:
            items=self._item_bias+self._rating_norm_per_user[userId]*np.sum(self._item_interactions[:,self._items_rated_per_user[userId]]*self._ratings_df.dev.loc[self._indexes_per_user[userId]].values.reshape(1,-1))+self._rating_norm_per_user[userId]*np.sum(self._item_interactions_passive[:,self._items_rated_per_user[userId]])
        else:
            items=pd.Series([0.0]*self.nitems)+self._rating_norm_per_user[userId]*np.sum(self._item_interactions[:,self._items_rated_per_user[userId]]*self._ratings_df.dev.loc[self._indexes_per_user[userId]].values.reshape(1,-1))+self._rating_norm_per_user[userId]*np.sum(self._item_interactions_passive[:,self._items_rated_per_user[userId]])
        
        # returning the items with highest scores
        already_rated=set(self._items_rated_per_user[userId])
        recc_items=list()
        if not scores:
            items=list(items.sort_values(ascending=False).index)
            for i in items:
                if i not in already_rated:
                    recc_items.append(self._item_int_to_orig[i])
                    if len(recc_items)>=n:
                        break
        else:
            items=items.sort_values(ascending=False)
            for i in items.iteritems():
                if i[0] not in already_rated:
                    recc_items.append((self._item_int_to_orig[i[0]],i[1]))
                    if len(recc_items)>=n:
                        break
        
        return recc_items
    
    def top_n(self, rating_history, n, user=None, use_item_bias=True, scores=False):
        """Get Top-N recommended items for a user from its rating history.

        Note
        ----
        If the user is new, will try to estimate user bias as:
            est_user_bias=mean(ratings-global_bias)
        If the step size for biases was set to zero, this is the same bias that was used to train the model.
        Can also produce a non-personalized most popular item list if called with an empty rating list, e.g.:
            top_n([]n=10,user=None)
        
        Parameters
        ----------
        rating_history: list of tuples, dict, or data frame
            Rating history of the user.
            If a list, must contain tuples in the format (ItemId,Rating).
            If a dict, must contain entries in the format {ItemId:Rating}.
            If a data frame, must contain columns 'ItemId','Rating'.
            If passing an empty list or only 1 element, will recommend the most popular items.
            Ratings for items that were not in the training set will be ignored.
        n: int
            Number of items to recommend.
        user
            User for which to make recommendations. If None, assumes a new user.
            If the user is not new and the model included biases, it will look up the bias for that user.
            Otherwise, it will estimate it from the rating history provided using a heuristic.
        use_item_bias: bool
            Whether to include the item bias (favoring overall better-rated items) when predicting ratings.
            Recommendations that don't take this into account are more diverse and more customized for each user,
            but overall not as good.
        scores: bool
            Whether to include model scores into the output, or only Item IDs

        Returns
        -------
        list
            Top-N recommended items for the user.
            If 'scores=True', list of tuples containing (Item,Score)
        """
        # checking input
        if type(rating_history)==list:
            if len(rating_history)<2:
                if self._use_biases:
                    return [self._item_int_to_orig[i] for i in self._item_bias.nlargest(n).index]
                else:
                    raise ValueError("Too few ratings. Recommendations by item popularity only available when using item biases")
            rating_norm=len(rating_history)**self._norm_nratings
            ratings=pd.DataFrame([(i[0],i[1]) for i in rating_history],columns=['ItemId','Rating'])
        elif type(rating_history)==dict:
            ratings=pd.DataFrame.from_dict(rating_history,orient='index').reset_index().rename(columns={'index':'ItemId',0:'Rating'})
            rating_norm=rating_history.shape[0]**self._norm_nratings
        elif type(rating_history)==pd.core.frame.DataFrame:
            ratings=rating_history[['ItemId','Rating']]
            rating_norm=rating_history.shape[0]**self._norm_nratings
        else:
            raise ValueError("Ratings must be a list of tuples or pandas data frame")
        
        # reindexing items
        ratings['ItemId']=ratings.ItemId.map(lambda x: self._item_orig_to_int[x] if x in self._item_orig_to_int else np.nan)
        ratings=ratings.loc[~ratings.ItemId.isnull()].sort_values('ItemId')
        rated_items=list(ratings.ItemId)
        if len(rated_items)<2:
            if self._use_biases:
                warnings.warn('Fewer than two of the items rated had parameters in the model. Recommending most popular items.')
                return [self._item_int_to_orig[i] for i in self._item_bias.nlargest(n).index]
            else:
                raise ValueError("Fewer than two of the items rated had parameters in the model. Recommendations by item popularity only available when using item biases")
        
        # getting bias when applicable
        if user==None:
            user_bias=np.mean(ratings.Rating-self._global_bias)
        else:
            user_bias=self._user_bias[self._user_orig_to_int[user]]

        # scoring the items
        if self._use_biases:
            deviations=ratings.apply(lambda x: x['Rating']-self._global_bias-user_bias-self._item_bias[x['ItemId']],axis=1)
            item_biases=self._item_bias
        else:
            deviations=ratings.Rating-user_bias
            item_biases=pd.Series([0.0]*self.nitems)
        items=item_biases+rating_norm*np.sum(self._item_interactions[:,rated_items]*deviations.values.reshape(1,-1),axis=1)+rating_norm*np.sum(self._item_interactions_passive[:,rated_items],axis=1)
        
        # returning the items with highest scores
        already_rated=set(rated_items)
        recc_items=list()
        if not scores:
            items=list(items.sort_values(ascending=False).index)
            for i in items:
                if i not in already_rated:
                    recc_items.append(self._item_int_to_orig[i])
                    if len(recc_items)>=n:
                        break
        else:
            items=items.sort_values(ascending=False)
            for i in items.iteritems():
                if i[0] not in already_rated:
                    recc_items.append((self._item_int_to_orig[i[0]],i[1]))
                    if len(recc_items)>=n:
                        break
        
        return recc_items
