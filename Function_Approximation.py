
def qLearnWithFA(env, episodes, estimator, discount=1.0, trials=1000 ,\
                 epsilon=0.2,gamma=0.05,lambda_=0.5,randomTrials=10,sampleSize=5000):
    # always resample from env, instead of following the path so that we know the total size
    # of experience. Use TD(0) off-policy update and experience replay, where the "on-policy"
    # follows epsilon-greedy of current estimator(s,a).
    # estimator needs to have fit with mask/predict method and could do multiple output prediction, one
    # for each action.
    A = env.action_space.n
    sDim = env.sDim # needs to implement this attribute in env
    aDim = env.aDim # needs to implement this attribute in env
    experience = np.zeros((episodes*trials,2*sDim+aDim+2)) # each dim corresponds to(s,a,r,s',done)
    
    index_ = 0 # track number of experience
    for i in range(episodes):
        for j in range(trials): # "on-policy" exploration
            s=env.reset() # restart s each trial
            
            if i < randomTrials: # random sample at the begining              
                a = env.action_space.sample() 
            else: # epsilon-greedy         
                a = np.argmax(estimator.predict(np.reshape(np.array(s),(1,-1))),1)[0] \
                        if np.random.rand()>epsilon else env.action_space.sample()
            s_next, r, done, _ = env.step(a)

            experience[index_,:sDim]=s
            experience[index_,sDim:sDim+aDim]=a
            experience[index_,sDim+aDim]=r
            experience[index_,sDim+aDim+1:sDim+aDim+1+sDim]=s_next
            experience[index_,sDim+aDim+1+sDim]=done                    
            index_+=1 
        
        # off-policy update estimator
        if index_ >= sampleSize:
            shuffleIndex=np.random.permutation(index_) # shuffle index for subsampling
            experience[:index_] = experience[shuffleIndex]
            X = experience[:sampleSize,:sDim]
            y = np.where(experience[:sampleSize,sDim+aDim+1+sDim]==1,\
                         experience[:sampleSize,sDim+aDim],\
                         experience[:sampleSize,sDim+aDim]+\
                         discount*np.max(estimator.predict(experience[:sampleSize,sDim+aDim+1:sDim+aDim+1+sDim]),1)
                         )
            mask = experience[:sampleSize,sDim] # works only for aDim=1
            estimator.fit(X,y,mask.astype('int64'))
        
    return estimator

class GBM(BaseEstimator, ClassifierMixin):
    def __init__(self,BaseEst,M_est,learnRate,BasePara,subFold):
        self.BaseEst=BaseEst
        self.M_est=M_est
        self.learnRate=learnRate
        self.estimator_=[]
        self.BasePara=BasePara
        self.subFold=subFold
        self.init_ = .0

    # Incorporate mask for RL     
    def fit(self,X,y,mask):
        n=y.shape[0]
        k_ = np.max(mask) + 1 # action starts at 0
        self.init_ = np.array([np.mean(y[mask==i]) for i in range(k_)])
        yhat = np.broadcast_to(self.init_, (n,k_))[np.arange(n),mask]
        kf = KFold(n, n_folds=self.subFold)
        for i in range(self.M_est):
            index=np.random.permutation(n) # shuffle index for subsampling
            X,y,yhat,mask = X[index,:],y[index],yhat[index],mask[index]
            for _,test in kf:
                target = np.zeros((test.shape[0],k_))
                target[np.arange(test.shape[0]),mask[test]] = y[test] - yhat[test]

                self.estimator_.append(self.BaseEst(**self.BasePara).fit(X[test],target))
                yhat+=self.learnRate*self.estimator_[-1].predict(X)[np.arange(n),mask]
        return self  
          
    def predict(self,X):
        yhat=np.reshape(self.init_,(1,-1))
        for clf in self.estimator_:
            yhat= yhat + self.learnRate*clf.predict(X)
        return yhat
    
#q_FA = qLearnWithFA(env1, 1000, GBM(DecisionTreeRegressor,100,0.1,{'max_depth':4,'splitter':'random','max_features':1},2))
