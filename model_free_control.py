def state2act(q_s, epsilon):
    # q_s is a matrix, where each row is one action, 
    # first col is q(s,a) and second col is count(s,a)
    if np.random.rand()<epsilon:
        return np.argmin(q_s[:,1])
    else:
        return np.argmax(q_s[:,0])
    
def mc_control(env, episodes, discount=1.0, epsilon=0.1,gamma=0.05):
    dict_q = {}
    A = env.action_space.n
    for i in range(episodes):
        s=env.reset()
        done = False
        cum_r = []
        while not done:            
            if s not in dict_q:
                dict_q[s] = np.zeros((A,2))
            a = state2act(dict_q[s], epsilon)
            s_next, r, done, _ = env.step(a)
            cum_r.append([[s,a],[1,r]])
            for list_ in cum_r[:-1]:
                list_[1][0]*=discount
                list_[1][1]+=list_[1][0]*r
            s = s_next
            
        for [[s,a],[_, cum]] in cum_r:
            dict_q[s][a,0] = gamma*cum + (1-gamma)*dict_q[s][a,0]
            dict_q[s][a,1] += 1
    return dict_q
    
    
def td_control(env, episodes, discount=1.0, epsilon=0.1,gamma=0.05,lambda_=0.5):
    # implment TD lambda, dict_q is a dict that maps S -> matrix, where each 
    # row represents a action, and first col is the value q(s,a), second col 
    # tracks the counts of (s,a) have been visited.
    dict_q = {}
    A = env.action_space.n
    for i in range(episodes):
        s=env.reset()
        done = False
        E = {}
        while not done:            
            if s not in dict_q:
                dict_q[s] = np.zeros((A,2))
            a = state2act(dict_q[s], epsilon)
            s_next, r, done, _ = env.step(a)
            if done:
                
                delta = r  - dict_q[s][a,0]
                if (s,a) in E:
                    E[(s,a)] +=1
                else:
                    E[(s,a)] = 1
                    
                for (s,a),items in E.iteritems():
                    dict_q[s][a,0] += gamma*delta*items
                    dict_q[s][a,1] += 1
                    items *= lambda_*discount
            else:    
                
                if s_next not in dict_q:
                    dict_q[s_next] = np.zeros((A,2))
                a_next = state2act(dict_q[s_next], epsilon)

                delta = r + discount*dict_q[s_next][a_next,0] - dict_q[s][a,0]
                if (s,a) in E:
                    E[(s,a)] +=1
                else:
                    E[(s,a)] = 1
                for (s,a),items in E.iteritems():
                    dict_q[s][a,0] += gamma*delta*items
                    dict_q[s][a,1] += 1
                    items *= lambda_*discount

                s = s_next
                a = a_next
            
    return dict_q
    
#q1 = td_control(env, 100000) 
#M_true = np.zeros((18,10))
#M_False = np.zeros((18,10))

#for (sum_,deal,face), act in q1.iteritems():
#    if face:
#        M_true[sum_-4,deal-1] = np.argmax(act[:,0])
#    else:
#        M_False[sum_-4,deal-1] = np.argmax(act[:,0])    
    
    

def decay_fun_gen(power=1.0/3, start=1.0,base=1.0):
    def fun(i):    
        return start/(base+i**power)    
    return fun
    
def td_control2(env, episodes, epsilon,gamma, discount=1.0, lambda_=0.5):
    # implment TD lambda, dict_q is a dict that maps S -> matrix, where each 
    # row represents a action, and first col is the value q(s,a), second col 
    # tracks the counts of (s,a) have been visited.
    # epsilon is a fun that maps episode to epsilon    
    
    dict_q = {}
    A = env.action_space.n
    for i in range(episodes):
        s=env.reset()
        done = False
        E = {}
        while not done:            
            if s not in dict_q:
                dict_q[s] = np.zeros((A,2))
            a = state2act(dict_q[s], epsilon(i))
            s_next, r, done, _ = env.step(a)
            if done:
                
                delta = r  - dict_q[s][a,0]
                if (s,a) in E:
                    E[(s,a)] +=1
                else:
                    E[(s,a)] = 1
                    
                for (s,a),items in E.iteritems():
                    dict_q[s][a,0] += gamma(i)*delta*items
                    dict_q[s][a,1] += 1
                    items *= lambda_*discount
            else:    
                
                if s_next not in dict_q:
                    dict_q[s_next] = np.zeros((A,2))
                a_next = state2act(dict_q[s_next], epsilon(i))

                delta = r + discount*dict_q[s_next][a_next,0] - dict_q[s][a,0]
                if (s,a) in E:
                    E[(s,a)] +=1
                else:
                    E[(s,a)] = 1
                for (s,a),items in E.iteritems():
                    dict_q[s][a,0] += gamma(i)*delta*items
                    dict_q[s][a,1] += 1
                    items *= lambda_*discount

                s = s_next
                a = a_next
            
    return dict_q    
    
def mc_eval(env, episodes, policy, discount=1.0):
    # eval expected value over both s and a
    # policy is a dict maps from s to a
    val = 0
    
    for i in range(episodes):
        s=env.reset()
        done = False
        factor = 1 # discount
        while not done:         
            s, r, done, _ = env.step(policy[s])
            val += factor*r
            factor *= discount
    return val/episodes 

#q1 = td_control2(env1, 100000,decay_fun_gen(),decay_fun_gen(start=0.4))     
#val1 = mc_eval(env1,1000000,{key:np.argmax(item[:,0]) for key,item in q1.iteritems()})


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
