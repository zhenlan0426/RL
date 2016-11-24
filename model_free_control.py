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


