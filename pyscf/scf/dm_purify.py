from scipy.sparse import csr_matrix
import numpy as np

def invsqrt_ovlp_diag(S):
    S_eigval, S_eigvec = np.linalg.eigh(S)
    return np.matmul(S_eigvec,np.matmul(np.diag(S_eigval**(-0.5) ), S_eigvec.T))

def get_focktilde(F,Sinvsqrt):
    if (F.ndim == 2 ):  
        Ftilde = np.matmul(np.matmul(Sinvsqrt,F),Sinvsqrt)  
        return Ftilde
    elif(F.ndim == 3 ):   
        Ftilde_a = np.matmul(np.matmul(Sinvsqrt,F[0]),Sinvsqrt)  
        Ftilde_b = np.matmul(np.matmul(Sinvsqrt,F[1]),Sinvsqrt)  
        return np.array([Ftilde_a,Ftilde_b])

def get_dm(Dtilde,Sinvsqrt):
    if (Dtilde.ndim == 2 ):  
        D = np.matmul(np.matmul(Sinvsqrt,Dtilde),Sinvsqrt)  
        return D*2
    elif(Dtilde.ndim == 3 ):   
        D_a = np.matmul(np.matmul(Sinvsqrt,Dtilde[0]),Sinvsqrt)  
        D_b = np.matmul(np.matmul(Sinvsqrt,Dtilde[1]),Sinvsqrt)  
        return np.array([D_a,D_b])


def dm_purify(H,N,Ne,method,thr,maxiter):
    
    if (method == 'hpcp') :
    
        X0 = hpcp_guess(H,N,Ne)    
        X, niter = hpcp_purify(X0,Ne,thr=1e-10,maxiter=50)
        
    return X, niter

def hpcp_guess(H,N,Ne,*args):
    
    if (Ne[0] == Ne[1]):
        Ne = Ne[0]
    else:
        print('WARNING: Ne',Ne)
        
    I = np.eye(N, N)    

    #Restricted = 1 density matrix
    if ( H.ndim == 2 ):    
        
        # 
        epsi_0 = epsi_min(H,N) 
        epsi_N = epsi_max(H,N) 

        mu = np.trace(H) / float(N)

        theta = np.float64(Ne / N)

        lambd1 = np.float64(    Ne) / ( N*(epsi_N - mu) )
        lambd2 = np.float64(N - Ne) / ( N*(mu - epsi_0) )
    
        beta1 = theta
        beta2 = min(lambd1, lambd2)
    
        X0 = beta1*I + beta2*(mu*I - H) 

        return X0
    
    # Unrestricted = 2 density matrices   
    elif ( H.ndim == 3 ):    

        epsi_0_a  = epsi_min(H[0],N) 
        epsi_N_a  = epsi_max(H[0],N) 
        epsi_0_b  = epsi_min(H[1],N) 
        epsi_N_b  = epsi_max(H[1],N) 

        mu_a = np.trace(H[0]) / float(N)
        mu_b = np.trace(H[1]) / float(N)

        theta_a = np.float64(Ne[0] / N)
        theta_b = np.float64(Ne[1] / N)

        lambd1_a = np.float64(    Ne[0]) / ( N*(epsi_N_a - mu_a) )
        lambd2_a = np.float64(N - Ne[0]) / ( N*(mu_a - epsi_0_a) )
        lambd1_b = np.float64(    Ne[1]) / ( N*(epsi_N_b - mu_b) )
        lambd2_b = np.float64(N - Ne[1]) / ( N*(mu_b - epsi_0_b) )
    
        beta1_a = theta_a
        beta2_a = min(lambd1_a, lambd2_a)
        beta1_b = theta_b
        beta2_b = min(lambd1_b, lambd2_b)
    
        X0_a = beta1_a*I + beta2_a*(mu_a*I - H[0]) 
        X0_b = beta1_b*I + beta2_b*(mu_b*I - H[1]) 

        return np.array([X0_a, X0_b])
       
def hpcp(X,Ne):  
    
    
    c = np.float64(0.0)
    c1= np.float64(0.0)
    c2= np.float64(0.0)    
    
    #X = csr_matrix(X)

    X_2 = X @ X
    X_3 = X_2 @ X
    
    c1 = np.float64( np.trace(X_2 - X_3) )
    c2 = np.float64( np.trace(X   - X_2) )   

    #c1 = float64( csr_matrix.trace(X_2 - X_3) )
    #c2 = float64( csr_matrix.trace(X   - X_2) )   

    if ( abs(c1) < 1e-8 ):
        c = np.float64(0.50)

    else:
        c = c1/c2
  
    X  = X + 2 * ( X_2 - X_3 - c * (X - X_2) )

    p = [c1,c2,c,0.0,0.0,0.0,0.0]
    #X = numpy.array(X.toarray()) #; print(numpy.shape(X))
    return X, 0, p

def hpcp_purify(X0,Ne,thr=1e-8,maxiter=50):

    #Restricted = 1 density matrix
    if ( X0.ndim == 2 ): 
        threshold = thr
        test = threshold*10
        iter_ = 0
        X = X0
    
        while ( test > threshold ) and ( iter_ < maxiter ):
           old_X = X
        
           X, diag, p = hpcp(X,Ne)
        
           test = np.linalg.norm(X - old_X, ord='fro')
           #print(test,numpy.shape(X),type(X))
           #    print(test,numpy.trace(X))
           iter_ += 1
        
        #print(linalg.eigh(X))
        return X, iter_
    
    # Unrestricted = 2 density matrices
    elif ( X0.ndim == 3 ):  
        
        threshold = thr
        test_a = threshold*10
        test_b = threshold*10
        iter_a = 0
        iter_b = 0
        X_a = X0[0]
        X_b = X0[1]
    
        while ( test_a > threshold ) and ( iter_a < maxiter ):
           old_X_a = X_a
        
           X_a, diag_a, p_a = hpcp(X_a,Ne[0])
        
           test_a = np.linalg.norm(X_a - old_X_a, ord='fro')
           #occ, _ = linalg.eigh(X_a)
           #print('X_a',test_a,numpy.trace(X_a),p_a[0],p_a[1],p_a[2])
           #print(occ)
           #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
           #print(test_b,numpy.shape(X_b),type(X_b),numpy.trace(X_b))
           iter_a += 1

        while ( test_b > threshold ) and ( iter_b < maxiter ):
           old_X_b = X_b
        
           X_b, diag_b, p_b = hpcp(X_b,Ne[1])
        
           test_b = np.linalg.norm(X_b - old_X_b, ord='fro')
           #occ, _ = linalg.eigh(X_b)
           #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
           #print('X_b',test_b,numpy.trace(X_b))
           #print(occ)
           
           iter_b += 1
        #print('X_b',numpy.trace(X_b))
        #print('X_a',numpy.trace(X_a))
        #print(linalg.eigh(X_a))
        #print(linalg.eigh(X_b))
        
        return np.array([X_a,X_b]), [iter_a,iter_b]

    
def epsi_max(W,n):
    #
    v = np.zeros(n) 
    #
    for i in range(n):
        #
        sum = 0
        #
        for j in range(n):
            #
            if (i != j):
                #
                sum = sum + abs(W[i,j])
                
        v[i] = W[i,i] + sum   
        #        
    return v.max()#, y.argmax(), x[y.argmax(),y.argmax()]  
    #
#
#
#============================================================================    
def epsi_min(W,n):
    #    
    v = np.zeros(n)
    #
    for i in range(n):
        #
        sum = 0
        #
        for j in range(n):
            #
            if (i != j):
                #
                sum = sum + abs(W[i,j])                
            #
        #        
        v[i] = W[i,i] - sum   
        #
    return v.min()#, y.argmin(), x[y.argmin(),y.argmin()]