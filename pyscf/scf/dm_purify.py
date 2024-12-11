from scipy.linalg  import sqrtm, logm
from numpy         import zeros, eye, size, ones, sqrt, log, sqrt, pi, roots, exp
from numpy         import trace, dot, float64, linalg, array, transpose, asarray, diag
from scipy.special import erf
import sys
import numpy as np

from numpy.linalg import matrix_power as mat_pow
from numpy        import matmul       as mat_mul
from numpy        import trace
import numpy
from scipy.sparse import csr_matrix

def invsqrt_ovlp_diag(S):
    S_eigval, S_eigvec = np.linalg.eigh(S)
    return np.matmul(S_eigvec,np.matmul(np.diag(S_eigval**(-0.5) ), S_eigvec.T))


def hpcp_guess(H,N,Ne,*args):
    
    I  = eye(N, N)    
    
    epsi_0   = epsi_min(H,N) 
    epsi_N   = epsi_max(H,N) 

    mu = trace(H) / float(N)

    theta = float64(Ne / N)

    lambd1 = float64(    Ne) / ( N*(epsi_N - mu) )
    lambd2 = float64(N - Ne) / ( N*(mu - epsi_0) )
    
    beta1 = theta
    beta2 = min(lambd1, lambd2)
    
    X0 = beta1*I + beta2*(mu*I - H) 

    return X0


def hpcp(X,Ne):  
    
    
    c = float64(0.0)
    c1= float64(0.0)
    c2= float64(0.0)    
    
    #X = csr_matrix(X)
    #X_2 = mat_pow(X,2)  
    #X_3 = mat_mul(X_2,X)  
    X_2 = X @ X
    X_3 = X_2 @ X
    
    c1 = float64( trace(X_2 - X_3) )
    c2 = float64( trace(X   - X_2) )   

    #c1 = float64( csr_matrix.trace(X_2 - X_3) )
    #c2 = float64( csr_matrix.trace(X   - X_2) )   

    if ( c1 < 1e-6 ):
        c = float64(0.50)

    else:
        c = c1/c2
  
    X  = X + float64(2.0) * ( X_2 - X_3 - c * (X - X_2) )

    p = [c1,c2,c,0.0,0.0,0.0,0.0]
    #X = numpy.array(X.toarray()) #; print(numpy.shape(X))
    return X, 0, p

def hpcp_purify(X0,Ne,thr=1e-8,maxiter=50):
    
    threshold = thr
    test = threshold*10
    maxiter = 50 ; iter_ = 0
    X = X0
    
    while ( test > threshold ) and ( iter_ < maxiter ):
        old_X = X
        
        X, diag, p = hpcp(X,Ne)
        
        test = numpy.linalg.norm(X - old_X, ord='fro')
        #print(test,numpy.shape(X),type(X))
        #print(test,numpy.trace(X))
        iter_ += 1
        
    return X, iter_

def epsi_max(W,n):
    #
    v = zeros(n) 
    #
    for i in range(n):
        #
        sum = float64(0)
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
    v = zeros(n)
    #
    for i in range(n):
        #
        sum = float64(0) 
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
