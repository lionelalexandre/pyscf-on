from scipy.linalg  import sqrtm, logm
from numpy         import zeros, eye, size, ones, sqrt, log, sqrt, pi, roots, exp
from numpy         import trace, dot, float64, linalg, array, transpose, asarray, diag
from scipy.special import erf
import sys
import numpy as np

def invsqrt_ovlp_diag(S):
    S_eigval, S_eigvec = np.linalg.eigh(S)
    return np.matmul(S_eigvec,np.matmul(np.diag(S_eigval**(-0.5) ), S_eigvec.T))


def hpcp_guess(H,N,Ne,eigval,occ,H_bounds,*args):
    
    I  = eye(N, N)
    
    exact_ener = 0.0
    for i in range(Ne):
        exact_ener = exact_ener + eigval[i]

    
    exact_mu = 0.0
    for i in range(N):
        exact_mu = exact_ener + eigval[i] 

    exact_homo = eigval[Ne-1] 
    exact_lumo = eigval[Ne  ] 

    exact_mu = exact_mu / N
    
    epsi_0   = epsi_min(H,N) 
    epsi_N   = epsi_max(H,N) 

    epsi_0_ = epsi_0#epsi_min_new(H,N,H_bounds)
    epsi_N_ = epsi_N#epsi_max_new(H,N,H_bounds)

    mu = trace(H) / float(N)

    theta = float64(Ne / N)

    lambd1 = float64(    Ne) / ( N*(epsi_N_ - mu) )
    lambd2 = float64(N - Ne) / ( N*(mu - epsi_0_) )
    
    lambd3 = float64(    Ne) / ( N*(mu - epsi_N_) )
    lambd4 = float64(N - Ne) / ( N*(epsi_0_ - mu) )


    lambd_o  = min(lambd1, lambd2)
    lambd_q  = max(lambd1, lambd2)

    lambd_r  = min(lambd3, lambd4)
    lambd_s  = max(lambd3, lambd4)


    X0min  = lambd_o * (mu*I - H) + (Ne / N)*I
    X0max  = lambd_q * (mu*I - H) + (Ne / N)*I

    Y0min  = lambd_r * (mu*I - H) + ( (N - Ne) / N)*I
    Y0max  = lambd_s * (mu*I - H) + ( (N - Ne) / N)*I

    print ('<< Before optimization:')
    print ('   norm2(D_min) + trace(D_minQ_max) : ', linalg.norm(X0min, ord=None)**2 + trace(dot(X0min,Y0max)))

    print ('   trace(D_min  )  : ', trace(X0min))
    print ('   trace(D_min^2)  : ', trace(dot(X0min,X0min)))
    print ('   trace(D_min^3)  : ', trace(dot(dot(X0min,X0min),X0min)))
    #print '   trace(D_min^4)  : ', trace(dot(dot(dot(X0min,X0min),X0min),X0min)))
    print ('   2*trace(D_min^2) - trace(D_min): ', 2*trace(dot(X0min,X0min)) - trace(X0min))
    #print '   F_norm(D_min)     : ', linalg.norm(X0min, ord='fro')
    #print '   F_norm(D_min)^2   : ', linalg.norm(X0min, ord='fro')**2
    #print '   F_norm(D_min)^3   : ', linalg.norm(X0min, ord='fro')**3
    #print '   F_norm(D_min)^4   : ', linalg.norm(X0min, ord='fro')**4
    #print
    #print '   3_norm(D_min)^3   : ', entrywise_norm(X0min,p=3)**3
    #print    
    #print '   trace(D_min  Q_min):', trace(dot(X0min,I-X0min))
    #print '   trace(D_min^2Q_min):', trace(dot(dot(X0min,X0min),I-X0min))
    print
    print ('   trace(D_max)    : ', trace(X0max))
    print ('   trace(D_max^2)  : ', trace(dot(X0max,X0max)))
    print ('   trace(D_max^3)  : ', trace(dot(dot(X0max,X0max),X0max)))
    print ('   2*trace(D_max^2) - trace(D_max): ', 2*trace(dot(X0max,X0max)) - trace(X0max))
    print ('>>')
    print()

    omega = 1./3.

    if   ( theta <= omega ):
        test = ( omega ) * Ne
    elif ( theta > omega ):
        test = Ne - ( 1.0 - omega ) * (N - Ne)

    print('<< second-order alpha optimization:')
    a = trace(     dot(X0min, X0min)   - 2.0*dot(X0min, X0max) + dot(X0max, X0max) )
    b = trace( 2.0*dot(X0min, X0max)   - 2.0*dot(X0max, X0max) ) 
    c = trace(     dot(X0max, X0max) ) - test

    coef     = [a,b,c]

    if ( (a and b) == 0.0 ) :
        roots_np = [0.0, 0.0]
    else:
        roots_np = roots(coef)

    #check if a =/ 0 and b =/0
    delta = b**2 - 4.0*a*c 

    if ( delta <= 0.0 ):
        alpha_p = 0.0
        alpha_m = 0.0
        alpha   = 1.0/2.0
       
    else:    
        alpha_p = ( -b + sqrt(delta) ) / ( 2.0*a )
        alpha_m = ( -b - sqrt(delta) ) / ( 2.0*a )
        alpha   = min(alpha_p, alpha_m)
        if ( alpha > 1.0 or alpha < 0.0 ):
            alpha = 1.0/2.0

    X0 = alpha * X0min + (1.0 - alpha) * X0max
    
    print(' a, b, c   :', a, b, c)
    print(' b**2, 4*ac:', b**2, 4.0*a*c)
    print(' delta     :', delta, sqrt( sqrt(delta**2) ))
    print(' alpha +/- :', alpha_p, alpha_m)
    print(' numpy +/- :', roots_np[0], roots_np[1])
    print(' alpha     :', alpha)
    print('   trace(D)          :', trace(X0) )
    print('   trace(D^2)        :', trace(dot(X0,X0)) )
    print('   trace(D^3)        :', trace(dot(dot(X0,X0),X0)) )
    tmp = trace(dot(dot(X0,X0),(I-X0))) / trace(dot(X0,(I-X0))) 
    print('   cn from HPCP guess:', tmp    )
    print('   (2*cn+1)*trace(D_0^2) - 2*cn*trace(D_0): ', (2*tmp+1)*trace(dot(X0,X0)) - 2*tmp*trace(X0))
    print()
    print('>>')
    print()

    #print '<< third-order alpha optimization:'
    #a = trace( dot(dot(X0min, X0min),X0min) - 3.0*dot(dot(X0min, X0min),X0max) + 3.0*dot(dot(X0max, X0max),X0min) - dot(dot(X0max, X0max),X0max) )
    #b = trace( 3.0*dot(dot(X0min, X0min),X0max) - 6.0*dot(dot(X0max, X0max),X0min) + 3.0*dot(dot(X0max, X0max),X0max) ) 
    #c = trace( 3.0*dot(dot(X0max, X0max),X0min) - 3.0*dot(dot(X0max, X0max),X0max) )
    #d = trace( dot(dot(X0max, X0max),X0max) ) - test               
    #coef     = [a,b,c,d]
    #roots_np = roots(coef)
    #print ' a, b, c, d:', a, b, c, d
    #print ' numpy +/- :', roots_np[0], roots_np[1], roots_np[2] 
    #print '>>'
    #print

    
    print('<< third-order cn optimization:')

    beta = 0.5   
    
    a = trace( dot(dot(X0min, X0min),X0min) - 3.0*dot(dot(X0min, X0min),X0max) + 3.0*dot(dot(X0max, X0max),X0min) - dot(dot(X0max, X0max),X0max) )
    
    b = trace( 3.0*dot(dot(X0min, X0min),X0max) - 6.0*dot(dot(X0max, X0max),X0min) + 3.0*dot(dot(X0max, X0max),X0max) ) \
        - (1.0 + beta ) * trace( dot(X0min, X0min) - 2.0*dot(X0min, X0max) + dot(X0max, X0max) )
        
    c = trace( 3.0*dot(dot(X0max, X0max),X0min) - 3.0*dot(dot(X0max, X0max),X0max) ) \
        - (1.0 + beta ) * trace( 2.0*dot(X0min, X0max) - 2.0*dot(X0max, X0max) ) + beta * trace(X0min - X0max)
        
    d = trace( dot(dot(X0max, X0max),X0max) ) \
        - (1.0 + beta ) * trace( dot(X0max, X0max) ) + beta * trace(X0max)

    coef     = [a,b,c,d]

    if ( (a and b) == 0.0 ) :
        roots_np = [0.0, 0.0, 0.0]
        alpha_   = 0.0
        
    else:
        roots_np = roots(coef)
        alpha_   = min(abs(roots_np))

    #if ( alpha > 1.0 or alpha < 0.0 ):
    #    alpha = 1.0/2.0
    
    print(' a, b, c, d:', a, b, c, d)
    print(' numpy +/- :', roots_np[0], roots_np[1], roots_np[2] )
    print(' alpha     :', alpha_)

    print('>>')
       
    X0 = alpha * X0min + (1.0 - alpha ) * X0max

    #toto = ( 0.5 * trace(X0min) - trace(dot(X0min,X0min)) ) / ( 0.5 * trace(dot(X0min,X0max)) - trace(dot(dot(X0min,X0min),X0max)) )     
    #alpha = toto    
    #X0 = X0min 
    #X0 = 4.8 * X0min - 3.8*(Ne/N)*I
    #print '    trace(D^3)   : ', trace(dot(dot(X0,X0),X0))
    #print '   3_norm(D)^3   : ', entrywise_norm(X0,p=3)**3
    print('   trace(D)          :', trace(X0) )
    print('   trace(D^2)        :', trace(dot(X0,X0)) )
    print('   trace(D^3)        :', trace(dot(dot(X0,X0),X0)) )
    tmp = trace(dot(dot(X0,X0),(I-X0))) / trace(dot(X0,(I-X0))) 
    print('   cn from HPCP guess:', tmp    )
    print('   (2*cn+1)*trace(D_0^2) - 2*cn*trace(D_0): ', (2*tmp+1)*trace(dot(X0,X0)) - 2*tmp*trace(X0)    )
    print()
    
    #X0 = alpha*(theta*I + lambd_o * (mu* I - H)) + (1 - alpha)*(I - (1.0-theta)*I + lambd_q*(mu* I - H))
    X0 = theta*I + (mu* I - H) * (alpha*lambd_o + (1.0 - alpha) * lambd_q)
    
    #toto = ( 0.5 * trace(X0min) - trace(dot(X0min,X0min)) ) / ( 0.5 * trace(dot(X0min,X0max)) - trace(dot(dot(X0min,X0min),X0max)) )     
    #alpha = toto    
    #X0 = X0min 
    #X0 = 4.8 * X0min - 3.8*(Ne/N)*I
    #print '    trace(D^3)   : ', trace(dot(dot(X0,X0),X0))
    #print '   3_norm(D)^3   : ', entrywise_norm(X0,p=3)**3
    print('   trace(D)          :', trace(X0) )
    print('   trace(D^2)        :', trace(dot(X0,X0)) )
    print('   trace(D^3)        :', trace(dot(dot(X0,X0),X0)) )
    tmp = trace(dot(dot(X0,X0),(I-X0))) / trace(dot(X0,(I-X0))) 
    print('   cn from HPCP guess:', tmp    )
    print('   (2*cn+1)*trace(D_0^2) - 2*cn*trace(D_0): ', (2*tmp+1)*trace(dot(X0,X0)) - 2*tmp*trace(X0)   )
    print()
    

    lambd = lambd_o    
    return X0, mu, lambd, lambd1, lambd2, epsi_0_, epsi_N_, exact_ener, alpha, test

def hpcp(X,H,Ne):  
    
    from numpy.linalg import matrix_power as mat_pow
    from numpy        import matmul       as mat_mul
    from numpy        import trace
    import numpy
    from scipy.sparse import csr_matrix
    
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
    #print(numpy.shape(X))
    #X = numpy.array(X.toarray()) #; print(numpy.shape(X))
    return X, 0, p


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
