from scipy.sparse import csr_matrix
import numpy as np
from pyscf import lib
from pyscf.scf.lib_dm import mm
import tensorflow as tf
from tensorflow import convert_to_tensor
from time import perf_counter
#from pyscf.scf import get_occ

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


def dm_purify(H,N,Ne,method,fmt,thr,maxiter):

    if (method == 'hpcp') :
        X0 = hpcp_guess(H,N,Ne)
        if (fmt == 'np'):
            X, niter = hpcp_purify_np(X0,Ne,thr=thr,maxiter=50)
        elif (fmt == 'es'):
            X, niter = hpcp_purify_es(X0,Ne,thr=thr,maxiter=50)
        elif (fmt == 'tf'):
            X, niter = hpcp_purify_tf(X0,Ne,thr=thr,maxiter=50)

    if (method == 'tc2') :
        X0 = tc2_guess(H,N,Ne)
        if (fmt == 'np'):
            X, niter = tc2_purify_np(X0,Ne,thr=thr,maxiter=50)
        elif (fmt == 'es'):
            X, niter = tc2_purify_es(X0,Ne,thr=thr,maxiter=50)
        elif (fmt == 'tf'):
            X, niter = tc2_purify_tf(X0,Ne,thr=thr,maxiter=50)

    if (method == 'trs4') :
        X0 = trs4_guess(H,N,Ne)
        if (fmt == 'np'):
            X, niter = trs4_purify_np(X0,Ne,thr=thr,maxiter=50)
        elif (fmt == 'es'):
            X, niter = trs4_purify_es(X0,Ne,thr=thr,maxiter=50)
        elif (fmt == 'tf'):
            X, niter = trs4_purify_tf(X0,Ne,thr=thr,maxiter=50)
    
    if (method == 'tc2acc') : 
        X0,betal,betah = tc2acc_guess(H,N,Ne)
        if (fmt == 'np'):
            X, niter = tc2acc_purify_np(X0,Ne,betal,betah,thr=thr,maxiter=50)
        elif (fmt == 'es'):
            X, niter = tc2acc_purify_es(X0,Ne,thr=thr,maxiter=50)
        # elif (fmt == 'tf'):
        #     X, niter = tc2_purify_tf(X0,Ne,thr=thr,maxiter=50)


    return X, niter

def hpcp_guess(H,N,Ne,*args):

    if (Ne[0] == Ne[1]):
        Ne = Ne[0]
    else:
        print('WARNING: Ne',Ne)

    I = np.eye(N, N)

    #Restricted = 1 density matrix
    if ( H.ndim == 2 ):

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

def hpcp_np(X,Ne):
    c = np.float64(0.0)
    c1= np.float64(0.0)
    c2= np.float64(0.0)

    #X = csr_matrix(X)

    X_2 = mm(X,X,method='np',dtype='float32')
    X_3 = mm(X_2,X,method='np',dtype='float32')

    c1 = np.float64( np.trace(X_2 - X_3) )
    c2 = np.float64( np.trace(X   - X_2) )

    #c1 = float64( csr_matrix.trace(X_2 - X_3) )
    #c2 = float64( csr_matrix.trace(X   - X_2) )

    if ( abs(c1) < 1e-8 ):
        c = np.float64(0.50)

    else:
        c = c1/c2

    X = X + 2 * ( X_2 - X_3 - c * (X - X_2) )

    p = [c1,c2,c,0.0,0.0,0.0,0.0]
    #X = numpy.array(X.toarray()) #; print(numpy.shape(X))
    return X, 0, p

def hpcp_es(X,Ne):
    c = np.float64(0.0)
    c1= np.float64(0.0)
    c2= np.float64(0.0)

    #X = csr_matrix(X)

    X_2 = mm(X,X,method='es',dtype='float32')
    X_3 = mm(X_2,X,method='es',dtype='float32')

    c1 = np.float64( np.trace(X_2 - X_3) )
    c2 = np.float64( np.trace(X   - X_2) )

    #c1 = float64( csr_matrix.trace(X_2 - X_3) )
    #c2 = float64( csr_matrix.trace(X   - X_2) )

    if ( abs(c1) < 1e-8 ):
        c = np.float64(0.50)

    else:
        c = c1/c2

    X = X + 2 * ( X_2 - X_3 - c * (X - X_2) )

    p = [c1,c2,c,0.0,0.0,0.0,0.0]
    #X = numpy.array(X.toarray()) #; print(numpy.shape(X))
    return X, 0, p

def hpcp_tf(X,Ne):
    c = (0.0)
    c1= (0.0)
    c2= (0.0)

    #X = csr_matrix(X)
    #X_tf = convert_to_tensor(X)

    X_2 = mm(X,X,method='tf',dtype='float32')
    X_3 = mm(X_2,X,method='tf',dtype='float32')

    c1 = ( tf.linalg.trace(X_2 - X_3) )
    c2 = ( tf.linalg.trace(X   - X_2) )

    #c1 = float64( csr_matrix.trace(X_2 - X_3) )
    #c2 = float64( csr_matrix.trace(X   - X_2) )

    if ( abs(c1) < 1e-8 ):
        c = (0.50)

    else:
        c = c1/c2

    X = X_tf + 2 * ( X_2 - X_3 - c * (X_tf - X_2) )

    p = [c1,c2,c,0.0,0.0,0.0,0.0]
    #X = numpy.array(X.toarray()) #; print(numpy.shape(X))
    return X, 0, p

def hpcp_purify_np(X0,Ne,thr=1e-8,maxiter=50):

    #Restricted = 1 density matrix
    if ( X0.ndim == 2 ):
        threshold = thr
        test = threshold*10
        iter_ = 0
        X = X0

        while ( test > threshold ) and ( iter_ < maxiter ):
            old_X = X
            X, diag, p = hpcp_np(X,Ne)

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

            X_a, diag_a, p_a = hpcp_np(X_a,Ne[0])

            test_a = np.linalg.norm(X_a - old_X_a, ord='fro')
            # occ, _ = linalg.eigh(X_a)
            # print('X_a',test_a,numpy.trace(X_a),p_a[0],p_a[1],p_a[2])
            # print(occ)
            # print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            # print(test_b,numpy.shape(X_b),type(X_b),numpy.trace(X_b))
            iter_a += 1

        while ( test_b > threshold ) and ( iter_b < maxiter ):
            old_X_b = X_b

            X_b, diag_b, p_b = hpcp_np(X_b,Ne[1])

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

def hpcp_purify_es(X0,Ne,thr=1e-8,maxiter=50):

    #Restricted = 1 density matrix
    if ( X0.ndim == 2 ):
        threshold = thr
        test = threshold*10
        iter_ = 0
        X = X0

        while ( test > threshold ) and ( iter_ < maxiter ):
            old_X = X
            X, diag, p = hpcp_es(X,Ne)

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

            X_a, diag_a, p_a = hpcp_es(X_a,Ne[0])

            test_a = np.linalg.norm(X_a - old_X_a, ord='fro')
            # occ, _ = linalg.eigh(X_a)
            # print('X_a',test_a,numpy.trace(X_a),p_a[0],p_a[1],p_a[2])
            # print(occ)
            # print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            # print(test_b,numpy.shape(X_b),type(X_b),numpy.trace(X_b))
            iter_a += 1

        while ( test_b > threshold ) and ( iter_b < maxiter ):
            old_X_b = X_b

            X_b, diag_b, p_b = hpcp_es(X_b,Ne[1])

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

def hpcp_purify_tf(X0,Ne,thr=1e-8,maxiter=50):

    #Restricted = 1 density matrix
    if ( X0.ndim == 2 ):
        threshold = thr
        test = threshold*10
        iter_ = 0
        X0_tf = convert_to_tensor(X0)
        X = X0_tf

        while ( test > threshold ) and ( iter_ < maxiter ):
            old_X = X
            X, diag, p = hpcp_tf(X,Ne)

            test = tf.norm(X - old_X)#, ord='fro')
            #print(test,numpy.shape(X),type(X))
            #    print(test,numpy.trace(X))
            iter_ += 1
        X = X.numpy()
        #print(linalg.eigh(X))
        return X, iter_

    # Unrestricted = 2 density matrices
    elif ( X0.ndim == 3 ):

        threshold = thr
        test_a = threshold*10
        test_b = threshold*10
        iter_a = 0
        iter_b = 0
        X0_tf = convert_to_tensor(X0)
        X_a = X0_tf[0]
        X_b = X0_tf[1]

        while ( test_a > threshold ) and ( iter_a < maxiter ):
            old_X_a = X_a

            X_a, diag_a, p_a = hpcp_tf(X_a,Ne[0])

            test_a = tf.norm(X_a - old_X_a, ord='fro')
            # occ, _ = linalg.eigh(X_a)
            # print('X_a',test_a,numpy.trace(X_a),p_a[0],p_a[1],p_a[2])
            # print(occ)
            # print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            # print(test_b,numpy.shape(X_b),type(X_b),numpy.trace(X_b))
            iter_a += 1

        while ( test_b > threshold ) and ( iter_b < maxiter ):
            old_X_b = X_b

            X_b, diag_b, p_b = hpcp_tf(X_b,Ne[1])

            test_b = tf.norm(X_b - old_X_b, ord='fro')
            #occ, _ = linalg.eigh(X_b)
            #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            #print('X_b',test_b,numpy.trace(X_b))
            #print(occ)

            iter_b += 1
        #print('X_b',numpy.trace(X_b))
        #print('X_a',numpy.trace(X_a))
        #print(linalg.eigh(X_a))
        #print(linalg.eigh(X_b))
        X_a = X_a.numpy()
        X_b = X_b.numpy()
        return np.array([X_a,X_b]), [iter_a,iter_b]

def tc2_guess(H,N,Ne,*args):

    if (Ne[0] == Ne[1]):
        Ne = Ne[0]
    else:
        print('WARNING: Ne',Ne)

    I = np.eye(N, N)

    #Restricted = 1 density matrix
    if ( H.ndim == 2 ):

        epsi_0 = epsi_min(H,N)
        epsi_N = epsi_max(H,N)

        X0 = (epsi_N*I - H) / (epsi_N - epsi_0)

        return X0

    # Unrestricted = 2 density matrices
    elif ( H.ndim == 3 ):

        epsi_0_a  = epsi_min(H[0],N)
        epsi_N_a  = epsi_max(H[0],N)
        epsi_0_b  = epsi_min(H[1],N)
        epsi_N_b  = epsi_max(H[1],N)

        X0_a = (epsi_N_a*I - H) / (epsi_N_a - epsi_0_a)
        X0_b = (epsi_N_b*I - H) / (epsi_N_b - epsi_0_b)

        return np.array([X0_a, X0_b])

def tc2_np(X,Ne):

    #X = csr_matrix(X)

    X_2 = mm(X,X,method='np',dtype='float32')
    trace_X = np.trace(X)
    if np.all(trace_X >= Ne):
        X = X_2

    else:
        X = 2*X - X_2
    return X

def tc2_es(X,Ne):

    #X = csr_matrix(X)

    X_2 = mm(X,X,method='es',dtype='float32')
    trace_X = np.trace(X)
    if np.all(trace_X >= Ne):
        X = X_2

    else:
        X = 2*X - X_2
    return X

def tc2_tf(X,Ne):

    #X = csr_matrix(X)

    X_2 = mm(X,X,method='tf',dtype='float32')
    trace_X = tf.linalg.trace(X)
    if np.all(trace_X >= Ne):
        X = X_2

    else:
        X = 2*X - X_2
    return X

def tc2_purify_np(X0,Ne,thr=1e-8,maxiter=50):

    #Restricted = 1 density matrix
    if ( X0.ndim == 2 ):
        threshold = thr
        test = threshold*10
        iter_ = 0
        X = X0

        while ( test > threshold ) and ( iter_ < maxiter ):
            old_X = X

            X = tc2_np(X,Ne)

            test = np.linalg.norm(X - old_X, ord='fro')
            #print(test,numpy.shape(X),type(X))
            #print(test,numpy.trace(X))
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

            X_a = tc2_np(X_a,Ne[0])

            test_a = np.linalg.norm(X_a - old_X_a, ord='fro')
            #occ, _ = linalg.eigh(X_a)
            #print('X_a',test_a,numpy.trace(X_a),p_a[0],p_a[1],p_a[2])
            #print(occ)
            #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            #print(test_b,numpy.shape(X_b),type(X_b),numpy.trace(X_b))
            iter_a += 1

        while ( test_b > threshold ) and ( iter_b < maxiter ):
            old_X_b = X_b

            X_b = tc2_np(X_b,Ne[1])

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

def tc2_purify_es(X0,Ne,thr=1e-8,maxiter=50):

    #Restricted = 1 density matrix
    if ( X0.ndim == 2 ):
        threshold = thr
        test = threshold*10
        iter_ = 0
        X = X0

        while ( test > threshold ) and ( iter_ < maxiter ):
            old_X = X

            X = tc2_es(X,Ne)

            test = np.linalg.norm(X - old_X, ord='fro')
            #print(test,numpy.shape(X),type(X))
            #print(test,numpy.trace(X))
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

            X_a = tc2_es(X_a,Ne[0])

            test_a = np.linalg.norm(X_a - old_X_a, ord='fro')
            #occ, _ = linalg.eigh(X_a)
            #print('X_a',test_a,numpy.trace(X_a),p_a[0],p_a[1],p_a[2])
            #print(occ)
            #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            #print(test_b,numpy.shape(X_b),type(X_b),numpy.trace(X_b))
            iter_a += 1

        while ( test_b > threshold ) and ( iter_b < maxiter ):
            old_X_b = X_b

            X_b = tc2_es(X_b,Ne[1])

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

def tc2_purify_tf(X0,Ne,thr=1e-8,maxiter=50):

    #Restricted = 1 density matrix
    if ( X0.ndim == 2 ):
        threshold = thr
        test = threshold*10
        iter_ = 0
        X0_tf = convert_to_tensor(X0)
        X = X0_tf

        while ( test > threshold ) and ( iter_ < maxiter ):
            old_X = X

            X = tc2_tf(X,Ne)

            test = tf.norm(X - old_X)#, ord='fro')
            #print(test,numpy.shape(X),type(X))
            #print(test,numpy.trace(X))
            iter_ += 1

        #print(linalg.eigh(X))
        X = X.numpy()
        return X, iter_
    # Unrestricted = 2 density matrices
    elif ( X0.ndim == 3 ):

        threshold = thr
        test_a = threshold*10
        test_b = threshold*10
        iter_a = 0
        iter_b = 0
        X0_tf = convert_to_tensor(X0)
        X_a = X0_tf[0]
        X_b = X0tf[1]

        while ( test_a > threshold ) and ( iter_a < maxiter ):
            old_X_a = X_a

            X_a = tc2_tf(X_a,Ne[0])

            test_a = tf.norm(X_a - old_X_a, ord='fro')
            #occ, _ = linalg.eigh(X_a)
            #print('X_a',test_a,numpy.trace(X_a),p_a[0],p_a[1],p_a[2])
            #print(occ)
            #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            #print(test_b,numpy.shape(X_b),type(X_b),numpy.trace(X_b))
            iter_a += 1

        while ( test_b > threshold ) and ( iter_b < maxiter ):
            old_X_b = X_b

            X_b = tc2_tf(X_b,Ne[1])

            test_b = tf.norm(X_b - old_X_b, ord='fro')
            #occ, _ = linalg.eigh(X_b)
            #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            #print('X_b',test_b,numpy.trace(X_b))
            #print(occ)

            iter_b += 1
        #print('X_b',numpy.trace(X_b))
        #print('X_a',numpy.trace(X_a))
        #print(linalg.eigh(X_a))
        #print(linalg.eigh(X_b))
        X_a = X_a.numpy()
        X_b = X_b.numpy()
        return np.array([X_a,X_b]), [iter_a,iter_b]

def tc2acc_guess(H,N,Ne,*args):
    #
    eigs = np.linalg.eigvalsh(H)
    #
    #occ = get_occ(mo_energy=eigs)
    t_hl0 = perf_counter()
    ehomo = eigs[Ne[0]-1]
    elumo = eigs[Ne[0]]
    print(ehomo)
    print(elumo)
    t_hl1 = perf_counter()
    t_homolumo = t_hl1-t_hl0
    print('t_homolumo =', t_homolumo)

    if (Ne[0] == Ne[1]):
        Ne = Ne[0]
    else:
        print('WARNING: Ne',Ne)

    I = np.eye(N, N)

    #Restricted = 1 density matrix
    if ( H.ndim == 2 ):

        epsi_0 = epsi_min(H,N)
        epsi_N = epsi_max(H,N)
        betal = (epsi_N-elumo)/(epsi_N-epsi_0)
        betah = (epsi_N-ehomo)/(epsi_N-epsi_0)

        X0 = (epsi_N*I - H) / (epsi_N - epsi_0)

        return X0, betal, betah

    # Unrestricted = 2 density matrices
    elif ( H.ndim == 3 ):

        epsi_0_a  = epsi_min(H[0],N)
        epsi_N_a  = epsi_max(H[0],N)
        epsi_0_b  = epsi_min(H[1],N)
        epsi_N_b  = epsi_max(H[1],N)

        X0_a = (epsi_N_a*I - H) / (epsi_N_a - epsi_0_a)
        X0_b = (epsi_N_b*I - H) / (epsi_N_b - epsi_0_b)

        return np.array([X0_a, X0_b]), betal, betah


def tc2acc_np(X,Ne, betal, betah):

    #X = csr_matrix(X)

    trace_X = np.trace(X)
    I = np.eye(np.shape(X)[0])
    if np.all(trace_X >= Ne):
        alpha = 2/(2-betal)
        X = (1-alpha)*I + alpha*X
        X = mm(X,X,method='np',dtype='float32')
        betal = (alpha*betal + 1-alpha)**2
        betah = (alpha*betah + 1-alpha)**2

    else:
        alpha = 2/(1+betah)
        X = alpha*X
        X_2 = mm(X,X,method='np',dtype='float32')
        X = 2*X - X_2
        betal = 2*alpha*betal - alpha**2*betal**2
        betah = 2*alpha*betah - alpha**2*betah**2
    return X, betal, betah

def tc2acc_es(X,Ne, betal, betah):

    #X = csr_matrix(X)

    trace_X = np.trace(X)
    I = np.eye(np.shape(X)[0])
    if np.all(trace_X >= Ne):
        alpha = 2/(2-betal)
        X = (1-alpha)*I + alpha*X
        X = mm(X,X,method='es',dtype='float32')
        betal = (alpha*betal + 1-alpha)**2
        betah = (alpha*betah + 1-alpha)**2

    else:
        alpha = 2/(1+betah)
        X = alpha*X
        X_2 = mm(X,X,method='es',dtype='float32')
        X = 2*X - X_2
        betal = 2*alpha*betal - alpha**2*betal**2
        betah = 2*alpha*betah - alpha**2*betah**2
    return X, betal, betah

# def tc2acc_tf(X,Ne, betal, betah)):

#     #X = csr_matrix(X)

#     trace_X = tf.linalg.trace(X)
#     I = np.eye(np.shape(X)[0])
#     if np.all(trace_X >= Ne):
#        alpha = 2/(2-betal)
#        X = (1-alpha)*I + alpha*X
#        X = mm(X,X,method='es',dtype='float32')
#        betal = (alpha*betal + 1-alpha)**2
#        betah = (alpha*betah + 1-alpha)**2

#    else:
#        alpha = 2/(1+betah)
#        X = alpha*X
#        X_2 = mm(X,X,method='es',dtype='float32')
#        X = 2*X - X_2
#        betal = 2*alpha*betal - alpha**2*betal**2
#        betah = 2*alpha*betah - alpha**2*betah**2
#    return X, betal, betah


def tc2acc_purify_np(X0,Ne,betal,betah,thr=1e-8,maxiter=50):

    #Restricted = 1 density matrix
    if ( X0.ndim == 2 ):
        threshold = thr
        test = threshold*10
        iter_ = 0
        X = X0

        while ( test > threshold ) and ( iter_ < maxiter ):
            old_X = X

            X,betal,betah = tc2acc_np(X,Ne, betal, betah)

            test = np.linalg.norm(X - old_X, ord='fro')
            #print(test,numpy.shape(X),type(X))
            #print(test,numpy.trace(X))
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

            X_a,betal,betah = tc2acc_np(X_a,Ne[0])

            test_a = np.linalg.norm(X_a - old_X_a, ord='fro')
            #occ, _ = linalg.eigh(X_a)
            #print('X_a',test_a,numpy.trace(X_a),p_a[0],p_a[1],p_a[2])
            #print(occ)
            #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            #print(test_b,numpy.shape(X_b),type(X_b),numpy.trace(X_b))
            iter_a += 1

        while ( test_b > threshold ) and ( iter_b < maxiter ):
            old_X_b = X_b

            X_b,betal,betah = tc2_np(X_b,Ne[1])

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

def tc2acc_purify_es(X0,Ne,betal,betah,thr=1e-8,maxiter=50):

    #Restricted = 1 density matrix
    if ( X0.ndim == 2 ):
        threshold = thr
        test = threshold*10
        iter_ = 0
        X = X0

        while ( test > threshold ) and ( iter_ < maxiter ):
            old_X = X

            X,betal,betah = tc2acc_es(X,Ne, betal, betah)

            test = np.linalg.norm(X - old_X, ord='fro')
            #print(test,numpy.shape(X),type(X))
            #print(test,numpy.trace(X))
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

            X_a,betal,betah = tc2acc_es(X_a,Ne[0])

            test_a = np.linalg.norm(X_a - old_X_a, ord='fro')
            #occ, _ = linalg.eigh(X_a)
            #print('X_a',test_a,numpy.trace(X_a),p_a[0],p_a[1],p_a[2])
            #print(occ)
            #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            #print(test_b,numpy.shape(X_b),type(X_b),numpy.trace(X_b))
            iter_a += 1

        while ( test_b > threshold ) and ( iter_b < maxiter ):
            old_X_b = X_b

            X_b,betal,betah = tc2_es(X_b,Ne[1])

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

# def tc2acc_purify_tf(X0,Ne,betal,betah,thr=1e-8,maxiter=50):

#     #Restricted = 1 density matrix
#     if ( X0.ndim == 2 ):
#         threshold = thr
#         test = threshold*10
#         iter_ = 0
#         X0_tf = convert_to_tensor(X0)
#         X = X0_tf

#         while ( test > threshold ) and ( iter_ < maxiter ):
#             old_X = X

#             X,betal,betah = tc2acc_tf(X,Ne, betal, betah)

#             test = tf.norm(X - old_X, ord='fro')
#             #print(test,numpy.shape(X),type(X))
#             #print(test,numpy.trace(X))
#             iter_ += 1

#         #print(linalg.eigh(X))
#         X = X.numpy()
#         return X, iter_
#     # Unrestricted = 2 density matrices
#     elif ( X0.ndim == 3 ):

#         threshold = thr
#         test_a = threshold*10
#         test_b = threshold*10
#         iter_a = 0
#         iter_b = 0
#         X0_tf = convert_to_tensor(X0)
#         X_a = X0_tf[0]
#         X_b = X0tf[1]

#         while ( test_a > threshold ) and ( iter_a < maxiter ):
#             old_X_a = X_a

#             X_a, betal, betah = tc2acc_tf(X_a,Ne[0], betal, betah)

#             test_a = tf.norm(X_a - old_X_a, ord='fro')
#             #occ, _ = linalg.eigh(X_a)
#             #print('X_a',test_a,numpy.trace(X_a),p_a[0],p_a[1],p_a[2])
#             #print(occ)
#             #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
#             #print(test_b,numpy.shape(X_b),type(X_b),numpy.trace(X_b))
#             iter_a += 1

#         while ( test_b > threshold ) and ( iter_b < maxiter ):
#             old_X_b = X_b

#             X_b, betal, betah = tc2acc_tf(X_b,Ne[1], betal, betah)

#             test_b = tf.norm(X_b - old_X_b, ord='fro')
#             #occ, _ = linalg.eigh(X_b)
#             #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
#             #print('X_b',test_b,numpy.trace(X_b))
#             #print(occ)

#             iter_b += 1
#         #print('X_b',numpy.trace(X_b))
#         #print('X_a',numpy.trace(X_a))
#         #print(linalg.eigh(X_a))
#         #print(linalg.eigh(X_b))
#         X_a = X_a.numpy()
#         X_b = X_b.numpy()
#         return np.array([X_a,X_b]), [iter_a,iter_b]

def trs4_guess(H,N,Ne,*args):

    if (Ne[0] == Ne[1]):
        Ne = Ne[0]
    else:
        print('WARNING: Ne',Ne)

    I = np.eye(N, N)

    #Restricted = 1 density matrix
    if ( H.ndim == 2 ):

        epsi_0 = epsi_min(H,N)
        epsi_N = epsi_max(H,N)

        X0 = (epsi_N*I - H) / (epsi_N - epsi_0)

        return X0

    # Unrestricted = 2 density matrices
    elif ( H.ndim == 3 ):

        epsi_0_a  = epsi_min(H[0],N)
        epsi_N_a  = epsi_max(H[0],N)
        epsi_0_b  = epsi_min(H[1],N)
        epsi_N_b  = epsi_max(H[1],N)

        X0_a = (epsi_N_a*I - H) / (epsi_N_a - epsi_0_a)
        X0_b = (epsi_N_b*I - H) / (epsi_N_b - epsi_0_b)

        return np.array([X0_a, X0_b])

def trs4_np(X,Ne):

    #X = csr_matrix(X)

    X_2 = mm(X,X,method='np',dtype='float32')
    I = np.eye(X.shape[0])
    I_X = I - X
    F = mm(X_2,(4*X - 3*X_2),method='np',dtype='float32')
    G = mm(X_2,((I_X) @ (I_X)),method='np',dtype='float32')
    trace_F = np.trace(F)
    trace_G = np.trace(G)
    gamma_n = (Ne[0] - trace_F) / trace_G
    gamma_min = 0
    gamma_max = 6

    if np.all(gamma_n<gamma_min):
        X = X_2

    elif np.all(gamma_n>gamma_max):
        X = 2*X - X_2

    else:
        X = F + gamma_n * G

    return X

def trs4_es(X,Ne):

    #X = csr_matrix(X)

    X_2 = mm(X,X,method='es',dtype='float32')
    I = np.eye(X.shape[0])
    I_X = I - X
    F = mm(X_2,(4*X - 3*X_2),method='es',dtype='float32')
    G = mm(X_2,((I_X) @ (I_X)),method='es',dtype='float32')
    trace_F = np.trace(F)
    trace_G = np.trace(G)
    gamma_n = (Ne[0] - trace_F) / trace_G
    gamma_min = 0
    gamma_max = 6

    if np.all(gamma_n<gamma_min):
        X = X_2

    elif np.all(gamma_n>gamma_max):
        X = 2*X - X_2

    else:
        X = F + gamma_n * G

    return X

def trs4_tf(X,Ne):

    #X = csr_matrix(X)

    X_2 = mm(X,X,method='tf',dtype='float32')
    I = np.eye(X.shape[0])
    I_X = I - X
    F = mm(X_2,(4*X - 3*X_2),method='tf',dtype='float32')
    G = mm(X_2,((I_X) @ (I_X)),method='tf',dtype='float32')
    trace_F = tf.linalg.trace(F)
    trace_G = tf.linalg.trace(G)
    gamma_n = (Ne[0] - trace_F) / trace_G
    gamma_min = 0
    gamma_max = 6

    if np.all(gamma_n<gamma_min):
        X = X_2

    elif np.all(gamma_n>gamma_max):
        X = 2*X - X_2

    else:
        X = F + gamma_n * G

    return X

def trs4_purify_np(X0,Ne,thr=1e-8,maxiter=50):

    #Restricted = 1 density matrix
    if ( X0.ndim == 2 ):
        threshold = thr
        test = threshold*10
        iter_ = 0
        X = X0

        while ( test > threshold ) and ( iter_ < maxiter ):
            old_X = X

            X = trs4_np(X,Ne)

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

            X_a = trs4_np(X_a,Ne[0])

            test_a = np.linalg.norm(X_a - old_X_a, ord='fro')
            #occ, _ = linalg.eigh(X_a)
            #print('X_a',test_a,numpy.trace(X_a),p_a[0],p_a[1],p_a[2])
            #print(occ)
            #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            #print(test_b,numpy.shape(X_b),type(X_b),numpy.trace(X_b))
            iter_a += 1

        while ( test_b > threshold ) and ( iter_b < maxiter ):
            old_X_b = X_b

            X_b = trs4_np(X_b,Ne[1])

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

def trs4_purify_es(X0,Ne,thr=1e-8,maxiter=50):

    #Restricted = 1 density matrix
    if ( X0.ndim == 2 ):
        threshold = thr
        test = threshold*10
        iter_ = 0
        X = X0

        while ( test > threshold ) and ( iter_ < maxiter ):
            old_X = X

            X = trs4_es(X,Ne)

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

            X_a = trs4_es(X_a,Ne[0])

            test_a = np.linalg.norm(X_a - old_X_a, ord='fro')
            #occ, _ = linalg.eigh(X_a)
            #print('X_a',test_a,numpy.trace(X_a),p_a[0],p_a[1],p_a[2])
            #print(occ)
            #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            #print(test_b,numpy.shape(X_b),type(X_b),numpy.trace(X_b))
            iter_a += 1

        while ( test_b > threshold ) and ( iter_b < maxiter ):
            old_X_b = X_b

            X_b = trs4_es(X_b,Ne[1])

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

def trs4_purify_tf(X0,Ne,thr=1e-8,maxiter=50):

    #Restricted = 1 density matrix
    if ( X0.ndim == 2 ):
        threshold = thr
        test = threshold*10
        iter_ = 0
        X0_tf = convert_to_tensor(X0)
        X = X0_tf

        while ( test > threshold ) and ( iter_ < maxiter ):
            old_X = X

            X = trs4_tf(X,Ne)

            test = tf.norm(X - old_X)#, ord='fro')
            #print(test,numpy.shape(X),type(X))
            #    print(test,numpy.trace(X))
            iter_ += 1

        #print(linalg.eigh(X))
        X = X.numpy()
        return X, iter_
    # Unrestricted = 2 density matrices
    elif ( X0.ndim == 3 ):

        threshold = thr
        test_a = threshold*10
        test_b = threshold*10
        iter_a = 0
        iter_b = 0
        X0_tf = convert_to_tensor(X0)
        X_a = X0_tf[0]
        X_b = X0_tf[1]

        while ( test_a > threshold ) and ( iter_a < maxiter ):
            old_X_a = X_a

            X_a = trs4_tf(X_a,Ne[0])

            test_a = tf.norm(X_a - old_X_a, ord='fro')
            #occ, _ = linalg.eigh(X_a)
            #print('X_a',test_a,numpy.trace(X_a),p_a[0],p_a[1],p_a[2])
            #print(occ)
            #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            #print(test_b,numpy.shape(X_b),type(X_b),numpy.trace(X_b))
            iter_a += 1

        while ( test_b > threshold ) and ( iter_b < maxiter ):
            old_X_b = X_b

            X_b = trs4_tf(X_b,Ne[1])

            test_b = tf.norm(X_b - old_X_b, ord='fro')
            #occ, _ = linalg.eigh(X_b)
            #print(test_a,numpy.shape(X_a),type(X_a),numpy.trace(X_a))
            #print('X_b',test_b,numpy.trace(X_b))
            #print(occ)

            iter_b += 1
        #print('X_b',numpy.trace(X_b))
        #print('X_a',numpy.trace(X_a))
        #print(linalg.eigh(X_a))
        #print(linalg.eigh(X_b))
        X_a = X_a.numpy()
        X_b = X_b.numpy()
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
