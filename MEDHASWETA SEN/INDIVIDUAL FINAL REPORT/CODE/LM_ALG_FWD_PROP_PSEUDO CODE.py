import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def gen_err (theta,ar,data):
    theta = np.array(theta).flatten()
    ana = theta[:ar]
    bnb = theta[ar:]
    ana1 = np.insert(ana, 0, 1)
    bnb1 = np.insert(bnb, 0, 1)
    diff = len(ana1) - len(bnb1)
    if diff > 0:
        bnb1 = np.pad(bnb1, (0, diff), mode='constant')
    else:
        ana1 = np.pad(ana1, (0, -diff), mode='constant')
    system = (ana1, bnb1, 1)
    t, e_dlsim = signal.dlsim(system, data)
    return e_dlsim

def gen_xmatrix(theta,er,data,ar):
    theta = np.array(theta).flatten()
    delta = 1e-06
    x = np.empty((len(data), 0))
    for i in range (0,len(theta)):
        change = []
        for j in range (0,len(theta)):
            if i == j:
                c = theta[j] + delta
                change.append(c)
            else:
                change.append(theta[j])

        e1 = gen_err (change,ar,data)
        xc = (er-e1)/delta
        x = np.hstack((x, xc))
    return x

def endr (theta_new,SSEnew,N,n,A,ar,SSEarray,nb,true):
    theta_new = np.array(theta_new).flatten()
    theta_hat = theta_new
    sigma_sq = SSEnew / (N - n)
    cov_theta_hat = sigma_sq * np.linalg.inv(A)
    print("The final estimated parameters are :", theta_hat)
    print("The true parameters are :", true)
    print("The Covariance Matrix :\n", cov_theta_hat)
    print("Variance :", sigma_sq)
    print("The Confidence Intervals are: ")
    for i in range(0, n):
        k = theta_hat[i]
        kc = cov_theta_hat[i, i]
        up = k + (2 * (kc ** 0.5))
        lw = k - (2 * (kc ** 0.5))
        if (i < ar):
            print(lw, "< a", (i + 1), "<", up)
        else:
            print(lw, "< b", (i + 1 - ar), "<", up)
    num1 = theta_new[:ar]
    num = np.insert(num1, 0, 1)
    den1 = theta_new[ar:]
    den = np.insert(den1, 0, 1)
    rootnum = np.roots(num)
    rootden = np.roots(den)
    if (ar > 0):
        print("The roots of the denominator are :", rootnum)
    if (nb > 0):
        print("The roots of the numerator are :", rootden)
    title_g = "SSE WITH EACH ITERATION FOR ARMA ("+ str(ar)+","+str(nb)+")"
    SSEser = pd.Series(SSEarray)
    plt.figure(figsize=(16,8))
    SSEser.plot()
    plt.xlabel("ITERATIONS")
    plt.ylabel("SSE")
    plt.grid()
    plt.title(title_g)
    plt.tight_layout()
    plt.show()
    return

def LM_ALG(data,ar,nb,true):

    SSEarray = []
    numi = 0
    maxi = 110
    mu = 0.01
    n = ar+nb
    N = len(data)
    # STEP ZERO:
    theta = np.zeros((n,1))
    # STEP ONE:
    e = gen_err (theta,ar,data)
    SSE = (e.T)@e
    SSE = SSE[0,0]
    SSEarray.append(SSE)
    X = gen_xmatrix(theta,e,data,ar)
    A = (X.T)@X
    g = (X.T)@e
    # STEP TWO:
    del_theta = np.linalg.inv(A+mu*np.eye(n))@g
    theta_new = theta + del_theta
    enew = gen_err (theta_new,ar,data)
    SSEnew = (enew.T)@enew
    SSEnew = SSEnew[0,0]
    #SSEarray.append(SSEnew)

    while (numi<maxi):

        if (SSEnew<SSE):

            if (np.linalg.norm(del_theta)<0.001):

                return endr(theta_new, SSEnew, N, n, A, ar, SSEarray, nb,true)
                return

            else:

                theta = theta_new
                mu = mu/10

        while (SSEnew>=SSE):

            mu = mu * 10

            if ( mu > 10**20 ):

                endr(theta_new, SSEnew, N, n, A, ar, SSEarray, nb,true)
                return

            del_theta = np.linalg.inv(A + mu * np.eye(n)) @ g
            theta_new = theta + del_theta
            enew = gen_err(theta_new, ar, data)
            SSEnew = (enew.T) @ enew
            SSEnew = SSEnew[0,0]
            #SSEarray.append(SSEnew)


        numi = numi + 1

        if (numi  > maxi) :

            endr(theta_new, SSEnew, N, n, A, ar, SSEarray, nb,true)
            return

        theta = theta_new

        e = gen_err(theta,ar, data)
        SSE = (e.T) @ e
        SSE = SSE[0,0]
        SSEarray.append(SSE)
        X = gen_xmatrix(theta,e, data, ar)
        A = (X.T) @ X
        g = (X.T) @ e
        # STEP TWO:
        del_theta = np.linalg.inv(A + mu * np.eye(n)) @ g
        theta_new = theta + del_theta
        enew = gen_err(theta_new, ar, data)
        SSEnew = (enew.T) @ enew
        SSEnew = SSEnew[0,0]
        #SSEarray.append(SSEnew)

