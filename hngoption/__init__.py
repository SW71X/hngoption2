# Python implementation by Dustin Zacharias (2017), method based on Fabrice Rouah

# Log Likelihood Estimation

from LogLikelihood import LogLike
from NMFiles import NelderMead


def main():
    # Settings for Nelder Mead Algorithm
    NumIters = 1  # First Iteration
    MaxIters = 1e3  # Maximum number of iterations
    Tolerance = 1e-5  # Tolerance on best and worst function values
    N = 5  # Number of Heston and Nandi parameters
    r = 0.05 / 252.0  # Risk Free Rate

    # Heston and Nandi parameter starting values (vertices) in vector form

    x = [[0 for i in range(N + 1)] for j in range(N)]
    x[0][0] = 5.02e-6;
    x[0][1] = 5.12e-6;
    x[0][2] = 5.00e-6;
    x[0][3] = 4.90e-6;
    x[0][4] = 4.95e-6;
    x[0][5] = 4.99e-6  # omega
    x[1][0] = 1.32e-6;
    x[1][1] = 1.25e-6;
    x[1][2] = 1.35e-6;
    x[1][3] = 1.36e-6;
    x[1][4] = 1.30e-6;
    x[1][5] = 1.44e-6  # alpha
    x[2][0] = 0.79;
    x[2][1] = 0.80;
    x[2][2] = 0.78;
    x[2][3] = 0.77;
    x[2][4] = 0.81;
    x[2][5] = 0.82  # beta
    x[3][0] = 427.0;
    x[3][1] = 421.0;
    x[3][2] = 425.0;
    x[3][3] = 419.1;
    x[3][4] = 422.1;
    x[3][5] = 430.0  # gamma
    x[4][0] = 0.21;
    x[4][1] = 0.20;
    x[4][2] = 0.22;
    x[4][3] = 0.19;
    x[4][4] = 0.18;
    x[4][5] = 0.205  # lambda

    # Run Nelder Mead and output Nelder Mead results
    B = NelderMead(LogLike, N, NumIters, MaxIters, Tolerance, x, r)

    #	print("Nelder Mead Minimization of Log-Likelihood for Heston and Nandi parameters")
    #	print("---------------------------------")
    #	print("omega  = ", B[0])
    #	print("alpha  = ", B[1])
    #	print("beta   = ", B[2])
    #	print("gamma  = ", B[3])
    #	print("lambda = ", B[4])
    #	print("Value of Objective Function = ", B[N])
    #	print("Number of Iterations = ", B[N+1])
    #	print("Persistence ", B[2]+B[1]*(B[3]**2) )
    #	print("---------------------------------")

    # alpha,beta,gamma,omega,lambda
    return [B[1], B[2], B[3], B[0], B[4]]


if __name__ == '__main__':
    main()





# Python implementation by Dustin Zacharias (2017), method based on Fabrice Rouah (volopta.com)


# Log Likelihood function for Heston and Nandi Model
from math import log, sqrt


# Returns the sum of a vector's elements
def VecSum(x):
    n = len(x)
    Sum = 0.0
    for i in range(0, n):
        Sum += x[i]
    return Sum


# Returns the sample variance
def VecVar(x):
    n = len(x)
    mean = VecSum(x) / n
    sumV = 0.0
    for i in range(0, n):
        sumV += (x[i] - mean) ** 2
    return sumV / (n - 1)


# Returns the log-likelihood based on timeseries
def LogLike(B, r):
    # pass a timeseries named prices with newest vals on top

    # i  = 0
    # with open('SP500.txt', 'r') as inPrices:
    #	for line in inPrices:
    #		try:
    #			Price.append(float(line))
    #			i += 1
    #		except:
    #			continue

    N = len(prices)
    # Calculate S&P500 returns
    ret = [0.0] * (N - 1)
    for i in range(0, N - 1):
        ret[i] = (log(prices.ix[i] / prices.ix[i + 1]))

    Variance = VecVar(ret)
    h = [0 * i for i in range(N - 1)]
    Z = [0 * i for i in range(N - 1)]
    L = [0 * i for i in range(N - 1)]

    # Construct GARCH(1,1) process by working back in time
    h[N - 2] = Variance
    Z[N - 2] = (ret[N - 2] - r - B[4] * h[N - 2]) / h[N - 2] ** 0.5
    L[N - 2] = -log(h[N - 2]) - (ret[N - 2] ** 2) / h[N - 2]

    for i in range(N - 3, -1, -1):
        h[i] = B[0] + B[2] * h[i + 1] + B[1] * pow(Z[i + 1] - B[3] * sqrt(h[i + 1]), 2)
        Z[i] = (ret[i] - r - B[4] * h[i]) / (h[i] ** 0.5)
        L[i] = -log(h[i]) - (ret[i] ** 2) / h[i]

    LogL = VecSum(L)
    if ((B[0] < 0) | (B[1] < 0) | (B[2] < 0) | (B[3] < 0) | (B[4] < 0)):  # (B[2]+B[1]*pow(B[3],2)>=1))
        return 1e50
    else:
        return -LogL  # Minimize -Log-Like(Beta)



# Python implementation by Dustin Zacharias (2017), method based on Fabrice Rouah

# HN Integral

import cmath
import math


# Trapezoidal Rule passing two vectors
def trapz(X, Y):
    n = len(X)
    sum = 0.0
    for i in range(1, n):
        sum += 0.5 * (X[i] - X[i - 1]) * (Y[i - 1] + Y[i])
    return sum


# HNC_f returns the real part of the Heston & Nandi integral
def HNC_f(complex_phi, d_alpha, d_beta, d_gamma, d_omega,
          d_lambda, d_V, d_S, d_K, d_r, i_T, i_FuncNum):
    A = [x for x in range(i_T + 1)]
    B = [x for x in range(i_T + 1)]
    complex_zero = complex(0.0, 0.0)
    complex_one = complex(1.0, 0.0)
    complex_i = complex(0.0, 1.0)
    A[i_T] = complex_zero
    B[i_T] = complex_zero

    for t in range(i_T - 1, -1, -1):
        if i_FuncNum == 1:
            A[t] = A[t + 1] + (complex_phi + complex_one) * d_r + B[t + 1] * d_omega \
                   - 0.5 * cmath.log(1.0 - 2.0 * d_alpha * B[t + 1])
            B[t] = (complex_phi + complex_one) * (d_lambda + d_gamma) - 0.5 * d_gamma ** 2 \
                   + d_beta * B[t + 1] + (0.5 * (complex_phi + complex_one - d_gamma) ** 2) \
                                         / (1.0 - 2.0 * d_alpha * B[t + 1])
        else:
            A[t] = A[t + 1] + (complex_phi) * d_r + B[t + 1] * d_omega \
                   - 0.5 * cmath.log(1.0 - 2.0 * d_alpha * B[t + 1])
            B[t] = complex_phi * (d_lambda + d_gamma) - 0.5 * d_gamma ** 2 + d_beta * B[t + 1] \
                   + (0.5 * (complex_phi - d_gamma) ** 2) / (1.0 - 2.0 * d_alpha * B[t + 1])
    if i_FuncNum == 1:
        z = (d_K ** (-complex_phi)) * (d_S ** (complex_phi + complex_one)) \
            * cmath.exp(A[0] + B[0] * d_V) / complex_phi
        return z.real
    else:
        z = (d_K ** (-complex_phi)) * (d_S ** (complex_phi)) * cmath.exp(A[0] + B[0] * d_V) / complex_phi
        return z.real


# Returns the Heston and Nandi option price
def HNC(alpha, beta, gamma, omega, d_lambda, V, S, K, r, T, PutCall):
    const_pi = 4.0 * math.atan(1.0)
    High = 100
    Increment = 0.25
    NumPoints = int(High / Increment)
    X, Y1, Y2 = [], [], []
    i = complex(0.0, 1.0)
    phi = complex(0.0, 0.0)
    for j in range(0, NumPoints):
        if j == 0:
            X.append(0.0000001)
        else:
            X.append(j * Increment)
        phi = X[j] * i
        Y1.append(HNC_f(phi, alpha, beta, gamma, omega, d_lambda, V, S, K, r, T, 1))
        Y2.append(HNC_f(phi, alpha, beta, gamma, omega, d_lambda, V, S, K, r, T, 2))

    int1 = trapz(X, Y1)
    int2 = trapz(X, Y2)
    P1 = 0.5 + math.exp(-r * T) * int1 / S / const_pi
    P2 = 0.5 + int2 / const_pi
    if P1 < 0:
        P1 = 0
    if P1 > 1:
        P1 = 1
    if P2 < 0:
        P2 = 0
    if P2 > 1:
        P2 = 1

    Call = S / 2 + math.exp(-r * T) * int1 / const_pi - K * math.exp(-r * T) * (0.5 + int2 / const_pi)
    Put = Call + K * math.exp(-r * T) - S
    if PutCall == 1:
        return Call
    else:
        return Put
    return 0



# Python implementation by Dustin Zacharias (2017), method based on Fabrice Rouah (volopta.com)



# HN GARCH Price

def HNP(prices,V, S, K, r, T, PutCall,fit):		#PutCall=1 -> Call
        #prices should be a reverse time series of S with newest values on top
    if (fit==1):  #Choose if whole loglike thing should run again


        global output
        output = main()


    return HNC(output[0],output[1],output[2],output[3],output[4],V, S, K, r, T, PutCall)





