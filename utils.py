import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

# One-of-K codification.
def one_of_K(y, K):
    Yhat = np.ones((y.shape[0], K))
    for k in range(K):
        Yhat[:,k,None] = (y==k+1).astype(int)
    return Yhat

#Defining the Sigmoid function and Softmax function
def Sigmoid(f_r):
    lam_r = 1/(1 + np.exp(-f_r))
    return lam_r

def Softmax(A):
    num = np.exp(A)
    den = np.sum(num, 1)
    den = den[:,np.newaxis]
    zeta_k = num/den
    return zeta_k

#Hard estimation of the ground truth by using the MAjority Voting scheme.
def MAjVot(Y, K):
    N,R = Y.shape
    Yhat = np.zeros((N,1))
    for n in range(N):
        votes = np.zeros((K,1))
        for r in range(R):
            for k in range(K):
                if Y[n,r] == k+1:
                    votes[k] = votes[k]+1
        Yhat[n] = np.argmax(votes) + 1
    return Yhat

# Multiple annotators' simulation

def multiple_annotators(R, NrP, Xtrain, ytrain):

  '''
  To simulate labels from multiple annotators, we assume them to correspond to
  corrupted versions of the ground truth.
  Thus, the labels are simulated by following approach:
  1. For each annotator $r$, we compute a function $f_r(\cdot)$ as a combination
  of $Q$ functions $u_q(\cdot)$, with $q\in\{1, \dots , Q\}$.
  2. We compute the annotators' reliability
  $\z_{r}(\cdot) = \sigma({f_r(\cdot)})$, where $\sigma(\cdot)$ is the Sigmoid function
  $$\sigma(\cdot) = \frac{1}{1 + e^{-f_r(\cdot)}}.$$
  3. If $\z_{r,n}>0.5$, $y_n^r=y_n$, and $y_n^r=\tilde{y}_n$ if $\z_{r,n}\le 0.5$,
  where $\tilde{y}_n$ is the flipped version of $y_n$
  '''
  N = Xtrain.shape[0]
  K = len(np.unique(ytrain))
  Kn = np.unique(ytrain)
  aux = 0
  A = np.zeros((K,1))
  for k in Kn:
      A[aux] = (ytrain == k).sum()
      aux = aux + 1
  per = np.min(A)

  tsne = TSNE(n_components=1,perplexity=per/2)
  if N < 25000:
      Xtrain = tsne.fit_transform(Xtrain)
  else:
      Xtrain = np.sum(Xtrain,1)

  Xtrain = Xtrain - Xtrain.min()
  Xtrain = Xtrain/Xtrain.max()
  Xtrain = Xtrain.reshape((N,1))
  yprueba = np.ones((N,1))


  u_q = np.empty((Xtrain.shape[0],3))
  u_q[:,0,None] = 4.5*np.cos(2*np.pi*Xtrain + 1.5*np.pi) - \
                             3*np.sin(4.3*np.pi*Xtrain + 0.3*np.pi)

  u_q[:,1,None] = 4.5*np.cos(1.5*np.pi*Xtrain + 0.5*np.pi) + \
                     5*np.sin(3*np.pi*Xtrain + 1.5*np.pi)

  u_q[:,2,None] = 1

  # Now, we define the combination parameters.
  W = []
  # q=1
  Wq1 = np.array(([[0.4],[-1.7],[-0.5],[0],[0.7]]))
  W.append(Wq1)
  # q=2
  Wq2 = np.array(([[0.4],[1.0],[-0.1],[-0.8],[-1.0]]))
  W.append(Wq2)
  # q=3
  Wq3 = np.array(([[3.6],[-5.6],[-0.0],[1.2],[10.0]]))
  W.append(Wq3)


  F_r = []
  Lam_r = []
  for r in range(R):
      f_r = np.zeros((Xtrain.shape[0], 1))
      # rho_r = np.zeros((Xtrain.shape[0], 1))
      for q in range(3):
          f_r += W[q][r].T*u_q[:,q,None]
      F_r.append(f_r)
      lam_r = Sigmoid(f_r)
      lam_r[lam_r>0.5] = 1
      lam_r[lam_r<=0.5] = 0
      Lam_r.append(lam_r)

  seed = 0
  np.random.seed(seed)
  Ytrain = np.ones((N, R))
  for r in range(R):
      aux = ytrain.copy()
      for n in range(N):
          if Lam_r[r][n] == 0:
              labels = np.arange(1, K+1)
              a = np.where(labels==ytrain[n])
              labels = np.delete(labels, a)
              idxlabels = np.random.permutation(K-1)
              aux[n] = labels[idxlabels[0]]
      Ytrain[:,r] = aux.flatten()

 # Ytrain = (Ytrain*maxy) + miny

  iAnn = np.zeros((N, R), dtype=int) # this indicates if the annotator r labels the nth sample.
  Nr = [int(np.floor(N*nrp)) for nrp in NrP]
  # Nr = np.ones((R), dtype=int)*int(np.floor(N*NrP))
  for r in range(R):
      if r < R-1:
          indexR = np.random.permutation(range(N))[:Nr[r]]
          iAnn[indexR,r] = 1
      else:
          iSimm = np.sum(iAnn, axis=1)
          idxZero = np.asarray([i for (i, val) in enumerate(iSimm) if val == 0])
          Nzeros = idxZero.shape[0]
          idx2Choose = np.arange(N)
          if Nzeros == 0:
              indexR = np.random.permutation(range(N))[:Nr[r]]
              iAnn[indexR,r] = 1
          else:
              idx2Choose = np.delete(idx2Choose, idxZero)
              N2chose = idx2Choose.shape[0]
              idxNoZero = np.random.permutation(N2chose)[:(Nr[r] - Nzeros)]
              idxTot = np.concatenate((idxZero, idx2Choose[idxNoZero]))
              iAnn[idxTot,r] = 1

  # Now, we verify that all the samples were labeled at least once
  Nr = (np.sum(iAnn,0))
  iSimm = np.sum(iAnn, axis=1)
  if np.asarray([i for (i, val) in enumerate(iSimm) if val == 0]).sum() == 0:
      ValueError("all the samples must be labeled at least once")

  # Finally, if iAnn=0 we assign a reference value to indicate a missing value
  Vref = -1e-20
  for r in range(R):
      Ytrain[iAnn[:,r] == 0, r] = Vref

  return Ytrain, iAnn, Lam_r, per
