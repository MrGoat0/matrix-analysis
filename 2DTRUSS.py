import numpy as np
import math
E = 29000
A = 0.0555556
nodes = []
bars = []

nodes.append([0, 0])
nodes.append([12, 0])
nodes.append([24, 0])
nodes.append([12, 16])

bars.append([0, 3])
bars.append([1, 3])
bars.append([2, 3])

nodes = np.array(nodes).astype(float)
bars = np.array(bars)

P = np.zeros_like(nodes)
P[3, 1] = -300
P[3, 0] = -150


DOFCON = np.ones_like(nodes).astype(int)
DOFCON[0, :] = 0
DOFCON[1, :] = 0
DOFCON[2, :] = 0
ur = [0, 0, 0, 0, 0, 0]


def trussAnalysis():
    NN = len(nodes)
    NE = len(bars)
    DOF = 2
    NDOF = DOF*NN

    d = nodes[bars[:, 1], :] - nodes[bars[:, 0], :]
    L = np.sqrt((d**2).sum(axis=1))
    angle = (d.T)/L
    a = np.concatenate((-angle.T, angle.T), axis=1)
    K = np.zeros([NDOF, NDOF])
    for i in range(NE):
        aux = 2*bars[i, :]
        index = np.r_[aux[0]:aux[0]+2, aux[1]:aux[1]+2]

        ES = np.dot(a[i][np.newaxis].T*E*A, a[i][np.newaxis])/L[i]
        K[np.ix_(index, index)] = K[np.ix_(index, index)] + ES

    freeDOF = DOFCON.flatten().nonzero()[0]
    supportDOF = (DOFCON.flatten() == 0).nonzero()[0]
    kff = K[np.ix_(freeDOF, freeDOF)]
    kfr = K[np.ix_(freeDOF, supportDOF)]
    krf = kfr.T
    krr = K[np.ix_(supportDOF, supportDOF)]

    pf = P.flatten()[freeDOF]
    uf = np.linalg.solve(kff, pf)
    U = DOFCON.astype(float).flatten()
    U[freeDOF] = uf
    U[supportDOF] = ur
    U = U.reshape(NN, DOF)
    u = np.concatenate((U[bars[:, 0]], U[bars[:, 1]]), axis=1)
    N = E*A/L[:]*(a[:]*u[:]).sum(axis=1)
    R = (krf[:]*uf).sum(axis=1) + (krr[:]*ur).sum(axis=1)
    R = R.reshape(2, DOF)
    return np.array(N), np.array(R), U


N, R, U = trussAnalysis()
print(N)
print(R)
print(U)
