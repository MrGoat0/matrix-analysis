import numpy as np
import math
qwer = 0
jointData, supportData, materialData, crossSeccionData, memberData, loadData = [
], [], 0, 0, [], []


def main():
    global qwer
    print("*****Computer Program for Analysis of Plane Trusses*****".upper() +
          '\n'+'\n'+"Select a option (1 - 8) or 'E' to exit" +
          '\n'+"1. Joint data" +
          '\n'+"2. Support data" +
          '\n'+"3. Material property data" +
          '\n'+"4. Cross-sectional property data"
          + '\n'+"5. Member data"
          + '\n'+"6. Load data" +
          '\n'+"7. Print data" +
            '\n'+"8. Run analysis!" +
          '\n'+"E: Exit (clean data)")
    x = str(input())
    if x == "E":
        qwer = 1
        return
    elif x == "1":
        jointAnalysis()
    elif x == "2":
        supportAnalysis()
    elif x == "3":
        materialAnalysis()
    elif x == "4":
        crossSeccionAnalysis()
    elif x == "5":
        memberAnalysis()
    elif x == "6":
        loadAnalysis()
    elif x == "7":
        printData()
    elif x == "8":
        analysis()
    else:
        print("Try again...")
    if qwer != 1:
        main()


def jointAnalysis():
    global jointData
    print("*****Joint Data*****".upper() + '\n'+"Enter the number of joints:")
    n = int(input())
    print("Enter the coordinates of the joints:")
    for i in range(n):
        print("Joint", i+1, ":")
        x = float(input("Coordinate x: "))
        y = float(input("coordinate y: "))
        jointData.append([x, y])
    print("Joint data saved")


def supportAnalysis():
    global supportData
    print("*****Support Data*****".upper() +
          '\n'+"Enter the number of supports:")
    n = int(input())
    print("Enter the coordinates of the supports:")
    for i in range(n):
        print("Support", i+1, ":")
        jointMember = float(input("Joint support: "))
        x = float(input("Restaint both directin (1=Free, 0=Restrained): "))
        supportData.append([jointMember, x])
    print("Support data saved")
    # print(supportData)


def materialAnalysis():
    global materialData
    print("*****Material Data*****".upper())
    E = float(input("Modulus of elasticity: "))
    # I = float(input("Moment of inertia: "))
    materialData = E
    print("Material data saved")
    # print(materialData)


def crossSeccionAnalysis():
    global crossSeccionData
    print("*****Cross-sectional Data*****".upper())
    A = float(input("Area: "))
    # I = float(input("Moment of inertia: "))
    crossSeccionData = A
    print("Cross-sectional data saved")
    # print(crossSeccionData)


def memberAnalysis():
    global memberData
    print("*****Member Data*****".upper() +
          '\n'+"Enter the number of members:")
    n = int(input())
    print("Enter the member joints:")
    for i in range(n):
        print("Member", i+1, ":")
        start = float(input("Beginning joint number: "))
        end = float(input("End joint number: "))
        memberData.append([start, end])
    print("Member data saved")
    # print(memberData)


def loadAnalysis():
    global loadData

    loadData = np.zeros_like(jointData)
    print("*****Load Data*****".upper() +
          '\n'+"Enter the number of loads:")
    n = int(input())
    print("Enter the load properties:")
    for i in range(n):
        print("Load", i+1, ":")
        joint = int(input("Joint number: "))
        direction = int(input("0: x-axis, 1: y-axis: "))
        magnitude = float(input("Magnitude force: "))
        loadData[joint, direction] = magnitude
    print("Load data saved")
    # print(loadData)


def printData():
    print("*****Data imput*****".upper() + '\n'+"Joint data:")
    print(jointData)
    print("Support data:")
    print(supportData)
    print("Material data:")
    print(materialData)
    print("Cross-sectional data:")
    print(crossSeccionData)
    print("Member data:")
    print(memberData)
    print("Load data:")
    print(loadData)


def analysis():
    ur = []
    for i in range(len(supportData)):
        ur.append(supportData[i][1])
        ur.append(supportData[i][1])

    DOFCON = np.ones_like(jointData).astype(int)
    for i in range(len(supportData)):
        if supportData[i][1] != 1.0:
            DOFCON[int(supportData[i][0]), :] = supportData[i][1]
    NN = len(jointData)
    NE = len(memberData)
    DOF = 2
    NDOF = DOF*NN

    d = jointData[memberData[:, 1], :] - jointData[memberData[:, 0], :]
    L = np.sqrt((d**2).sum(axis=1))
    angle = (d.T)/L
    a = np.concatenate((-angle.T, angle.T), axis=1)
    K = np.zeros([NDOF, NDOF])
    for i in range(NE):
        aux = 2*memberData[i, :]
        index = np.r_[aux[0]:aux[0]+2, aux[1]:aux[1]+2]

        ES = np.dot(a[i][np.newaxis].T*materialData *
                    crossSeccionData, a[i][np.newaxis])/L[i]
        K[np.ix_(index, index)] = K[np.ix_(index, index)] + ES

    freeDOF = DOFCON.flatten().nonzero()[0]
    supportDOF = (DOFCON.flatten() == 0).nonzero()[0]
    kff = K[np.ix_(freeDOF, freeDOF)]
    kfr = K[np.ix_(freeDOF, supportDOF)]
    krf = kfr.T
    krr = K[np.ix_(supportDOF, supportDOF)]

    pf = loadData.flatten()[freeDOF]
    uf = np.linalg.solve(kff, pf)
    U = DOFCON.astype(float).flatten()
    U[freeDOF] = uf
    U[supportDOF] = ur
    U = U.reshape(NN, DOF)
    u = np.concatenate((U[memberData[:, 0]], U[memberData[:, 1]]), axis=1)
    N = materialData*crossSeccionData/L[:]*(a[:]*u[:]).sum(axis=1)
    R = (krf[:]*uf).sum(axis=1) + (krr[:]*ur).sum(axis=1)
    R = R.reshape(2, DOF)
    N, R, U = np.array(N), np.array(R), U
    print(N)
    print(R)
    print(U)


main()
