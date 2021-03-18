"""
Tp de Génie Mathématique Ma 223
TP 1 Méthode de Gauss pour la résolution de systèmes linéaires
Nadhir OFKIR
Tom POIZAT
Classe: 2PF1
"""

import numpy as np
import time
import matplotlib.pyplot as plt

list_nb = []

list_tps_gauss = []
list_tps_linalg = []
list_tps_LU = []
list_tps_pivotP = []
list_tps_pivotT = []

erreurgauss = []
erreurLU = []
erreurpivotP = []
erreurpivotT = []

# Algorithme de Gauss

def ReductionGauss(Aaug):
    n, m = np.shape(Aaug)

    for i in range(0, n-1):
        if Aaug[i,i] == 0 :
            Aaug[i, :] = Aaug[i+1]
        else :
            for j in range(i+1,n):
                Aaug[j, :] = Aaug[j, :] - (Aaug[j,i] / Aaug[i,i]) * Aaug[i, :]
    return Aaug


def ResolutionSystTriSup(Taug):
    n, m = np.shape(Taug)
    x = np.zeros(n)
    x[n-1] = Taug[n-1][m-1] / Taug[n-1][n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = Taug[i][m-1]
        for j in range(i+1, n):
            x[i] = x[i] - Taug[i][j] *x[j]
        
        x[i] = x[i] /Taug[i][i]

    return x


def Gauss(A,B):
    n, m = np.shape(A)
    Aaug = np.column_stack((A,B)) 

    Xgauss = ResolutionSystTriSup(ReductionGauss(Aaug))
    print("x gauss",Xgauss)

    Egauss = np.linalg.norm(np.dot(A,Xgauss) - (np.ravel(B)))
    erreurgauss.append(Egauss)
    print("\nerreur gauss",erreurgauss)

    return Xgauss


#LU

def ResolutionSystTriInf(Taug):
    n,m = np.shape(Taug)
    x=np.ones(n)
    for i in range (n):
        l=0
        for j in range(n):
            l+=Taug[i][j]*x[j]
        x[i]=(Taug[i][n]-l+Taug[i][i]) / Taug[i][i]

    return x


def DecompositionLU(A):
    U = np.copy(A)
    m,n = U.shape
    L = np.eye(n)
    
    for i in range (0, n-1):
        if U[i,i] == 0:
            return ("Erreur")

        else:
            for j in range(i+1,n):
                g = U[j,i] / U[i,i]
                U[j,:] = U[j,:] - g*U[i,:]
                L[j,i] = g
                  
    #print("L",L)
    #print("U",U)
    return(L,U)


def ResolutionLU(L,U,B,A):
    Laug = np.concatenate((L,B),axis = 1)

    RSTI = ResolutionSystTriInf(Laug)
    RSTI = np.reshape(RSTI,(n,1))

    XLU = ResolutionSystTriSup(np.concatenate((U,RSTI),axis=1))

    print("\nx LU",XLU)

    ELU = np.linalg.norm(np.dot(A,XLU) - (np.ravel(B)))
    erreurLU.append(ELU)
    print("\nerreur LU",erreurLU)
    
    return XLU


#Pivot partiel

def GaussChoixPivotPartiel(A,B):
    Aaug = np.hstack((A,B))
    n,m = Aaug.shape

    for i in range(0, n-1):
        for j in range(i+1, n):
            if abs(Aaug[j,i]) > abs(Aaug[i,i]) :  
                T = Aaug[i, :].copy()  
                Aaug[i, :] = Aaug[j, :] 
                Aaug[j, :] = T

            g = Aaug[j][i]/Aaug[i][i]
            Aaug[j] = Aaug[j] - g*Aaug[i]

    X = ResolutionSystTriSup(Aaug)
    print("\nX pivot partiel",X)

    EpivotP = np.linalg.norm(np.dot(A,X) - (np.ravel(B)))
    erreurpivotP.append(EpivotP)
    print("\nerreur pivot P", erreurpivotP)

    return X


#Pivot total

def GaussChoixPivotTotal(A, B):
    A2 = np.copy(A)   # On copie la matrice A pour pouvoir utiliser la matrice A dans le calcul de l'erreur 
    A2 = np.column_stack((A2, B))
    n, m = A2.shape
    x = np.zeros(n)

    for i in range(n):
        for j in range(0, n-i):
            if  abs(A2[i+j, i]) > abs(A2[i, i]):
                T = A2[i, :].copy()
                A2[i, :] = A2[i+j, :]
                A2[i+j, :] = T
            sol = A2[i,i]

        if sol != 0:
            for k in range(i, n-1):
                A2[k+1, :] = A2[k+1, :] - (A2[k+1, i] / sol)*A2[i, :]

    for k in range(n-1, -1, -1):
        S = 0
        for j in range(k+1, n):
            S = S + A2[k,j] * x[j]
        x[k] = (A2[k, m-1] - S) / A2[k, k]

    EpivotT = np.linalg.norm(np.dot(A, x) - (np.ravel(B)))
    erreurpivotT.append(EpivotT)
    print("\nx pivot T", x)
    print("\nerreur pivot T", erreurpivotT)

    return x


#taille=[1]
taille =[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
for i in taille:

    A = np.random.rand(i,i)
    B = np.random.rand(i,1)
    #A = np.array([[1.0,1.0,1.0,1.0],[2.0,4.0,-3.0,2.0],[-1.0,-1.0,0.0,-3.0],[1.0,-1.0,4.0,9.0]])
    #B = np.array([[1.0],[1.0],[2.0],[-8.0]])

    n = len(A)
    list_nb.append(n)

    #print("Matrice A :\n", A, "\n")
    #print("Matrice B :\n", B, "\n")
    
    #================GAUSS=========================
    t_start_gauss = time.time()

    Gauss(A, B)

    t_final_gauss = time.time()
    t_total_gauss = t_final_gauss - t_start_gauss
    list_tps_gauss.append(t_total_gauss)
    
    #================Linalg=========================
    t_start_linalg = time.time()

    S_lin = np.linalg.solve(A, B)
    print("\nX linalg", np.ravel(S_lin),"\n")

    t_final_linalg = time.time()
    t_total_linalg = t_final_linalg - t_start_linalg
    list_tps_linalg.append(t_total_linalg)
    
    #===================LU==========================
    t_start_LU = time.time()

    L,U = DecompositionLU(A)
    ResolutionLU(L,U,B,A)

    t_final_LU = time.time()
    t_total_LU = t_final_LU - t_start_LU
    list_tps_LU.append(t_total_LU)
    
    #==============Pivot Partiel=====================
    t_start_pivotP = time.time()

    GaussChoixPivotPartiel(A,B)

    t_final_pivotP = time.time()
    t_total_pivotP = t_final_pivotP - t_start_pivotP
    list_tps_pivotP.append(t_total_pivotP)
    
    #==============Pivot Total=======================
    t_start_pivotT = time.time()

    GaussChoixPivotTotal(A,B)

    t_final_pivotT = time.time()
    t_total_pivotT = t_final_pivotT - t_start_pivotT
    list_tps_pivotT.append(t_total_pivotT)
    
    print("\n Matrice suivante \n")
    

def graph_tps():
    x = list_nb 
    y1 = list_tps_gauss
    y2 = list_tps_linalg
    y3 = list_tps_LU
    y4 = list_tps_pivotP
    y5 = list_tps_pivotT
    
    plt.title("Temps d'exécution du programme en fonction de la taille de la matrice A")
    plt.plot(x, y1, label="Gauss",color="b")
    plt.plot(x, y2, label="Linalg",color="k")
    plt.plot(x, y3, label="LU",color="g")
    plt.plot(x, y4, label="Pivot Partiel",color="r")
    plt.plot(x, y5, label="Pivot Total",color="y")
    plt.legend()

    plt.xlabel("Taille de la matrice A (nombre de ligne/colonne)")
    plt.ylabel("Temps d'exécution du calcul (en seconde)")

    plt.grid()
    plt.show()


def graph_tps_loglog():
    x = list_nb 

    y1 = list_tps_gauss
    y2 = list_tps_linalg
    y3 = list_tps_LU
    y4 = list_tps_pivotP
    y5 = list_tps_pivotT
    
    plt.title("Temps d'exécution du programme en fonction de la taille de la matrice A")
    plt.loglog(x, y1, label="Gauss",color="b")
    plt.loglog(x, y2, label="Linalg",color="k")
    plt.loglog(x, y3, label="LU",color="g")
    plt.loglog(x, y4, label="Pivot Partiel",color="r")
    plt.loglog(x, y5, label="Pivot Total",color="y")
    plt.legend()

    plt.xlabel("Taille de la matrice A (nombre de ligne/colonne)")
    plt.ylabel("Temps d'exécution du calcul (en seconde)")

    plt.grid()
    plt.show()


def graph_erreur():

    x_erreur = list_nb
    y_erreurgauss = erreurgauss
    y_erreurLU = erreurLU
    y_erreurpivotP = erreurpivotP
    y_erreurpivotT = erreurpivotT

    plt.title("Erreur des différentes fonctions en fonction de la taille de la matrice A")
    plt.semilogy(x_erreur, y_erreurgauss, label="Erreur Gauss",color="b")
    plt.semilogy(x_erreur, y_erreurLU, label="Erreur LU",color="g")
    plt.semilogy(x_erreur, y_erreurpivotP, label="Erreur Pivot Partiel",color="r")
    plt.semilogy(x_erreur, y_erreurpivotT, label="Erreur Pivot Total",color="y")
    plt.legend()

    plt.xlabel("Taille de la matrice A (nombre de ligne/colonne)")
    plt.ylabel("Erreur ||AX-B||")

    plt.grid()
    plt.show()


graph_tps()
graph_tps_loglog()
graph_erreur()
