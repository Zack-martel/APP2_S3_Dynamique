import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin
from scipy.integrate import solve_ivp
import sympy as sp
import sympy.physics.mechanics as mec

# Définition des variables globales
LA = 0.07
LBcm = 0.242
LBhitch = 0.472
LCw = 0.206
LCcm = 0.0206
rW = 0.02
wTrailer = 0.2

# Lecture des fichier textes
data_stagiaire = np.loadtxt("Data_Stagiaire.txt", comments="%")
solution_qB = np.loadtxt("Solution_qB.txt", comments="%")

t = data_stagiaire[:, 0]
qc = data_stagiaire[:, 6]
qb = solution_qB[:, 1]

# Définition des référentielle
N = mec.ReferenceFrame("N")
B = mec.ReferenceFrame("B")
C = mec.ReferenceFrame("C")

"""# Définition des points d'intérêt
Ao = mec.Point("Ao")

Bo = mec.Point("Bo")
Bo.set_pos(Ao, (LA*cos(qb)*N.y) - (LA*sin(qb)*N.x))

Bcm = mec.Point("Bcm")
Bcm.set_pos(Bo, (-LBcm*B.x))

Bc = mec.Point("Bc")
Bc.set_pos(Bo, (-LBhitch*B.x))

Cb = mec.Point("Cb")
Cb.set_pos(Bc, 0)

Ccm = mec.Point("Ccm")
Ccm.set_pos(Cb, (-LCcm*C.x))

Cw = mec.Point("Cw")
Cw.set_pos(Ccm, (-wTrailer/2)*C.y)

Wt= mec.Point("Wt")
Wt.set_pos(Cw, (-rW*C.z))

# Définition des vecteurs position
r_BowAo = Bo.pos_from(Ao)
r_BcwBo = Bc.pos_from(Bo)
r_CcmwBc = Ccm.pos_from(Bc)
r_CwwCcm = Cw.pos_from(Ccm)

r_CwwAo = r_BowAo + r_BcwBo + r_CcmwBc + r_CwwCcm"""

# Fonction pour déterminer la vitesse linéaire de Cw
def vitesse_lin_Cw(qb, dqb, qc, dqc):
    dr_BowA0 = LA * (-(cos(qb) * dqb)*N.x - (sin(qb) * dqb)*N.y)
    dr_BcwBo = (-dqb * LBhitch)*B.y
    dr_CcmwBc = (-(dqb + dqc) * LCcm)*C.y
    dr_CwwCcm = ((dqb + dqc) * (wTrailer / 2))*C.x
    v_N_Cw = dr_BowA0 + dr_BcwBo + dr_CcmwBc + dr_CwwCcm
    return v_N_Cw

# Fonction pour déterminer l'accélération linéaire de Cw
def acceleration_lin_Cw(qb, dqb, ddqb, qc, dqc, ddqc):
    ddr_BowA0 = LA * (((sin(qb) * (dqb**2)) - (cos(qb) * ddqb))*N.x + ((-cos(qb) * (dqb**2)) - (sin(qb) * ddqb))*N.y)
    ddr_BcwBo = LBhitch * ((dqb**2)*B.x - (ddqb)*B.y)
    ddr_CcmwBc = LCcm * (((dqb + dqc)**2)*C.x - (ddqb + ddqc)*C.y)
    ddr_CwwCcm = (wTrailer / 2) * ((ddqb + ddqc)*C.x + ((dqb + dqc)**2)*C.y)
    a_N_Cw = ddr_BowA0 + ddr_BcwBo + ddr_CcmwBc + ddr_CwwCcm
    return (a_N_Cw.dot(N.x)), (a_N_Cw.dot(N.y)), (a_N_Cw.dot(N.z)), 

# Fonction pour calculer la dérivée première
def derivee_1_num(f):
    df = np.zeros(len(f))
    for i in range(0, len(f)):
        if i == 0:
            h = f[i + 1] - f[i]
            df[i] = (f[i + 1] - f[i]) / h
        if i == (len(f) - 1):
            h = f[i] - f[i - 1]
            df[i] = (f[i] - f[i - 1]) / h
        else:
            h = f[i + 1] - f[i]
            df[i] = (f[i + 1] - f[i - 1]) / (2 * h)  
    return df

# Fonction pour calculer la dérivée seconde
def derivee_2_num(df):
    ddf = np.zeros(len(df))
    for i in range(0, len(df)):
        if i == 0:
            h = df[i + 1] - df[i]
            ddf[i] = (df[i + 1] - df[i]) / h
        if i == (len(df) - 1):
            h = df[i] - df[i - 1]
            ddf[i] = (df[i] - df[i - 1]) / h
        else:
            h = df[i + 1] - df[i]
            ddf[i] = (df[i + 1] - df[i - 1]) / (2 * h)  
    return ddf


qb_1 = derivee_1_num(qb)
qb_2 = derivee_2_num(derivee_1_num(qb))

qc_1 = derivee_1_num(qc)
qc_2 = derivee_2_num(derivee_1_num(qc))

acc_lin = np.zeros((len(t), 3))

for i in range(0, len(t)):
    B.orient_axis(N, N.z, qb[i])
    C.orient_axis(B, B.z, qc[i])

    acc_lin[i][0] = acceleration_lin_Cw(qb[i], qb_1[i], qb_2[i], qc[i], qc_1[i], qc_2[i])[0]
    acc_lin[i][1] = acceleration_lin_Cw(qb[i], qb_1[i], qb_2[i], qc[i], qc_1[i], qc_2[i])[1]
    acc_lin[i][2] = acceleration_lin_Cw(qb[i], qb_1[i], qb_2[i], qc[i], qc_1[i], qc_2[i])[2]

acc_max = 0
for i, element in enumerate(acc_lin):
    if np.linalg.norm(element) >= acc_max:
        acc_max = element
        indexe = i

print(f"Accélération linéaire maximale au point Cw : {np.max(acc_lin, axis=0)}N.x + {np.max(acc_lin, axis=0)}N.y")