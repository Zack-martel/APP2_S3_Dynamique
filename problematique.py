import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import sympy.physics.vector as vec

# Définition des variables globales
LA = 0.07
LBcm = 0.242
LBhitch = 0.472
LCw = 0.206
LCcm = 0.0206
rW = 0.02
wTrailer = 0.2

v_Tapis = 2.222

# Lecture des fichier textes
data_stagiaire = np.loadtxt("Data_Stagiaire.txt", comments="%")
solution_qB = np.loadtxt("Solution_qB.txt", comments="%")

t_val = data_stagiaire[:, 0]
qC_val = data_stagiaire[:, 6]
qB_val = solution_qB[:, 1]
Box_val = data_stagiaire[:, 1]
Boy_val = data_stagiaire[:, 2]
wb_val = data_stagiaire[:, 5]

Box, Boy = vec.dynamicsymbols("Box Boy")

qB, qC = vec.dynamicsymbols("qB qC")
dqB, dqC = vec.dynamicsymbols("qB qC", 1)

# Définition des référentielle
N = vec.ReferenceFrame("N")
B = N.orientnew("B", "Axis", [qB, N.z])
C = B.orientnew("C", "Axis", [qC, B.z])

B.set_ang_vel(N, (dqB * N.z))
C.set_ang_vel(B, (dqC * B.z))

Ao = vec.Point("Ao")
Ao.set_vel(N, 0)

Bo = Ao.locatenew("Bo", (Box*N.x + Boy*N.y))
Bo.set_vel(B, 0)

Bc = Bo.locatenew("Bc", (-LBhitch*B.x))
Bc.set_vel(B, 0)

Cb = Bc.locatenew("Cb", 0)
Cb.set_vel(C, 0)

Ccm = Cb.locatenew("Ccm", (-LCcm*C.x))
Ccm.set_vel(C, 0)

Cw = Ccm.locatenew("Cw", (-(wTrailer / 2)*C.y))
Cw.set_vel(C, 0)

acc_lin = Cw.a2pt_theory(Ao, N, C)

print(f"acc_lin = {acc_lin}")

def derivee_num(f):
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

# Définition des variables de substitution
dqc_val = derivee_num(qC_val)
ddqc_val = derivee_num(dqc_val)

ddqb_val = derivee_num(wb_val)

t_symbol = vec.dynamicsymbols._t

ddqb = sp.Derivative(dqB, t_symbol)

ddqc = sp.Derivative(dqC, t_symbol)

Box_dot_val = derivee_num(Box_val)
Boy_dot_val = derivee_num(Boy_val)

Box_dot = sp.Derivative(Box, (t_symbol, 2))
Boy_dot = sp.Derivative(Boy, (t_symbol, 2))

# Résolution
acc_lineaire = np.zeros((len(t_val), 3))

for i in range(0, len(t_val)):
    acc_num = acc_lin.subs([
        (ddqb, ddqb_val[i]),
        (ddqc, ddqc_val[i]),
        (Box_dot, Box_dot_val[i]),
        (Boy_dot, Boy_dot_val[i]),
        (Box, Box_val[i]),
        (Boy, Boy_val[i]),
        (qB, qB_val[i]),
        (qC, qC_val[i]),
        (dqB, wb_val[i]),
        (dqC, dqc_val[i])
    ])

    #print(acc_num)

    acc_lineaire[i, 0] = (acc_num.dot(N.x)).subs([(qB, qB_val[i]), (qC, qC_val[i])])
    acc_lineaire[i, 1] = (acc_num.dot(N.y)).subs([(qB, qB_val[i]), (qC, qC_val[i])])
    acc_lineaire[i, 2] = (acc_num.dot(N.z)).subs([(qB, qB_val[i]), (qC, qC_val[i])])


fig1, axes = plt.subplots(2, 1, sharex=True)

ax = axes[0]
ax.plot(t_val, acc_lineaire[:, 1], color="red")
ax.grid()
ax.set_ylabel("Accélération (m/s²)")
ax.set_title("Composante Cy de l'accélération linéaire au point Cw")

plt.show()


