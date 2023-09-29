import matplotlib.pyplot as plt
import numpy as np

from Analysis import *
from GraphicsFrequency import *

from scipy.integrate import *
from numpy.fft import fft, fftfreq


# Турбина К-200-130
I1 = 4.296*10**2
I2 = 2.779*10**3
I3 = 6.158*10**3
I4 = 3.56*10**3
C1 = 4.78*10**7
C2 = 5.2*10**7
C3 = 7.23*10**7
l1 = 4.720
l2 = 4.321
l3 = 4.860
l = [l1, l2, l3]

C = np.array([[-C1, C1, 0.0, 0.0], [C1, -C1-C2, C2, 0.0], [0.0, C2, -C2-C3, C3], [0.0, 0.0, C3, -C3]])
J = np.array([[I1, 0.0, 0.0, 0.0], [0.0, I2, 0.0, 0.0], [0.0, 0.0, I3, 0.0], [0.0, 0.0, 0.0, I4]])
D = np.array([[3000, 00.0, 00.0, 00.0], [0.0, 3000, 0.0, 0.0], [0.0, 0.0, 3000, 0.0], [0.0, 0.0, 0.0, 3000]])


V1, V2, V3, V4 = Analysis(C, J).formation_of_natural_frequencies_and_shapes()
GraphicsFrequency(V1, l).plotting(1)
GraphicsFrequency(V2, l).plotting(2)
GraphicsFrequency(V3, l).plotting(3)
GraphicsFrequency(V4, l).plotting(4)
plt.grid()
plt.show()

# Построение АЧХ
A = np.array([[], [], [], []])
w = np.arange(1, 400, 0.025001142864743)
for i in w:
    J1 = J*i**2
    v1 = np.array([[0], [0], [0], [1000]])
    a = np.linalg.solve((J1 + C), v1)
    A = np.hstack((A, a))
plt.figure(5)
plt.axis([0, 400, -0.001, 0.001])
plt.plot(w, A[0, :])
plt.title(f'А1')
plt.grid()
plt.figure(6)
plt.axis([0, 400, -0.001, 0.001])
plt.plot(w, A[1, :])
plt.title(f'А2')
plt.grid()
plt.figure(7)
plt.axis([0, 400, -0.001, 0.001])
plt.plot(w, A[2, :])
plt.title(f'А3')
plt.grid()
plt.figure(8)
plt.axis([0, 400, -0.001, 0.001])
plt.plot(w, A[3, :])
plt.title(f'А4')
plt.grid()
plt.show()


Jj = np.linalg.inv(J)

k_1 = np.zeros((4, 4))
k_2 = np.eye(4)
k_3 = np.nan_to_num(np.dot(Jj, C))
k_4 = np.nan_to_num(np.dot(Jj, -D))

K_1 = np.hstack((k_1, k_2))
K_2 = np.hstack((k_3, k_4))

K = np.vstack((K_1, K_2))


def pend(y, t, K, M):
    # Время импульса tt1
    tt1 = 0.2
    sol1 = np.dot(K, y)
    # Внешняя нагрузка в виде прямоугольного импульса
    if t > 0 and t <= tt1:
        sol = np.transpose(sol1)+[0, 0, 0, M, 0, 0, 0, 0]
    elif t > tt1:
        sol = np.transpose(sol1)
    elif t == 0:
        sol = np.transpose(sol1)
    # Внешняя нагрузка в виде тригонометрической функции
    sol = np.transpose(sol1) + [0, 0, 0, M*np.cos(314*t), 0, 0, 0, 0]
    # Внешняя нагрузка в виде суммы двух тригонометрических функций
    sol = np.transpose(sol1) + [0, 0, 0, M*(np.sin(314*t)-1/2*np.sin(2*314*t)), 0, 0, 0, 0]
    return sol


# Начальные условия
y0 = np.array([0, 0, 0, 0, 0, 0, 0, 0])
# Временной интервал интегрирования
tt2 = 1.0
t1 = np.arange(0, tt2, 0.000001)
# Решение системы дифференциальных уравнений
sol = odeint(pend, y0, t1, args=(K, (3*0.65*10**6)*Jj[3][3]))


G = 80 * 10**9
d1 = (32*C1*l1/(G*np.pi))**(1/4)
d2 = (32*C2*l2/(G*np.pi))**(1/4)
d3 = (32*C3*l3/(G*np.pi))**(1/4)
Wp1 = (np.pi*d1**3)/16
Wp2 = (np.pi*d2**3)/16
Wp3 = (np.pi*d3**3)/16

# Пострение спектра
N1 = len((sol[:, 1]-sol[:, 0])*C1/Wp1)
yf1 = fft((sol[:, 1]-sol[:, 0])*C1)/N1*2

N2 = len((sol[:, 2]-sol[:, 1])*C2/Wp2)
yf2 = fft((sol[:, 2]-sol[:, 1])*C2)/N2*2

N3 = len((sol[:, 3]-sol[:, 2])*C3/Wp3)
yf3 = fft((sol[:, 3]-sol[:, 2])*C3)/N3*2

xf1 = fftfreq(N1, tt2/N1)
xf2 = fftfreq(N2, tt2/N2)
xf3 = fftfreq(N3, tt2/N3)

plt.figure(9)
plt.plot(xf1, (np.abs(yf1))/10**6, 'r')
plt.xlim([0, 150])
plt.grid()
plt.ylabel('$𝜏$1, МПа')
plt.xlabel('Частота, Гц')
plt.legend(loc='best')
plt.figure(10)
plt.plot(xf2, (np.abs(yf2))/10**6, 'g')
plt.xlim([0, 150])
plt.grid()
plt.ylabel('$𝜏$2, МПа')
plt.xlabel('Частота, Гц')
plt.legend(loc='best')
plt.figure(11)
plt.plot(xf3, (np.abs(yf3))/10**6, 'b')
plt.xlim([0, 150])
plt.grid()
plt.ylabel('$𝜏$3, МПа')
plt.xlabel('Частота, Гц')
plt.legend(loc='best')
plt.show()


# Касательные напряжения
Mk1 = ((sol[:, 1]-sol[:, 0])*C1)
Mk2 = ((sol[:, 2]-sol[:, 1])*C1)
Mk3 = ((sol[:, 3]-sol[:, 2])*C1)

tao1 = Mk1/Wp1
tao2 = Mk2/Wp2
tao3 = Mk3/Wp3

t = np.arange(0, 0.2, 0.0001)
plt.figure(12)
print(3*0.65/Wp1)
plt.plot(t1, Mk1/(10**6), 'r')
plt.legend(loc='best')
plt.ylabel('$𝜏$1, МПа')
plt.xlabel('t, с')
plt.grid()
plt.xlim([0, 0.4])
plt.figure(13)
plt.plot(t1, Mk2/(10**6), 'g')
plt.legend(loc='best')
plt.ylabel('$𝜏$2, МПа')
plt.xlabel('t, с')
plt.grid()
plt.xlim([0, 0.4])
plt.figure(14)
plt.plot(t1, Mk3/(10**6), 'b')
plt.legend(loc='best')
plt.ylabel('$𝜏$3, МПа')
plt.xlabel('t, с')
plt.xlim([0, 0.4])
plt.grid()
plt.show()




# Виды нагрузок

# Тригонометрическая функция
plt.plot(t1, (3*0.65*10**6)*np.cos(314*t1), 'b')
plt.ylabel('Mp, Нм')
plt.xlabel('t, с')
plt.xlim([0, 0.1])
plt.grid()
plt.show()

# Сумма двух тригонометрических функций

plt.plot(t1, (3*0.65*10**6)*(np.sin(314*t1)-1/2*np.sin(2*314*t1)), 'b')
plt.ylabel('Mp, Нм')
plt.xlabel('t, с')
plt.xlim([0, 0.1])
plt.grid()
plt.show()

# Прямоугольный импульс
t = np.arange(0, 0.3, 0.0001875)
M11 = np.dot(np.ones(100), (3*0.65*10**6))
Mm = np.zeros(50)
for i in range(len(t)):
    if t[i] <= 0.02:
        t1 = np.arange(0, 0.2, 0.002)
        plt.plot(t1, M11, 'b')
    else:
        t2 = np.arange(0.2, 0.3, 0.002)
        plt.plot(t2, Mm, 'b')

plt.xlabel('t, с')
plt.ylabel('Mp, Нм')
plt.xlim([0, 0.35])
plt.grid()
plt.show()

