import matplotlib.pyplot as plt

efficiency_gains = [0, 1, 2, 5, 10, 15, 20, 30, 50, 70, 80, 90]
net_impact_optimistic = [0.07, -0.15, -0.38, -1.06, -2.2, -3.33, -4.47, -6.74, -11.28, -15.82, -18.09, -20.36]
net_impact_realistic = [2.61, 2.38, 2.15, 1.47, 0.34, -0.8, -1.93, -4.20, -8.74, -13.28, -15.55, -17.82] 
net_impact_pessimistic = [68.69, 68.46, 68.23, 67.55, 66.42, 65.28, 64.15, 61.88, 57.34, 52.8, 50.53, 48.26]


plt.plot(efficiency_gains, net_impact_optimistic)
plt.plot(efficiency_gains, net_impact_realistic)
plt.plot(efficiency_gains, net_impact_pessimistic)
plt.hlines(0, xmin=0, xmax=100, linestyles='--')
plt.show()
