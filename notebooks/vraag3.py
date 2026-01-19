
#%%
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 400)
y = -x + (1 + np.pi**2 / 4) * (np.arctan(x + np.pi / 2) - np.arctan(np.pi / 2))

# plot of the function
plt.plot(x, y, lw=3, label=r'$f$')

# axes
plt.axhline(0, color='black', lw=.2)
plt.axvline(0, color='black', lw=.2)

plt.plot([-np.pi, -np.pi], [-10, 10], 'k--', lw=1)  # Local minimum at x = -π
plt.plot([0, 0], [-10, 10], 'k--', lw=1)  # Local maximum at x = 0

plt.plot([-np.pi/2, -np.pi/2], [-10, 10], 'r-.', lw=1)  # inflection point at x = -π/2

# add info to vertical lines
offset = 0.25
vert_pos = -9
plt.text(-np.pi - offset, vert_pos, 'Lokaal minimum', ha='center', va='bottom', rotation=90, fontsize=8)
plt.text(0 - offset, vert_pos, 'Lokaal maximum', ha='center', va='bottom', rotation=90, fontsize=8)
plt.text(-np.pi/2 - offset, vert_pos, 'Buigpunt', ha='center', va='bottom', rotation=90, fontsize=8)

plt.text(8.5, offset, r'$y=0$', ha='center', va='bottom', fontsize=8, color='grey')
plt.text(-.3-offset, 8.2, r'$x=0$', ha='center', va='bottom', fontsize=8, color='grey', rotation=90)

plt.plot([-10, 10], [10, -10], 'g:', lw=1)  # y = -x line
plt.text(-7.5, 7.5, r'$y = -x$', ha='center', va='bottom', fontsize=8, color='green', rotation=-45)

# clean up the plot
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xticks(np.arange(-10, 11, 2))
plt.yticks(np.arange(-10, 11, 2))

plt.xlabel(r'x')
plt.ylabel(r'$y$')

plt.legend()

# make the figur square
plt.gca().set_aspect('equal', adjustable='box')

plt.savefig('functie_vraag3.png', dpi=300)