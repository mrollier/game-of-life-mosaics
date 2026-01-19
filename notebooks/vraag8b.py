
#%%
import numpy as np
import matplotlib.pyplot as plt

a=3
epsilon=0.01
x = np.linspace(epsilon, 2*a-epsilon, 400)
y = np.sqrt(x**3 / (2*a - x))

# plot of the function
plt.plot(x, y, lw=3, label=r'Cissoïde', color='C0')
plt.plot(x, -y, lw=3, color='C0')

# plot the box
index = 250
jump = 15
plt.fill_between([x[index],x[-1]], [y[index]-2, y[index]-2],[y[index], y[index]], lw=2, color='C1', zorder=5, label='Strook')

# add a hor arrow from x=6 to the orange box
plt.annotate('',
             xy=(x[index]-.05, -1),
             xytext=(2*a, -1),
             arrowprops=dict(arrowstyle='<->', color='k',lw=1),
             zorder=6)
plt.text(4.9, -3.8, r'$2a-x$', fontsize=12, ha='center', va='bottom')

# add a dx arrow on top of the orange box
# offset = .12
# plt.annotate('',
#              xy=(x[index]-offset, y[index]+2),
#              xytext=(x[index+jump]+offset, y[index]+2),
#              arrowprops=dict(arrowstyle='<->', color='k',lw=1),
#              zorder=6)
# plt.text(x[index]+.1, y[index]+3, r'$dx$', fontsize=12, ha='center', va='bottom')

# add a vert arrow from y=0 to the orange box
plt.annotate('',
             xy=(2*a+0.2, 2),
             xytext=(2*a+0.2, 5.7),
             arrowprops=dict(arrowstyle='<->', color='k',lw=1),
             zorder=6)
plt.text(2*a+.5, 2.5, r'$dy$', fontsize=12, ha='center', va='bottom')

# axes
plt.axhline(0, color='black', lw=.2)
plt.axvline(0, color='black', lw=.2)

plt.axvline(2*a, color='k', lw=1, ls='--')

# add info to vertical lines
offset = 0.15
vert_pos = -15

plt.text(7.5, -2.5, r'$y=0$', ha='center', va='bottom', fontsize=8, color='grey')
plt.text(-.2, 8.2, r'$x=0$', ha='center', va='bottom', fontsize=8, color='grey', rotation=90)

plt.text(2*a - offset, vert_pos, r'$x=2a$', ha='center', va='bottom', rotation=90, fontsize=8)

# clean up the plot
plt.xlim(-1, 8)
plt.ylim(-25, 25)
plt.xticks([0, 2, 4, 6, 8])
plt.yticks([-20, -10, 0, 10, 20])

plt.xlabel(r'x')
plt.ylabel(r'$y$')

# locate legend in upper center
plt.legend(loc='upper center')

# make the figur square
# plt.gca().set_aspect('equal', adjustable='box')

plt.savefig('functie_vraag8b.png', dpi=300)