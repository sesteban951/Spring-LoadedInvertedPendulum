###########################################################################
# just messing with the leg kinematics
###########################################################################

import numpy as np
import matplotlib.pyplot as plt
import math

# TODO: try for 3D. shoudl be easy to do

if __name__ == '__main__':

    p_com = np.array([0, 2]).reshape(2, 1)
    v_com = np.array([-1, 0]).reshape(2, 1)

    p_foot = np.array([1, 0]).reshape(2, 1)
    v_foot = np.array([0, 0]).reshape(2, 1)

    # leg length state
    r_vec = p_foot - p_com
    rdot_vec = v_foot - v_com
    r = np.linalg.norm(r_vec)
    r_unit = r_vec / r
    rdot = -v_com.T @ r_unit

    # angle state
    theta = -np.arctan2(r_vec[0], -r_vec[1])
    # theta = -math.atan(r_vec[0]/-r_vec[1])
    r_x = r_vec[0]
    r_z = r_vec[1]
    rdot_x = rdot_vec[0]
    rdot_z = rdot_vec[1]
    thetadot = (r_z * rdot_x - r_x * rdot_z) / r

    print('*' * 50)
    print('r: ', r)
    print('rdot: ', rdot)
    print('angle: ', theta * (180/np.pi))
    print('thetadot: ', thetadot * (180/np.pi))

plt.figure()
plt.grid()
plt.axis('equal')

plt.plot([p_com[0], p_foot[0]], [p_com[1], p_foot[1]], 'k')
plt.quiver(p_com[0], p_com[1], v_com[0], v_com[1], angles='xy', scale_units='xy', scale=1, color='r')
plt.plot(0, 0, 'k+', markersize=25)
plt.plot(p_com[0], p_com[1], 'ko', markersize=15)
plt.title('angle: ' + str(theta * (180/np.pi)))

plt.show()

