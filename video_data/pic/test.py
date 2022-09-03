import numpy as np
import os
poses_bounds = np.load(os.path.join('../capture_image', 'poses_bounds.npy'))
print(poses_bounds)
# poses_bounds = np.concatenate((poses_bounds,poses_bounds,poses_bounds,poses_bounds))
# print(poses_bounds)
# print(len(poses_bounds))
# np.save('./poses_bounds.npy',poses_bounds)