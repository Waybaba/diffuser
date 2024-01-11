import pdb

import pyrootutils
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from src.datamodule import QuickdrawDataset

def interpolate_trajectory(drawing, max_dist=5):
    
    max_speed = 0

    for stroke in drawing:
        x, y = stroke[0], stroke[1]
        t = np.arange(len(x))
        max_speed = max(np.max(np.abs(np.diff(x))), max_speed)
        max_speed = max(np.max(np.abs(np.diff(y))), max_speed)

    dt = max_dist / max_speed

    all_traj = []

    for k, stroke in enumerate(drawing):

        x, y = stroke[0], stroke[1]
        t = np.arange(len(x))

        # non-drawing moving of pen
        if k >= 1:
            n =  max(int(np.abs(x[0] - last_x) / max_dist) + 1,  int(np.abs(y[0] - last_y) / max_dist) + 1)
            dt_emp = 1 / n

            t_emp = np.arange(0, 1 + dt_emp, dt_emp)
            fx_emp = interp1d([0, 1], [last_x, x[0]], kind='linear', fill_value="extrapolate")
            fy_emp = interp1d([0, 1], [last_y, y[0]], kind='linear', fill_value="extrapolate")
            x_emp = fx_emp(t_emp)
            y_emp = fy_emp(t_emp) 

            all_traj.append(np.vstack([np.diff(x_emp), np.diff(y_emp), - 0.5 * np.ones_like(np.diff(x_emp))])) # non-drawing trajectory

        t_new = np.arange(t.min(), t.max() + dt, dt)

        fx = interp1d(t, x, kind='linear', fill_value="extrapolate")
        fy = interp1d(t, y, kind='linear', fill_value="extrapolate")

        x_new = fx(t_new)
        y_new = fy(t_new)

        all_traj.append(np.vstack([np.diff(x_new), np.diff(y_new), np.ones_like(np.diff(x_new)) * 0.5])) # drawing trajectory

        last_x = x_new[-1]
        last_y = y_new[-1]


    actions = np.concatenate(all_traj, axis=-1).swapaxes(0, 1)

    return actions
# 

if __name__ == "__main__":
	import os
	ds = QuickdrawDataset(data_dir=os.environ['UDATADIR'] + '/quickdraw').get_datadict()
	print("finish")