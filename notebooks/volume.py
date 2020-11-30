"""
Simulations pour faire les graphes de variabilités.
Pour different patron d'apoptose, tester toutes les conditions (w/o polatiry; w/o perturbation)
"""

import os
import time
import json
import random
import datetime
import numpy as np
import pandas as pd

from IPython.display import display

from pathlib import Path
from tyssue import config
from tyssue import Sheet
from tyssue.io import hdf5
from tyssue.io.hdf5 import load_datasets
from tyssue.topology import all_rearangements
from tyssue.core.history import HistoryHdf5
from tyssue.dynamics import SheetModel as basemodel
from tyssue.solvers.quasistatic import QSSolver
from tyssue.draw import sheet_view
from tyssue.draw.plt_draw import quick_edge_draw
from tyssue.draw.ipv_draw import sheet_view as ipv_draw

from tyssue.behaviors.event_manager import EventManager
from tyssue.behaviors.sheet.apoptosis_events import apoptosis
from tyssue.behaviors.sheet.basic_events import reconnect

from tyssue.utils import to_nd

import matplotlib.pyplot as plt

import ipyvolume as ipv

from polarity.apoptosis import apoptosis, apoptosis_patterning
from polarity.dynamics import EllipsoidLameGeometry as geom
from polarity.toolbox import (init,
                              define_fold_position,
                              apoptosis_ventral,
                              apoptosis_lateral,
                              apoptosis_dorsal,
                              define_polarity_old
                              )
from tyssue.dynamics import units, effectors, model_factory

from polarity.delamination import delamination


SIM_DIR = Path('')

today = datetime.date.today().strftime('%Y%m%d')
sim_save_dir = SIM_DIR / f'{today}-variability'

try:
    os.mkdir(sim_save_dir)
except FileExistsError:
    pass

from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime


solver = QSSolver(with_t1=False, with_t3=False, with_collisions=False)

model = model_factory(
    [
        effectors.BarrierElasticity,
        effectors.RadialTension,
        effectors.PerimeterElasticity,
        effectors.FaceAreaElasticity,
        effectors.LumenVolumeElasticity,
    ], effectors.FaceAreaElasticity)


apopto_pattern_kwargs = {'t': 0.,
                         'dt': 1.,
                         'time_of_last_apoptosis': 30.}
# apoptose
apoptosis_settings = {
    "critical_area_pulling": 10,
    "critical_area": 0.5,
    "contract_rate": 1.08,
    "basal_contract_rate": 1.01,
    "contract_neighbors": True,
    "contract_span": 3,
    "radial_tension": 30.,
    "max_traction": 30.,
    "current_traction": 0.,
    "geom": geom,
}

# pour la propagation aux voisins
contraction_lt_kwargs = {
    'face_id': -1,
    'face': -1,
    'shrink_rate': 1.05,
    'critical_area': 5.,
    "contraction_column": "line_tension",
    "model": model,
}

# Add events to limit rosette
rosette_kwargs = {
    'threshold_length': 1e-6,
    'p_4': 0.9,
    'p_5p': 0.9}

# clone perturbateur
delaminate_settings = {
    'radial_tension': 30,
    "contract_rate": 1.08,
    "critical_area_pulling": 15,
    "critical_area": 1e-2,
    'current_traction': 0,
    'max_traction': 150,
    'contract_neighbors': True,
    'contract_span': 3,
    'basal_contract_rate': 1.01,
    'geom': geom,
    'model': model}


def init_tissue():

    sheet = init(hf5_filename='super_egg.hf5',
                 json_filename='superegg_final.json')

    sheet.settings['geometry'] = "spherical"
    sheet.settings['lumen_vol_elasticity'] = 1e-5
    sheet.settings['lumen_volume_elasticity'] = 1e-5
    sheet.settings['barrier_radius'] = 100

    sheet.vert_df['barrier_elasticity'] = 280.0

    # A définir en fonction de l'angle de la jonction
    sheet.edge_df['weight'] = 1.
    sheet.edge_df['weight_length'] = sheet.edge_df.weight * \
        sheet.edge_df.length

    sheet.face_df['apoptosis'] = 0
    sheet.face_df['current_traction'] = 0.0
    sheet.face_df['radial_tension'] = 0.0
    sheet.face_df['prefered_perimeter'] = 2. * \
        np.sqrt(sheet.face_df['prefered_area'])
    sheet.face_df['perimeter_elasticity'] = 10.
    sheet.face_df['area_elasticity'] = 1.

    sheet.settings['apopto_pattern_kwargs'] = apopto_pattern_kwargs
    sheet.settings['apoptosis'] = apoptosis_settings
    sheet.settings['rosette_kwargs'] = rosette_kwargs
    sheet.settings['contraction_lt_kwargs'] = contraction_lt_kwargs
    sheet.settings['delaminate_setting'] = delaminate_settings

    geom.update_all(sheet)

    define_fold_position(sheet, fold_number=1, position=[-8, 8])
    sheet.face_df.apoptosis = 0

    apoptosis_ventral(sheet, 1)
    apoptosis_lateral(sheet, 1)
    apoptosis_dorsal(sheet, 1)

    sheet.face_df['is_mesoderm'] = 0

    return sheet


def run_sim(sim_save_dir, original_tissue, polarity, perturbation, ve, iteration=0):
    time.sleep(np.random.rand())
    # without copy, dataframe is on read only...
    sheet = original_tissue.copy()
    sheet.settings['lumen_prefered_vol'] = ve

    if perturbation != -1:
        for p in perturbation:
            sheet.face_df.loc[int(p), 'is_mesoderm'] = 1

    define_polarity_old(sheet, 1, polarity)
    geom.normalize_weights(sheet)

    res = solver.find_energy_min(sheet, geom, model, options={"gtol": 1e-8})

    filename = '{}_polarity_{}_perturbation_{}_ve_{}'.format(polarity,
                                                      perturbation, ve, iteration)
    dirname = os.path.join(sim_save_dir, filename)

    print('starting {}'.format(dirname))
    try:
        os.mkdir(dirname)
    except IOError:
        pass

    # Add some information to the sheet and copy initial sheet
    sheet.face_df['id'] = sheet.face_df.index.values

    # Initiate history
    history = HistoryHdf5(sheet,
                          extra_cols={"face": sheet.face_df.columns,
                                      "edge": list(sheet.edge_df.columns),
                                      "vert": list(sheet.vert_df.columns)},
                          hf5file=os.path.join(dirname, filename+'.hf5'))

    # Initiate manager
    manager = EventManager('face')

    # Update kwargs...
    sheet.settings['apoptosis'].update(
        {
            'contract_rate': 1.08,
            'radial_tension': 50,
        })

    # save settings
    pd.Series(sheet.settings).to_csv(os.path.join(dirname, 'settings.csv'))

    manager.append(reconnect, **sheet.settings['rosette_kwargs'])
    manager.append(apoptosis_patterning, **
                   sheet.settings['apopto_pattern_kwargs'])

    t = 0.
    stop = 150.
    # Run simulation
    while t < stop:
        if t == 5:
            for i in sheet.face_df[sheet.face_df.is_mesoderm == 1].index:
                delamination_kwargs = sheet.settings[
                    'delaminate_setting'].copy()
                delamination_kwargs.update(
                    {
                        "face_id": i,
                        #"radial_tension": radial_tension,
                        "radial_tension": 50,
                        "contract_rate": 1.08,
                        "max_traction": 90,
                        "current_traction": 0,
                    }
                )
                manager.append(delamination, **delamination_kwargs)

        # Reset radial tension at each time step
        sheet.vert_df.radial_tension = 0.

        manager.execute(sheet)
        res = solver.find_energy_min(
            sheet, geom, model, options={"gtol": 1e-8})

        # add noise on vertex position to avoid local minimal.
        sheet.vert_df[
            ['x', 'y']] += np.random.normal(scale=1e-3, size=(sheet.Nv, 2))
        geom.update_all(sheet)

        # Save result in each time step.
        """figname = os.path.join(
            dirname, 'invagination_{:04.0f}.png'.format(t))
        hdfname = figname[:-3] + 'hf5'
        hdf5.save_datasets(hdfname, sheet)
		"""
        history.record(time_stamp=float(t))

        manager.update()
        t += 1.

    print('{} done'.format(dirname))
    print('~~~~~~~~~~~~~~~~~~~~~\n')


# Main

global_start = datetime.now()
print("start : " + str(global_start))
num_cores = multiprocessing.cpu_count()
#""" Initiale find minimal energy
# To be sure we are at the equilibrium before executing simulation
# (it will be done only once if multiprocessing...)
#res = solver.find_energy_min(sheet, geom, model, options={"gtol": 1e-8})

tissue = init_tissue()

list_perturbator = tissue.face_df[(tissue.face_df.z > -45) & (tissue.face_df.z < 45)].index.to_numpy()


for i in range(0, 1):

    position_aleatory_perturbation = [list_perturbator[
        random.randint(0, len(list_perturbator))]]
    position_aleatory_perturbation.append(list_perturbator[
        random.randint(0, len(list_perturbator))])
    position_aleatory_perturbation.append(list_perturbator[
        random.randint(0, len(list_perturbator))])

    volume = []

    volume.append(tissue.settings['lumen_prefered_vol'])
    volume.append(tissue.settings['lumen_prefered_vol']*0.9)
    volume.append(tissue.settings['lumen_prefered_vol']*0.8)
    volume.append(tissue.settings['lumen_prefered_vol']*0.7)
    volume.append(tissue.settings['lumen_prefered_vol']*0.6)

    #polarity = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
    polarity=[0.5]
    #perturbation = [-1, position_aleatory_perturbation]
    perturbation = [-1, [1157, 232, 509]]
    pola, perturb, vol = np.meshgrid(polarity, perturbation, volume)

    results = Parallel(n_jobs=3)(delayed(run_sim)(
        sim_save_dir, tissue, po, pe, v, i)
        for po, pe, v in zip(pola.ravel(), perturb.ravel(), vol.ravel()))


global_end = datetime.now()
print("end : " + str(global_end))
print('Duree totale d execution : \n\t\t')
print(global_end - global_start)
