import os
import random
import numpy as np
import pandas as pd

from polarity.dynamics import EllipsoidLameGeometry as geom
from polarity.toolbox import (define_fold_position,
                              define_apoptotic_pattern,
                              define_polarity)
from polarity.apoptosis import apoptosis_patterning
from polarity.delamination import delamination

from tyssue.dynamics import effectors
from tyssue.dynamics.factory import model_factory

from tyssue.solvers.quasistatic import QSSolver
from tyssue.core.history import HistoryHdf5
from tyssue.behaviors.event_manager import EventManager
from tyssue.behaviors.sheet.basic_events import reconnect


model = model_factory(
    [
        effectors.BarrierElasticity,
        effectors.RadialTension,
        effectors.PerimeterElasticity,
        effectors.FaceAreaElasticity,
        effectors.LumenVolumeElasticity,
    ], effectors.FaceAreaElasticity)


def polarity_process(sim_save_dir,
                     sheet,
                     polarity=1.,
                     perturbation=-1
                     ):
    """
    Initiate simulation before running according to parameters.

    Parameters
    ----------
    sim_save_dir : directory where saving simulation
    sheet :
    contracts : contractility rate of
    apoptotic_pattern : default None. If is None, create an apoptotic
    pattern according to the specific pattern. Else must be a list of face indices.
    """

    # Define solver
    solver = QSSolver(with_t1=False, with_t3=False, with_collisions=False)

    # Define polarity
    define_polarity(sheet, polarity, 1)
    geom.normalize_weights(sheet)

    res = solver.find_energy_min(sheet, geom, model, options={"gtol": 1e-8})
    if res.success is False:
        raise ('Stop because solver didn''t succeed', res)

    sheet_ = run_sim(sim_save_dir, sheet, polarity, perturbation)

    return sheet_


def run_sim(sim_save_dir,
            _sheet,
            polarity,
            perturbation=-1,
            stop=150.,
            iteration=0,
            ):

    # Define solver
    solver = QSSolver(with_t1=False, with_t3=False, with_collisions=False)

    filename = '{}_polarity{}_perturbation.hf5'.format(
        polarity, perturbation)
    try:
        os.mkdir(sim_save_dir)
    except IOError:
        pass

    # without copy, dataframe is on read only...
    sheet = _sheet.copy()

    sheet.face_df['is_mesoderm'] = 0
    if perturbation != -1:
        for p in perturbation:
            sheet.face_df.loc[int(p), 'is_mesoderm'] = 1

    define_polarity(sheet, 1, polarity)
    geom.normalize_weights(sheet)

    # Add some information to the sheet
    sheet.face_df['id'] = sheet.face_df.index.values

    # Initiate history
    history = HistoryHdf5(sheet,
                          extra_cols={"face": sheet.face_df.columns,
                                      "edge": list(sheet.edge_df.columns),
                                      "vert": list(sheet.vert_df.columns)},
                          hf5file=os.path.join(sim_save_dir, filename))

    # Initiate manager
    manager = EventManager('face')


    # save settings
    pd.Series(sheet.settings).to_csv(
        os.path.join(sim_save_dir, (filename[:-4] + '_settings.csv')))

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
                    }
                )
                manager.append(delamination, **delamination_kwargs)

        # Reset radial tension at each time step
        sheet.vert_df.radial_tension = 0.

        manager.execute(sheet)
        res = solver.find_energy_min(
            sheet, geom, model, options={"gtol": 1e-8})
        if res.success is False:
            raise ('Stop because solver didn''t succeed at time t ' + str(t), res)

        # add noise on vertex position to avoid local minimal.
        sheet.vert_df[
            ['x', 'y']] += np.random.normal(scale=1e-3, size=(sheet.Nv, 2))
        geom.update_all(sheet)

        history.record(time_stamp=float(t))

        manager.update()
        t += 1.

    return sheet
