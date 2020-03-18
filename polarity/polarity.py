import os
import random
import numpy as np
import pandas as pd

from polarity.dynamics import EllipsoidLameGeometry as geom
from polarity.toolbox import (define_fold_position,
                              define_apoptotic_pattern,
                              define_polarity_old)
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
                     contracts,
                     critical_area,
                     radial_tension,
                     apoptotic_pattern=None,
                     polarity=1.,
                     clone=False,
                     pos_clone=None,
                     ablation=False,
                     a_face=None,
                     it=0,
                     stop=150.,
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
    define_fold_position(sheet, fold_number=1, position=[-8, 8])

    # Define if we use a predefined apoptotic pattern of apoptotic cell
    # or if we create one.
    if apoptotic_pattern is None:
        define_apoptotic_pattern(sheet)
    elif isinstance(apoptotic_pattern, list):
        for i in apoptotic_pattern:
            sheet.face_df.loc[i, 'apoptosis'] = 1
    else:
        raise ("apoptotic_pattern must be a list or stay as None.")

    # Define if there is a mechanical perturbation in the tissue
    # modelled as a delaminated cell.
    if clone:
        sheet.face_df['is_mesoderm'] = 0
        if pos_clone is None:
            # define default/aleatory position
            print('need to be coded')
        else:
            sheet.face_df.loc[pos_clone, 'is_mesoderm'] = 1

    # Define solver
    solver = QSSolver(with_t1=False, with_t3=False, with_collisions=False)

    # Define polarity
    define_polarity_old(sheet, polarity, 1)
    geom.normalize_weights(sheet)

    res = solver.find_energy_min(sheet, geom, model, options={"gtol": 1e-8})
    if res.success is False:
        raise ('Stop because solver didn''t succeed', res)

    sheet2 = run_sim(sim_save_dir, sheet, contracts, radial_tension,
                     stop=stop, iteration=it, clone=clone,
                     ablation=ablation, ablated_face=a_face)


def run_sim(sim_save_dir, _sheet, constriction, radial_tension, stop=150., iteration=0,
            clone=False, ablation=False, ablated_face=None):

    # Define solver
    solver = QSSolver(with_t1=False, with_t3=False, with_collisions=False)

    # without copy, dataframe is on read only...
    sheet = _sheet.copy()

    filename = '{}_constriction_{}_radialtension{}.hf5'.format(
        constriction, radial_tension, iteration)

    try:
        os.mkdir(sim_save_dir)
    except IOError:
        pass

    # Add some information to the sheet
    sheet.face_df['id'] = sheet.face_df.index.values
    if ablation:
        sheet.face_df['ablate'] = 0
        if ablated_face is None:
            sheet_ = sheet.extract_bounding_box(
                x_boundary=(-30, 30), z_boundary=(-70, 70))
            face_ = random.choice(sheet_.face_df.id.to_numpy())
            face = sheet.face_df[sheet.face_df.id == face_].index.to_numpy()[0]
            sheet.face_df.loc[face, 'ablate'] = 1
        else:
            sheet.face_df.loc[ablated_face, 'ablate'] = 1

    # Initiate history
    history = HistoryHdf5(sheet,
                          extra_cols={"face": sheet.face_df.columns,
                                      "edge": list(sheet.edge_df.columns),
                                      "vert": list(sheet.vert_df.columns)},
                          hf5file=os.path.join(sim_save_dir, filename))

    # Initiate manager
    manager = EventManager('face')

    # Update kwargs...
    sheet.settings['apoptosis'].update(
        {
            'contract_rate': constriction,
            'radial_tension': radial_tension,
        })

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

        if clone and t == 5:
            for i in sheet.face_df[sheet.face_df.is_mesoderm == 1].index:
                delamination_kwargs = sheet.settings[
                    'delaminate_setting'].copy()
                delamination_kwargs.update(
                    {
                        "face_id": i,
                        "radial_tension": radial_tension,
                        "contract_rate": constriction,
                        "max_traction": 90,
                        "current_traction": 0,
                    }
                )
                manager.append(delamination, **delamination_kwargs)

        # Mise à 0 des poids des jonctions autours d'une cellule choisie
        if ablation:
            face = sheet.face_df[sheet.face_df.ablate == 1].index.to_numpy()[0]
            if t > 10:
                sheet.get_opposite()
                edge_opp = sheet.edge_df[
                    sheet.edge_df.face == face].opposite.to_numpy()
                for v in sheet.edge_df[sheet.edge_df.face == face]['srce'].values:
                    edge = np.concatenate(
                        (sheet.edge_df[(sheet.edge_df.srce == v)
                                       & (sheet.edge_df.face != face)].index.values,
                         sheet.edge_df[(sheet.edge_df.trgt == v)
                                       & (sheet.edge_df.face != face)].index.values))

                    for e in edge:
                        if (sheet.edge_df.loc[e, 'trgt'] not in sheet.edge_df.loc[edge_opp].trgt.to_numpy()
                                or sheet.edge_df.loc[e, 'srce'] not in sheet.edge_df.loc[edge_opp].srce.to_numpy()):
                            sheet.edge_df.loc[e, 'weight'] = 0

        # Reset radial tension at each time step
        sheet.vert_df.radial_tension = 0.

        manager.execute(sheet)
        res = solver.find_energy_min(
            sheet, geom, model, options={"gtol": 1e-8})

        if res.success is False:
            raise ('Stop because solver didn''t succeed at time t ' + str(t), res)

        # add noise on vertex position to avoid local minima.
        sheet.vert_df[
            ['x', 'y']] += np.random.normal(scale=1e-3, size=(sheet.Nv, 2))
        geom.update_all(sheet)

        # Save result of each time step in the history file
        history.record(time_stamp=float(t))

        manager.update()
        t += 1.

        if t % 10 == 0:
            print(filename + ' ' + str(t) + ' timestep')

    return sheet
