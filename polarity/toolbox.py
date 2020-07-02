import json
import random
import os

import numpy as np

from tyssue import Sheet
from tyssue.io import hdf5
from tyssue import config



def init_tissue(hf5_filename='superegg_final.hf5',
                json_filename='superegg_final.json'):
    """
    Initialisation of the superegg tissue
    """
    dsets = hdf5.load_datasets('../examples/' + hf5_filename,
                               data_names=['vert', 'edge', 'face'])

    with open('../examples/' + json_filename, 'r+') as fp:
        specs = json.load(fp)

    sheet = Sheet('spherical', dsets, specs)

    modify_some_initial_settings(sheet)

    return sheet


def modify_some_initial_settings(sheet):
    """
    Need to find an other name function...
    """
    sheet.settings['geometry'] = "spherical"
    sheet.settings['lumen_vol_elasticity'] = 1.e-5

    sheet.settings['barrier_radius'] = 100
    sheet.vert_df['barrier_elasticity'] = 280.0

    sheet.settings['apopto_pattern_kwargs'] = apopto_pattern_kwargs
    sheet.settings['apoptosis'] = apoptosis_settings
    sheet.settings['contraction_lt_kwargs'] = contraction_lt_kwargs
    sheet.settings['delaminate_setting'] = delaminate_settings

    sheet.edge_df['weight'] = 1.
    sheet.edge_df['weight_length'] = sheet.edge_df.weight * \
        sheet.edge_df.length

    sheet.face_df['apoptosis'] = 0
    sheet.face_df['current_traction'] = 0.0
    sheet.face_df['radial_tension'] = 0.0
    sheet.face_df['prefered_perimeter'] = 2 * \
        np.sqrt(sheet.face_df['prefered_area'])
    sheet.face_df['perimeter_elasticity'] = 10.
    sheet.face_df['area_elasticity'] = 1.
    sheet.face_df['apoptosis'] = 0

    sheet.vert_df['viscosity'] = 0.1


def define_fold_position(sheet,
                         fold_number=2,
                         position=[-36, -22, 22, 36]):
    """
    Define fold position in the superegg tissue
    along the z axis

    Parameters
    ----------
    fold_number : number of fold
    position : [min1, max1, min2, max2, ...]
    """
    sheet.face_df['fold'] = 0
    for i in range(fold_number):
        i = i * 2
        sheet.face_df.loc[sheet.face_df[(sheet.face_df.z > position[i])
                                        & (sheet.face_df.z < position[i + 1])].index, 'fold'] = i + 1


def apoptosis_ventral(sheet, code_fold=1):

    fold_cell = sheet.face_df[sheet.face_df.fold == code_fold]

    for i in fold_cell.itertuples():
        proba = 0.6 * np.exp(-(i.phi + 3)**2 / 0.7**2) + \
            0.6 * np.exp(-(i.phi - 3)**2 / 0.7**2)
        aleatory_number = random.uniform(0, 1)
        if aleatory_number < proba:
            sheet.face_df.loc[i.Index, "apoptosis"] = 1


def apoptosis_lateral(sheet, code_fold=1):

    fold_cell = sheet.face_df[sheet.face_df.fold == code_fold]

    for i in fold_cell.itertuples():
        proba = 0.3 * np.exp(-(i.phi + 1.5)**2 / 0.8**2) + \
            0.3 * np.exp(-(i.phi - 1.5)**2 / 0.8**2)
        aleatory_number = random.uniform(0, 1)
        if aleatory_number < proba:
            sheet.face_df.loc[i.Index, "apoptosis"] = 1


def apoptosis_dorsal(sheet, code_fold=1):

    fold_cell = sheet.face_df[sheet.face_df.fold == code_fold]

    for i in fold_cell.itertuples():
        proba = 0.25 * np.exp(-(i.phi)**2 / 0.4 * 2)
        aleatory_number = random.uniform(0, 1)
        if aleatory_number < proba:
            sheet.face_df.loc[i.Index, "apoptosis"] = 1


def define_apoptotic_pattern(sheet, code_fold=1):
    """
    Define apoptotic cell according to a probability to enter in apoptotis
    More apoptotic cell in the ventral side of the tissue than in the dorsal side.
    """
    sheet.face_df.apoptosis = 0
    apoptosis_ventral(sheet, code_fold)
    apoptosis_lateral(sheet, code_fold)
    apoptosis_dorsal(sheet, code_fold)


def decrease_polarity_lateral(sheet, face, parallel_weight, perpendicular_weight):
    edges = sheet.edge_df[sheet.edge_df["face"] == face]
    for index, edge in edges.iterrows():
        angle_ = np.arctan2(
            sheet.edge_df.loc[edge.name, "dx"], sheet.edge_df.loc[
                edge.name, "dz"]
        )
        if (((np.abs(angle_) < np.pi / 6) and (np.abs(angle_) > -np.pi / 6)) or
                ((np.abs(angle_) > -np.pi / 6) and (np.abs(angle_) < np.pi / 6)) or
            ((np.abs(angle_) > 5 * np.pi / 6) and (np.abs(angle_) < 7 * np.pi / 6)) or
                ((np.abs(angle_) < -7 * np.pi / 6) and (np.abs(angle_) > -5 * np.pi / 6))):

            sheet.edge_df.loc[edge.name, "weight"] = perpendicular_weight
        else:
            sheet.edge_df.loc[edge.name, "weight"] = parallel_weight


def decrease_polarity_dv(sheet, face, parallel_weight, perpendicular_weight):
    edges = sheet.edge_df[sheet.edge_df["face"] == face]
    for index, edge in edges.iterrows():
        angle_ = np.arctan2(
            sheet.edge_df.loc[edge.name, "dy"], sheet.edge_df.loc[
                edge.name, "dz"]
        )
        if (((np.abs(angle_) < np.pi / 6) and (np.abs(angle_) > -np.pi / 6)) or
                ((np.abs(angle_) > -np.pi / 6) and (np.abs(angle_) < np.pi / 6)) or
            ((np.abs(angle_) > 5 * np.pi / 6) and (np.abs(angle_) < 7 * np.pi / 6)) or
                ((np.abs(angle_) < -7 * np.pi / 6) and (np.abs(angle_) > -5 * np.pi / 6))):
            sheet.edge_df.loc[edge.name, "weight"] = perpendicular_weight
        else:
            sheet.edge_df.loc[edge.name, "weight"] = parallel_weight


def define_polarity(sheet, parallel_weight, perpendicular_weight):
    sheet.edge_df['id_'] = sheet.edge_df.index

    sheet2 = sheet.extract_bounding_box(y_boundary=(30, 150))
    [decrease_polarity_lateral(
        sheet2, i, parallel_weight, perpendicular_weight) for i in range(sheet2.Nf)]
    for i in (sheet2.edge_df.index):
        sheet.edge_df.loc[sheet.edge_df[sheet.edge_df.id_ == sheet2.edge_df.loc[
            i, 'id_']].index, 'weight'] = sheet2.edge_df.loc[i, 'weight']

    sheet2 = sheet.extract_bounding_box(y_boundary=(-150, -30))
    [decrease_polarity_lateral(
        sheet2, i, parallel_weight, perpendicular_weight) for i in range(sheet2.Nf)]
    for i in (sheet2.edge_df.index):
        sheet.edge_df.loc[sheet.edge_df[sheet.edge_df.id_ == sheet2.edge_df.loc[
            i, 'id_']].index, 'weight'] = sheet2.edge_df.loc[i, 'weight']

    sheet2 = sheet.extract_bounding_box(x_boundary=(-150, -30))
    [decrease_polarity_dv(
        sheet2, i, parallel_weight, perpendicular_weight) for i in range(sheet2.Nf)]
    for i in (sheet2.edge_df.index):
        sheet.edge_df.loc[sheet.edge_df[sheet.edge_df.id_ == sheet2.edge_df.loc[
            i, 'id_']].index, 'weight'] = sheet2.edge_df.loc[i, 'weight']

    sheet2 = sheet.extract_bounding_box(x_boundary=(30, 150))
    [decrease_polarity_dv(
        sheet2, i, parallel_weight, perpendicular_weight) for i in range(sheet2.Nf)]
    for i in (sheet2.edge_df.index):
        sheet.edge_df.loc[sheet.edge_df[sheet.edge_df.id_ == sheet2.edge_df.loc[
            i, 'id_']].index, 'weight'] = sheet2.edge_df.loc[i, 'weight']

    # Pour une cellule apoptotic, toutes les jonctions ont le même poids
    if 'apoptosis' in sheet.face_df.columns:
        for f in sheet.face_df[sheet.face_df.apoptosis == 1].index:
            for e in sheet.edge_df[sheet.edge_df.face == f].index:
                sheet.edge_df.loc[e, 'weight'] = 1.

    if 'is_mesoderm' in sheet.face_df.columns:
        for f in sheet.face_df[sheet.face_df.is_mesoderm == 1].index:
            for e in sheet.edge_df[sheet.edge_df.face == f].index:
                sheet.edge_df.loc[e, 'weight'] = 1.


def open_sheet(dirname, t=0, file_name=None, data_names=['vert', 'edge', 'face', 'cell']):
    """Open hdf5 file

    Open HDF5 file correspond to t time from dirname directory.

    Parameters
    ----------
    directory : str
        complete directory path
    t : int
        time step
    """
    if file_name is None:
        file_name = 'invagination_{:04d}.hf5'.format(t)
    dsets = hdf5.load_datasets(os.path.join(dirname, file_name),
                               data_names=data_names)

    specs = config.geometry.cylindrical_sheet()
    sheet = Sheet('ellipse', dsets, specs)
    return sheet


def depth_calculation(sheet, zmin_, zmax_):

    sheet_fold = sheet.extract_bounding_box(z_boundary=(zmin_, zmax_))

    r = np.mean(np.sqrt(sheet_fold.face_df.x**2 + sheet_fold.face_df.y**2))

    return r


def all_depth_calculation(directory, zmin_=-7, zmax_=7):
    sheet_init = open_sheet(directory, 0)
    depth_init = depth_calculation(sheet_init, zmin_, zmax_)

    depths = []
    for t in range(0, 200):
        try:
            sheet = open_sheet(directory, t)
            depths.append(depth_calculation(sheet, zmin_, zmax_) / depth_init)

        except Exception:
            pass

    return depths


from polarity.polarity import model
from polarity.dynamics import EllipsoidLameGeometry as geom

apopto_pattern_kwargs = {'t': 0.,
                         'dt': 1.,
                         'time_of_last_apoptosis': 30.}
# Apoptosis
apoptosis_settings = {
    "critical_area_pulling": 10,
    "critical_area": 0.5,
    "contract_rate": 1.08,
    "basal_contract_rate": 1.01,
    "contract_neighbors": True,
    "contract_span": 3,
    "radial_tension": 50.,
    "max_traction": 30.,
    "current_traction": 0.,
    "geom": geom,
}

# Neighbors propagation
contraction_lt_kwargs = {
    'face_id': -1,
    'face': -1,
    'shrink_rate': 1.05,
    'critical_area': 5.,
    "contraction_column": "line_tension",
    "model": model,
}


# Mechanical perturbation
delaminate_settings = {
    'radial_tension': 80,
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
