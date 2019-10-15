import json
import random

import numpy as np

from tyssue import Sheet
from tyssue.io import hdf5


def init(hf5_filename='superegg_final.hf5',
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
    sheet.settings['lumen_volume_elasticity'] = 3.e-6

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


def predefined_apoptotic_cell(sheet):
    list_apopto = [0, 1, 2, 3]

    for i in list_apopto:
        sheet.face_df.loc[i, 'apoptosis'] = 1


def decrease_polarity_lateral(sheet, face, parallel_weighted, perpendicular_weighted):
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

            sheet.edge_df.loc[edge.name, "weighted"] = perpendicular_weighted
        else:
            sheet.edge_df.loc[edge.name, "weighted"] = parallel_weighted


def decrease_polarity_dv(sheet, face, parallel_weighted, perpendicular_weighted):
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
            sheet.edge_df.loc[edge.name, "weighted"] = perpendicular_weighted
        else:
            sheet.edge_df.loc[edge.name, "weighted"] = parallel_weighted


def define_polarity(sheet, parallel_weighted, perpendicular_weighted):
    sheet.edge_df['id_'] = sheet.edge_df.index

    sheet2 = sheet.extract_bounding_box(y_boundary=(30, 150))
    [decrease_polarity_lateral(
        sheet2, i, parallel_weighted, perpendicular_weighted) for i in range(sheet2.Nf)]
    for i in (sheet2.edge_df.index):
        sheet.edge_df.loc[sheet.edge_df[sheet.edge_df.id_ == sheet2.edge_df.loc[
            i, 'id_']].index, 'weighted'] = sheet2.edge_df.loc[i, 'weighted']

    sheet2 = sheet.extract_bounding_box(y_boundary=(-150, -30))
    [decrease_polarity_lateral(
        sheet2, i, parallel_weighted, perpendicular_weighted) for i in range(sheet2.Nf)]
    for i in (sheet2.edge_df.index):
        sheet.edge_df.loc[sheet.edge_df[sheet.edge_df.id_ == sheet2.edge_df.loc[
            i, 'id_']].index, 'weighted'] = sheet2.edge_df.loc[i, 'weighted']

    sheet2 = sheet.extract_bounding_box(x_boundary=(-150, -30))
    [decrease_polarity_dv(
        sheet2, i, parallel_weighted, perpendicular_weighted) for i in range(sheet2.Nf)]
    for i in (sheet2.edge_df.index):
        sheet.edge_df.loc[sheet.edge_df[sheet.edge_df.id_ == sheet2.edge_df.loc[
            i, 'id_']].index, 'weighted'] = sheet2.edge_df.loc[i, 'weighted']

    sheet2 = sheet.extract_bounding_box(x_boundary=(30, 150))
    [decrease_polarity_dv(
        sheet2, i, parallel_weighted, perpendicular_weighted) for i in range(sheet2.Nf)]
    for i in (sheet2.edge_df.index):
        sheet.edge_df.loc[sheet.edge_df[sheet.edge_df.id_ == sheet2.edge_df.loc[
            i, 'id_']].index, 'weighted'] = sheet2.edge_df.loc[i, 'weighted']

    # Pour une cellule apoptotic, toutes les jonctions ont le même poids
    if 'apoptosis' in sheet.face_df.columns:
        for f in sheet.face_df[sheet.face_df.apoptosis == 1].index:
            for e in sheet.edge_df[sheet.edge_df.face == f].index:
                sheet.edge_df.loc[e, 'weighted'] = 1.

    if 'is_mesoderm' in sheet.face_df.columns:
        for f in sheet.face_df[sheet.face_df.is_mesoderm == 1].index:
            for e in sheet.edge_df[sheet.edge_df.face == f].index:
                sheet.edge_df.loc[e, 'weighted'] = 1.
