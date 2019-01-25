import numpy as np
import pandas as pd
from tyssue import config
from scipy import optimize
from scipy.sparse import coo_matrix
from tyssue.utils import data_at_opposite


def find_energy_min(sheet, geom, model, pos_idx=None, **settings_kw):
    #coords = ['x', 'y', 'z']
    # Loads 'tyssue/config/solvers/minimize.json
    settings = config.solvers.minimize_spec()
    settings.update(**settings_kw)

    c0 = sheet.edge_df['C_a']

    row = sheet.edge_df['face'].values
    col = sheet.edge_df.index.values
    matrix = coo_matrix((sheet.edge_df.length.values, (row, col)),
                        shape=(sheet.Nf, sheet.Ne)).toarray()
    linear_constraint = optimize.LinearConstraint(
        matrix, sheet.face_df['prefered_N_a'], sheet.face_df['prefered_N_a'])

    bounds = optimize.Bounds(np.zeros(sheet.Ne), np.full((sheet.Ne), np.inf))

    res = optimize.minimize(
        chem_energy,
        c0,
        args=(sheet),
        bounds=bounds,
        constraints=linear_constraint,
        jac=chem_grad,
        **settings["minimize"]
    )
    return res


def set_concentration(sheet, c):
    sheet.edge_df['C_a'] = c

    sheet.edge_df['N_a'] = sheet.edge_df['C_a'] * sheet.edge_df['length']


def chem_energy(c, sheet):

    set_concentration(sheet, c)

    # Auto enrichissement
    P1 = (- sheet.settings['epsilon_0'] / 2 *
          sheet.edge_df['length'] * sheet.edge_df['C_a']**2).values

    # Enrichissment in the edge of the opposite cell
    sheet.get_opposite()
    P2 = (- sheet.settings['epsilon_1'] * sheet.edge_df['length'] * sheet.edge_df[
          'C_a'] * data_at_opposite(sheet, sheet.edge_df['C_a'], 0)).values

    # Part 3
    diff_couple1, diff_couple2 = _calculate_diff_C_neighbors(sheet)
    P3 = - sheet.settings['J'] / 2 * (diff_couple1**2 + diff_couple2**2)

    # Constraint on concentration
    #P4 = (sheet.settings['lambda_c'] * (sheet.sum_face(sheet.edge_df['C_a'])['C_a'] - sheet.face_df['prefered_C_a'])**2)

    return (P1 + P2 + P3).sum()


def chem_grad(c, sheet):
    # Part1
    P1 = (- sheet.settings['epsilon_0'] *
          sheet.edge_df['length'] * sheet.edge_df['C_a']).values

    # Part2
    P2 = (- sheet.settings['epsilon_1'] * sheet.edge_df['length']
          * data_at_opposite(sheet, sheet.edge_df['C_a'], 0)).values

    # Part3
    diff_couple1, diff_couple2 = _calculate_diff_C_neighbors(sheet)
    P3 = - sheet.settings['J'] * (diff_couple1 + diff_couple2)

    #P4 = 2 * sheet.settings['lambda_c'] * (sheet.sum_face(sheet.edge_df['C_a'])['C_a'] - sheet.face_df['prefered_C_a'])
    #P4 = sheet.upcast_face(P4)

    grad = (P1 + P2 + P3)

    return grad.values.ravel()


def _calculate_diff_C_neighbors(sheet):
    C1_ij = sheet.edge_df.sort_values(['face', 'trgt'])['C_a']
    C1_jm = sheet.edge_df.sort_values(['face', 'srce'])['C_a']
    couple1 = pd.DataFrame(
        {'ij': C1_ij.values, 'jm': C1_jm.values}, index=C1_ij.index)
    diff_couple1 = couple1['ij'] - couple1['jm']
    diff_couple1.sort_index(axis='index', inplace=True)

    C2_ij = sheet.edge_df.sort_values(['face', 'srce'])['C_a']
    C2_ni = sheet.edge_df.sort_values(['face', 'trgt'])['C_a']
    couple2 = pd.DataFrame(
        {'ij': C2_ij.values, 'ni': C2_ni.values}, index=C2_ij.index)
    diff_couple2 = couple2['ij'] - couple2['ni']
    diff_couple2.sort_index(axis='index', inplace=True)

    return (diff_couple1, diff_couple2)
