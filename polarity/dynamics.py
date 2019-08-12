import numpy as np

from tyssue.utils.utils import to_nd, _to_3d
from tyssue.dynamics.sheet_gradients import height_grad
from tyssue.dynamics import units, effectors, model_factory
from tyssue.geometry.sheet_geometry import ClosedSheetGeometry, SheetGeometry


class EllipsoidLameGeometry(ClosedSheetGeometry):
    # Entoure l'ellipsoide de lame d'une sphere,
    # Afin de "compresser" le tissu
    @classmethod
    def update_all(cls, eptm):
        super().update_all(eptm)
        cls.center(eptm)
        #cls.update_height(eptm)

    @staticmethod
    def update_height2(eptm):
        # Barriere sur les "extremités" avec une sphere
        r = eptm.settings['barrier_ray']

        eptm.vert_df["barrier_rho"] = r

        eptm.vert_df["rho"] = np.sqrt(eptm.vert_df['x']**2 +
                                      eptm.vert_df['y']**2 +
                                      eptm.vert_df['z']**2)

        eptm.vert_df["delta_rho"] = (
            eptm.vert_df["rho"] - eptm.vert_df["barrier_rho"])

        eptm.vert_df["delta_rho"] *= (eptm.vert_df["delta_rho"] > 0).astype(float)

        eptm.vert_df["height"] = eptm.vert_df["rho"]

        edge_height = eptm.upcast_srce(eptm.vert_df[["height", "rho"]])
        edge_height.set_index(eptm.edge_df["face"], append=True, inplace=True)

        """# Barriere sur le "centre" avec un cylindre
                                r = eptm.settings['barrier_ray_cylinder']
                                sheet.vert_df["rho"] = np.hypot(sheet.vert_df['x'], sheet.vert_df['y'])
                                eptm.vert_df["delta_rho"] = eptm.vert_df['rho'] - r

                                eptm.vert_df["height"] = eptm.vert_df["rho"] - eptm.vert_df["basal_shift"]

                                edge_height = eptm.upcast_srce(eptm.vert_df[["height", "rho"]])
                                edge_height.set_index(eptm.edge_df["face"], append=True, inplace=True)
                                eptm.face_df[["height", "rho"]] = edge_height.mean(level="face")"""


class RadialTension(effectors.AbstractEffector):

    dimensions = units.line_tension
    magnitude = 'radial_tension'
    label = 'Apical basal tension'
    element = 'vert'
    specs = {'vert': {'is_active',
                      'height',
                      'radial_tension'}}

    @staticmethod
    def energy(eptm):
        return eptm.vert_df.eval(
            'height * radial_tension * is_active')

    @staticmethod
    def gradient(eptm):
        grad = height_grad(eptm) * to_nd(
            eptm.vert_df.eval('radial_tension'), 3)
        grad.columns = ['g' + c for c in eptm.coords]
        return grad, None


class BarrierElasticity(effectors.AbstractEffector):

    dimensions = units.line_elasticity
    magnitude = 'barrier_elasticity'
    label = 'Barrier elasticity'
    element = 'vert'
    specs = {
        'vert': {
            'barrier_elasticity',
            'is_active',
            'delta_rho'}
    }  # distance to a barrier membrane

    @staticmethod
    def energy(eptm):
        """eptm.vert_df['energy'] = sheet.vert_df.eval('delta_rho**2 * barrier_elasticity/2')
        energy = [0 if v.delta_rho < 0 else v.energy for v in eptm.vert_df.itertuples()]
        return energy
        """
        return eptm.vert_df.eval(
            'delta_rho**2 * barrier_elasticity/2')

    @staticmethod
    def gradient(eptm):
        grad = height_grad(eptm) * _to_3d(
            eptm.vert_df.eval('barrier_elasticity * delta_rho'))
        grad.columns = ['g' + c for c in eptm.coords]
        return grad, None


model = model_factory(
    [
        RadialTension,
        #BarrierElasticity,
        effectors.FaceContractility,
        effectors.FaceAreaElasticity,
        effectors.LumenVolumeElasticity,
    ], effectors.FaceAreaElasticity)
