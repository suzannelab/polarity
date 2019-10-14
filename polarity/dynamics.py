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
        # cls.update_height(eptm)
        cls.update_height2(eptm)
        cls.update_perimeters(eptm)

    @staticmethod
    def update_perimeters(eptm):
        """
        Updates the perimeter of each face.
        """
        eptm.edge_df['weighted_length'] = eptm.edge_df.weighted * eptm.edge_df.length
        eptm.face_df["perimeter"] = eptm.sum_face(eptm.edge_df["weighted_length"])
        #/eptm.sum_face(eptm.edge_df['weighted'])*eptm.sum_face(eptm.edge_df['is_valid'])

    @staticmethod
    def normalize_weights(sheet):
        sheet.edge_df["num_sides"] = sheet.upcast_face('num_sides')
        sheet.edge_df["weighted"] = sheet.edge_df.groupby('face').apply(
            lambda df: (df["num_sides"] * df["weighted"]
                        / df["weighted"].sum())
        ).sort_index(level='edge').to_numpy()


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

        eptm.vert_df[
            "delta_rho"] *= (eptm.vert_df["delta_rho"] > 0).astype(float)

        eptm.vert_df["height"] = eptm.vert_df["rho"]

        edge_height = eptm.upcast_srce(eptm.vert_df[["height", "rho"]])
        edge_height.set_index(eptm.edge_df["face"], append=True, inplace=True)

        eptm.face_df[["height", "rho"]] = edge_height.mean(level="face")

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
    element = 'face'
    specs = {'face': {'height',
                      'radial_tension'}}

    @staticmethod
    def energy(eptm):
        return eptm.face_df.eval(
            'height * radial_tension')

    @staticmethod
    def gradient(eptm):

        # upcast_face(radial_tension) / numsides * upcast_srce (height)

        upcast_f = eptm.upcast_face(
            eptm.face_df[['radial_tension', 'num_sides']])
        upcast_tension = (upcast_f['radial_tension'] / upcast_f['num_sides'])

        upcast_height = eptm.upcast_srce(height_grad(eptm))
        grad_srce = to_nd(upcast_tension, 3) * upcast_height
        grad_srce.columns = ["g" + u for u in eptm.coords]
        return grad_srce, None


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
        # BarrierElasticity,
        effectors.FaceContractility,
        effectors.FaceAreaElasticity,
        effectors.LumenVolumeElasticity,
    ], effectors.FaceAreaElasticity)
