import numpy as np

from tyssue.geometry.sheet_geometry import ClosedSheetGeometry


class EllipsoidLameGeometry(ClosedSheetGeometry):
    """
    Sphere surrounding the sheet.
    Sphere compress the tissue at its extremity
    """

    @classmethod
    def update_all(cls, eptm):
        cls.center(eptm)
        super().update_all(eptm)

    @staticmethod
    def update_perimeters(eptm):
        """
        Updates the perimeter of each face according to the weight of each junction.
        """
        eptm.edge_df['weighted_length'] = eptm.edge_df.weight * \
            eptm.edge_df.length
        eptm.face_df["perimeter"] = eptm.sum_face(
            eptm.edge_df["weighted_length"])

    @staticmethod
    def normalize_weights(sheet):
        sheet.edge_df["num_sides"] = sheet.upcast_face('num_sides')
        sheet.edge_df["weight"] = sheet.edge_df.groupby('face').apply(
            lambda df: (df["num_sides"] * df["weight"] / df["weight"].sum())
        ).sort_index(level=1).to_numpy()

    @staticmethod
    def update_height(eptm):

        eptm.vert_df["rho"] = np.linalg.norm(
            eptm.vert_df[eptm.coords], axis=1)
        r = eptm.settings['barrier_radius']
        eptm.vert_df["delta_rho"] = (eptm.vert_df["rho"] - r).clip(0)
        eptm.vert_df["height"] = eptm.vert_df["rho"]
