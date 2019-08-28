import numpy as np
from tyssue.utils.decorators import face_lookup
from tyssue.geometry.sheet_geometry import SheetGeometry
from tyssue.behaviors.sheet.actions import (ab_pull,
                                            exchange,
                                            remove,
                                            decrease,
                                            increase_linear_tension)
from tyssue.behaviors.sheet.basic_events import contraction_line_tension


def apoptosis_patterning(sheet, manager, **kwargs):
    # ViscousSolver...
    # Need to be imported from a notebook...

    return 0


default_apoptosis_spec = {
    "face_id": -1,
    "face": -1,
    "shrink_rate": 1.1,
    "critical_area": 1e-2,
    "radial_tension": 0.1,
    "contract_rate": 0.1,
    "basal_contract_rate": 1.001,
    "contract_span": 2,
    "max_traction": 10,
    "current_traction": 0,
    "geom": SheetGeometry,
}


@face_lookup
def apoptosis(sheet, manager, **kwargs):
    """Apoptotic behavior

    While the cell's apical area is bigger than a threshold, the
    cell contracts, and the contractility of its neighbors is increased.
    once the critical area is reached, the cell is eliminated
    from the apical surface through successive type 1 transition. Once
    only three sides are left, the cell is eliminated from the tissue.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    manager : a :class:`EventManager` object
    face_id : int,
        the id of the apoptotic cell
    contract_rate : float, default 0.1
        the rate of reduction of the cell's prefered volume
        e.g. the prefered volume is devided by a factor 1+contract_rate
    critical_area : area at which the face is eliminated from the sheet
    radial_tension : amount of radial tension added at each contraction steps
    contractile_increase : increase in contractility at the cell neighbors
    contract_span : number of neighbors affected by the contracitity increase
    geom : the geometry class used
    """

    apoptosis_spec = default_apoptosis_spec
    apoptosis_spec.update(**kwargs)
    dt = sheet.settings.get('dt', 1.0)
    # Small variable name for some spec
    face = apoptosis_spec["face"]
    current_traction = apoptosis_spec["current_traction"]
    face_area = sheet.face_df.loc[face, "area"]

    if (face_area > apoptosis_spec["critical_area"]):
        # reduce prefered_area
        decrease(sheet,
                 'face',
                 face,
                 apoptosis_spec["shrink_rate"],
                 col="prefered_area",
                 divide=True,
                 bound=apoptosis_spec["critical_area"] / 2,
                 )

        increase_linear_tension(
            sheet,
            face,
            apoptosis_spec['contract_rate'] * dt,
            multiple=True,
            isotropic=True,
            limit=100)

        # contract neighbors
        neighbors = sheet.get_neighborhood(
            face, apoptosis_spec["contract_span"]
        ).dropna()
        neighbors["id"] = sheet.face_df.loc[neighbors.face, "id"].values
        manager.extend(
            [
                (
                    contraction_line_tension,
                    _neighbor_contractile_increase(
                        neighbor, dt, apoptosis_spec),
                )
                for _, neighbor in neighbors.iterrows()
            ])

    if face_area < apoptosis_spec["critical_area"]:
        if current_traction < apoptosis_spec["max_traction"]:
            ab_pull(sheet, face, apoptosis_spec[
                "radial_tension"], distributed=True)
            current_traction = current_traction + dt
            apoptosis_spec.update({"current_traction": current_traction})

        elif current_traction >= apoptosis_spec["max_traction"]:
            if sheet.face_df.loc[face, "num_sides"] > 3:
                exchange(sheet, face, apoptosis_spec["geom"])
            else:
                remove(sheet, face, apoptosis_spec["geom"])
                return

    manager.append(apoptosis, **apoptosis_spec)


def _neighbor_contractile_increase(neighbor, dt, apoptosis_spec):

    contract = apoptosis_spec["contract_rate"]
    basal_contract = apoptosis_spec["basal_contract_rate"]

    increase = (
        -(contract - basal_contract) / apoptosis_spec["contract_span"]
    ) * neighbor["order"] + contract

    specs = {
        "face_id": neighbor["id"],
        "contractile_increase": increase * dt,
        "critical_area": apoptosis_spec["critical_area"],
        "max_contractility": 50,
        "multiple": True,
        "unique": False,
    }

    return specs


def propagate_line_tension(sheet, face):
    neighbors = sheet.get_neighborhood(face, 8).dropna()

    for _, neighbor in neighbors.iterrows():
        print(neighbor.face, neighbor.order)
        V1 = (sheet.face_df.loc[face][sheet.coords] -
              sheet.face_df.loc[neighbor][sheet.coords])
        # Axe proximo-distale -> à choisir en le passant en paramètre ?
        V2 = [0, 0, 1]
        dot_product = np.dot(V1, V2)
        d_V1 = np.sqrt(V1.x**2 + V1.y**2 + V1.z**2)
        d_V2 = np.sqrt(V2[0]**2 + V2[1]**2 + V2[2]**2)

        angle = np.arccosh(dot_product / (d_V1 * d_V2))

        if (angle >= (np.pi / 3)) and (angle <= (2 * np.pi / 3)):

            # augmenter "beaucoup" la tension linéaire sur 7/8 voisins
            # ou augmenter la tension linéaire que pour les jonctions parralèle
            # au pli ?
            pass
        else:
            # augmenter "légèrement" la tension linéaire sur 2/3 voisins
            # ou augmenter la tension linéaire que pour les jonctions
            # perpendiculaire au pli ?
            pass

    return
