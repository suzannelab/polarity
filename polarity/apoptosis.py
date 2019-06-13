import random
import numpy as np

from tyssue.utils.decorators import face_lookup
from tyssue.geometry.sheet_geometry import SheetGeometry

from tyssue.behaviors.sheet.actions import contract, ab_pull, exchange, remove, contract
from tyssue.behaviors.sheet.basic_events import contraction

default_apoptosis_spec = {
    "face_id": -1,
    "face": -1,
    "critical_area": 1e-2,
    "radial_tension": 0.1,
    "contract_rate": 0.1,
    "basal_contract_rate": 1.001,
    "contract_span": 2,
    "max_traction": 30,
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

    # Small variable name for some spec
    face = apoptosis_spec["face"]
    current_traction = apoptosis_spec["current_traction"]
    face_area = sheet.face_df.loc[face, "area"]

    if face_area > apoptosis_spec["critical_area"]:
        # contract
        contract(
            sheet,
            face,
            apoptosis_spec["contract_rate"],
            True)
        # contract neighbors
        neighbors = sheet.get_neighborhood(
            face, apoptosis_spec["contract_span"]
        ).dropna()
        neighbors["id"] = sheet.face_df.loc[neighbors.face, "id"].values

        manager.extend(
            [
                (
                    contraction,
                    _neighbor_contractile_increase(neighbor, apoptosis_spec),
                )
                for _, neighbor in neighbors.iterrows()
            ])

    proba_tension = np.exp(-face_area / apoptosis_spec["critical_area"])
    aleatory_number = random.uniform(0, 1)

    if current_traction < apoptosis_spec["max_traction"]:
        if aleatory_number < proba_tension:
            current_traction = current_traction + 1
            ab_pull(sheet, face, apoptosis_spec[
                "radial_tension"], distributed=False)
            apoptosis_spec.update({"current_traction": current_traction})

    elif current_traction >= apoptosis_spec["max_traction"]:
        if sheet.face_df.loc[face, "num_sides"] > 3:
            exchange(sheet, face, apoptosis_spec["geom"])
        else:
            remove(sheet, face, apoptosis_spec["geom"])
            return

    manager.append(apoptosis, **apoptosis_spec)


def _neighbor_contractile_increase(neighbor, apoptosis_spec):

    contract = apoptosis_spec["contract_rate"]
    basal_contract = apoptosis_spec["basal_contract_rate"]

    increase = (
        -(contract - basal_contract) / apoptosis_spec["contract_span"]
    ) * neighbor["order"] + contract

    specs = {
        "face_id": neighbor["id"],
        "contractile_increase": increase,
        "critical_area": apoptosis_spec["critical_area"],
        "max_contractility": 50,
        "multiple": True,
        "unique": False,
    }

    return specs
