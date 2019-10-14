import numpy as np
from tyssue.utils.decorators import face_lookup
from tyssue.geometry.sheet_geometry import SheetGeometry
from tyssue.behaviors.sheet.actions import (exchange,
                                            remove,
                                            decrease,
                                            increase,
                                            increase_linear_tension,
                                            set_value)


default_constriction_spec = {
    "face_id": -1,
    "face": -1,
    "contract_rate": 2,
    "critical_area": 1e-2,
    "radial_tension": 1.0,
    "contract_neighbors": True,
    "critical_area_neighbors": 10,
    "critical_area_pulling": 10,
    "contract_span": 2,
    "basal_contract_rate": 1.001,
    "current_traction": 0,
    "max_traction": 30,
}


@face_lookup
def delamination(sheet, manager, **kwargs):
    """Constriction process
    This function corresponds to the process called "apical constriction"
    in the manuscript
    The cell undergoing delamination first contracts its apical
    area until it reaches a critical area. A probability
    dependent to the apical area allow an apico-basal
    traction of the cell. The cell can pull during max_traction
    time step, not necessarily consecutively.
    Parameters
    ----------
    sheet : a :class:`tyssue.sheet` object
    manager : a :class:`tyssue.events.EventManager` object
    face_id : int
       the Id of the face undergoing delamination.
    contract_rate : float, default 2
       rate of increase of the face contractility.
    critical_area : float, default 1e-2
       face's area under which the cell starts loosing sides.
    radial_tension : float, default 1.
       tension applied on the face vertices along the
       apical-basal axis.
    contract_neighbors : bool, default `False`
       if True, the face contraction triggers contraction of the neighbor
       faces.
    contract_span : int, default 2
       rank of neighbors contracting if contract_neighbor is True. Contraction
       rate for the neighbors is equal to `contract_rate` devided by
       the rank.
    """
    constriction_spec = default_constriction_spec
    constriction_spec.update(**kwargs)

    # initialiser une variable face
    # aller chercher la valeur dans le dictionnaire à chaque fois ?
    face = constriction_spec["face"]
    current_traction = constriction_spec["current_traction"]

    face_area = sheet.face_df.loc[face, "area"]

    if face_area > constriction_spec["critical_area"]:
        # reduce prefered_area
        decrease(sheet,
                 'face',
                 face,
                 constriction_spec["contract_rate"],
                 col="prefered_area",
                 divide=True,
                 bound=constriction_spec["critical_area"],
                 )
        decrease(sheet,
                 'face',
                 face,
                 np.sqrt(constriction_spec["contract_rate"]),
                 col="prefered_perimeter",
                 divide=True,
                 )

        neighbors = sheet.get_neighborhood(
            face, constriction_spec["contract_span"]
        ).dropna()

        manager.extend(
            [
                (
                    contraction_line_tension,
                    _neighbor_contractile_increase_delamination(
                        neighbor, 1, constriction_spec, sheet),
                )
                for _, neighbor in neighbors.iterrows()
            ]
        )

    if face_area < constriction_spec["critical_area_pulling"]:
        if current_traction < constriction_spec["max_traction"]:
            # AB pull
            set_value(sheet,
                      'face',
                      face,
                      constriction_spec['radial_tension'],
                      col="radial_tension")
            set_value(sheet,
                      'face',
                      face,
                      current_traction,
                      col="current_traction")
            constriction_spec.update({"current_traction": current_traction})

        if (current_traction > 10) and (sheet.face_df.loc[face, "num_sides"] > 3):
            exchange(sheet, face, constriction_spec["geom"])

    if constriction_spec["face_id"] in (sheet.face_df.id):
        manager.append(delamination, **constriction_spec)


def _neighbor_contractile_increase_delamination(neighbor, dt, constriction_spec, sheet):

    specs = sheet.settings['delaminate_setting'].copy()

    contract = specs["contract_rate"]
    basal_contract = specs["basal_contract_rate"]

    increase_ = (
        -(contract - basal_contract) / constriction_spec["contract_span"]
    ) * neighbor["order"] + contract

    specs.update({
        "face_id": neighbor.face,
        "contractile_increase": increase_,
        "critical_area": constriction_spec["critical_area"],
        "multiple": True,
        "unique": False,
    })

    return specs


default_contraction_line_tension_spec = {
    "face_id": -1,
    "face": -1,
    "contract_rate": 1.05,
    "critical_area": 5.,
    "model": None,
}


@face_lookup
def contraction_line_tension(sheet, manager, **kwargs):
    """
    Single step contraction event
    """
    contraction_spec = default_contraction_line_tension_spec
    contraction_spec.update(**kwargs)
    face = contraction_spec["face"]

    if sheet.face_df.loc[face, "area"] > contraction_spec["critical_area"]:
        # reduce prefered_area
        decrease(sheet,
                 'face',
                 face,
                 contraction_spec["contract_rate"],
                 col="prefered_area",
                 divide=True,
                 bound=contraction_spec["critical_area"],
                 )
        decrease(sheet,
                 'face',
                 face,
                 np.sqrt(contraction_spec["contract_rate"]),
                 col="prefered_perimeter",
                 divide=True,
                 bound=1,
                 )
