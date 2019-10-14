import numpy as np
from tyssue.utils.decorators import face_lookup
from tyssue.geometry.sheet_geometry import SheetGeometry
from tyssue.behaviors.sheet.actions import (exchange,
                                            remove,
                                            decrease,
                                            increase_linear_tension,
                                            set_value)


default_pattern = {
    "t": 0.,
    "dt": 0.1,
    "time_of_last_apoptosis": 30
}


def apoptosis_patterning(sheet, manager, **kwargs):
    """Pattern of face apoptosis entry.

    Apoptotic cell enter in apoptosis by following a time dependent pattern from
    ventral to dorsal part of the leg tissue.

    Parameters
    ----------
    sheet : a :class:`Sheet` object
    manager : a :class:`EventManager` object
    t : float : current simulation time
    dt : float : time step
    time_of_last_apoptosis : float : time spend to go to dorsal from ventral.

    """
    specs = default_pattern
    specs.update(**kwargs)
    t = specs['t']
    end = specs["time_of_last_apoptosis"]

    # -t * max(phi)/tfin + max(phi)
    phi_min = -t * max(np.abs(sheet.face_df.phi)) / \
        end + max(np.abs(sheet.face_df.phi))

    l_index_apoptosis_cell = sheet.face_df[(np.abs(sheet.face_df.phi) > phi_min) &
                                           (sheet.face_df.apoptosis > 0)
                                           ].id.values
    apopto_kwargs = sheet.settings['apoptosis'].copy()
    for c in l_index_apoptosis_cell:
        if c in sheet.face_df.id:
            apopto_kwargs.update(
                {
                    'face_id': c,
                }
            )
            notpresent = True
            for tup in manager.current:
                if c in tup[1]:
                    notpresent = False
            if notpresent:
                manager.append(apoptosis, **apopto_kwargs)

    specs.update({"t": specs['t'] + specs['dt']})
    manager.append(apoptosis_patterning, **specs)


default_apoptosis_spec = {
    "face_id": -1,
    "face": -1,
    "critical_area": 1e-2,
    "critical_area_pulling": 10,
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
    # reutiliser cette ligne après verification du manager
    #current_traction = apoptosis_spec["current_traction"]
    current_traction = sheet.face_df.loc[face, "current_traction"]
    face_area = sheet.face_df.loc[face, "area"]

    if (face_area > apoptosis_spec["critical_area"]):
        # reduce prefered_area
        decrease(sheet,
                 'face',
                 face,
                 apoptosis_spec["contract_rate"],
                 col="prefered_area",
                 divide=True,
                 bound=apoptosis_spec["critical_area"],
                 )

        decrease(sheet,
                 'face',
                 face,
                 np.sqrt(apoptosis_spec["contract_rate"]),
                 col="prefered_perimeter",
                 divide=True,
                 )

        # contract neighbors
        neighbors = sheet.get_neighborhood(
            face, apoptosis_spec["contract_span"]
        ).dropna()
        manager.extend(
            [
                (
                    neighbor_contraction,
                    _neighbor_contractile_increase(
                        neighbor, dt, apoptosis_spec, sheet),
                )
                for _, neighbor in neighbors.iterrows()
            ])

    if face_area < apoptosis_spec["critical_area_pulling"]:
        if current_traction < apoptosis_spec["max_traction"]:
            # AB pull
            set_value(sheet,
                      'face',
                      face,
                      apoptosis_spec['radial_tension'],
                      col="radial_tension")
            # Verifier que le manager fonctionne avant de supprimer
            current_traction = current_traction + dt
            set_value(sheet,
                      'face',
                      face,
                      current_traction,
                      col="current_traction")
            apoptosis_spec.update({"current_traction": current_traction})

        else:
            if sheet.face_df.loc[face, "num_sides"] > 3:
                exchange(sheet, face, apoptosis_spec["geom"])
            else:
                remove(sheet, face, apoptosis_spec["geom"])
                return

    if apoptosis_spec["face_id"] in (sheet.face_df.id):
        manager.append(apoptosis, **apoptosis_spec)


def _neighbor_contractile_increase(neighbor, dt, apoptosis_spec, sheet):

    specs = sheet.settings['contraction_lt_kwargs'].copy()

    increase = (
        -(apoptosis_spec['contract_rate'] - apoptosis_spec['basal_contract_rate']
          ) / apoptosis_spec["contract_span"]
    ) * neighbor["order"] + apoptosis_spec['contract_rate']

    specs.update({
        "face_id": neighbor.face,
        "contract_rate": increase,
        "unique": False
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
def neighbor_contraction(sheet, manager, **kwargs):
    """
    Single step contraction event
    """
    contraction_spec = default_contraction_line_tension_spec
    contraction_spec.update(**kwargs)
    face = contraction_spec["face"]
    if sheet.face_df.loc[face].apoptosis == 1:
        return

    if sheet.face_df.loc[face, "prefered_area"] > contraction_spec['critical_area']:

        decrease(sheet,
                 'face',
                 face,
                 contraction_spec["contract_rate"],
                 col="prefered_area",
                 divide=True,
                 bound=contraction_spec['critical_area'],
                 )
        decrease(sheet,
                 'face',
                 face,
                 np.sqrt(contraction_spec["contract_rate"]),
                 col="prefered_perimeter",
                 divide=True,
                 bound=1,
                 )

