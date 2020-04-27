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
