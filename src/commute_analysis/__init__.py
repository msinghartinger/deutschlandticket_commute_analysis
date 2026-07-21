from .routing import build_transport_network, route_between_points
from .synthetic_population import (
    download_population_grid,
    random_point_in_polygon,
    sample_population_weighted_locations,
)
from .geometry import create_target_area
from .scoring import (
    EmployeeRoutingBatchResult,
    calculate_car_record,
    calculate_public_transport_record,
    _extract_number_of_transfers,
    _extract_total_walking_time,
)

try:
    from .visualization import (
        plot_commute_map,
        plot_routes,
        plot_sampling_probability_map,
        plot_synthetic_employees,
    )
    _HAS_VISUALIZATION = True
except ImportError:
    _HAS_VISUALIZATION = False

__all__ = [
    "build_transport_network",
    "download_population_grid",
    "random_point_in_polygon",
    "route_between_points",
    "sample_population_weighted_locations",
    "create_target_area",
    "EmployeeRoutingBatchResult",
    "calculate_car_record",
    "calculate_public_transport_record",
]

if _HAS_VISUALIZATION:
    __all__.extend(
        [
            "plot_commute_map",
            "plot_routes",
            "plot_sampling_probability_map",
            "plot_synthetic_employees",
        ]
    )
