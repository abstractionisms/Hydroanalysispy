"""
HUC-2 (Hydrologic Unit Code) Region Definitions.

The US is divided into 21 major water resource regions (HUC-2).
Each region contains nested subregions (HUC-4, HUC-6, HUC-8, etc.).

This module provides metadata for the 21 HUC-2 regions including:
- Region names
- Geographic center coordinates
- Associated states
- Map display settings

References:
- USGS WBD: https://www.usgs.gov/national-hydrography/watershed-boundary-dataset
- USGS HUC: https://water.usgs.gov/GIS/huc.html
"""

from typing import Dict, List, Any

# HUC-2 Region Definitions
# Each region has: name, center coordinates (lat, lon), and primary states
HUC2_REGIONS: Dict[str, Dict[str, Any]] = {
    '01': {
        'name': 'New England',
        'center': [43.5, -71.5],
        'zoom': 7,
        'states': ['CT', 'MA', 'ME', 'NH', 'RI', 'VT', 'NY'],
        'major_rivers': ['Connecticut', 'Merrimack', 'Penobscot', 'Kennebec'],
    },
    '02': {
        'name': 'Mid-Atlantic',
        'center': [40.5, -76.0],
        'zoom': 7,
        'states': ['DE', 'MD', 'NJ', 'NY', 'PA', 'VA', 'WV', 'DC'],
        'major_rivers': ['Delaware', 'Susquehanna', 'Hudson', 'Potomac'],
    },
    '03': {
        'name': 'South Atlantic-Gulf',
        'center': [32.0, -83.0],
        'zoom': 6,
        'states': ['AL', 'FL', 'GA', 'MS', 'NC', 'SC', 'VA'],
        'major_rivers': ['Savannah', 'Altamaha', 'Apalachicola', 'Mobile'],
    },
    '04': {
        'name': 'Great Lakes',
        'center': [44.0, -85.0],
        'zoom': 6,
        'states': ['IL', 'IN', 'MI', 'MN', 'NY', 'OH', 'PA', 'WI'],
        'major_rivers': ['St. Lawrence', 'Maumee', 'Grand (MI)', 'Fox'],
    },
    '05': {
        'name': 'Ohio',
        'center': [39.0, -83.0],
        'zoom': 6,
        'states': ['IL', 'IN', 'KY', 'MD', 'NY', 'OH', 'PA', 'TN', 'VA', 'WV'],
        'major_rivers': ['Ohio', 'Allegheny', 'Monongahela', 'Kanawha', 'Wabash'],
    },
    '06': {
        'name': 'Tennessee',
        'center': [35.5, -86.0],
        'zoom': 7,
        'states': ['AL', 'GA', 'KY', 'MS', 'NC', 'TN', 'VA'],
        'major_rivers': ['Tennessee', 'Clinch', 'French Broad', 'Holston'],
    },
    '07': {
        'name': 'Upper Mississippi',
        'center': [43.0, -91.0],
        'zoom': 6,
        'states': ['IA', 'IL', 'MN', 'MO', 'WI', 'SD'],
        'major_rivers': ['Mississippi', 'Minnesota', 'Wisconsin', 'Rock', 'Des Moines'],
    },
    '08': {
        'name': 'Lower Mississippi',
        'center': [33.0, -91.0],
        'zoom': 6,
        'states': ['AR', 'LA', 'MS', 'MO', 'TN', 'KY'],
        'major_rivers': ['Mississippi', 'Yazoo', 'Big Black', 'St. Francis'],
    },
    '09': {
        'name': 'Souris-Red-Rainy',
        'center': [48.0, -97.0],
        'zoom': 6,
        'states': ['MN', 'ND', 'SD', 'MT'],
        'major_rivers': ['Red River of the North', 'Souris', 'Rainy'],
    },
    '10': {
        'name': 'Missouri',
        'center': [43.0, -104.0],
        'zoom': 5,
        'states': ['CO', 'IA', 'KS', 'MN', 'MO', 'MT', 'ND', 'NE', 'SD', 'WY'],
        'major_rivers': ['Missouri', 'Yellowstone', 'Platte', 'Kansas', 'James'],
    },
    '11': {
        'name': 'Arkansas-White-Red',
        'center': [36.0, -97.0],
        'zoom': 6,
        'states': ['AR', 'CO', 'KS', 'LA', 'MO', 'NM', 'OK', 'TX'],
        'major_rivers': ['Arkansas', 'Red', 'White', 'Canadian', 'Cimarron'],
    },
    '12': {
        'name': 'Texas-Gulf',
        'center': [30.0, -97.0],
        'zoom': 6,
        'states': ['NM', 'TX'],
        'major_rivers': ['Brazos', 'Colorado (TX)', 'Trinity', 'Sabine', 'Nueces'],
    },
    '13': {
        'name': 'Rio Grande',
        'center': [33.0, -106.0],
        'zoom': 6,
        'states': ['CO', 'NM', 'TX'],
        'major_rivers': ['Rio Grande', 'Pecos', 'Devils'],
    },
    '14': {
        'name': 'Upper Colorado',
        'center': [39.0, -109.0],
        'zoom': 6,
        'states': ['AZ', 'CO', 'NM', 'UT', 'WY'],
        'major_rivers': ['Colorado', 'Green', 'San Juan', 'Gunnison', 'Dolores'],
    },
    '15': {
        'name': 'Lower Colorado',
        'center': [34.0, -112.0],
        'zoom': 6,
        'states': ['AZ', 'CA', 'NV', 'UT'],
        'major_rivers': ['Colorado', 'Gila', 'Salt', 'Verde', 'Little Colorado'],
    },
    '16': {
        'name': 'Great Basin',
        'center': [40.0, -117.0],
        'zoom': 6,
        'states': ['CA', 'ID', 'NV', 'OR', 'UT', 'WY'],
        'major_rivers': ['Humboldt', 'Bear', 'Truckee', 'Walker', 'Carson'],
    },
    '17': {
        'name': 'Pacific Northwest',
        'center': [46.0, -120.0],
        'zoom': 6,
        'states': ['ID', 'MT', 'OR', 'WA', 'WY', 'NV'],
        'major_rivers': ['Columbia', 'Snake', 'Willamette', 'Spokane', 'Yakima'],
    },
    '18': {
        'name': 'California',
        'center': [37.0, -120.0],
        'zoom': 6,
        'states': ['CA', 'NV', 'OR'],
        'major_rivers': ['Sacramento', 'San Joaquin', 'Klamath', 'Eel', 'Russian'],
    },
    '19': {
        'name': 'Alaska',
        'center': [64.0, -153.0],
        'zoom': 4,
        'states': ['AK'],
        'major_rivers': ['Yukon', 'Kuskokwim', 'Copper', 'Susitna', 'Tanana'],
    },
    '20': {
        'name': 'Hawaii',
        'center': [20.5, -157.0],
        'zoom': 7,
        'states': ['HI'],
        'major_rivers': ['Wailuku', 'Waimea'],
    },
    '21': {
        'name': 'Caribbean',
        'center': [18.2, -66.0],
        'zoom': 8,
        'states': ['PR', 'VI'],
        'major_rivers': ['Rio Grande de Loiza', 'Rio de la Plata'],
    },
}

# Contiguous US regions (excludes Alaska, Hawaii, Caribbean)
CONUS_REGIONS = [f'{i:02d}' for i in range(1, 19)]

# Region groupings for UI
REGION_GROUPS = {
    'Northeast': ['01', '02'],
    'Southeast': ['03', '06'],
    'Great Lakes': ['04'],
    'Midwest': ['05', '07', '08'],
    'Northern Plains': ['09', '10'],
    'Southern Plains': ['11', '12', '13'],
    'Mountain West': ['14', '15', '16'],
    'Pacific': ['17', '18'],
    'Non-Contiguous': ['19', '20', '21'],
}


def get_region_name(huc2: str) -> str:
    """Get the name of a HUC-2 region."""
    region = HUC2_REGIONS.get(huc2)
    return region['name'] if region else f"Unknown ({huc2})"


def get_region_center(huc2: str) -> List[float]:
    """Get the center coordinates [lat, lon] for a HUC-2 region."""
    region = HUC2_REGIONS.get(huc2)
    return region['center'] if region else [39.8, -98.5]  # Default: US center


def get_region_zoom(huc2: str) -> int:
    """Get the recommended zoom level for a HUC-2 region."""
    region = HUC2_REGIONS.get(huc2)
    return region.get('zoom', 6) if region else 6


def get_huc2_from_huc(huc_code: str) -> str:
    """Extract HUC-2 code from any HUC code (HUC-4, HUC-8, etc.)."""
    if not huc_code:
        return ""
    return str(huc_code)[:2].zfill(2)


def get_states_for_region(huc2: str) -> List[str]:
    """Get list of states associated with a HUC-2 region."""
    region = HUC2_REGIONS.get(huc2)
    return region.get('states', []) if region else []


def get_regions_for_state(state_code: str) -> List[str]:
    """Get list of HUC-2 regions that include a given state."""
    state_code = state_code.upper()
    return [
        huc2 for huc2, info in HUC2_REGIONS.items()
        if state_code in info.get('states', [])
    ]


def get_all_region_options() -> List[Dict[str, str]]:
    """Get list of all regions formatted for UI dropdown."""
    return [
        {'value': huc2, 'label': f"{huc2} - {info['name']}"}
        for huc2, info in sorted(HUC2_REGIONS.items())
    ]


# National map center
US_CENTER = [39.8, -98.5]
US_ZOOM = 4
