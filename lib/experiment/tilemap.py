from ..env.tilemap import TilemapConstraints

import numpy as np

# TILEMAP_CONSTRAINTS = TilemapConstraints.from_rules([
#     ('sand', 'NESW', 'sand'),
#     ('water', 'NESW', 'water'),
    
#     ('sand_water_north', 'N', 'water'),
#     ('sand_water_north', 'S', 'sand'),
#     ('sand_water_north', 'EW', 'sand_water_north'),
#     ('sand_water_north', 'E', 'sand_water_northeast'),
#     ('sand_water_north', 'W', 'sand_water_northwest'),
#     ('sand_water_north', 'S', 'sand_water_south'),
    
#     ('sand_water_east', 'E', 'water'),
#     ('sand_water_east', 'W', 'sand'),
#     ('sand_water_east', 'NS', 'sand_water_east'),
#     ('sand_water_east', 'N', 'sand_water_northeast'),
#     ('sand_water_east', 'S', 'sand_water_southeast'),
#     ('sand_water_east', 'W', 'sand_water_west'),
    
#     # Top Right
#     ('sand_water_northeast', 'NE', 'water'),
    
#     ('sand_water_northeast', 'NE', 'sand_water_southwest'),
#     ('sand_water_northeast', 'E', 'sand_water_west'),
#     ('sand_water_northeast', 'E', 'sand_water_northwest'),
    
#     ('sand_water_northeast', 'S', 'sand_water_southeast'),
#     ('sand_water_northeast', 'W', 'sand_water_northwest'),
#     #('sand_water_northeast', 'SW', 'sand'),
    
#     ('sand_water_south', 'S', 'water'),
#     ('sand_water_south', 'N', 'sand'),
#     ('sand_water_south', 'EW', 'sand_water_south'),
#     ('sand_water_south', 'E', 'sand_water_southeast'),
#     ('sand_water_south', 'W', 'sand_water_southwest'),
#     ('sand_water_south', 'N', 'sand_water_north'),
    
#     # Bottom Right
#     ('sand_water_southeast', 'SE', 'water'),
    
#     ('sand_water_southeast', 'SE', 'sand_water_northwest'),
#     ('sand_water_southeast', 'E', 'sand_water_west'),
#     ('sand_water_southeast', 'E', 'sand_water_southwest'),
    
#     ('sand_water_southeast', 'N', 'sand_water_northeast'),
#     ('sand_water_southeast', 'S', 'sand_water_northeast'),
#     ('sand_water_southeast', 'W', 'sand_water_southwest'),
#     #('sand_water_southeast', 'NW', 'sand'),
    
#     ('sand_water_west', 'W', 'water'),
#     ('sand_water_west', 'E', 'sand'),
#     ('sand_water_west', 'NS', 'sand_water_west'),
#     ('sand_water_west', 'N', 'sand_water_northwest'),
#     ('sand_water_west', 'S', 'sand_water_southwest'),
#     ('sand_water_west', 'E', 'sand_water_east'),
    
#     # Bottom Left
#     ('sand_water_southwest', 'SW', 'water'),
    
#     ('sand_water_southwest', 'SW', 'sand_water_northeast'),
#     ('sand_water_southwest', 'W', 'sand_water_east'),
#     ('sand_water_southwest', 'W', 'sand_water_southeast'),
    
#     ('sand_water_southwest', 'N', 'sand_water_northwest'),
#     ('sand_water_southwest', 'E', 'sand_water_southeast'),
#     #('sand_water_southwest', 'NE', 'sand'),
    
#     # Top Left
#     ('sand_water_northwest', 'NW', 'water'),
    
#     ('sand_water_northwest', 'W', 'sand_water_east'),
#     ('sand_water_northwest', 'W', 'sand_water_northeast'),
#     ('sand_water_northwest', 'W', 'sand_water_southeast'),
    
#     ('sand_water_northwest', 'NS', 'sand_water_southwest'),
#     ('sand_water_northwest', 'E', 'sand_water_northeast'),
#     #('sand_water_northwest', 'SE', 'sand')
# ])

TILEMAP_CONSTRAINTS = TilemapConstraints.from_rules([
    ('sand', 'NESW', 'sand'),
    ('water', 'NESW', 'water'),
    
    ('sand_water_north', 'N', 'water'),
    ('sand_water_north', 'S', 'sand'),
    ('sand_water_north', 'EW', 'sand_water_north'),
    ('sand_water_north', 'E', 'sand_water_northeast'),
    ('sand_water_north', 'W', 'sand_water_northwest'),
    
    ('sand_water_east', 'E', 'water'),
    ('sand_water_east', 'W', 'sand'),
    ('sand_water_east', 'NS', 'sand_water_east'),
    ('sand_water_east', 'N', 'sand_water_northeast'),
    ('sand_water_east', 'S', 'sand_water_southeast'),
    
    ('sand_water_northeast', 'NE', 'water'),
    #('sand_water_northeast', 'SW', 'sand'),
    
    ('sand_water_south', 'S', 'water'),
    ('sand_water_south', 'N', 'sand'),
    ('sand_water_south', 'EW', 'sand_water_south'),
    ('sand_water_south', 'E', 'sand_water_southeast'),
    ('sand_water_south', 'W', 'sand_water_southwest'),
    
    ('sand_water_southeast', 'SE', 'water'),
    #('sand_water_southeast', 'NW', 'sand'),
    
    ('sand_water_west', 'W', 'water'),
    ('sand_water_west', 'E', 'sand'),
    ('sand_water_west', 'NS', 'sand_water_west'),
    ('sand_water_west', 'N', 'sand_water_northwest'),
    ('sand_water_west', 'S', 'sand_water_southwest'),
    
    ('sand_water_southwest', 'SW', 'water'),
    #('sand_water_southwest', 'NE', 'sand'),
    
    ('sand_water_northwest', 'NW', 'water'),
    #('sand_water_northwest', 'SE', 'sand')
])

grass = (0, 1.0, 0)
sand = (0.76, 0.7, 0.5)
water = (0, 0, 1.0)

TILEMAP_IMAGES = {
    'grass': np.full((16, 16, 3), fill_value=grass),
    'sand': np.full((16, 16, 3), fill_value=sand),
    'water': np.full((16, 16, 3), fill_value=water),
}
TILEMAP_IMAGES['sand_water_north'] = np.full((16, 16, 3), fill_value=sand)
TILEMAP_IMAGES['sand_water_north'][:8, :, :] = water
TILEMAP_IMAGES['sand_water_north'][8, :, :] = (1.0, 0, 0)

TILEMAP_IMAGES['sand_water_east'] = np.full((16, 16, 3), fill_value=sand)
TILEMAP_IMAGES['sand_water_east'][:, 8:, :] = water
TILEMAP_IMAGES['sand_water_east'][:, 8, :] = (1.0, 0, 0)

TILEMAP_IMAGES['sand_water_south'] = np.full((16, 16, 3), fill_value=sand)
TILEMAP_IMAGES['sand_water_south'][8:, :, :] = water
TILEMAP_IMAGES['sand_water_south'][8, :, :] = (1.0, 0, 0)

TILEMAP_IMAGES['sand_water_west'] = np.full((16, 16, 3), fill_value=sand)
TILEMAP_IMAGES['sand_water_west'][:, :8, :] = water
TILEMAP_IMAGES['sand_water_west'][:, 8, :] = (1.0, 0, 0)

TILEMAP_IMAGES['sand_water_northeast'] = np.full((16, 16, 3), fill_value=water)
TILEMAP_IMAGES['sand_water_northeast'][8:, :8, :] = sand
TILEMAP_IMAGES['sand_water_northeast'][8, :8, :] = (1.0, 0, 0)
TILEMAP_IMAGES['sand_water_northeast'][8:, 8, :] = (1.0, 0, 0)

TILEMAP_IMAGES['sand_water_southeast'] = np.full((16, 16, 3), fill_value=water)
TILEMAP_IMAGES['sand_water_southeast'][:8, :8, :] = sand
TILEMAP_IMAGES['sand_water_southeast'][8, :8, :] = (1.0, 0, 0)
TILEMAP_IMAGES['sand_water_southeast'][:8, 8, :] = (1.0, 0, 0)

TILEMAP_IMAGES['sand_water_southwest'] = np.full((16, 16, 3), fill_value=water)
TILEMAP_IMAGES['sand_water_southwest'][:8, 8:, :] = sand
TILEMAP_IMAGES['sand_water_southwest'][:8, 8, :]= (1.0, 0, 0)
TILEMAP_IMAGES['sand_water_southwest'][8, 8:, :] = (1.0, 0, 0)

TILEMAP_IMAGES['sand_water_northwest'] = np.full((16, 16, 3), fill_value=water)
TILEMAP_IMAGES['sand_water_northwest'][8:, 8:, :] = sand
TILEMAP_IMAGES['sand_water_northwest'][8:, 8, :] = (1.0, 0, 0)
TILEMAP_IMAGES['sand_water_northwest'][8, 8:, :] = (1.0, 0, 0)

for k in TILEMAP_IMAGES:
    TILEMAP_IMAGES[k][15, :, :] = (0, 0, 0)
    TILEMAP_IMAGES[k][:, 15, :] = (0, 0, 0)