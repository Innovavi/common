from typing import Tuple, List, Union, Optional, Dict
import pandas as pd
from common.data_manipulation.pandas_tools import get_landmarks_from_pandas, get_bbox_from_pandas


DIGIFACE_COEF_COLUMNS = [
    'Brows Arch', 'Brows Centre Depth', 'Brows Depth', 'Brows Distance',
    'Brows Height', 'Brows Inner Height', 'Brows Middle Height',
    'Brows Outer Height', 'Brows Outer Width', 'ForeHead Depth',
    'Cheeks Bones Size', 'Cheeks Define', 'Cheeks Crease',
    'Cheeks Size (Alt)', 'Chin Cleft', 'Chin Define', 'Chin Depth',
    'Chin Length', 'Chin Valley Abrupt', 'Chin Valley Depth',
    'Jaw Corner Height', 'Jaw Corner Width (Alt)', 'Jaw Curve (Alt)',
    'Jaw Definition', 'Jaw Depth', 'Jaw Height (Alt)', 'Jaw Width',
    'Under Jaw Neck Height', 'Face Height', 'Face Length', 'Eye Lids Depth',
    'Eyes Distance', 'Eyes Inner Height', 'Eyes Outer Height',
    'Eye Lids Height A', 'Eyes Depth (Alt)', 'Eyes Arched', 'Eyes Height B',
    'Eyes Inner Width', 'Eyes Outer Width', 'Nose Angle Down',
    'Nose Angle Up', 'Nose Curve In', 'Nose Curve Out', 'Nose Depth (Alt)',
    'Nose Scale', 'Nostrils Size', 'Nostrils Length',
    'Nostrils Front Height', 'Nose Wing Width', 'Nose Wing Volume',
    'Nose Wing Upper Height', 'Nose Wing Height', 'Nose Wing Arc',
    'Nose Width Center Line', 'Nose Size Depth', 'Nose Bridge Depth (Alt)',
    'Nose Lower Height', 'Philtrum Curve', 'Mouth Width (Alt)',
    'Mouth Height (Alt)', 'Lips Thin (Alt)', 'Lip Upper Width',
    'Lip Upper Volume', 'Lip Upper Peaks Smooth',
    'Lip Upper Peaks Distance', 'Lip Upper Enlarge',
    'Lip Upper Corners Pout', 'Lip Lower Volume', 'Lip Lower Width',
    'Lip Lower Curves A', 'Lip Lower Curves C', 'Lip Line Height',
    'Lip Line Depth', 'Lip Line Curves Outer', 'Lip Line Curves Inner',
    'Lip Upper Depth', 'Lip Upper Fuller', 'Lip Lower Depth',
    'Lip Lower Fuller', 'Lip Upper Peak Height'
]


def parse_digiface_pandas_row(pandas_row: Union[pd.Series, pd.DataFrame]):
    metadata = pandas_row['metadata']

    image_id = metadata.get('image_id', -1)
    face_id = metadata.get('face_id', -1)
    dataset = metadata.get('dataset', '')
    set_name = metadata.get('set', '')
    image_name = metadata.get('image_name', '')
    batch_id = metadata.get('batch_id', -1)
    camera_type = metadata.get('camera_type', '')

    dome_rotation = pandas_row['dome_rotation'].values
    bbox = get_bbox_from_pandas(pandas_row['bbox'])
    coefs = pandas_row['coefs'].values

    full_landmarks = get_landmarks_from_pandas(pandas_row['landmarks'], to_lm=2987, filter_default=False)

    return image_id, face_id, image_name, set_name, camera_type, dome_rotation, bbox, coefs, full_landmarks



