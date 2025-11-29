from typing import Tuple, List, Union, Optional, Dict
import pandas as pd
from common.data_manipulation.pandas_tools import get_landmarks_from_pandas, get_bbox_from_pandas

DIGIFACE_G9_COEF_COLUMNS = [
    '200_head_bs_BrowsDepth', '200_head_bs_BrowsDistance', '200_head_bs_BrowsHeight', '200_head_bs_BrowsInnerHeight', '200_head_bs_BrowsMiddleHeight', '200_head_bs_BrowsOuterHeight',
    '200_head_bs_BrowsOuterWidth', 'head_bs_BrowSkinSlack', '200_head_bs_ForeHeadWidth', '200_head_bs_ForeHeadDepth', '200_head_bs_CheeksBonesSize', '200_head_bs_CheeksDefine',
    '200_head_bs_CheeksNasolabialDetail', '200_head_bs_CheeksDepthLower', '200_head_bs_CheeksDepthUpper', '200_head_bs_CheeksVolumeLower', '200_head_bs_CheeksVolumeMiddle', 'head_bs_ChinCleft',
    '200_head_bs_ChinDefine', 'head_bs_ChinDepth', 'head_bs_ChinHeight', 'head_bs_ChintoLipDepth', '200_head_bs_ChinValleyHeight', '200_head_bs_ChinWidth',
    '200_head_bs_JawCornerHeight', '200_head_bs_JawCornerWidth(Alt)', '200_head_bs_JawCurve(Alt)', '200_head_bs_JawDefinition', '200_head_bs_JawDepth', 'head_bs_JawWholeHeight',
    'head_bs_JawtoNeckSlack', 'head_bs_JawWholeWidth', '200_head_bs_FaceHeight', '200_head_bs_FaceLength', '200_head_bs_CraniumWidth', '200_head_bs_FaceWeightAdjust',
    '200_head_bs_EyesHeightB', '200_head_bs_EyesDistance', '200_head_bs_EyesDepth(Alt)', '200_head_bs_EyeLidsHeightB', 'head_bs_EyelidCreaseUpperSmooth', 'head_bs_EyeWholeSize',
    'head_bs_EyeAngleInner', 'head_bs_EyeAngleOuter', '200_head_bs_NoseLength', '200_head_bs_NoseTipAngle', 'head_bs_NoseBaseSize', '200_head_bs_NoseDepthLower',
    '200_head_bs_NoseDepthUpper', 'head_bs_NoseWholeSize', 'head_bs_NoseTipDepth', 'head_bs_NoseBridgeUpperDepth', 'head_bs_NoseTipSize', '200_head_bs_NoseWidthAll',
    '200_head_bs_NoseWidthLower', 'head_bs_NoseWingHeight', 'head_bs_NoseWingSizeVertical', 'head_bs_NoseBridgeSize', 'head_bs_NoseBridgeSlope', 'head_bs_MouthHeight',
    '200_head_bs_LipsWidth', 'head_bs_MouthDepth', 'head_bs_MouthSize', 'head_bs_LipThinLowerInner', 'head_bs_LipThinLowerOuter', 'head_bs_LipThinUpperInner',
    'head_bs_LipThinUpperOuter', '200_head_bs_LipUpperWidth', '200_head_bs_LipLowerWidthHard', '200_head_bs_LipUpperDepth', '200_head_bs_LipLowerDepth', '200_head_bs_LipUpperPeakHeight',
    '200_head_bs_LipUpperPeaksShape', '200_head_bs_LipLineHeight', 'head_bs_LipSizeLower', 'head_bs_LipSizeUpper', '200_head_bs_LipsDefine', 'head_bs_MouthApertureCurveUpper',
    '200_head_bs_LipsOuterVertical', '200_head_bs_LipUpperFuller', '200_head_bs_LipLowerFuller', '200_head_bs_PhiltrumCurve', '200_head_bs_PhiltrumWidth', 'HeavyFaceOnly',
    'LitheHeadOnly', 'MassBodyHeadOnly', 'EmaciatedHeadOnly', '200_head_bs_Ears_HeightMovement', '200_head_bs_EarsSizeHorizontal', '200_head_bs_EarsSizeVertical',
    '200_head_bs_EarsAngle1', 'head_bs_EarlobeLength', 'head_bs_EarlobeSize', 'head_bs_EarDepth', '200_head_bs_EarsCurl2', '200_head_bs_EarsLowerCurve',
    'head_ctrl_ProportionHeadSize_scl', 'head_bs_NoseWingLowerHeight', '200_head_bs_NoseFlesh', '200_head_bs_CraniumLowerWidth', '200_head_bs_CraniumUpperWidth', '200_head_bs_JawMiddleIndent',
    'SFDBMK_body_bs_AdamsAppleShape', '200_head_bs_ChinLength', 'head_bs_NoseBridgeBump', 'head_bs_EyeWholeAngle', '200_head_bs_NostrilsFrontHeight', '200_head_bs_NoseNostrilsSize',
    '200_head_bs_PhiltrumDepthCenter', '200_head_bs_SeptumCurve', '200_head_bs_NoseSeptumWidth', '200_head_bs_NoseWingArc', 'head_bs_EyeAlmondInner', 'head_bs_EyeAlmondOuter',
    '200_head_bs_EyesCornerOuterDown','200_head_bs_EyesCornerOuterLonger','head_bs_NoseBridgeDefinition','200_head_bs_LipUpperCurveOut','200_head_bs_LipsCenterVertical2','head_bs_FaceDepth',
    'head_bs_ChintoLipSharpen','200_head_bs_ChinScale','head_bs_EyeIrisSizeLarger','head_bs_EyeIrisSizeSmaller','head_bs_NoseTipSeptumContour','head_bs_NoseWingDepth',
    '200_head_bs_PhiltrumDepth','200_head_bs_NoseWingVolume','200_head_bs_LipPart','200_head_bs_Ears_Rotate','200_head_bs_LipsCenterVertical1','200_head_bs_FaceAngleTilt',
    '200_head_bs_CraniumHeight','head_bs_BrowArchSizeInner','200_head_bs_LipLowerCenterHeight','200_head_bs_LipUpperEnlarge','head_bs_FaceOlder','head_bs_FaceRound',
    'head_bs_FaceHeart','head_bs_FaceSquare','head_bs_FaceCenterDepth','head_bs_MouthBelowSizeInner','head_bs_NoseBridgeUpperHeight'
]

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

DIGIFACE_COEF_COLUMNS_S = [
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
    'Eyes Inner Width', 'Eyes Outer Width', 'Nose Angle', 'Nose Curve',
    'Nose Depth (Alt)', 'Nose Scale', 'Nostrils Size', 'Nostrils Length',
    'Nostrils Front Height', 'Nose Wing Width', 'Nose Wing Volume',
    'Nose Wing Upper Height', 'Nose Wing Height', 'Nose Wing Arc',
    'Nose Width Center Line', 'Nose Size Depth', 'Nose Bridge Depth (Alt)',
    'Nose Lower Height', 'Philtrum Curve', 'Mouth Width (Alt)',
    'Mouth Height (Alt)', 'Lips Thin (Alt)', 'Lip Upper Width',
    'Lip Upper Volume', 'Lip Upper Peaks Smooth',
    'Lip Upper Peaks Distance', 'Lip Upper Enlarge',
    'Lip Upper Corners Pout', 'Lip Lower Volume', 'Lip Lower Width',
    'Lip Lower Curves', 'Lip Line Height', 'Lip Line Depth',
    'Lip Line Curves Outer', 'Lip Line Curves Inner', 'Lip Upper Depth',
    'Lip Upper Fuller', 'Lip Lower Depth', 'Lip Lower Fuller',
    'Lip Upper Peak Height'
]

DIGIFACE_COEF_COLUMNS_B = [
    'Brows Arch', 'Brows Centre Depth', 'Brows Depth', 'Brows Distance', 'Brows Height',
    'Brows Inner Height', 'Brows Middle Height', 'Brows Outer Height', 'Brows Outer Width',
    'Cheeks Bones Size', 'Cheeks Crease', 'Cheeks Define', 'Cheeks Size (Alt)', 'Chin Cleft',
    'Chin Define', 'Chin Depth', 'Chin Length', 'Chin Valley Abrupt', 'Chin Valley Depth',
    'Eyes Arched', 'Eyes Depth (Alt)', 'Eyes Distance', 'Eyes Height B', 'Eyes Inner Height',
    'Eyes Inner Width', 'Eyes Outer Height', 'Eyes Outer Width', 'Eye Lids Depth', 'Eye Lids Height A',
    'Face Height', 'Face Length', 'ForeHead Depth', 'Jaw Corner Height', 'Jaw Corner Width (Alt)',
    'Jaw Curve (Alt)', 'Jaw Definition', 'Jaw Depth', 'Jaw Height (Alt)', 'Jaw Width',
    'Lips Thin (Alt)', 'Lip Line Curves Inner', 'Lip Line Curves Outer', 'Lip Line Depth',
    'Lip Line Height', 'Lip Lower Curves A', 'Lip Lower Curves C', 'Lip Lower Depth', 'Lip Lower Fuller',
    'Lip Lower Volume', 'Lip Lower Width', 'Lip Upper Corners Pout', 'Lip Upper Depth', 'Lip Upper Enlarge',
    'Lip Upper Fuller', 'Lip Upper Peaks Distance', 'Lip Upper Peaks Smooth', 'Lip Upper Volume', 'Lip Upper Width',
    'Mouth Height (Alt)', 'Mouth Width (Alt)', 'Nose Angle Down', 'Nose Angle Up', 'Nose Bridge Depth (Alt)',
    'Nose Curve In', 'Nose Curve Out', 'Nose Depth (Alt)', 'Nose Lower Height', 'Nose Scale', 'Nose Size Depth',
    'Nose Width Center Line', 'Nose Wing Arc', 'Nose Wing Height', 'Nose Wing Upper Height', 'Nose Wing Volume',
    'Nose Wing Width', 'Nostrils Front Height', 'Nostrils Length', 'Nostrils Size', 'Philtrum Curve',
    'Under Jaw Neck Height', 'Lip Upper Peak Height'
]

DIGIFACE_COEF_COLUMNS_C = [
    'Brows Arch', 'Brows Centre Depth', 'Brows Depth', 'Brows Distance', 'Brows Height', 'Brows Inner Height',
    'Brows Middle Height', 'Brows Outer Height', 'Brows Outer Width', 'ForeHead Depth',
    'Cheeks Bones Size', 'Cheeks Define', 'Cheeks Crease', 'Cheeks Size (Alt)', 'Chin Cleft', 'Chin Define',
    'Chin Depth', 'Chin Length', 'Chin Valley Abrupt', 'Chin Valley Depth', 'Jaw Corner Height',
    'Jaw Corner Width (Alt)', 'Jaw Curve (Alt)', 'Jaw Definition', 'Jaw Depth', 'Jaw Height (Alt)', 'Jaw Width',
    'Under Jaw Neck Height', 'Face Height', 'Face Length', 'Eye Lids Depth', 'Eyes Distance',
    'Eyes Inner Height', 'Eyes Outer Height', 'Eye Lids Height A', 'Eyes Depth (Alt)', 'Eyes Arched', 'Eyes Height B',
    'Eyes Inner Width', 'Eyes Outer Width', 'Nose Angle Down', 'Nose Angle Up', 'Nose Curve In',
    'Nose Curve Out', 'Nose Depth (Alt)', 'Nose Scale', 'Nostrils Size', 'Nostrils Length', 'Nostrils Front Height',
    'Nose Wing Width', 'Nose Wing Volume', 'Nose Wing Upper Height', 'Nose Wing Height', 'Nose Wing Arc',
    'Nose Width Center Line', 'Nose Size Depth', 'Nose Bridge Depth (Alt)', 'Nose Lower Height', 'Philtrum Curve',
    'Mouth Width (Alt)', 'Mouth Height (Alt)', 'Lips Thin (Alt)', 'Lip Upper Width', 'Lip Upper Volume',
    'Lip Upper Peaks Smooth', 'Lip Upper Peaks Distance', 'Lip Upper Enlarge', 'Lip Upper Corners Pout',
    'Lip Lower Volume', 'Lip Lower Width', 'Lip Lower Curves A', 'Lip Lower Curves C', 'Lip Line Height',
    'Lip Line Depth', 'Lip Line Curves Outer', 'Lip Line Curves Inner', 'Lip Upper Depth', 'Lip Upper Fuller',
    'Lip Lower Depth', 'Lip Lower Fuller', 'Lip Upper Peak Height'
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

    full_landmarks = get_landmarks_from_pandas(pandas_row['landmarks'], to_lm=3555, filter_default=False)

    return image_id, face_id, image_name, set_name, camera_type, dome_rotation, bbox, coefs, full_landmarks



