from typing import Union

import pandas as pd

from common.data_manipulation.pandas_tools import get_bbox_from_pandas, get_landmarks_from_pandas


DIGIBODY_COEF_COLUMNS = [
    'Shoulder_Width', 'Arm_Size', '4_Belly_Move_InOut', 'PBMGlutesSize', 'Hip_Size_2', 'PBMThighsSize', 'PBMShinsSize', 'PBMBreastsGone', 'PBMBreastsSize','CTRLBreastsNatural',
    'PBMBreastsShape04', 'CTRLFitness', 'CTRLBodybuilder', 'Glute_Depth', 'Torso_Depth', 'Arm_Flab', 'PBMRibcageSize', 'Waist_Depth', 'Waist_Straightness', 'Neck_Width',
    'FBMHeavy', 'FBMEmaciated', 'Wrist_Thickness', '5_Belly_Shape_Pregnant', '5_Belly_Shape_Fat_1', 'PBMStomachLowerDepth', 'P3DTLBackHollow', 'P3DTLBackStraight', 'PBMBreastsDiameter',
    'PBMBreastsHeavy', 'PBMBreastShape05', 'Torso_Width', 'Arm_Length', 'Shape_Width_Adjust', 'PBMSternumDepth', 'Abdomen_Length', 'SCLNeckLength', 'Thigh_Length', 'Shin_Length',
    'Pelvic_Length', 'Traps_shape', 'PBMGlutesUtilitiesScaleX', 'PBMGlutesUtilitiesScaleY', 'PBMGlutesUtilitiesScaleZ', 'Glute_UpDown', 'Hip_UpDown', 'Hip_Depth', 'GHD_Glutes_Square',
    'GHD_Hip_Square', 'Glute_Width_Lower', 'Glute_Width_Upper', 'GHD_Glutes_Bubble', 'GHD_Glutes_LargeUp', 'GHD_Glutes_Round_Large', 'GHD_Glutes_Saggy', 'P3DTLButtocksNarrow',
    'Glute_Angle', 'PBMGlutesUtilitiesShape04', 'PBMGlutesUtilitiesShape34', 'PBMGlutesUtilitiesShape32', 'Hip_Shape_4', 'Hip_Rotate_2', 'Glute_Line_sharpness', 'Glute_Height_Lower',
    'Glute_Height_Inner', 'Pelvic_Depth', 'PBMBreastsDownwardSlope', 'CTRLBreastsImplants', 'PBMBreastsPerkSide', 'GHD_Breasts_Shape_06', 'P3DTLBreastsSaggy', 'PBMBreastShape07',
    'PBMBreastShape21', 'PBMBreastShape25', 'PBMBreastShape04', 'PBMBreastShape10', 'PBMBreastShape26', 'PBMBreastsShape05', 'PBMBreastSeparation', 'PBMBreastsSmall',
    'PBMBreastUnderCurve', 'BD_Figure_Curvy_Breasts', 'BD_Figure_Pear_Breasts', 'PBMPregnant', 'GHD_Stomach_Heavy', # 'GHD_Ribcage_Thin',
    'Back_Rotate', 'Back_Updown', 'Abdomen_Width', 'Abdomen_Size', 'RibCage_Defintion', 'PBMSternumHeight', '0_Belly_Fold_Horizontal', '0_Belly_Size', '1_Belly_Upper_Bulge',
    '2_Belly_Lower_Width', '2_Belly_Lower_UpDown', '4_Belly_Move_Rotate', '4_Belly_Move_UpDown', '5_Belly_Shape_Fat_2', '5_Belly_Shape_Fat_3', 'Torso_Muscular', 'Waist_Size',
    'Waist_Shape', 'Waist_Rotate', 'Waist_Height', 'Waist_FrontBack', 'GHD_Abdominal_Muscle', 'PBMLoveHandles', 'GHD_Stomach_In', 'GHD_Stomach_Puffy', 'GHD_Stomach_Sitting_Rolls',
    'P3DTLBellyBigger', 'GHD_Back_Heavy', 'PBMLatsSize', 'GHD_Scapula_Out', 'PBMShouldersSize', 'PBMForearmsSize', 'PBMUpperArmsSize', 'GHD_Arm_Muscles', 'SCLPropagatingHand',
    'SCLFingersLength', 'P3DTLArmsThin', 'Neck_Depth', 'FK-ThighsGapClosed1', 'Thigh_Front', 'Thigh_Apart', 'Thigh_Back', 'Thigh_Inner_2', 'Thigh_Outer', 'Thigh_Straight_Shape',
    'FK-ThighShape4', 'FK-ThighsThick', 'GHD_Thighs_Outer_Heavy_2', 'GHD_Thigh_Back_Def', 'GHD_Thighs_Muscle', 'Thigh_Muscular', 'Shin_Muscular', 'P3DTLThighGap', 'P3DTLThighsUpperSmall',
    'P3DTLThighsUpperWidth', 'P3DTLThighsRound', '1FK-LegsCurved', 'GHD_Shins_Muscle_Fit', 'GHD_Calves_Inner_Wide', 'GHD_Calves_Outer_Wide', 'FK-KneeSize', 'Knee_Inner', 'Knee_Outer',
    'FK-KneeBoneFrontDepth', 'BD_Shins_Calf_Strength', 'FK-CalfUniform', 'Calf_Type_3', '1FK-AnklesThin', 'SCLPropagatingHead', 'SCLPropagatingFoot'
]
DIGIBODY_FULL_COEF_COLUMNS = [
    'Shoulder_Width', 'Arm_Size', '4_Belly_Move_InOut', 'PBMGlutesSize', 'Hip_Size_2', 'PBMThighsSize', 'PBMShinsSize', 'PBMBreastsGone', 'PBMBreastsSize','CTRLBreastsNatural',
    'PBMBreastsShape04', 'CTRLFitness', 'CTRLBodybuilder', 'Glute_Depth', 'Torso_Depth', 'Arm_Flab', 'PBMRibcageSize', 'Waist_Depth', 'Waist_Straightness', 'Neck_Width',
    'FBMHeavy', 'FBMEmaciated', 'Wrist_Thickness', '5_Belly_Shape_Pregnant', '5_Belly_Shape_Fat_1', 'PBMStomachLowerDepth', 'P3DTLBackHollow', 'P3DTLBackStraight', 'PBMBreastsDiameter',
    'PBMBreastsHeavy', 'PBMBreastShape05', 'Torso_Width', 'Arm_Length', 'Shape_Width_Adjust', 'PBMSternumDepth', 'Abdomen_Length', 'SCLNeckLength', 'Thigh_Length', 'Shin_Length',
    'Pelvic_Length', 'Traps_shape', 'PBMGlutesUtilitiesScaleX', 'PBMGlutesUtilitiesScaleY', 'PBMGlutesUtilitiesScaleZ', 'Glute_UpDown', 'Hip_UpDown', 'Hip_Depth', 'GHD_Glutes_Square',
    'GHD_Hip_Square', 'Glute_Width_Lower', 'Glute_Width_Upper', 'GHD_Glutes_Bubble', 'GHD_Glutes_LargeUp', 'GHD_Glutes_Round_Large', 'GHD_Glutes_Saggy', 'P3DTLButtocksNarrow',
    'Glute_Angle', 'PBMGlutesUtilitiesShape04', 'PBMGlutesUtilitiesShape34', 'PBMGlutesUtilitiesShape32', 'Hip_Shape_4', 'Hip_Rotate_2', 'Glute_Line_sharpness', 'Glute_Height_Lower',
    'Glute_Height_Inner', 'Pelvic_Depth', 'PBMBreastsDownwardSlope', 'CTRLBreastsImplants', 'PBMBreastsPerkSide', 'GHD_Breasts_Shape_06', 'P3DTLBreastsSaggy', 'PBMBreastShape07',
    'PBMBreastShape21', 'PBMBreastShape25', 'PBMBreastShape04', 'PBMBreastShape10', 'PBMBreastShape26', 'PBMBreastsShape05', 'PBMBreastSeparation', 'PBMBreastsSmall',
    'PBMBreastUnderCurve', 'BD_Figure_Curvy_Breasts', 'BD_Figure_Pear_Breasts', 'PBMPregnant', 'GHD_Stomach_Heavy', 'GHD_Ribcage_Thin',
    'Back_Rotate', 'Back_Updown', 'Abdomen_Width', 'Abdomen_Size', 'RibCage_Defintion', 'PBMSternumHeight', '0_Belly_Fold_Horizontal', '0_Belly_Size', '1_Belly_Upper_Bulge',
    '2_Belly_Lower_Width', '2_Belly_Lower_UpDown', '4_Belly_Move_Rotate', '4_Belly_Move_UpDown', '5_Belly_Shape_Fat_2', '5_Belly_Shape_Fat_3', 'Torso_Muscular', 'Waist_Size',
    'Waist_Shape', 'Waist_Rotate', 'Waist_Height', 'Waist_FrontBack', 'GHD_Abdominal_Muscle', 'PBMLoveHandles', 'GHD_Stomach_In', 'GHD_Stomach_Puffy', 'GHD_Stomach_Sitting_Rolls',
    'P3DTLBellyBigger', 'GHD_Back_Heavy', 'PBMLatsSize', 'GHD_Scapula_Out', 'PBMShouldersSize', 'PBMForearmsSize', 'PBMUpperArmsSize', 'GHD_Arm_Muscles', 'SCLPropagatingHand',
    'SCLFingersLength', 'P3DTLArmsThin', 'Neck_Depth', 'FK-ThighsGapClosed1', 'Thigh_Front', 'Thigh_Apart', 'Thigh_Back', 'Thigh_Inner_2', 'Thigh_Outer', 'Thigh_Straight_Shape',
    'FK-ThighShape4', 'FK-ThighsThick', 'GHD_Thighs_Outer_Heavy_2', 'GHD_Thigh_Back_Def', 'GHD_Thighs_Muscle', 'Thigh_Muscular', 'Shin_Muscular', 'P3DTLThighGap', 'P3DTLThighsUpperSmall',
    'P3DTLThighsUpperWidth', 'P3DTLThighsRound', '1FK-LegsCurved', 'GHD_Shins_Muscle_Fit', 'GHD_Calves_Inner_Wide', 'GHD_Calves_Outer_Wide', 'FK-KneeSize', 'Knee_Inner', 'Knee_Outer',
    'FK-KneeBoneFrontDepth', 'BD_Shins_Calf_Strength', 'FK-CalfUniform', 'Calf_Type_3', '1FK-AnklesThin', 'SCLPropagatingHead', 'SCLPropagatingFoot'
]


def parse_digibody_pandas_row(pandas_row: Union[pd.Series, pd.DataFrame]):
    metadata = pandas_row['metadata']

    image_id = metadata.get('body_id', -1)  # TODO: Regenerate df to have image_id and body_id in the right columns
    body_id = metadata.get('image_id', -1)
    cloth_id = metadata.get('cloth_id', -1)
    hair_id = metadata.get('hair_id', -1)
    texture_map_id = metadata.get('texture_map_id', -1)
    set_name = metadata.get('set', '')
    image_name = metadata.get('image_name', '')
    camera_type = metadata.get('camera_type', '')

    measures = pandas_row['measures'].values
    bones_measures = pandas_row['bones_measures'].values

    # dome_rotation = pandas_row['dome_rotation'].values
    bbox = get_bbox_from_pandas(pandas_row['bbox'])
    coefs = pandas_row['parameters'].values

    full_landmarks = get_landmarks_from_pandas(pandas_row['landmarks'], to_lm=5873)

    return image_id, body_id, cloth_id, hair_id, texture_map_id, set_name, image_name, camera_type, bbox, measures, bones_measures, coefs, full_landmarks
