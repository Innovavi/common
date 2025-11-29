from typing import Union

import numpy as np
import pandas as pd

from common.data_manipulation.pandas_tools import get_bbox_from_pandas, get_landmarks_from_pandas

DIGIBODY_G9_COEF_COLUMNS = [
    'SS_body_bs_2BellyLowerUpDown', 'SS_body_bs_2BellyLowerWidth', 'SS_body_bs_4BellyMoveInOut', 'SS_body_bs_4BellyMoveRotate', 'SS_body_bs_4BellyMoveUpDown',
    'SS_body_bs_1BellyUpperWidth', 'SS_body_bs_PelvicDepth', 'SS_body_bs_WaistDepth', 'SS_body_bs_WaistFrontBack', 'SS_body_bs_AbdomenUpDown', 'SS_body_bs_WaistRotate',
    'TM_body_bs_WaistWidth', 'SS_body_bs_0BellyFoldHorizontal', 'SS_body_bs_0BellySize', 'SS_body_bs_1BellyUpperBulge', 'SFDBMK_body_bs_BellyLoFlat', 'SS_body_bs_5BellyShapeFat1',
    'SS_body_bs_5BellyShapeFat2', 'SFDBMK_body_bs_AbdomenSmall', 'body_bs_LoveHandles', 'body_bs_StomachDepthLower', 'SFDBMK_body_bs_BellySize', 'SFDBMK_body_bs_BellyLoSize',
    'SS_body_bs_5BellyShapePregnant', 'SFDBMK_body_bs_UpperArmsLength', 'SFDBMK_body_bs_ArmsLowerLength', 'body_bs_TaperForearmA', 'body_bs_TaperForearmB', 'body_bs_MassShoulders',
    'body_bs_TaperUpperArmA', 'body_bs_TaperUpperArmB', 'SFDBMK_body_bs_FingersLength', 'body_bs_FingersWidth', 'body_ctrl_ProportionHandSize', 'body_bs_ProportionShoulderWidth',
    'TM_body_bs_ShoulderWidth', 'SFDBMK_body_bs_ShouldersHeight', 'SS_body_bs_WristThickness', 'TM_body_bs_ShoulderHeight', 'SFDBMK_body_bs_ShouldersWeak', 'P3DTL_body_bs_ArmsMuscular',
    'SS_body_bs_ArmFlab', 'TM_body_bs_ArmpitHeight', 'SFDBMK_body_bs_ArmsElbowsShape', 'body_bs_SternumDepth', 'body_bs_SternumWidth', 'body_bs_SternumHeight',
    'body_bs_PectoralsCleavage', 'body_bs_PectoralsDiameter', 'body_bs_PectoralsHeight', 'body_bs_PectoralsUnderCurve', 'body_bs_PectoralsHeightOuter', 'body_bs_PectoralsWidth',
    'SFDBMK_body_bs_ChestMDeveloped', 'SFDBMK_body_bs_BreastCleavage', 'body_bs_BC_!BreastDiameter', 'P3DTL_body_bs_BreastsFlat', 'body_bs_BreastsLargeHigh',
    'body_bs_BreastsFullnessLower', 'body_bs_BreastsFullnessUpper', 'body_bs_BreastsPerkSide', 'body_bs_BreastsDownwardSlope', 'body_bs_BC_!BreastHeightLower',
    'body_bs_BC_!BreastHeightUpper', 'body_bs_BC_!BreastUnderWeight', 'body_bs_BC_!BreastUnderHeight', 'body_bs_BreastsSidesDepth', 'SFDBMK_body_bs_ChestSidesSmaller2',
    'SFDBMK_body_bs_TorsoSidesSmall', 'body_bs_PectoralsSag', 'SFDBMK_body_bs_BreastBigger', 'body_bs_BreastsNatural', 'DF-BreastUtilities_BaseFeminine_chest_bs_BreastsShape02',
    'TM_body_bs_ChestBreastDroop', 'TM_body_bs_ChestBreastImplant02', 'P3DTL_body_bs_BreastsSaggy', 'P3DTL_body_bs_BreastsXL', 'body_bs_BreastsHeavy', 'body_bs_BreastsShape06',
    'SFDBMK_body_bs_BreastFShape1', 'body_bs_BC_!BreastSag1', 'body_bs_BC_!BreastPointed', 'SS_body_bs_GluteAngle', 'body_bs_GluteDepthLower', 'body_bs_GluteDepthUpper',
    'SS_body_bs_GluteHeightInner', 'SS_body_bs_GluteHeightLower', 'SS_body_bs_GluteHeightUpper', 'TM_body_bs_GlutesDefined', 'SS_body_bs_GluteUpDown', 'SS_body_bs_HipDepth',
    'SS_body_bs_HipUpDown', 'SS_body_bs_HipWidth1', 'body_bs_GluteSize', 'SS_body_bs_GluteWidthLower', 'SS_body_bs_GluteWidthUpper', 'SS_body_bs_PelvicRotate',
    'SS_body_bs_PelvicHeight1', 'SS_body_bs_HipWidth3', 'body_bs_HipSize', 'SS_body_bs_HipDiameter', 'SS_body_bs_HipShape4', 'SS_body_bs_HipSize2', 'TM_body_bs_HipBoneStrength',
    'SFDBMK_body_bs_Glutes1', 'SFDBMK_body_bs_GlutesSidesShape', 'body_bs_GluteCrease', 'SS_body_bs_AnatomyBulge1', 'TM_body_bs_NeckThickness', 'TM_body_bs_NeckWidth',
    'head_ctrl_ProportionHeadSize_scl', '200_head_bs_CraniumWidth', 'head_bs_JawtoNeckSlack', 'SS_body_bs_AbdomenLength', 'head_bs_ProportionNeckLength', 'SS_body_bs_PelvicLength',
    'SS_body_bs_ShinLength', 'SS_body_bs_ThighLength', 'body_bs_ProportionFootLength', 'body_ctrl_ProportionFootSize', 'SS_body_bs_KneeInner', 'SS_body_bs_KneeOuter',
    'SFDBMK_body_bs_KneesShape', 'SFDBMK_body_bs_KneesHollowShape', 'body_bs_TaperShinA', 'body_bs_TaperShinB', 'SFDBMK_body_bs_ShinsShape', 'body_bs_TaperThighA',
    'body_bs_TaperThighB', 'body_bs_ThighDepth', 'SFDBMK_body_bs_LegsUpperBackShape', 'SFDBMK_body_bs_LegsUpperFrontShape', 'SS_body_bs_ThighInner2',
    'SFDBMK_body_bs_LegsThighOuterSmall', 'body_bs_CalvesSize', 'SS_body_bs_CalfFlexed', 'SS_body_bs_ThighThicknessWidth', 'body_bs_MassAnkles', 'TM_body_bs_AnkleSize',
    'SS_body_bs_FeetWidth', 'SS_body_bs_HeelBallSize', 'SS_body_bs_HeelDepth', 'body_bs_ProportionToesLength', 'SS_body_bs_CalfType3', 'SS_body_bs_ShinMuscular',
    'SS_body_bs_ThighApart', 'P3DTL_body_bs_ThighGap', 'SS_body_bs_ThighMuscular', 'SS_body_bs_ThighStraightShape', 'body_ctrl_BodyMuscular', 'body_ctrl_BodyFitness',
    'HeavyBodyOnly', 'LitheBodyOnly', 'SS_body_bs_ShapeWidthAdjust', 'EmaciatedBodyOnly', 'MassBodyBodyOnly', 'SS_body_bs_AbdomenWidth', 'SS_body_bs_BackRotate',
    'SS_body_bs_BackUpdown', 'SS_body_bs_BackDepth', 'body_bs_RibcageSize', 'SS_body_bs_RibCageDefine', 'SS_body_bs_TorsoDepth', 'SS_body_bs_TorsoWidth',
    'TM_body_bs_RibsWidth', 'body_bs_LatsSize', 'SFDBMK_body_bs_ShoulderBlades', 'SS_body_bs_TorsoMuscular', 'body_bs_TrapsSize', 'SFDBMK_body_bs_TrapeziusShape',
    'P3DTL_body_bs_NeckTrapeziusDown', 'body_bs_NailsLengthSquare', 'P3Design_body_bs_Nails_Oval', 'LKKatharina_body_bs_nails', '200_head_bs_ForeHeadWidth'
]

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


def parse_G9_digibody_pandas_row(pandas_row: Union[pd.Series, pd.DataFrame]):
    metadata = pandas_row['metadata']

    image_id = metadata.get('image_id', -1)
    body_id = metadata.get('body_id', -1)
    cloth_id = metadata.get('cloth_id', -1)
    clothTightness = metadata.get('clothTightness', -1)
    hair_id = metadata.get('hair_id', -1)
    skinMapID = metadata.get('skinMapID', -1)
    set_name = metadata.get('set', '')
    image_name = metadata.get('image_name', '')
    view_name = metadata.get('view_name', '')
    is_male = metadata.get('is_male', '')

    measures = pandas_row['measures'].values
    bone_measures = pandas_row['bone_measures'].values

    # dome_rotation = pandas_row['dome_rotation'].values
    bbox = get_bbox_from_pandas(pandas_row['bbox'])
    coefs = pandas_row['coefs'].values

    # full_landmarks = get_landmarks_from_pandas(pandas_row['landmarks'], to_lm=11917)
    full_landmarks = pandas_row['2D_landmarks'].values.reshape(-1, 2).astype(np.float32)

    return image_id, body_id, cloth_id, clothTightness, hair_id, skinMapID, set_name, image_name, view_name, bbox, measures, bone_measures, coefs, full_landmarks