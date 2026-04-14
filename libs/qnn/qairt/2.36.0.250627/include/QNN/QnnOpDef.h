//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

//=============================================================================
// !!! This is an auto-generated file. Do NOT modify manually !!!
//=============================================================================

/**
 * @file
 * @brief QNN operation definition related names and constants.
 *
 *        Supported QNN operations are named alphabetically and belong to the
 *        QNN_OP_PACKAGE_NAME_QTI_AISW.
 */

#ifndef QNN_OP_DEF_H
#define QNN_OP_DEF_H

// The Op package name
#define QNN_OP_PACKAGE_NAME_QTI_AISW "qti.aisw"

#define QNN_OPSET_VERSION_MAJOR 2
#define QNN_OPSET_VERSION_MINOR 4
#define QNN_OPSET_VERSION_PATCH 0

#define QNN_OP_ARGB_TO_RGB                      "ArgbToRgb"
#define QNN_OP_ARGB_TO_RGB_PARAM_INPUT_ORDER    "input_order"
#define QNN_OP_ARGB_TO_RGB_INPUT_ORDER_ARGB     0
#define QNN_OP_ARGB_TO_RGB_INPUT_ORDER_RGBA     1
#define QNN_OP_ARGB_TO_RGB_PARAM_REVERSE_OUTPUT "reverse_output"

#define QNN_OP_ARGMAX                 "Argmax"
#define QNN_OP_ARGMAX_PARAM_AXIS      "axis"
#define QNN_OP_ARGMAX_PARAM_KEEP_DIMS "keep_dims"

#define QNN_OP_ARGMIN                 "Argmin"
#define QNN_OP_ARGMIN_PARAM_AXIS      "axis"
#define QNN_OP_ARGMIN_PARAM_KEEP_DIMS "keep_dims"

#define QNN_OP_AXIS_ALIGNED_BBOX_TRANSFORM               "AxisAlignedBboxTransform"
#define QNN_OP_AXIS_ALIGNED_BBOX_TRANSFORM_PARAM_WEIGHTS "weights"

#define QNN_OP_BATCHNORM "Batchnorm"

#define QNN_OP_BATCH_PERMUTATION "BatchPermutation"

#define QNN_OP_BATCH_TO_SPACE                  "BatchToSpace"
#define QNN_OP_BATCH_TO_SPACE_PARAM_BLOCK_SIZE "block_size"
#define QNN_OP_BATCH_TO_SPACE_PARAM_CROPS      "crops"

#define QNN_OP_BBOX_TRANSFORM                            "BboxTransform"
#define QNN_OP_BBOX_TRANSFORM_PARAM_WEIGHTS              "weights"
#define QNN_OP_BBOX_TRANSFORM_PARAM_APPLY_SCALE          "apply_scale"
#define QNN_OP_BBOX_TRANSFORM_PARAM_ANGLE_BOUNDS         "angle_bounds"
#define QNN_OP_BBOX_TRANSFORM_PARAM_ANGLE_CLIP_THRESHOLD "angle_clip_threshold"

#define QNN_OP_BOX_WITH_NMS_LIMIT                            "BoxWithNmsLimit"
#define QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_NMS_KERNEL_METHOD    "nms_kernel_method"
#define QNN_OP_BOX_WITH_NMS_LIMIT_NMS_KERNEL_METHOD_HARD     0
#define QNN_OP_BOX_WITH_NMS_LIMIT_NMS_KERNEL_METHOD_LINEAR   1
#define QNN_OP_BOX_WITH_NMS_LIMIT_NMS_KERNEL_METHOD_GAUSSIAN 2
#define QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_NMS_SCORE_THRESHOLD  "nms_score_threshold"
#define QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_SCORE_THRESHOLD      "score_threshold"
#define QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_PRE_NMS_LIMIT        "pre_nms_limit"
#define QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_IOU_THRESHOLD        "iou_threshold"
#define QNN_OP_BOX_WITH_NMS_LIMIT_PARAM_SIGMA                "sigma"

#define QNN_OP_BUFFER                         "Buffer"
#define QNN_OP_BUFFER_PARAM_BUFFER_SIZE       "buffer_size"
#define QNN_OP_BUFFER_PARAM_BUFFER_DIM        "buffer_dim"
#define QNN_OP_BUFFER_PARAM_STRIDE            "stride"
#define QNN_OP_BUFFER_PARAM_MODE              "mode"
#define QNN_OP_BUFFER_MODE_BLOCKING           0
#define QNN_OP_BUFFER_MODE_NON_BLOCKING_LEFT  1
#define QNN_OP_BUFFER_MODE_NON_BLOCKING_RIGHT 2
#define QNN_OP_BUFFER_PARAM_BUFFER_PADDING    "buffer_padding"

#define QNN_OP_CAST "Cast"

#define QNN_OP_CHANNEL_SHUFFLE                  "ChannelShuffle"
#define QNN_OP_CHANNEL_SHUFFLE_PARAM_NUM_GROUPS "num_groups"
#define QNN_OP_CHANNEL_SHUFFLE_PARAM_AXIS       "axis"

#define QNN_OP_COL2_IM                   "Col2Im"
#define QNN_OP_COL2_IM_PARAM_KERNEL_SIZE "kernel_size"
#define QNN_OP_COL2_IM_PARAM_STRIDE      "stride"
#define QNN_OP_COL2_IM_PARAM_PAD_AMOUNT  "pad_amount"
#define QNN_OP_COL2_IM_PARAM_DILATION    "dilation"

#define QNN_OP_COLLECT_RPN_PROPOSALS                     "CollectRpnProposals"
#define QNN_OP_COLLECT_RPN_PROPOSALS_PARAM_RPN_MIN_LEVEL "rpn_min_level"
#define QNN_OP_COLLECT_RPN_PROPOSALS_PARAM_RPN_MAX_LEVEL "rpn_max_level"
#define QNN_OP_COLLECT_RPN_PROPOSALS_PARAM_POST_NMS_TOP  "post_nms_top"

#define QNN_OP_COMBINED_NMS                           "CombinedNms"
#define QNN_OP_COMBINED_NMS_PARAM_MAX_BOXES_PER_CLASS "max_boxes_per_class"
#define QNN_OP_COMBINED_NMS_PARAM_MAX_TOTAL_BOXES     "max_total_boxes"
#define QNN_OP_COMBINED_NMS_PARAM_IOU_THRESHOLD       "iou_threshold"
#define QNN_OP_COMBINED_NMS_PARAM_SCORE_THRESHOLD     "score_threshold"
#define QNN_OP_COMBINED_NMS_PARAM_PAD_PER_CLASS       "pad_per_class"
#define QNN_OP_COMBINED_NMS_PARAM_CLIP_BOXES          "clip_boxes"

#define QNN_OP_CONCAT            "Concat"
#define QNN_OP_CONCAT_PARAM_AXIS "axis"

#define QNN_OP_CONSTANT_OF_SHAPE             "ConstantOfShape"
#define QNN_OP_CONSTANT_OF_SHAPE_PARAM_VALUE "value"

#define QNN_OP_CONV_1D                  "Conv1d"
#define QNN_OP_CONV_1D_PARAM_STRIDE     "stride"
#define QNN_OP_CONV_1D_PARAM_PAD_AMOUNT "pad_amount"
#define QNN_OP_CONV_1D_PARAM_GROUP      "group"
#define QNN_OP_CONV_1D_PARAM_DILATION   "dilation"

#define QNN_OP_CONV_2D                            "Conv2d"
#define QNN_OP_CONV_2D_PARAM_STRIDE               "stride"
#define QNN_OP_CONV_2D_PARAM_PAD_AMOUNT           "pad_amount"
#define QNN_OP_CONV_2D_PARAM_GROUP                "group"
#define QNN_OP_CONV_2D_PARAM_DILATION             "dilation"
#define QNN_OP_CONV_2D_PARAM_REUSE_SPARSE_INDICES "reuse_sparse_indices"

#define QNN_OP_CONV_3D                             "Conv3d"
#define QNN_OP_CONV_3D_PARAM_STRIDE                "stride"
#define QNN_OP_CONV_3D_PARAM_PAD_AMOUNT            "pad_amount"
#define QNN_OP_CONV_3D_PARAM_GROUP                 "group"
#define QNN_OP_CONV_3D_PARAM_DILATION              "dilation"
#define QNN_OP_CONV_3D_PARAM_REUSE_SPARSE_INDICIES "reuse_sparse_indicies"

#define QNN_OP_CONVERT                           "Convert"
#define QNN_OP_CONVERT_PARAM_DYNAMIC_INPUT_DATA  "dynamic_input_data"
#define QNN_OP_CONVERT_PARAM_DYNAMIC_OUTPUT_DATA "dynamic_output_data"

#define QNN_OP_CORRELATION_1D                    "Correlation1D"
#define QNN_OP_CORRELATION_1D_PARAM_DISPLACEMENT "displacement"
#define QNN_OP_CORRELATION_1D_PARAM_SHIFT        "shift"

#define QNN_OP_CREATE_SPARSE "CreateSparse"

#define QNN_OP_CROP_AND_RESIZE                                     "CropAndResize"
#define QNN_OP_CROP_AND_RESIZE_PARAM_RESIZE_DIMS                   "resize_dims"
#define QNN_OP_CROP_AND_RESIZE_PARAM_INTERPOLATION_MODE            "interpolation_mode"
#define QNN_OP_CROP_AND_RESIZE_INTERPOLATION_MODE_BILINEAR         0
#define QNN_OP_CROP_AND_RESIZE_INTERPOLATION_MODE_NEAREST_NEIGHBOR 1
#define QNN_OP_CROP_AND_RESIZE_PARAM_EXTRAPOLATION_VALUE           "extrapolation_value"

#define QNN_OP_CUMULATIVE_SUM                 "CumulativeSum"
#define QNN_OP_CUMULATIVE_SUM_PARAM_AXIS      "axis"
#define QNN_OP_CUMULATIVE_SUM_PARAM_EXCLUSIVE "exclusive"
#define QNN_OP_CUMULATIVE_SUM_PARAM_REVERSE   "reverse"

#define QNN_OP_DEPTH_TO_SPACE                  "DepthToSpace"
#define QNN_OP_DEPTH_TO_SPACE_PARAM_BLOCK_SIZE "block_size"
#define QNN_OP_DEPTH_TO_SPACE_PARAM_MODE       "mode"
#define QNN_OP_DEPTH_TO_SPACE_MODE_DCR         0
#define QNN_OP_DEPTH_TO_SPACE_MODE_CRD         1

#define QNN_OP_DEPTH_WISE_CONV_1D                  "DepthWiseConv1d"
#define QNN_OP_DEPTH_WISE_CONV_1D_PARAM_STRIDE     "stride"
#define QNN_OP_DEPTH_WISE_CONV_1D_PARAM_PAD_AMOUNT "pad_amount"
#define QNN_OP_DEPTH_WISE_CONV_1D_PARAM_DILATION   "dilation"

#define QNN_OP_DEPTH_WISE_CONV_2D                  "DepthWiseConv2d"
#define QNN_OP_DEPTH_WISE_CONV_2D_PARAM_STRIDE     "stride"
#define QNN_OP_DEPTH_WISE_CONV_2D_PARAM_PAD_AMOUNT "pad_amount"
#define QNN_OP_DEPTH_WISE_CONV_2D_PARAM_DILATION   "dilation"

#define QNN_OP_DEQUANTIZE "Dequantize"

#define QNN_OP_DETECTION_OUTPUT                             "DetectionOutput"
#define QNN_OP_DETECTION_OUTPUT_PARAM_DELTA_SCALING_FACTORS "delta_scaling_factors"
#define QNN_OP_DETECTION_OUTPUT_PARAM_CONFIDENCE_THRESHOLD  "confidence_threshold"
#define QNN_OP_DETECTION_OUTPUT_PARAM_IOU_THRESHOLD         "iou_threshold"
#define QNN_OP_DETECTION_OUTPUT_PARAM_NMS_TYPE              "nms_type"
#define QNN_OP_DETECTION_OUTPUT_NMS_TYPE_FAST               0
#define QNN_OP_DETECTION_OUTPUT_NMS_TYPE_REGULAR            1
#define QNN_OP_DETECTION_OUTPUT_PARAM_BACKGROUND_CLASS_IDX  "background_class_idx"
#define QNN_OP_DETECTION_OUTPUT_PARAM_USE_BG_IN_NMS         "use_bg_in_nms"
#define QNN_OP_DETECTION_OUTPUT_PARAM_OUTPUT_BACKGROUND     "output_background"
#define QNN_OP_DETECTION_OUTPUT_PARAM_SHARE_LOCATION        "share_location"
#define QNN_OP_DETECTION_OUTPUT_PARAM_NMS_ETA               "nms_eta"
#define QNN_OP_DETECTION_OUTPUT_PARAM_DETECTION_LIMIT       "detection_limit"

#define QNN_OP_DISTRIBUTE_FPN_PROPOSALS                           "DistributeFpnProposals"
#define QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_MIN_LEVEL       "roi_min_level"
#define QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_MAX_LEVEL       "roi_max_level"
#define QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_CANONICAL_SCALE "roi_canonical_scale"
#define QNN_OP_DISTRIBUTE_FPN_PROPOSALS_PARAM_ROI_CANONICAL_LEVEL "roi_canonical_level"

#define QNN_OP_ELEMENT_WISE_ABS "ElementWiseAbs"

#define QNN_OP_ELEMENT_WISE_ADD "ElementWiseAdd"

#define QNN_OP_ELEMENT_WISE_AND "ElementWiseAnd"

#define QNN_OP_ELEMENT_WISE_ASIN "ElementWiseAsin"

#define QNN_OP_ELEMENT_WISE_ATAN "ElementWiseAtan"

#define QNN_OP_ELEMENT_WISE_BINARY                              "ElementWiseBinary"
#define QNN_OP_ELEMENT_WISE_BINARY_PARAM_OPERATION              "operation"
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_ADD                0
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_AND                1
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_DIVIDE             2
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_EQUAL              3
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FLOOR_DIV          4
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_FMOD               5
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER            6
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_GREATER_EQUAL      7
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS               8
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_LESS_EQUAL         9
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MAXIMUM            10
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MINIMUM            11
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MOD                12
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_MULTIPLY           13
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_NOT_EQUAL          14
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_OR                 15
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_POWER              16
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SQUARED_DIFFERENCE 17
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_SUBTRACT           18
#define QNN_OP_ELEMENT_WISE_BINARY_OPERATION_XOR                19

#define QNN_OP_ELEMENT_WISE_CEIL "ElementWiseCeil"

#define QNN_OP_ELEMENT_WISE_COS "ElementWiseCos"

#define QNN_OP_ELEMENT_WISE_DIVIDE "ElementWiseDivide"

#define QNN_OP_ELEMENT_WISE_EQUAL "ElementWiseEqual"

#define QNN_OP_ELEMENT_WISE_EXP "ElementWiseExp"

#define QNN_OP_ELEMENT_WISE_FLOOR "ElementWiseFloor"

#define QNN_OP_ELEMENT_WISE_FLOOR_DIV "ElementWiseFloorDiv"

#define QNN_OP_ELEMENT_WISE_FMOD "ElementWiseFmod"

#define QNN_OP_ELEMENT_WISE_GREATER "ElementWiseGreater"

#define QNN_OP_ELEMENT_WISE_GREATER_EQUAL "ElementWiseGreaterEqual"

#define QNN_OP_ELEMENT_WISE_LESS "ElementWiseLess"

#define QNN_OP_ELEMENT_WISE_LESS_EQUAL "ElementWiseLessEqual"

#define QNN_OP_ELEMENT_WISE_LOG "ElementWiseLog"

#define QNN_OP_ELEMENT_WISE_MAXIMUM "ElementWiseMaximum"

#define QNN_OP_ELEMENT_WISE_MINIMUM "ElementWiseMinimum"

#define QNN_OP_ELEMENT_WISE_MOD "ElementWiseMod"

#define QNN_OP_ELEMENT_WISE_MULTIPLY "ElementWiseMultiply"

#define QNN_OP_ELEMENT_WISE_NEG "ElementWiseNeg"

#define QNN_OP_ELEMENT_WISE_NEURON                        "ElementWiseNeuron"
#define QNN_OP_ELEMENT_WISE_NEURON_PARAM_OPERATION        "operation"
#define QNN_OP_ELEMENT_WISE_NEURON_OPERATION_ELU          0
#define QNN_OP_ELEMENT_WISE_NEURON_OPERATION_GELU         1
#define QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SIGMOID 2
#define QNN_OP_ELEMENT_WISE_NEURON_OPERATION_HARD_SWISH   3
#define QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU         4
#define QNN_OP_ELEMENT_WISE_NEURON_OPERATION_RELU_MIN_MAX 5
#define QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SIGMOID      6
#define QNN_OP_ELEMENT_WISE_NEURON_OPERATION_SOFTPLUS     7
#define QNN_OP_ELEMENT_WISE_NEURON_OPERATION_TANH         8
#define QNN_OP_ELEMENT_WISE_NEURON_PARAM_ALPHA            "alpha"
#define QNN_OP_ELEMENT_WISE_NEURON_PARAM_BETA             "beta"
#define QNN_OP_ELEMENT_WISE_NEURON_PARAM_MIN_VALUE        "min_value"
#define QNN_OP_ELEMENT_WISE_NEURON_PARAM_MAX_VALUE        "max_value"
#define QNN_OP_ELEMENT_WISE_NEURON_PARAM_THRESHOLD        "threshold"

#define QNN_OP_ELEMENT_WISE_NOT "ElementWiseNot"

#define QNN_OP_ELEMENT_WISE_NOT_EQUAL "ElementWiseNotEqual"

#define QNN_OP_ELEMENT_WISE_OR "ElementWiseOr"

#define QNN_OP_ELEMENT_WISE_POWER "ElementWisePower"

#define QNN_OP_ELEMENT_WISE_ROUND "ElementWiseRound"

#define QNN_OP_ELEMENT_WISE_RSQRT "ElementWiseRsqrt"

#define QNN_OP_ELEMENT_WISE_SELECT "ElementWiseSelect"

#define QNN_OP_ELEMENT_WISE_SIN "ElementWiseSin"

#define QNN_OP_ELEMENT_WISE_SIGN "ElementWiseSign"

#define QNN_OP_ELEMENT_WISE_SOFTPLUS                 "ElementWiseSoftplus"
#define QNN_OP_ELEMENT_WISE_SOFTPLUS_PARAM_BETA      "beta"
#define QNN_OP_ELEMENT_WISE_SOFTPLUS_PARAM_THRESHOLD "threshold"

#define QNN_OP_ELEMENT_WISE_SQUARED_DIFFERENCE "ElementWiseSquaredDifference"

#define QNN_OP_ELEMENT_WISE_SQUARE_ROOT "ElementWiseSquareRoot"

#define QNN_OP_ELEMENT_WISE_SUBTRACT "ElementWiseSubtract"

#define QNN_OP_ELEMENT_WISE_UNARY                      "ElementWiseUnary"
#define QNN_OP_ELEMENT_WISE_UNARY_PARAM_OPERATION      "operation"
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ABS        0
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ASIN       1
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ATAN       2
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_CEIL       3
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_COS        4
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_EXP        5
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_FLOOR      6
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_LOG        7
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NEG        8
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_NOT        9
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_RECIPROCAL 10
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_ROUND      11
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_RSQRT      12
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIGN       13
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SIN        14
#define QNN_OP_ELEMENT_WISE_UNARY_OPERATION_SQRT       15

#define QNN_OP_ELEMENT_WISE_XOR "ElementWiseXor"

#define QNN_OP_ELU             "Elu"
#define QNN_OP_ELU_PARAM_ALPHA "alpha"

#define QNN_OP_EXPAND_DIMS            "ExpandDims"
#define QNN_OP_EXPAND_DIMS_PARAM_AXIS "axis"
#define QNN_OP_EXPAND_DIMS_PARAM_AXES "axes"

#define QNN_OP_EXTRACT_GLIMPSE                  "ExtractGlimpse"
#define QNN_OP_EXTRACT_GLIMPSE_PARAM_SIZE       "size"
#define QNN_OP_EXTRACT_GLIMPSE_PARAM_CENTERED   "centered"
#define QNN_OP_EXTRACT_GLIMPSE_PARAM_NORMALIZED "normalized"
#define QNN_OP_EXTRACT_GLIMPSE_PARAM_NOISE      "noise"
#define QNN_OP_EXTRACT_GLIMPSE_NOISE_UNIFORM    0
#define QNN_OP_EXTRACT_GLIMPSE_NOISE_GAUSSIAN   1
#define QNN_OP_EXTRACT_GLIMPSE_NOISE_ZEROES     2

#define QNN_OP_EXTRACT_PATCHES               "ExtractPatches"
#define QNN_OP_EXTRACT_PATCHES_PARAM_SIZE    "size"
#define QNN_OP_EXTRACT_PATCHES_PARAM_STRIDE  "stride"
#define QNN_OP_EXTRACT_PATCHES_PARAM_RATE    "rate"
#define QNN_OP_EXTRACT_PATCHES_PARAM_PADDING "padding"
#define QNN_OP_EXTRACT_PATCHES_PADDING_VALID 0
#define QNN_OP_EXTRACT_PATCHES_PADDING_SAME  1

#define QNN_OP_FULLY_CONNECTED                 "FullyConnected"
#define QNN_OP_FULLY_CONNECTED_PARAM_KEEP_DIMS "keep_dims"

#define QNN_OP_GATHER            "Gather"
#define QNN_OP_GATHER_PARAM_AXIS "axis"

#define QNN_OP_GATHER_ELEMENTS            "GatherElements"
#define QNN_OP_GATHER_ELEMENTS_PARAM_AXIS "axis"

#define QNN_OP_GATHER_ND                  "GatherNd"
#define QNN_OP_GATHER_ND_PARAM_BATCH_DIMS "batch_dims"

#define QNN_OP_GELU "Gelu"

#define QNN_OP_GENERATE_PROPOSALS                       "GenerateProposals"
#define QNN_OP_GENERATE_PROPOSALS_PARAM_IMG_SIZE_RATIO  "img_size_ratio"
#define QNN_OP_GENERATE_PROPOSALS_PARAM_MIN_SIZE        "min_size"
#define QNN_OP_GENERATE_PROPOSALS_PARAM_PRE_NMS_LIMIT   "pre_nms_limit"
#define QNN_OP_GENERATE_PROPOSALS_PARAM_POST_NMS_LIMIT  "post_nms_limit"
#define QNN_OP_GENERATE_PROPOSALS_PARAM_IOU_THRESHOLD   "iou_threshold"
#define QNN_OP_GENERATE_PROPOSALS_PARAM_BBOX_XFORM_CLIP "bbox_xform_clip"

#define QNN_OP_GET_SPARSE_INDICES "GetSparseIndices"

#define QNN_OP_GET_SPARSE_VALUES "GetSparseValues"

#define QNN_OP_GRID_SAMPLE                         "GridSample"
#define QNN_OP_GRID_SAMPLE_PARAM_ALIGN_CORNERS     "align_corners"
#define QNN_OP_GRID_SAMPLE_PARAM_MODE              "mode"
#define QNN_OP_GRID_SAMPLE_MODE_BILINEAR           0
#define QNN_OP_GRID_SAMPLE_MODE_NEAREST            1
#define QNN_OP_GRID_SAMPLE_PARAM_PADDING_MODE      "padding_mode"
#define QNN_OP_GRID_SAMPLE_PADDING_MODE_ZEROS      0
#define QNN_OP_GRID_SAMPLE_PADDING_MODE_BORDER     1
#define QNN_OP_GRID_SAMPLE_PADDING_MODE_REFLECTION 2

#define QNN_OP_GROUP_NORM               "GroupNorm"
#define QNN_OP_GROUP_NORM_PARAM_EPSILON "epsilon"
#define QNN_OP_GROUP_NORM_PARAM_GROUP   "group"

#define QNN_OP_GRU                           "Gru"
#define QNN_OP_GRU_PARAM_DIRECTION           "direction"
#define QNN_OP_GRU_DIRECTION_FORWARD         0
#define QNN_OP_GRU_DIRECTION_REVERSE         1
#define QNN_OP_GRU_PARAM_LINEAR_BEFORE_RESET "linear_before_reset"
#define QNN_OP_GRU_PARAM_TIME_MAJOR          "time_major"

#define QNN_OP_HARD_SWISH "HardSwish"

#define QNN_OP_HEAT_MAP_MAX_KEY_POINT "HeatMapMaxKeyPoint"

#define QNN_OP_IM2_COL                   "Im2Col"
#define QNN_OP_IM2_COL_PARAM_KERNEL_SIZE "kernel_size"
#define QNN_OP_IM2_COL_PARAM_STRIDE      "stride"
#define QNN_OP_IM2_COL_PARAM_PAD_AMOUNT  "pad_amount"
#define QNN_OP_IM2_COL_PARAM_DILATION    "dilation"

#define QNN_OP_IF                  "If"
#define QNN_OP_IF_PARAM_THEN_GRAPH "then_graph"
#define QNN_OP_IF_PARAM_ELSE_GRAPH "else_graph"

#define QNN_OP_IMAGE_PROJECTION_TRANSFORM                                     "ImageProjectionTransform"
#define QNN_OP_IMAGE_PROJECTION_TRANSFORM_PARAM_INTERPOLATION_MODE            "interpolation_mode"
#define QNN_OP_IMAGE_PROJECTION_TRANSFORM_INTERPOLATION_MODE_BILINEAR         0
#define QNN_OP_IMAGE_PROJECTION_TRANSFORM_INTERPOLATION_MODE_NEAREST_NEIGHBOR 1

#define QNN_OP_INSTANCE_NORM                          "InstanceNorm"
#define QNN_OP_INSTANCE_NORM_PARAM_EPSILON            "epsilon"
#define QNN_OP_INSTANCE_NORM_PARAM_MODE               "mode"
#define QNN_OP_INSTANCE_NORM_MODE_MU_SIGMA            0
#define QNN_OP_INSTANCE_NORM_MODE_RMS                 1
#define QNN_OP_INSTANCE_NORM_PARAM_NORMALIZE_VARIANCE "normalize_variance"
#define QNN_OP_INSTANCE_NORM_PARAM_REGION             "region"
#define QNN_OP_INSTANCE_NORM_REGION_ACROSS_SPATIAL    0
#define QNN_OP_INSTANCE_NORM_REGION_ACROSS_CHANNEL    1
#define QNN_OP_INSTANCE_NORM_REGION_ACROSS_ALL        2

#define QNN_OP_IS_INF                       "IsInf"
#define QNN_OP_IS_INF_PARAM_DETECT_NEGATIVE "detect_negative"
#define QNN_OP_IS_INF_PARAM_DETECT_POSITIVE "detect_positive"

#define QNN_OP_L2_NORM               "L2Norm"
#define QNN_OP_L2_NORM_PARAM_AXIS    "axis"
#define QNN_OP_L2_NORM_PARAM_AXES    "axes"
#define QNN_OP_L2_NORM_PARAM_EPSILON "epsilon"

#define QNN_OP_L2_POOL_2D                   "L2Pool2d"
#define QNN_OP_L2_POOL_2D_PARAM_FILTER_SIZE "filter_size"
#define QNN_OP_L2_POOL_2D_PARAM_STRIDE      "stride"
#define QNN_OP_L2_POOL_2D_PARAM_PAD_AMOUNT  "pad_amount"

#define QNN_OP_LAYER_NORM               "LayerNorm"
#define QNN_OP_LAYER_NORM_PARAM_EPSILON "epsilon"
#define QNN_OP_LAYER_NORM_PARAM_AXES    "axes"

#define QNN_OP_LOG_SOFTMAX            "LogSoftmax"
#define QNN_OP_LOG_SOFTMAX_PARAM_AXIS "axis"
#define QNN_OP_LOG_SOFTMAX_PARAM_BETA "beta"

#define QNN_OP_LRN                       "Lrn"
#define QNN_OP_LRN_PARAM_ALPHA           "alpha"
#define QNN_OP_LRN_PARAM_BETA            "beta"
#define QNN_OP_LRN_PARAM_BIAS            "bias"
#define QNN_OP_LRN_PARAM_RADIUS          "radius"
#define QNN_OP_LRN_PARAM_REGION          "region"
#define QNN_OP_LRN_REGION_ACROSS_CHANNEL 0
#define QNN_OP_LRN_REGION_WITHIN_CHANNEL 1

#define QNN_OP_LSTM                             "Lstm"
#define QNN_OP_LSTM_PARAM_DIRECTION             "direction"
#define QNN_OP_LSTM_DIRECTION_FORWARD           0
#define QNN_OP_LSTM_DIRECTION_REVERSE           1
#define QNN_OP_LSTM_PARAM_CELL_CLIP_THRESHOLD   "cell_clip_threshold"
#define QNN_OP_LSTM_PARAM_OUTPUT_CLIP_THRESHOLD "output_clip_threshold"
#define QNN_OP_LSTM_PARAM_TIME_MAJOR            "time_major"
#define QNN_OP_LSTM_PARAM_INPUT_GATE_QSCALE     "input_gate_qscale"
#define QNN_OP_LSTM_PARAM_FORGET_GATE_QSCALE    "forget_gate_qscale"
#define QNN_OP_LSTM_PARAM_CELL_GATE_QSCALE      "cell_gate_qscale"
#define QNN_OP_LSTM_PARAM_OUTPUT_GATE_QSCALE    "output_gate_qscale"
#define QNN_OP_LSTM_PARAM_HIDDEN_STATE_OFFSET   "hidden_state_offset"
#define QNN_OP_LSTM_PARAM_HIDDEN_STATE_QSCALE   "hidden_state_qscale"

#define QNN_OP_MASKED_SOFTMAX                   "MaskedSoftmax"
#define QNN_OP_MASKED_SOFTMAX_PARAM_MODE        "mode"
#define QNN_OP_MASKED_SOFTMAX_MODE_UNCOMPRESSED 0
#define QNN_OP_MASKED_SOFTMAX_MODE_COMPRESSED   1

#define QNN_OP_MOMENTS                 "Moments"
#define QNN_OP_MOMENTS_PARAM_AXES      "axes"
#define QNN_OP_MOMENTS_PARAM_KEEP_DIMS "keep_dims"

#define QNN_OP_MULTI_CLASS_NMS                       "MultiClassNms"
#define QNN_OP_MULTI_CLASS_NMS_PARAM_IOU_THRESHOLD   "iou_threshold"
#define QNN_OP_MULTI_CLASS_NMS_PARAM_SCORE_THRESHOLD "score_threshold"
#define QNN_OP_MULTI_CLASS_NMS_PARAM_SOFT_NMS_SIGMA  "soft_nms_sigma"

#define QNN_OP_NON_MAX_SUPPRESSION                          "NonMaxSuppression"
#define QNN_OP_NON_MAX_SUPPRESSION_PARAM_IOU_THRESHOLD      "iou_threshold"
#define QNN_OP_NON_MAX_SUPPRESSION_PARAM_SCORE_THRESHOLD    "score_threshold"
#define QNN_OP_NON_MAX_SUPPRESSION_PARAM_MAX_BOXES_SELECTED "max_boxes_selected"

#define QNN_OP_NON_ZERO "NonZero"

#define QNN_OP_NV12_TO_RGB                    "Nv12ToRgb"
#define QNN_OP_NV12_TO_RGB_PARAM_OUTPUT_ORDER "output_order"
#define QNN_OP_NV12_TO_RGB_OUTPUT_ORDER_RGB   0
#define QNN_OP_NV12_TO_RGB_OUTPUT_ORDER_BGR   1

#define QNN_OP_NV21_TO_RGB                    "Nv21ToRgb"
#define QNN_OP_NV21_TO_RGB_PARAM_OUTPUT_ORDER "output_order"
#define QNN_OP_NV21_TO_RGB_OUTPUT_ORDER_RGB   0
#define QNN_OP_NV21_TO_RGB_OUTPUT_ORDER_BGR   1

#define QNN_OP_ONE_HOT                 "OneHot"
#define QNN_OP_ONE_HOT_PARAM_DEPTH     "depth"
#define QNN_OP_ONE_HOT_PARAM_AXIS      "axis"
#define QNN_OP_ONE_HOT_PARAM_ON_VALUE  "on_value"
#define QNN_OP_ONE_HOT_PARAM_OFF_VALUE "off_value"

#define QNN_OP_PACK            "Pack"
#define QNN_OP_PACK_PARAM_AXIS "axis"

#define QNN_OP_MAT_MUL                     "MatMul"
#define QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN0 "transpose_in0"
#define QNN_OP_MAT_MUL_PARAM_TRANSPOSE_IN1 "transpose_in1"

#define QNN_OP_PAD                          "Pad"
#define QNN_OP_PAD_PARAM_SCHEME             "scheme"
#define QNN_OP_PAD_SCHEME_CONSTANT          0
#define QNN_OP_PAD_SCHEME_MIRROR_SYMMETRIC  1
#define QNN_OP_PAD_SCHEME_MIRROR_REFLECT    2
#define QNN_OP_PAD_SCHEME_EDGE              3
#define QNN_OP_PAD_PARAM_PAD_AMOUNT         "pad_amount"
#define QNN_OP_PAD_PARAM_PAD_CONSTANT_VALUE "pad_constant_value"

#define QNN_OP_POOL_AVG_2D                           "PoolAvg2d"
#define QNN_OP_POOL_AVG_2D_PARAM_FILTER_SIZE         "filter_size"
#define QNN_OP_POOL_AVG_2D_PARAM_STRIDE              "stride"
#define QNN_OP_POOL_AVG_2D_PARAM_PAD_AMOUNT          "pad_amount"
#define QNN_OP_POOL_AVG_2D_PARAM_COUNT_PAD_FOR_EDGES "count_pad_for_edges"
#define QNN_OP_POOL_AVG_2D_PARAM_ROUNDING_MODE       "rounding_mode"
#define QNN_OP_POOL_AVG_2D_ROUNDING_MODE_FLOOR       0
#define QNN_OP_POOL_AVG_2D_ROUNDING_MODE_CEIL        1

#define QNN_OP_POOL_AVG_3D                           "PoolAvg3d"
#define QNN_OP_POOL_AVG_3D_PARAM_FILTER_SIZE         "filter_size"
#define QNN_OP_POOL_AVG_3D_PARAM_STRIDE              "stride"
#define QNN_OP_POOL_AVG_3D_PARAM_PAD_AMOUNT          "pad_amount"
#define QNN_OP_POOL_AVG_3D_PARAM_COUNT_PAD_FOR_EDGES "count_pad_for_edges"
#define QNN_OP_POOL_AVG_3D_PARAM_ROUNDING_MODE       "rounding_mode"
#define QNN_OP_POOL_AVG_3D_ROUNDING_MODE_FLOOR       0
#define QNN_OP_POOL_AVG_3D_ROUNDING_MODE_CEIL        1

#define QNN_OP_POOL_MAX_2D                     "PoolMax2d"
#define QNN_OP_POOL_MAX_2D_PARAM_FILTER_SIZE   "filter_size"
#define QNN_OP_POOL_MAX_2D_PARAM_STRIDE        "stride"
#define QNN_OP_POOL_MAX_2D_PARAM_PAD_AMOUNT    "pad_amount"
#define QNN_OP_POOL_MAX_2D_PARAM_ROUNDING_MODE "rounding_mode"
#define QNN_OP_POOL_MAX_2D_ROUNDING_MODE_FLOOR 0
#define QNN_OP_POOL_MAX_2D_ROUNDING_MODE_CEIL  1

#define QNN_OP_POOL_MAX_3D                     "PoolMax3d"
#define QNN_OP_POOL_MAX_3D_PARAM_FILTER_SIZE   "filter_size"
#define QNN_OP_POOL_MAX_3D_PARAM_STRIDE        "stride"
#define QNN_OP_POOL_MAX_3D_PARAM_PAD_AMOUNT    "pad_amount"
#define QNN_OP_POOL_MAX_3D_PARAM_ROUNDING_MODE "rounding_mode"
#define QNN_OP_POOL_MAX_3D_ROUNDING_MODE_FLOOR 0
#define QNN_OP_POOL_MAX_3D_ROUNDING_MODE_CEIL  1

#define QNN_OP_PRELU "Prelu"

#define QNN_OP_QUANTIZE "Quantize"

#define QNN_OP_REDUCE_MAX                 "ReduceMax"
#define QNN_OP_REDUCE_MAX_PARAM_AXES      "axes"
#define QNN_OP_REDUCE_MAX_PARAM_KEEP_DIMS "keep_dims"

#define QNN_OP_REDUCE_MEAN                 "ReduceMean"
#define QNN_OP_REDUCE_MEAN_PARAM_AXES      "axes"
#define QNN_OP_REDUCE_MEAN_PARAM_KEEP_DIMS "keep_dims"

#define QNN_OP_REDUCE_MIN                 "ReduceMin"
#define QNN_OP_REDUCE_MIN_PARAM_AXES      "axes"
#define QNN_OP_REDUCE_MIN_PARAM_KEEP_DIMS "keep_dims"

#define QNN_OP_REDUCE_PROD                 "ReduceProd"
#define QNN_OP_REDUCE_PROD_PARAM_AXES      "axes"
#define QNN_OP_REDUCE_PROD_PARAM_KEEP_DIMS "keep_dims"

#define QNN_OP_REDUCE_SUM                 "ReduceSum"
#define QNN_OP_REDUCE_SUM_PARAM_AXES      "axes"
#define QNN_OP_REDUCE_SUM_PARAM_KEEP_DIMS "keep_dims"

#define QNN_OP_REDUCE_SUM_SQUARE                 "ReduceSumSquare"
#define QNN_OP_REDUCE_SUM_SQUARE_PARAM_AXES      "axes"
#define QNN_OP_REDUCE_SUM_SQUARE_PARAM_KEEP_DIMS "keep_dims"

#define QNN_OP_RELU "Relu"

#define QNN_OP_RELU1 "Relu1"

#define QNN_OP_RELU6 "Relu6"

#define QNN_OP_RELU_MIN_MAX                 "ReluMinMax"
#define QNN_OP_RELU_MIN_MAX_PARAM_MIN_VALUE "min_value"
#define QNN_OP_RELU_MIN_MAX_PARAM_MAX_VALUE "max_value"

#define QNN_OP_RESHAPE "Reshape"

#define QNN_OP_RESIZE                                        "Resize"
#define QNN_OP_RESIZE_PARAM_EXCLUDE_OUTSIDE                  "exclude_outside"
#define QNN_OP_RESIZE_PARAM_TRANSFORMATION_MODE              "transformation_mode"
#define QNN_OP_RESIZE_TRANSFORMATION_MODE_HALF_PIXEL         0
#define QNN_OP_RESIZE_TRANSFORMATION_MODE_PYTORCH_HALF_PIXEL 1
#define QNN_OP_RESIZE_TRANSFORMATION_MODE_ALIGN_CORNERS      2
#define QNN_OP_RESIZE_TRANSFORMATION_MODE_ASYMMETRIC         3
#define QNN_OP_RESIZE_PARAM_INTERPOLATION_MODE               "interpolation_mode"
#define QNN_OP_RESIZE_INTERPOLATION_MODE_NEAREST             0
#define QNN_OP_RESIZE_INTERPOLATION_MODE_LINEAR              1
#define QNN_OP_RESIZE_INTERPOLATION_MODE_CUBIC               2
#define QNN_OP_RESIZE_PARAM_NEAREST_MODE                     "nearest_mode"
#define QNN_OP_RESIZE_NEAREST_MODE_ROUND_PREFER_FLOOR        0
#define QNN_OP_RESIZE_NEAREST_MODE_ROUND_PREFER_CEIL         1
#define QNN_OP_RESIZE_NEAREST_MODE_FLOOR                     2
#define QNN_OP_RESIZE_NEAREST_MODE_CEIL                      3
#define QNN_OP_RESIZE_PARAM_CUBIC_COEFF                      "cubic_coeff"

#define QNN_OP_RESIZE_BILINEAR                          "ResizeBilinear"
#define QNN_OP_RESIZE_BILINEAR_PARAM_ALIGN_CORNERS      "align_corners"
#define QNN_OP_RESIZE_BILINEAR_PARAM_HALF_PIXEL_CENTERS "half_pixel_centers"
#define QNN_OP_RESIZE_BILINEAR_PARAM_ANTIALIAS          "antialias"

#define QNN_OP_RESIZE_NEAREST_NEIGHBOR                          "ResizeNearestNeighbor"
#define QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_ALIGN_CORNERS      "align_corners"
#define QNN_OP_RESIZE_NEAREST_NEIGHBOR_PARAM_HALF_PIXEL_CENTERS "half_pixel_centers"

#define QNN_OP_RMS_NORM               "RmsNorm"
#define QNN_OP_RMS_NORM_PARAM_EPSILON "epsilon"
#define QNN_OP_RMS_NORM_PARAM_AXES    "axes"

#define QNN_OP_ROI_ALIGN                         "RoiAlign"
#define QNN_OP_ROI_ALIGN_PARAM_IMG_SIZE_RATIO    "img_size_ratio"
#define QNN_OP_ROI_ALIGN_PARAM_NUM_SAMPLES_Y     "num_samples_y"
#define QNN_OP_ROI_ALIGN_PARAM_NUM_SAMPLES_X     "num_samples_x"
#define QNN_OP_ROI_ALIGN_PARAM_ALIGNED           "aligned"
#define QNN_OP_ROI_ALIGN_PARAM_ALLOW_INVALID_ROI "allow_invalid_roi"

#define QNN_OP_ROI_POOLING                      "RoiPooling"
#define QNN_OP_ROI_POOLING_PARAM_IMG_SIZE_RATIO "img_size_ratio"

#define QNN_OP_SCATTER_ELEMENTS                 "ScatterElements"
#define QNN_OP_SCATTER_ELEMENTS_PARAM_AXIS      "axis"
#define QNN_OP_SCATTER_ELEMENTS_PARAM_REDUCTION "reduction"
#define QNN_OP_SCATTER_ELEMENTS_REDUCTION_NONE  0
#define QNN_OP_SCATTER_ELEMENTS_REDUCTION_ADD   1
#define QNN_OP_SCATTER_ELEMENTS_REDUCTION_MUL   2
#define QNN_OP_SCATTER_ELEMENTS_REDUCTION_MAX   3

#define QNN_OP_SCATTER_ND                 "ScatterNd"
#define QNN_OP_SCATTER_ND_PARAM_REDUCTION "reduction"
#define QNN_OP_SCATTER_ND_REDUCTION_NONE  0
#define QNN_OP_SCATTER_ND_REDUCTION_ADD   1
#define QNN_OP_SCATTER_ND_REDUCTION_MUL   2

#define QNN_OP_SHAPE             "Shape"
#define QNN_OP_SHAPE_PARAM_START "start"
#define QNN_OP_SHAPE_PARAM_END   "end"

#define QNN_OP_SIGMOID "Sigmoid"

#define QNN_OP_SOFTMAX            "Softmax"
#define QNN_OP_SOFTMAX_PARAM_AXIS "axis"
#define QNN_OP_SOFTMAX_PARAM_BETA "beta"

#define QNN_OP_SPACE_TO_BATCH                  "SpaceToBatch"
#define QNN_OP_SPACE_TO_BATCH_PARAM_BLOCK_SIZE "block_size"
#define QNN_OP_SPACE_TO_BATCH_PARAM_PAD_AMOUNT "pad_amount"

#define QNN_OP_SPACE_TO_DEPTH                  "SpaceToDepth"
#define QNN_OP_SPACE_TO_DEPTH_PARAM_BLOCK_SIZE "block_size"
#define QNN_OP_SPACE_TO_DEPTH_PARAM_MODE       "mode"
#define QNN_OP_SPACE_TO_DEPTH_MODE_DCR         0
#define QNN_OP_SPACE_TO_DEPTH_MODE_CRD         1

#define QNN_OP_SPARSE_TO_DENSE "SparseToDense"

#define QNN_OP_SPLIT                   "Split"
#define QNN_OP_SPLIT_PARAM_AXIS        "axis"
#define QNN_OP_SPLIT_PARAM_SPLIT_INDEX "split_index"

#define QNN_OP_SQUEEZE            "Squeeze"
#define QNN_OP_SQUEEZE_PARAM_AXES "axes"

#define QNN_OP_STRIDED_SLICE                     "StridedSlice"
#define QNN_OP_STRIDED_SLICE_PARAM_RANGES        "ranges"
#define QNN_OP_STRIDED_SLICE_PARAM_BEGIN_MASK    "begin_mask"
#define QNN_OP_STRIDED_SLICE_PARAM_END_MASK      "end_mask"
#define QNN_OP_STRIDED_SLICE_PARAM_SHRINK_AXES   "shrink_axes"
#define QNN_OP_STRIDED_SLICE_PARAM_NEW_AXES_MASK "new_axes_mask"

#define QNN_OP_TANH "Tanh"

#define QNN_OP_TILE                 "Tile"
#define QNN_OP_TILE_PARAM_MULTIPLES "multiples"

#define QNN_OP_TOP_K               "TopK"
#define QNN_OP_TOP_K_PARAM_K       "k"
#define QNN_OP_TOP_K_PARAM_LARGEST "largest"

#define QNN_OP_TRANSPOSE            "Transpose"
#define QNN_OP_TRANSPOSE_PARAM_PERM "perm"

#define QNN_OP_TRANSPOSE_CONV_1D                      "TransposeConv1d"
#define QNN_OP_TRANSPOSE_CONV_1D_PARAM_STRIDE         "stride"
#define QNN_OP_TRANSPOSE_CONV_1D_PARAM_PAD_AMOUNT     "pad_amount"
#define QNN_OP_TRANSPOSE_CONV_1D_PARAM_GROUP          "group"
#define QNN_OP_TRANSPOSE_CONV_1D_PARAM_OUTPUT_PADDING "output_padding"

#define QNN_OP_TRANSPOSE_CONV_2D                      "TransposeConv2d"
#define QNN_OP_TRANSPOSE_CONV_2D_PARAM_STRIDE         "stride"
#define QNN_OP_TRANSPOSE_CONV_2D_PARAM_PAD_AMOUNT     "pad_amount"
#define QNN_OP_TRANSPOSE_CONV_2D_PARAM_GROUP          "group"
#define QNN_OP_TRANSPOSE_CONV_2D_PARAM_OUTPUT_PADDING "output_padding"

#define QNN_OP_TRANSPOSE_CONV_3D                      "TransposeConv3d"
#define QNN_OP_TRANSPOSE_CONV_3D_PARAM_STRIDE         "stride"
#define QNN_OP_TRANSPOSE_CONV_3D_PARAM_PAD_AMOUNT     "pad_amount"
#define QNN_OP_TRANSPOSE_CONV_3D_PARAM_DILATION       "dilation"
#define QNN_OP_TRANSPOSE_CONV_3D_PARAM_GROUP          "group"
#define QNN_OP_TRANSPOSE_CONV_3D_PARAM_OUTPUT_PADDING "output_padding"

#define QNN_OP_UN_PACK            "UnPack"
#define QNN_OP_UN_PACK_PARAM_AXIS "axis"

#endif  // QNN_OP_DEF_H