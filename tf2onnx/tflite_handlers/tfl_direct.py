# SPDX-License-Identifier: Apache-2.0


"""
tfl_direct
"""

from tf2onnx.handler import tfl_op


# pylint: disable=unused-argument,missing-docstring,unused-variable,pointless-string-statement,invalid-name


@tfl_op("TFL_ABS", tf_op="Abs")
@tfl_op("TFL_BROADCAST_TO", tf_op="BroadcastTo")
@tfl_op("TFL_CEIL", tf_op="Ceil")
@tfl_op("TFL_COS", tf_op="Cos")
@tfl_op("TFL_ELU", tf_op="Elu")
@tfl_op("TFL_EQUAL", tf_op="Equal")
@tfl_op("TFL_EXP", tf_op="Exp")
@tfl_op("TFL_FLOOR", tf_op="Floor")
@tfl_op("TFL_FLOOR_DIV", tf_op="FloorDiv")
@tfl_op("TFL_FLOOR_MOD", tf_op="FloorMod")
@tfl_op("TFL_GREATER", tf_op="Greater")
@tfl_op("TFL_GREATER_EQUAL", tf_op="GreaterEqual")
@tfl_op("TFL_LESS", tf_op="Less")
@tfl_op("TFL_LESS_EQUAL", tf_op="LessEqual")
@tfl_op("TFL_LOG", tf_op="Log")
@tfl_op("TFL_LOG_SOFTMAX", tf_op="LogSoftmax")
@tfl_op("TFL_LOGICAL_AND", tf_op="LogicalAnd")
@tfl_op("TFL_LOGICAL_NOT", tf_op="LogicalNot")
@tfl_op("TFL_LOGICAL_OR", tf_op="LogicalOr")
@tfl_op("TFL_MATRIX_DIAG", tf_op="MatrixDiag")
@tfl_op("TFL_MATRIX_SET_DIAG", tf_op="MatrixSetDiag")
@tfl_op("TFL_MAXIMUM", tf_op="Maximum")
@tfl_op("TFL_MINIMUM", tf_op="Minimum")
@tfl_op("TFL_NEG", tf_op="Neg")
@tfl_op("TFL_NOT_EQUAL", tf_op="NotEqual")
@tfl_op("TFL_POW", tf_op="Pow")
@tfl_op("TFL_RANK", tf_op="Rank")
@tfl_op("TFL_RELU", tf_op="Relu")
@tfl_op("TFL_RELU6", tf_op="Relu6")
@tfl_op("TFL_ROUND", tf_op="Round")
@tfl_op("TFL_RSQRT", tf_op="Rsqrt")
@tfl_op("TFL_SELECT", tf_op="Select")
@tfl_op("TFL_SELECT_V2", tf_op="SelectV2")
@tfl_op("TFL_SIN", tf_op="Sin")
@tfl_op("TFL_SQRT", tf_op="Sqrt")
@tfl_op("TFL_SQUARE", tf_op="Square")
@tfl_op("TFL_SQUARED_DIFFERENCE", tf_op="SquaredDifference")
@tfl_op("TFL_TANH", tf_op="Tanh")
@tfl_op("TFL_WHERE", tf_op="Where")
@tfl_op("TFL_ZEROS_LIKE", tf_op="ZerosLike")
@tfl_op("TFL_FILL", tf_op="Fill")
@tfl_op("TFL_GATHER_ND", tf_op="GatherNd")
@tfl_op("TFL_PAD", tf_op="Pad")
@tfl_op("TFL_REVERSE_V2", tf_op="ReverseV2")
@tfl_op("TFL_SCATTER_ND", tf_op="ScatterNd")
@tfl_op("TFL_SEGMENT_SUM", tf_op="SegmentSum")
@tfl_op("TFL_SHAPE", tf_op="Shape")
@tfl_op("TFL_SLICE", tf_op="Slice")
@tfl_op("TFL_SQUEEZE", tf_op="Squeeze")
@tfl_op("TFL_TILE", tf_op="Tile")
@tfl_op("TFL_EXPAND_DIMS", tf_op="ExpandDims")
@tfl_op("TFL_TRANSPOSE", tf_op="Transpose")
@tfl_op("TFL_UNPACK", tf_op="Unpack")
@tfl_op("TFL_ADD_N", tf_op="AddN")
@tfl_op("TFL_ONE_HOT", tf_op="OneHot")
@tfl_op("TFL_DEPTH_TO_SPACE", tf_op="DepthToSpace")
@tfl_op("TFL_ARG_MIN", tf_op="ArgMin")
@tfl_op("TFL_ARG_MAX", tf_op="ArgMax")
@tfl_op("TFL_NON_MAX_SUPPRESSION_V5", tf_op="NonMaxSuppressionV5")
@tfl_op("TFL_RESIZE_NEAREST_NEIGHBOR", tf_op="ResizeNearestNeighbor")
@tfl_op("TFL_LEAKY_RELU", tf_op="LeakyRelu")
@tfl_op("TFL_STRIDED_SLICE", tf_op="StridedSlice")
@tfl_op("TFL_MEAN", tf_op="Mean")
@tfl_op("TFL_SUM", tf_op="Sum")
@tfl_op("TFL_MIRROR_PAD", tf_op="MirrorPad")
@tfl_op("TFL_RESIZE_BILINEAR", tf_op="ResizeBilinear")
@tfl_op("TFL_REVERSE_SEQUENCE", tf_op="ReverseSequence")
@tfl_op("TFL_SPARSE_TO_DENSE", tf_op="SparseToDense")
@tfl_op("TFL_CUMSUM", tf_op="Cumsum")
class TflDirectOp:
    @classmethod
    def to_tf(cls, ctx, node, **kwargs):
        pass
