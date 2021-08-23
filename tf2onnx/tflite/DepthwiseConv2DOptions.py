# SPDX-License-Identifier: Apache-2.0

# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class DepthwiseConv2DOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = DepthwiseConv2DOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsDepthwiseConv2DOptions(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def DepthwiseConv2DOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # DepthwiseConv2DOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # DepthwiseConv2DOptions
    def Padding(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def StrideW(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def StrideH(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def DepthMultiplier(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # DepthwiseConv2DOptions
    def DilationWFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

    # DepthwiseConv2DOptions
    def DilationHFactor(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 1

def Start(builder): builder.StartObject(7)
def DepthwiseConv2DOptionsStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddPadding(builder, padding): builder.PrependInt8Slot(0, padding, 0)
def DepthwiseConv2DOptionsAddPadding(builder, padding):
    """This method is deprecated. Please switch to AddPadding."""
    return AddPadding(builder, padding)
def AddStrideW(builder, strideW): builder.PrependInt32Slot(1, strideW, 0)
def DepthwiseConv2DOptionsAddStrideW(builder, strideW):
    """This method is deprecated. Please switch to AddStrideW."""
    return AddStrideW(builder, strideW)
def AddStrideH(builder, strideH): builder.PrependInt32Slot(2, strideH, 0)
def DepthwiseConv2DOptionsAddStrideH(builder, strideH):
    """This method is deprecated. Please switch to AddStrideH."""
    return AddStrideH(builder, strideH)
def AddDepthMultiplier(builder, depthMultiplier): builder.PrependInt32Slot(3, depthMultiplier, 0)
def DepthwiseConv2DOptionsAddDepthMultiplier(builder, depthMultiplier):
    """This method is deprecated. Please switch to AddDepthMultiplier."""
    return AddDepthMultiplier(builder, depthMultiplier)
def AddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(4, fusedActivationFunction, 0)
def DepthwiseConv2DOptionsAddFusedActivationFunction(builder, fusedActivationFunction):
    """This method is deprecated. Please switch to AddFusedActivationFunction."""
    return AddFusedActivationFunction(builder, fusedActivationFunction)
def AddDilationWFactor(builder, dilationWFactor): builder.PrependInt32Slot(5, dilationWFactor, 1)
def DepthwiseConv2DOptionsAddDilationWFactor(builder, dilationWFactor):
    """This method is deprecated. Please switch to AddDilationWFactor."""
    return AddDilationWFactor(builder, dilationWFactor)
def AddDilationHFactor(builder, dilationHFactor): builder.PrependInt32Slot(6, dilationHFactor, 1)
def DepthwiseConv2DOptionsAddDilationHFactor(builder, dilationHFactor):
    """This method is deprecated. Please switch to AddDilationHFactor."""
    return AddDilationHFactor(builder, dilationHFactor)
def End(builder): return builder.EndObject()
def DepthwiseConv2DOptionsEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)