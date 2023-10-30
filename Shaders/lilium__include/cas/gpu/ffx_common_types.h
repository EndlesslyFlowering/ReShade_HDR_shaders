// This file is part of the FidelityFX SDK.
//
// Copyright (C) 2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the “Software”), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//#ifndef FFX_COMMON_TYPES_H
//#define FFX_COMMON_TYPES_H

#if defined(FFX_CPU)
#define FFX_PARAMETER_IN
#define FFX_PARAMETER_OUT
#define FFX_PARAMETER_INOUT
#define FFX_PARAMETER_UNIFORM
#elif defined(FFX_HLSL)
#define FFX_PARAMETER_IN        in
#define FFX_PARAMETER_OUT       out
#define FFX_PARAMETER_INOUT     inout
#define FFX_PARAMETER_UNIFORM uniform
#elif defined(FFX_GLSL)
#define FFX_PARAMETER_IN        in
#define FFX_PARAMETER_OUT       out
#define FFX_PARAMETER_INOUT     inout
#define FFX_PARAMETER_UNIFORM const //[cacao_placeholder] until a better fit is found!
#endif // #if defined(FFX_CPU)

#if defined(FFX_CPU)
/// A typedef for a boolean value.
///
/// @ingroup CPUTypes
#define FfxBoolean bool

/// A typedef for a unsigned 8bit integer.
///
/// @ingroup CPUTypes
#define FfxUInt8 uint8_t

/// A typedef for a unsigned 16bit integer.
///
/// @ingroup CPUTypes
#define FfxUInt16 uint16_t

/// A typedef for a unsigned 32bit integer.
///
/// @ingroup CPUTypes
#define FfxUInt32 uint32_t

/// A typedef for a unsigned 64bit integer.
///
/// @ingroup CPUTypes
#define FfxUInt64 uint64_t

/// A typedef for a signed 8bit integer.
///
/// @ingroup CPUTypes
#define FfxInt8 int8_t

/// A typedef for a signed 16bit integer.
///
/// @ingroup CPUTypes
#define FfxInt16 int16_t

/// A typedef for a signed 32bit integer.
///
/// @ingroup CPUTypes
#define FfxInt32 int32_t

/// A typedef for a signed 64bit integer.
///
/// @ingroup CPUTypes
#define FfxInt64 int64_t

/// A typedef for a floating point value.
///
/// @ingroup CPUTypes
#define FfxFloat32 float

/// A typedef for a 2-dimensional floating point value.
///
/// @ingroup CPUTypes
typedef float FfxFloat32x2[2];

/// A typedef for a 3-dimensional floating point value.
///
/// @ingroup CPUTypes
typedef float FfxFloat32x3[3];

/// A typedef for a 4-dimensional floating point value.
///
/// @ingroup CPUTypes
typedef float FfxFloat32x4[4];

/// A typedef for a 2-dimensional 32bit unsigned integer.
///
/// @ingroup CPUTypes
typedef uint32_t FfxUInt32x2[2];

/// A typedef for a 3-dimensional 32bit unsigned integer.
///
/// @ingroup CPUTypes
typedef uint32_t FfxUInt32x3[3];

/// A typedef for a 4-dimensional 32bit unsigned integer.
///
/// @ingroup CPUTypes
typedef uint32_t FfxUInt32x4[4];
#endif // #if defined(FFX_CPU)

#if defined(FFX_HLSL)

#define FfxFloat32Mat4 matrix <float, 4, 4>
#define FfxFloat32Mat3 matrix <float, 3, 3>

/// A typedef for a boolean value.
///
/// @ingroup HLSLTypes
#define FfxBoolean bool

#if FFX_HLSL_6_2

/// @defgroup HLSL62Types HLSL 6.2 And Above Types
/// HLSL 6.2 and above type defines for all commonly used variables
///
/// @ingroup HLSLTypes

/// A typedef for a floating point value.
///
/// @ingroup HLSL62Types
#define FfxFloat32 float32_t

/// A typedef for a 2-dimensional floating point value.
///
/// @ingroup HLSL62Types
#define FfxFloat32x2 float32_t2

/// A typedef for a 3-dimensional floating point value.
///
/// @ingroup HLSL62Types
#define FfxFloat32x3 float32_t3

/// A typedef for a 4-dimensional floating point value.
///
/// @ingroup HLSL62Types
#define FfxFloat32x4 float32_t4

/// A [cacao_placeholder] typedef for matrix type until confirmed.
#define FfxFloat32x4x4 float4x4
#define FfxFloat32x3x3 float3x3
#define FfxFloat32x2x2 float2x2

/// A typedef for a unsigned 32bit integer.
///
/// @ingroup HLSL62Types
#define FfxUInt32 uint32_t

/// A typedef for a 2-dimensional 32bit unsigned integer.
///
/// @ingroup HLSL62Types
#define FfxUInt32x2 uint32_t2

/// A typedef for a 3-dimensional 32bit unsigned integer.
///
/// @ingroup HLSL62Types
#define FfxUInt32x3 uint32_t3

/// A typedef for a 4-dimensional 32bit unsigned integer.
///
/// @ingroup HLSL62Types
#define FfxUInt32x4 uint32_t4

/// A typedef for a signed 32bit integer.
///
/// @ingroup HLSL62Types
#define FfxInt32 int32_t

/// A typedef for a 2-dimensional signed 32bit integer.
///
/// @ingroup HLSL62Types
#define FfxInt32x2 int32_t2

/// A typedef for a 3-dimensional signed 32bit integer.
///
/// @ingroup HLSL62Types
#define FfxInt32x3 int32_t3

/// A typedef for a 4-dimensional signed 32bit integer.
///
/// @ingroup HLSL62Types
#define FfxInt32x4 int32_t4

#else // #if defined(FFX_HLSL_6_2)

/// @defgroup HLSLBaseTypes HLSL 6.1 And Below Types
/// HLSL 6.1 and below type defines for all commonly used variables
///
/// @ingroup HLSLTypes

#define FfxFloat32   float
#define FfxFloat32x2 float2
#define FfxFloat32x3 float3
#define FfxFloat32x4 float4

/// A [cacao_placeholder] typedef for matrix type until confirmed.
#define FfxFloat32x4x4 float4x4
#define FfxFloat32x3x3 float3x3
#define FfxFloat32x2x2 float2x2

/// A typedef for a unsigned 32bit integer.
///
/// @ingroup GPU
#define FfxUInt32 uint
#define FfxUInt32x2 uint2
#define FfxUInt32x3 uint3
#define FfxUInt32x4 uint4

#define FfxInt32 int
#define FfxInt32x2 int2
#define FfxInt32x3 int3
#define FfxInt32x4 int4

#endif // #if defined(FFX_HLSL_6_2)

#if FFX_HALF

#if FFX_HLSL_6_2

#define FfxFloat16 float16_t
#define FfxFloat16x2 float16_t2
#define FfxFloat16x3 float16_t3
#define FfxFloat16x4 float16_t4

/// A typedef for an unsigned 16bit integer.
///
/// @ingroup HLSLTypes
#define FfxUInt16 uint16_t
#define FfxUInt16x2 uint16_t2
#define FfxUInt16x3 uint16_t3
#define FfxUInt16x4 uint16_t4

/// A typedef for a signed 16bit integer.
///
/// @ingroup HLSLTypes
#define FfxInt16 int16_t
#define FfxInt16x2 int16_t2
#define FfxInt16x3 int16_t3
#define FfxInt16x4 int16_t4
#else // #if FFX_HLSL_6_2
#define FfxFloat16 min16float
#define FfxFloat16x2 min16float2
#define FfxFloat16x3 min16float3
#define FfxFloat16x4 min16float4

/// A typedef for an unsigned 16bit integer.
///
/// @ingroup HLSLTypes
#define FfxUInt16 min16uint
#define FfxUInt16x2 min16uint2
#define FfxUInt16x3 min16uint3
#define FfxUInt16x4 min16uint4

/// A typedef for a signed 16bit integer.
///
/// @ingroup HLSLTypes
#define FfxInt16 min16int
#define FfxInt16x2 min16int2
#define FfxInt16x3 min16int3
#define FfxInt16x4 min16int4
#endif  // #if FFX_HLSL_6_2

#endif // FFX_HALF

#endif // #if defined(FFX_HLSL)

#if defined(FFX_GLSL)

#define FfxFloat32Mat4 mat4
#define FfxFloat32Mat3 mat3

/// A typedef for a boolean value.
///
/// @ingroup GLSLTypes
#define FfxBoolean   bool
#define FfxFloat32   float
#define FfxFloat32x2 vec2
#define FfxFloat32x3 vec3
#define FfxFloat32x4 vec4
#define FfxUInt32    uint
#define FfxUInt32x2  uvec2
#define FfxUInt32x3  uvec3
#define FfxUInt32x4  uvec4
#define FfxInt32     int
#define FfxInt32x2   ivec2
#define FfxInt32x3   ivec3
#define FfxInt32x4   ivec4

/// A [cacao_placeholder] typedef for matrix type until confirmed.
#define FfxFloat32x4x4 mat4
#define FfxFloat32x3x3 mat3
#define FfxFloat32x2x2 mat2

#if FFX_HALF
#define FfxFloat16   float16_t
#define FfxFloat16x2 f16vec2
#define FfxFloat16x3 f16vec3
#define FfxFloat16x4 f16vec4
#define FfxUInt16    uint16_t
#define FfxUInt16x2  u16vec2
#define FfxUInt16x3  u16vec3
#define FfxUInt16x4  u16vec4
#define FfxInt16     int16_t
#define FfxInt16x2   i16vec2
#define FfxInt16x3   i16vec3
#define FfxInt16x4   i16vec4
#endif // FFX_HALF
#endif // #if defined(FFX_GLSL)

// Global toggles:
// #define FFX_HALF            (1)
// #define FFX_HLSL_6_2        (1)

#if FFX_HALF

#if FFX_HLSL_6_2

#define FFX_MIN16_SCALAR( TypeName, BaseComponentType )           typedef BaseComponentType##16_t TypeName;
#define FFX_MIN16_VECTOR( TypeName, BaseComponentType, COL )      typedef vector<BaseComponentType##16_t, COL> TypeName;
#define FFX_MIN16_MATRIX( TypeName, BaseComponentType, ROW, COL ) typedef matrix<BaseComponentType##16_t, ROW, COL> TypeName;

#define FFX_16BIT_SCALAR( TypeName, BaseComponentType )           typedef BaseComponentType##16_t TypeName;
#define FFX_16BIT_VECTOR( TypeName, BaseComponentType, COL )      typedef vector<BaseComponentType##16_t, COL> TypeName;
#define FFX_16BIT_MATRIX( TypeName, BaseComponentType, ROW, COL ) typedef matrix<BaseComponentType##16_t, ROW, COL> TypeName;

#else //FFX_HLSL_6_2

#define FFX_MIN16_SCALAR( TypeName, BaseComponentType )           typedef min16##BaseComponentType TypeName;
#define FFX_MIN16_VECTOR( TypeName, BaseComponentType, COL )      typedef vector<min16##BaseComponentType, COL> TypeName;
#define FFX_MIN16_MATRIX( TypeName, BaseComponentType, ROW, COL ) typedef matrix<min16##BaseComponentType, ROW, COL> TypeName;

#define FFX_16BIT_SCALAR( TypeName, BaseComponentType )           FFX_MIN16_SCALAR( TypeName, BaseComponentType );
#define FFX_16BIT_VECTOR( TypeName, BaseComponentType, COL )      FFX_MIN16_VECTOR( TypeName, BaseComponentType, COL );
#define FFX_16BIT_MATRIX( TypeName, BaseComponentType, ROW, COL ) FFX_MIN16_MATRIX( TypeName, BaseComponentType, ROW, COL );

#endif //FFX_HLSL_6_2

#else //FFX_HALF

#define FFX_MIN16_SCALAR( TypeName, BaseComponentType )           typedef BaseComponentType TypeName;
#define FFX_MIN16_VECTOR( TypeName, BaseComponentType, COL )      typedef vector<BaseComponentType, COL> TypeName;
#define FFX_MIN16_MATRIX( TypeName, BaseComponentType, ROW, COL ) typedef matrix<BaseComponentType, ROW, COL> TypeName;

#define FFX_16BIT_SCALAR( TypeName, BaseComponentType )           typedef BaseComponentType TypeName;
#define FFX_16BIT_VECTOR( TypeName, BaseComponentType, COL )      typedef vector<BaseComponentType, COL> TypeName;
#define FFX_16BIT_MATRIX( TypeName, BaseComponentType, ROW, COL ) typedef matrix<BaseComponentType, ROW, COL> TypeName;

#endif //FFX_HALF

#if defined(FFX_GPU)
// Common typedefs:
#if defined(FFX_HLSL)
#define FFX_MIN16_F float
#define FFX_MIN16_F2 vector<float, 2>
#define FFX_MIN16_F3 vector<float, 3>
#define FFX_MIN16_F4 vector<float, 4>

#define FFX_MIN16_I int
#define FFX_MIN16_I2 vector<int, 2>
#define FFX_MIN16_I3 vector<int, 3>
#define FFX_MIN16_I4 vector<int, 4>

#define FFX_MIN16_U uint
#define FFX_MIN16_U2 vector<uint, 2>
#define FFX_MIN16_U3 vector<uint, 3>
#define FFX_MIN16_U4 vector<uint, 4>

#define FFX_F16_t float
#define FFX_F16_t2 vector<float, 2>
#define FFX_F16_t3 vector<float, 3>
#define FFX_F16_t4 vector<float, 4>

#define FFX_I16_t int
#define FFX_I16_t2 vector<int, 2>
#define FFX_I16_t3 vector<int, 3>
#define FFX_I16_t4 vector<int, 4>

#define FFX_U16_t uint
#define FFX_U16_t2 vector<uint, 2>
#define FFX_U16_t3 vector<uint, 3>
#define FFX_U16_t4 vector<uint, 4>

#define TYPEDEF_MIN16_TYPES(Prefix)           \
typedef FFX_MIN16_F     Prefix##_F;           \
typedef FFX_MIN16_F2    Prefix##_F2;          \
typedef FFX_MIN16_F3    Prefix##_F3;          \
typedef FFX_MIN16_F4    Prefix##_F4;          \
typedef FFX_MIN16_I     Prefix##_I;           \
typedef FFX_MIN16_I2    Prefix##_I2;          \
typedef FFX_MIN16_I3    Prefix##_I3;          \
typedef FFX_MIN16_I4    Prefix##_I4;          \
typedef FFX_MIN16_U     Prefix##_U;           \
typedef FFX_MIN16_U2    Prefix##_U2;          \
typedef FFX_MIN16_U3    Prefix##_U3;          \
typedef FFX_MIN16_U4    Prefix##_U4;

#define TYPEDEF_16BIT_TYPES(Prefix)           \
typedef FFX_16BIT_F     Prefix##_F;           \
typedef FFX_16BIT_F2    Prefix##_F2;          \
typedef FFX_16BIT_F3    Prefix##_F3;          \
typedef FFX_16BIT_F4    Prefix##_F4;          \
typedef FFX_16BIT_I     Prefix##_I;           \
typedef FFX_16BIT_I2    Prefix##_I2;          \
typedef FFX_16BIT_I3    Prefix##_I3;          \
typedef FFX_16BIT_I4    Prefix##_I4;          \
typedef FFX_16BIT_U     Prefix##_U;           \
typedef FFX_16BIT_U2    Prefix##_U2;          \
typedef FFX_16BIT_U3    Prefix##_U3;          \
typedef FFX_16BIT_U4    Prefix##_U4;

#define TYPEDEF_FULL_PRECISION_TYPES(Prefix)  \
typedef FfxFloat32      Prefix##_F;           \
typedef FfxFloat32x2    Prefix##_F2;          \
typedef FfxFloat32x3    Prefix##_F3;          \
typedef FfxFloat32x4    Prefix##_F4;          \
typedef FfxInt32        Prefix##_I;           \
typedef FfxInt32x2      Prefix##_I2;          \
typedef FfxInt32x3      Prefix##_I3;          \
typedef FfxInt32x4      Prefix##_I4;          \
typedef FfxUInt32       Prefix##_U;           \
typedef FfxUInt32x2     Prefix##_U2;          \
typedef FfxUInt32x3     Prefix##_U3;          \
typedef FfxUInt32x4     Prefix##_U4;
#endif // #if defined(FFX_HLSL)

#if defined(FFX_GLSL)

#if FFX_HALF

#define  FFX_MIN16_F  float16_t
#define  FFX_MIN16_F2 f16vec2
#define  FFX_MIN16_F3 f16vec3
#define  FFX_MIN16_F4 f16vec4

#define  FFX_MIN16_I  int16_t
#define  FFX_MIN16_I2 i16vec2
#define  FFX_MIN16_I3 i16vec3
#define  FFX_MIN16_I4 i16vec4

#define  FFX_MIN16_U  uint16_t
#define  FFX_MIN16_U2 u16vec2
#define  FFX_MIN16_U3 u16vec3
#define  FFX_MIN16_U4 u16vec4

#define FFX_16BIT_F  float16_t
#define FFX_16BIT_F2 f16vec2
#define FFX_16BIT_F3 f16vec3
#define FFX_16BIT_F4 f16vec4

#define FFX_16BIT_I  int16_t
#define FFX_16BIT_I2 i16vec2
#define FFX_16BIT_I3 i16vec3
#define FFX_16BIT_I4 i16vec4

#define FFX_16BIT_U  uint16_t
#define FFX_16BIT_U2 u16vec2
#define FFX_16BIT_U3 u16vec3
#define FFX_16BIT_U4 u16vec4

#else // FFX_HALF

#define  FFX_MIN16_F  float
#define  FFX_MIN16_F2 vec2
#define  FFX_MIN16_F3 vec3
#define  FFX_MIN16_F4 vec4

#define  FFX_MIN16_I  int
#define  FFX_MIN16_I2 ivec2
#define  FFX_MIN16_I3 ivec3
#define  FFX_MIN16_I4 ivec4

#define  FFX_MIN16_U  uint
#define  FFX_MIN16_U2 uvec2
#define  FFX_MIN16_U3 uvec3
#define  FFX_MIN16_U4 uvec4

#define FFX_16BIT_F  float
#define FFX_16BIT_F2 vec2
#define FFX_16BIT_F3 vec3
#define FFX_16BIT_F4 vec4

#define FFX_16BIT_I  int
#define FFX_16BIT_I2 ivec2
#define FFX_16BIT_I3 ivec3
#define FFX_16BIT_I4 ivec4

#define FFX_16BIT_U  uint
#define FFX_16BIT_U2 uvec2
#define FFX_16BIT_U3 uvec3
#define FFX_16BIT_U4 uvec4

#endif // FFX_HALF

#endif // #if defined(FFX_GLSL)

#endif // #if defined(FFX_GPU)
//#endif // #ifndef FFX_COMMON_TYPES_H
