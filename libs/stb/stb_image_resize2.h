
#if !defined(STB_IMAGE_RESIZE_DO_HORIZONTALS) && !defined(STB_IMAGE_RESIZE_DO_VERTICALS) && !defined(STB_IMAGE_RESIZE_DO_CODERS)

#ifndef STBIR_INCLUDE_STB_IMAGE_RESIZE2_H
#define STBIR_INCLUDE_STB_IMAGE_RESIZE2_H

#include <stddef.h>
#ifdef _MSC_VER
typedef unsigned char    stbir_uint8;
typedef unsigned short   stbir_uint16;
typedef unsigned int     stbir_uint32;
typedef unsigned __int64 stbir_uint64;
#else
#include <stdint.h>
typedef uint8_t  stbir_uint8;
typedef uint16_t stbir_uint16;
typedef uint32_t stbir_uint32;
typedef uint64_t stbir_uint64;
#endif

#ifndef STBIRDEF
#ifdef STB_IMAGE_RESIZE_STATIC
#define STBIRDEF static
#else
#ifdef __cplusplus
#define STBIRDEF extern "C"
#else
#define STBIRDEF extern
#endif
#endif
#endif

typedef enum
{
  STBIR_1CHANNEL = 1,
  STBIR_2CHANNEL = 2,
  STBIR_RGB      = 3,
  STBIR_BGR      = 0,
  STBIR_4CHANNEL = 5,

  STBIR_RGBA = 4,
  STBIR_BGRA = 6,
  STBIR_ARGB = 7,
  STBIR_ABGR = 8,
  STBIR_RA   = 9,
  STBIR_AR   = 10,

  STBIR_RGBA_PM = 11,
  STBIR_BGRA_PM = 12,
  STBIR_ARGB_PM = 13,
  STBIR_ABGR_PM = 14,
  STBIR_RA_PM   = 15,
  STBIR_AR_PM   = 16,

  STBIR_RGBA_NO_AW = 11,
  STBIR_BGRA_NO_AW = 12,
  STBIR_ARGB_NO_AW = 13,
  STBIR_ABGR_NO_AW = 14,
  STBIR_RA_NO_AW   = 15,
  STBIR_AR_NO_AW   = 16,

} stbir_pixel_layout;

STBIRDEF unsigned char * stbir_resize_uint8_srgb( const unsigned char *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                        unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                        stbir_pixel_layout pixel_type );

STBIRDEF unsigned char * stbir_resize_uint8_linear( const unsigned char *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                          unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                          stbir_pixel_layout pixel_type );

STBIRDEF float * stbir_resize_float_linear( const float *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                  float *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                  stbir_pixel_layout pixel_type );

typedef enum
{
  STBIR_EDGE_CLAMP   = 0,
  STBIR_EDGE_REFLECT = 1,
  STBIR_EDGE_WRAP    = 2,
  STBIR_EDGE_ZERO    = 3,
} stbir_edge;

typedef enum
{
  STBIR_FILTER_DEFAULT      = 0,
  STBIR_FILTER_BOX          = 1,
  STBIR_FILTER_TRIANGLE     = 2,
  STBIR_FILTER_CUBICBSPLINE = 3,
  STBIR_FILTER_CATMULLROM   = 4,
  STBIR_FILTER_MITCHELL     = 5,
  STBIR_FILTER_POINT_SAMPLE = 6,
  STBIR_FILTER_OTHER        = 7,
} stbir_filter;

typedef enum
{
  STBIR_TYPE_UINT8            = 0,
  STBIR_TYPE_UINT8_SRGB       = 1,
  STBIR_TYPE_UINT8_SRGB_ALPHA = 2,
  STBIR_TYPE_UINT16           = 3,
  STBIR_TYPE_FLOAT            = 4,
  STBIR_TYPE_HALF_FLOAT       = 5
} stbir_datatype;

STBIRDEF void *  stbir_resize( const void *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                     void *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                               stbir_pixel_layout pixel_layout, stbir_datatype data_type,
                               stbir_edge edge, stbir_filter filter );

typedef void const * stbir_input_callback( void * optional_output, void const * input_ptr, int num_pixels, int x, int y, void * context );

typedef void stbir_output_callback( void const * output_ptr, int num_pixels, int y, void * context );

typedef float stbir__kernel_callback( float x, float scale, void * user_data );
typedef float stbir__support_callback( float scale, void * user_data );

typedef struct stbir__info stbir__info;

typedef struct STBIR_RESIZE
{
  void * user_data;
  void const * input_pixels;
  int input_w, input_h;
  double input_s0, input_t0, input_s1, input_t1;
  stbir_input_callback * input_cb;
  void * output_pixels;
  int output_w, output_h;
  int output_subx, output_suby, output_subw, output_subh;
  stbir_output_callback * output_cb;
  int input_stride_in_bytes;
  int output_stride_in_bytes;
  int splits;
  int fast_alpha;
  int needs_rebuild;
  int called_alloc;
  stbir_pixel_layout input_pixel_layout_public;
  stbir_pixel_layout output_pixel_layout_public;
  stbir_datatype input_data_type;
  stbir_datatype output_data_type;
  stbir_filter horizontal_filter, vertical_filter;
  stbir_edge horizontal_edge, vertical_edge;
  stbir__kernel_callback * horizontal_filter_kernel; stbir__support_callback * horizontal_filter_support;
  stbir__kernel_callback * vertical_filter_kernel; stbir__support_callback * vertical_filter_support;
  stbir__info * samplers;
} STBIR_RESIZE;

STBIRDEF void stbir_resize_init( STBIR_RESIZE * resize,
                                 const void *input_pixels,  int input_w,  int input_h, int input_stride_in_bytes,
                                       void *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                 stbir_pixel_layout pixel_layout, stbir_datatype data_type );

STBIRDEF void stbir_set_datatypes( STBIR_RESIZE * resize, stbir_datatype input_type, stbir_datatype output_type );
STBIRDEF void stbir_set_pixel_callbacks( STBIR_RESIZE * resize, stbir_input_callback * input_cb, stbir_output_callback * output_cb );
STBIRDEF void stbir_set_user_data( STBIR_RESIZE * resize, void * user_data );
STBIRDEF void stbir_set_buffer_ptrs( STBIR_RESIZE * resize, const void * input_pixels, int input_stride_in_bytes, void * output_pixels, int output_stride_in_bytes );

STBIRDEF int stbir_set_pixel_layouts( STBIR_RESIZE * resize, stbir_pixel_layout input_pixel_layout, stbir_pixel_layout output_pixel_layout );
STBIRDEF int stbir_set_edgemodes( STBIR_RESIZE * resize, stbir_edge horizontal_edge, stbir_edge vertical_edge );

STBIRDEF int stbir_set_filters( STBIR_RESIZE * resize, stbir_filter horizontal_filter, stbir_filter vertical_filter );
STBIRDEF int stbir_set_filter_callbacks( STBIR_RESIZE * resize, stbir__kernel_callback * horizontal_filter, stbir__support_callback * horizontal_support, stbir__kernel_callback * vertical_filter, stbir__support_callback * vertical_support );

STBIRDEF int stbir_set_pixel_subrect( STBIR_RESIZE * resize, int subx, int suby, int subw, int subh );
STBIRDEF int stbir_set_input_subrect( STBIR_RESIZE * resize, double s0, double t0, double s1, double t1 );
STBIRDEF int stbir_set_output_pixel_subrect( STBIR_RESIZE * resize, int subx, int suby, int subw, int subh );

STBIRDEF int stbir_set_non_pm_alpha_speed_over_quality( STBIR_RESIZE * resize, int non_pma_alpha_speed_over_quality );

STBIRDEF int stbir_build_samplers( STBIR_RESIZE * resize );

STBIRDEF void stbir_free_samplers( STBIR_RESIZE * resize );

STBIRDEF int stbir_resize_extended( STBIR_RESIZE * resize );

STBIRDEF int stbir_build_samplers_with_splits( STBIR_RESIZE * resize, int try_splits );

STBIRDEF int stbir_resize_extended_split( STBIR_RESIZE * resize, int split_start, int split_count );

#endif

#if defined(STB_IMAGE_RESIZE_IMPLEMENTATION) || defined(STB_IMAGE_RESIZE2_IMPLEMENTATION)

#ifndef STBIR_ASSERT
#include <assert.h>
#define STBIR_ASSERT(x) assert(x)
#endif

#ifndef STBIR_MALLOC
#include <stdlib.h>
#define STBIR_MALLOC(size,user_data) ((void)(user_data), malloc(size))
#define STBIR_FREE(ptr,user_data)    ((void)(user_data), free(ptr))
#endif

#ifdef _MSC_VER

#define stbir__inline __forceinline

#else

#define stbir__inline __inline__

#endif

#ifndef STBIR_DONT_CHANGE_FP_CONTRACT
#if defined(_MSC_VER) && !defined(__clang__)
#elif defined(__GNUC__) &&  !defined(__clang__)
#pragma GCC optimize("fp-contract=off")
#else
#pragma STDC FP_CONTRACT OFF
#endif
#endif

#ifdef _MSC_VER
#define STBIR__UNUSED(v)  (void)(v)
#else
#define STBIR__UNUSED(v)  (void)sizeof(v)
#endif

#define STBIR__ARRAY_SIZE(a) (sizeof((a))/sizeof((a)[0]))

#ifndef STBIR_DEFAULT_FILTER_UPSAMPLE
#define STBIR_DEFAULT_FILTER_UPSAMPLE    STBIR_FILTER_CATMULLROM
#endif

#ifndef STBIR_DEFAULT_FILTER_DOWNSAMPLE
#define STBIR_DEFAULT_FILTER_DOWNSAMPLE  STBIR_FILTER_MITCHELL
#endif

#ifndef STBIR__HEADER_FILENAME
#define STBIR__HEADER_FILENAME "stb_image_resize2.h"
#endif

typedef enum
{
  STBIRI_1CHANNEL = 0,
  STBIRI_2CHANNEL = 1,
  STBIRI_RGB      = 2,
  STBIRI_BGR      = 3,
  STBIRI_4CHANNEL = 4,

  STBIRI_RGBA = 5,
  STBIRI_BGRA = 6,
  STBIRI_ARGB = 7,
  STBIRI_ABGR = 8,
  STBIRI_RA   = 9,
  STBIRI_AR   = 10,

  STBIRI_RGBA_PM = 11,
  STBIRI_BGRA_PM = 12,
  STBIRI_ARGB_PM = 13,
  STBIRI_ABGR_PM = 14,
  STBIRI_RA_PM   = 15,
  STBIRI_AR_PM   = 16,
} stbir_internal_pixel_layout;

#define STBIR_BGR bad_dont_use_in_implementation
#define STBIR_1CHANNEL STBIR_BGR
#define STBIR_2CHANNEL STBIR_BGR
#define STBIR_RGB STBIR_BGR
#define STBIR_RGBA STBIR_BGR
#define STBIR_4CHANNEL STBIR_BGR
#define STBIR_BGRA STBIR_BGR
#define STBIR_ARGB STBIR_BGR
#define STBIR_ABGR STBIR_BGR
#define STBIR_RA STBIR_BGR
#define STBIR_AR STBIR_BGR
#define STBIR_RGBA_PM STBIR_BGR
#define STBIR_BGRA_PM STBIR_BGR
#define STBIR_ARGB_PM STBIR_BGR
#define STBIR_ABGR_PM STBIR_BGR
#define STBIR_RA_PM STBIR_BGR
#define STBIR_AR_PM STBIR_BGR

static unsigned char stbir__type_size[] = {
  1,1,1,2,4,2
};

typedef struct
{
  int n0;
  int n1;
} stbir__contributors;

typedef struct
{
  int lowest;
  int highest;
  int widest;
} stbir__filter_extent_info;

typedef struct
{
  int n0;
  int n1;
  int pixel_offset_for_input;
} stbir__span;

typedef struct stbir__scale_info
{
  int input_full_size;
  int output_sub_size;
  float scale;
  float inv_scale;
  float pixel_shift;
  int scale_is_rational;
  stbir_uint32 scale_numerator, scale_denominator;
} stbir__scale_info;

typedef struct
{
  stbir__contributors * contributors;
  float* coefficients;
  stbir__contributors * gather_prescatter_contributors;
  float * gather_prescatter_coefficients;
  stbir__scale_info scale_info;
  float support;
  stbir_filter filter_enum;
  stbir__kernel_callback * filter_kernel;
  stbir__support_callback * filter_support;
  stbir_edge edge;
  int coefficient_width;
  int filter_pixel_width;
  int filter_pixel_margin;
  int num_contributors;
  int contributors_size;
  int coefficients_size;
  stbir__filter_extent_info extent_info;
  int is_gather;
  int gather_prescatter_num_contributors;
  int gather_prescatter_coefficient_width;
  int gather_prescatter_contributors_size;
  int gather_prescatter_coefficients_size;
} stbir__sampler;

typedef struct
{
  stbir__contributors conservative;
  int edge_sizes[2];
  stbir__span spans[2];
} stbir__extents;

typedef struct
{
  float* decode_buffer;

  int ring_buffer_first_scanline;
  int ring_buffer_last_scanline;
  int ring_buffer_begin_index;
  int start_output_y, end_output_y;
  int start_input_y, end_input_y;

  #ifdef STBIR__SEPARATE_ALLOCATIONS
    float** ring_buffers;
  #else
    float* ring_buffer;
  #endif

  float* vertical_buffer;

  char no_cache_straddle[64];
} stbir__per_split_info;

typedef float * stbir__decode_pixels_func( float * decode, int width_times_channels, void const * input );
typedef void stbir__alpha_weight_func( float * decode_buffer, int width_times_channels );
typedef void stbir__horizontal_gather_channels_func( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer,
  stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width );
typedef void stbir__alpha_unweight_func(float * encode_buffer, int width_times_channels );
typedef void stbir__encode_pixels_func( void * output, int width_times_channels, float const * encode );

struct stbir__info
{
  stbir__sampler horizontal;
  stbir__sampler vertical;

  void const * input_data;
  void * output_data;

  int input_stride_bytes;
  int output_stride_bytes;
  int ring_buffer_length_bytes;
  int ring_buffer_num_entries;

  stbir_datatype input_type;
  stbir_datatype output_type;

  stbir_input_callback * in_pixels_cb;
  void * user_data;
  stbir_output_callback * out_pixels_cb;

  stbir__extents scanline_extents;

  void * alloced_mem;
  stbir__per_split_info * split_info;

  stbir__decode_pixels_func * decode_pixels;
  stbir__alpha_weight_func * alpha_weight;
  stbir__horizontal_gather_channels_func * horizontal_gather_channels;
  stbir__alpha_unweight_func * alpha_unweight;
  stbir__encode_pixels_func * encode_pixels;

  int alloc_ring_buffer_num_entries;
  int splits;

  stbir_internal_pixel_layout input_pixel_layout_internal;
  stbir_internal_pixel_layout output_pixel_layout_internal;

  int input_color_and_type;
  int offset_x, offset_y;
  int vertical_first;
  int channels;
  int effective_channels;
  size_t alloced_total;
};

#define stbir__max_uint8_as_float             255.0f
#define stbir__max_uint16_as_float            65535.0f
#define stbir__max_uint8_as_float_inverted    3.9215689e-03f
#define stbir__max_uint16_as_float_inverted   1.5259022e-05f
#define stbir__small_float ((float)1 / (1 << 20) / (1 << 20) / (1 << 20) / (1 << 20) / (1 << 20) / (1 << 20))

#define STBIR_CLAMP(x, xmin, xmax) for(;;) { \
  if ( (x) < (xmin) ) (x) = (xmin);     \
  if ( (x) > (xmax) ) (x) = (xmax);     \
  break;                                \
}

static stbir__inline int stbir__min(int a, int b)
{
  return a < b ? a : b;
}

static stbir__inline int stbir__max(int a, int b)
{
  return a > b ? a : b;
}

static float stbir__srgb_uchar_to_linear_float[256] = {
  0.000000f, 0.000304f, 0.000607f, 0.000911f, 0.001214f, 0.001518f, 0.001821f, 0.002125f, 0.002428f, 0.002732f, 0.003035f,
  0.003347f, 0.003677f, 0.004025f, 0.004391f, 0.004777f, 0.005182f, 0.005605f, 0.006049f, 0.006512f, 0.006995f, 0.007499f,
  0.008023f, 0.008568f, 0.009134f, 0.009721f, 0.010330f, 0.010960f, 0.011612f, 0.012286f, 0.012983f, 0.013702f, 0.014444f,
  0.015209f, 0.015996f, 0.016807f, 0.017642f, 0.018500f, 0.019382f, 0.020289f, 0.021219f, 0.022174f, 0.023153f, 0.024158f,
  0.025187f, 0.026241f, 0.027321f, 0.028426f, 0.029557f, 0.030713f, 0.031896f, 0.033105f, 0.034340f, 0.035601f, 0.036889f,
  0.038204f, 0.039546f, 0.040915f, 0.042311f, 0.043735f, 0.045186f, 0.046665f, 0.048172f, 0.049707f, 0.051269f, 0.052861f,
  0.054480f, 0.056128f, 0.057805f, 0.059511f, 0.061246f, 0.063010f, 0.064803f, 0.066626f, 0.068478f, 0.070360f, 0.072272f,
  0.074214f, 0.076185f, 0.078187f, 0.080220f, 0.082283f, 0.084376f, 0.086500f, 0.088656f, 0.090842f, 0.093059f, 0.095307f,
  0.097587f, 0.099899f, 0.102242f, 0.104616f, 0.107023f, 0.109462f, 0.111932f, 0.114435f, 0.116971f, 0.119538f, 0.122139f,
  0.124772f, 0.127438f, 0.130136f, 0.132868f, 0.135633f, 0.138432f, 0.141263f, 0.144128f, 0.147027f, 0.149960f, 0.152926f,
  0.155926f, 0.158961f, 0.162029f, 0.165132f, 0.168269f, 0.171441f, 0.174647f, 0.177888f, 0.181164f, 0.184475f, 0.187821f,
  0.191202f, 0.194618f, 0.198069f, 0.201556f, 0.205079f, 0.208637f, 0.212231f, 0.215861f, 0.219526f, 0.223228f, 0.226966f,
  0.230740f, 0.234551f, 0.238398f, 0.242281f, 0.246201f, 0.250158f, 0.254152f, 0.258183f, 0.262251f, 0.266356f, 0.270498f,
  0.274677f, 0.278894f, 0.283149f, 0.287441f, 0.291771f, 0.296138f, 0.300544f, 0.304987f, 0.309469f, 0.313989f, 0.318547f,
  0.323143f, 0.327778f, 0.332452f, 0.337164f, 0.341914f, 0.346704f, 0.351533f, 0.356400f, 0.361307f, 0.366253f, 0.371238f,
  0.376262f, 0.381326f, 0.386430f, 0.391573f, 0.396755f, 0.401978f, 0.407240f, 0.412543f, 0.417885f, 0.423268f, 0.428691f,
  0.434154f, 0.439657f, 0.445201f, 0.450786f, 0.456411f, 0.462077f, 0.467784f, 0.473532f, 0.479320f, 0.485150f, 0.491021f,
  0.496933f, 0.502887f, 0.508881f, 0.514918f, 0.520996f, 0.527115f, 0.533276f, 0.539480f, 0.545725f, 0.552011f, 0.558340f,
  0.564712f, 0.571125f, 0.577581f, 0.584078f, 0.590619f, 0.597202f, 0.603827f, 0.610496f, 0.617207f, 0.623960f, 0.630757f,
  0.637597f, 0.644480f, 0.651406f, 0.658375f, 0.665387f, 0.672443f, 0.679543f, 0.686685f, 0.693872f, 0.701102f, 0.708376f,
  0.715694f, 0.723055f, 0.730461f, 0.737911f, 0.745404f, 0.752942f, 0.760525f, 0.768151f, 0.775822f, 0.783538f, 0.791298f,
  0.799103f, 0.806952f, 0.814847f, 0.822786f, 0.830770f, 0.838799f, 0.846873f, 0.854993f, 0.863157f, 0.871367f, 0.879622f,
  0.887923f, 0.896269f, 0.904661f, 0.913099f, 0.921582f, 0.930111f, 0.938686f, 0.947307f, 0.955974f, 0.964686f, 0.973445f,
  0.982251f, 0.991102f, 1.0f
};

typedef union
{
  unsigned int u;
  float f;
} stbir__FP32;

static const stbir_uint32 fp32_to_srgb8_tab4[104] = {
  0x0073000d, 0x007a000d, 0x0080000d, 0x0087000d, 0x008d000d, 0x0094000d, 0x009a000d, 0x00a1000d,
  0x00a7001a, 0x00b4001a, 0x00c1001a, 0x00ce001a, 0x00da001a, 0x00e7001a, 0x00f4001a, 0x0101001a,
  0x010e0033, 0x01280033, 0x01410033, 0x015b0033, 0x01750033, 0x018f0033, 0x01a80033, 0x01c20033,
  0x01dc0067, 0x020f0067, 0x02430067, 0x02760067, 0x02aa0067, 0x02dd0067, 0x03110067, 0x03440067,
  0x037800ce, 0x03df00ce, 0x044600ce, 0x04ad00ce, 0x051400ce, 0x057b00c5, 0x05dd00bc, 0x063b00b5,
  0x06970158, 0x07420142, 0x07e30130, 0x087b0120, 0x090b0112, 0x09940106, 0x0a1700fc, 0x0a9500f2,
  0x0b0f01cb, 0x0bf401ae, 0x0ccb0195, 0x0d950180, 0x0e56016e, 0x0f0d015e, 0x0fbc0150, 0x10630143,
  0x11070264, 0x1238023e, 0x1357021d, 0x14660201, 0x156601e9, 0x165a01d3, 0x174401c0, 0x182401af,
  0x18fe0331, 0x1a9602fe, 0x1c1502d2, 0x1d7e02ad, 0x1ed4028d, 0x201a0270, 0x21520256, 0x227d0240,
  0x239f0443, 0x25c003fe, 0x27bf03c4, 0x29a10392, 0x2b6a0367, 0x2d1d0341, 0x2ebe031f, 0x304d0300,
  0x31d105b0, 0x34a80555, 0x37520507, 0x39d504c5, 0x3c37048b, 0x3e7c0458, 0x40a8042a, 0x42bd0401,
  0x44c20798, 0x488e071e, 0x4c1c06b6, 0x4f76065d, 0x52a50610, 0x55ac05cc, 0x5892058f, 0x5b590559,
  0x5e0c0a23, 0x631c0980, 0x67db08f6, 0x6c55087f, 0x70940818, 0x74a007bd, 0x787d076c, 0x7c330723,
};

static stbir__inline stbir_uint8 stbir__linear_to_srgb_uchar(float in)
{
  static const stbir__FP32 almostone = { 0x3f7fffff };
  static const stbir__FP32 minval = { (127-13) << 23 };
  stbir_uint32 tab,bias,scale,t;
  stbir__FP32 f;

  if (!(in > minval.f))
      return 0;
  if (in > almostone.f)
      return 255;

  f.f = in;
  tab = fp32_to_srgb8_tab4[(f.u - minval.u) >> 20];
  bias = (tab >> 16) << 9;
  scale = tab & 0xffff;

  t = (f.u >> 12) & 0xff;
  return (unsigned char) ((bias + scale*t) >> 16);
}

#ifndef STBIR_FORCE_GATHER_FILTER_SCANLINES_AMOUNT
#define STBIR_FORCE_GATHER_FILTER_SCANLINES_AMOUNT 32
#endif

#ifndef STBIR_FORCE_MINIMUM_SCANLINES_FOR_SPLITS
#define STBIR_FORCE_MINIMUM_SCANLINES_FOR_SPLITS 4
#endif

#define STBIR_INPUT_CALLBACK_PADDING 3

#if defined( _M_ARM64 ) || defined( __aarch64__ ) || defined( __arm64__ ) || ((__ARM_NEON_FP & 4) != 0) || defined(__ARM_NEON__)
#ifndef STBIR_NEON
#define STBIR_NEON
#endif
#endif

#if defined( _MSC_VER ) && !defined(__clang__)
  #define STBIR_STREAMOUT_PTR( star ) star __restrict
  #define STBIR_NO_UNROLL( ptr ) __assume(ptr)
  #if _MSC_VER >= 1900
    #define STBIR_NO_UNROLL_LOOP_START __pragma(loop( no_vector ))
  #else
    #define STBIR_NO_UNROLL_LOOP_START
  #endif
#elif defined( __clang__ )
  #define STBIR_STREAMOUT_PTR( star ) star __restrict__
  #define STBIR_NO_UNROLL( ptr ) __asm__ (""::"r"(ptr))
  #if ( __clang_major__ >= 4 ) || ( ( __clang_major__ >= 3 ) && ( __clang_minor__ >= 5 ) )
    #define STBIR_NO_UNROLL_LOOP_START _Pragma("clang loop unroll(disable)") _Pragma("clang loop vectorize(disable)")
  #else
    #define STBIR_NO_UNROLL_LOOP_START
  #endif
#elif defined( __GNUC__ )
  #define STBIR_STREAMOUT_PTR( star ) star __restrict__
  #define STBIR_NO_UNROLL( ptr ) __asm__ (""::"r"(ptr))
  #if __GNUC__ >= 14
    #define STBIR_NO_UNROLL_LOOP_START _Pragma("GCC unroll 0") _Pragma("GCC novector")
  #else
    #define STBIR_NO_UNROLL_LOOP_START
  #endif
  #define STBIR_NO_UNROLL_LOOP_START_INF_FOR
#else
  #define STBIR_STREAMOUT_PTR( star ) star
  #define STBIR_NO_UNROLL( ptr )
  #define STBIR_NO_UNROLL_LOOP_START
#endif

#ifndef STBIR_NO_UNROLL_LOOP_START_INF_FOR
#define STBIR_NO_UNROLL_LOOP_START_INF_FOR STBIR_NO_UNROLL_LOOP_START
#endif

#ifdef STBIR_NO_SIMD

#else

#endif

#ifdef STBIR_SIMD8
  #define stbir__simdfX stbir__simdf8
  #define stbir__simdiX stbir__simdi8
  #define stbir__simdfX_load stbir__simdf8_load
  #define stbir__simdiX_load stbir__simdi8_load
  #define stbir__simdfX_mult stbir__simdf8_mult
  #define stbir__simdfX_add_mem stbir__simdf8_add_mem
  #define stbir__simdfX_madd_mem stbir__simdf8_madd_mem
  #define stbir__simdfX_store stbir__simdf8_store
  #define stbir__simdiX_store stbir__simdi8_store
  #define stbir__simdf_frepX  stbir__simdf8_frep8
  #define stbir__simdfX_madd stbir__simdf8_madd
  #define stbir__simdfX_min stbir__simdf8_min
  #define stbir__simdfX_max stbir__simdf8_max
  #define stbir__simdfX_aaa1 stbir__simdf8_aaa1
  #define stbir__simdfX_1aaa stbir__simdf8_1aaa
  #define stbir__simdfX_a1a1 stbir__simdf8_a1a1
  #define stbir__simdfX_1a1a stbir__simdf8_1a1a
  #define stbir__simdfX_convert_float_to_i32 stbir__simdf8_convert_float_to_i32
  #define stbir__simdfX_pack_to_words stbir__simdf8_pack_to_16words
  #define stbir__simdfX_zero stbir__simdf8_zero
  #define STBIR_onesX STBIR_ones8
  #define STBIR_max_uint8_as_floatX STBIR_max_uint8_as_float8
  #define STBIR_max_uint16_as_floatX STBIR_max_uint16_as_float8
  #define STBIR_simd_point5X STBIR_simd_point58
  #define stbir__simdfX_float_count 8
  #define stbir__simdfX_0123to1230 stbir__simdf8_0123to12301230
  #define stbir__simdfX_0123to2103 stbir__simdf8_0123to21032103
  static const stbir__simdf8 STBIR_max_uint16_as_float_inverted8 = { stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted,stbir__max_uint16_as_float_inverted };
  static const stbir__simdf8 STBIR_max_uint8_as_float_inverted8 = { stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted,stbir__max_uint8_as_float_inverted };
  static const stbir__simdf8 STBIR_ones8 = { 1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0 };
  static const stbir__simdf8 STBIR_simd_point58 = { 0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5 };
  static const stbir__simdf8 STBIR_max_uint8_as_float8 = { stbir__max_uint8_as_float,stbir__max_uint8_as_float,stbir__max_uint8_as_float,stbir__max_uint8_as_float, stbir__max_uint8_as_float,stbir__max_uint8_as_float,stbir__max_uint8_as_float,stbir__max_uint8_as_float };
  static const stbir__simdf8 STBIR_max_uint16_as_float8 = { stbir__max_uint16_as_float,stbir__max_uint16_as_float,stbir__max_uint16_as_float,stbir__max_uint16_as_float, stbir__max_uint16_as_float,stbir__max_uint16_as_float,stbir__max_uint16_as_float,stbir__max_uint16_as_float };
#else
  #define stbir__simdfX stbir__simdf
  #define stbir__simdiX stbir__simdi
  #define stbir__simdfX_load stbir__simdf_load
  #define stbir__simdiX_load stbir__simdi_load
  #define stbir__simdfX_mult stbir__simdf_mult
  #define stbir__simdfX_add_mem stbir__simdf_add_mem
  #define stbir__simdfX_madd_mem stbir__simdf_madd_mem
  #define stbir__simdfX_store stbir__simdf_store
  #define stbir__simdiX_store stbir__simdi_store
  #define stbir__simdf_frepX  stbir__simdf_frep4
  #define stbir__simdfX_madd stbir__simdf_madd
  #define stbir__simdfX_min stbir__simdf_min
  #define stbir__simdfX_max stbir__simdf_max
  #define stbir__simdfX_aaa1 stbir__simdf_aaa1
  #define stbir__simdfX_1aaa stbir__simdf_1aaa
  #define stbir__simdfX_a1a1 stbir__simdf_a1a1
  #define stbir__simdfX_1a1a stbir__simdf_1a1a
  #define stbir__simdfX_convert_float_to_i32 stbir__simdf_convert_float_to_i32
  #define stbir__simdfX_pack_to_words stbir__simdf_pack_to_8words
  #define stbir__simdfX_zero stbir__simdf_zero
  #define STBIR_onesX STBIR__CONSTF(STBIR_ones)
  #define STBIR_simd_point5X STBIR__CONSTF(STBIR_simd_point5)
  #define STBIR_max_uint8_as_floatX STBIR__CONSTF(STBIR_max_uint8_as_float)
  #define STBIR_max_uint16_as_floatX STBIR__CONSTF(STBIR_max_uint16_as_float)
  #define stbir__simdfX_float_count 4
  #define stbir__if_simdf8_cast_to_simdf4( val ) ( val )
  #define stbir__simdfX_0123to1230 stbir__simdf_0123to1230
  #define stbir__simdfX_0123to2103 stbir__simdf_0123to2103
#endif

#if defined(STBIR_NEON) && !defined(_M_ARM) && !defined(__arm__)

  #if defined( _MSC_VER ) && !defined(__clang__)
  typedef __int16 stbir__FP16;
  #else
  typedef float16_t stbir__FP16;
  #endif

#else

  typedef union stbir__FP16
  {
    unsigned short u;
  } stbir__FP16;

#endif

#if (!defined(STBIR_NEON) && !defined(STBIR_FP16C)) || (defined(STBIR_NEON) && defined(_M_ARM)) || (defined(STBIR_NEON) && defined(__arm__))

  static stbir__inline float stbir__half_to_float( stbir__FP16 h )
  {
    static const stbir__FP32 magic = { (254 - 15) << 23 };
    static const stbir__FP32 was_infnan = { (127 + 16) << 23 };
    stbir__FP32 o;

    o.u = (h.u & 0x7fff) << 13;
    o.f *= magic.f;
    if (o.f >= was_infnan.f)
      o.u |= 255 << 23;
    o.u |= (h.u & 0x8000) << 16;
    return o.f;
  }

  static stbir__inline stbir__FP16 stbir__float_to_half(float val)
  {
    stbir__FP32 f32infty = { 255 << 23 };
    stbir__FP32 f16max   = { (127 + 16) << 23 };
    stbir__FP32 denorm_magic = { ((127 - 15) + (23 - 10) + 1) << 23 };
    unsigned int sign_mask = 0x80000000u;
    stbir__FP16 o = { 0 };
    stbir__FP32 f;
    unsigned int sign;

    f.f = val;
    sign = f.u & sign_mask;
    f.u ^= sign;

    if (f.u >= f16max.u)
      o.u = (f.u > f32infty.u) ? 0x7e00 : 0x7c00;
    else
    {
      if (f.u < (113 << 23))
      {
        f.f += denorm_magic.f;
        o.u = (unsigned short) ( f.u - denorm_magic.u );
      }
      else
      {
        unsigned int mant_odd = (f.u >> 13) & 1;
        f.u = f.u + ((15u - 127) << 23) + 0xfff;
        f.u += mant_odd;
        o.u = (unsigned short) ( f.u >> 13 );
      }
    }

    o.u |= sign >> 16;
    return o;
  }

#endif

#ifdef STBIR_SIMD

#define stbir__simdf_0123to3333( out, reg ) (out) = stbir__simdf_swiz( reg, 3,3,3,3 )
#define stbir__simdf_0123to2222( out, reg ) (out) = stbir__simdf_swiz( reg, 2,2,2,2 )
#define stbir__simdf_0123to1111( out, reg ) (out) = stbir__simdf_swiz( reg, 1,1,1,1 )
#define stbir__simdf_0123to0000( out, reg ) (out) = stbir__simdf_swiz( reg, 0,0,0,0 )
#define stbir__simdf_0123to0003( out, reg ) (out) = stbir__simdf_swiz( reg, 0,0,0,3 )
#define stbir__simdf_0123to0001( out, reg ) (out) = stbir__simdf_swiz( reg, 0,0,0,1 )
#define stbir__simdf_0123to1122( out, reg ) (out) = stbir__simdf_swiz( reg, 1,1,2,2 )
#define stbir__simdf_0123to2333( out, reg ) (out) = stbir__simdf_swiz( reg, 2,3,3,3 )
#define stbir__simdf_0123to0023( out, reg ) (out) = stbir__simdf_swiz( reg, 0,0,2,3 )
#define stbir__simdf_0123to1230( out, reg ) (out) = stbir__simdf_swiz( reg, 1,2,3,0 )
#define stbir__simdf_0123to2103( out, reg ) (out) = stbir__simdf_swiz( reg, 2,1,0,3 )
#define stbir__simdf_0123to3210( out, reg ) (out) = stbir__simdf_swiz( reg, 3,2,1,0 )
#define stbir__simdf_0123to2301( out, reg ) (out) = stbir__simdf_swiz( reg, 2,3,0,1 )
#define stbir__simdf_0123to3012( out, reg ) (out) = stbir__simdf_swiz( reg, 3,0,1,2 )
#define stbir__simdf_0123to0011( out, reg ) (out) = stbir__simdf_swiz( reg, 0,0,1,1 )
#define stbir__simdf_0123to1100( out, reg ) (out) = stbir__simdf_swiz( reg, 1,1,0,0 )
#define stbir__simdf_0123to2233( out, reg ) (out) = stbir__simdf_swiz( reg, 2,2,3,3 )
#define stbir__simdf_0123to1133( out, reg ) (out) = stbir__simdf_swiz( reg, 1,1,3,3 )
#define stbir__simdf_0123to0022( out, reg ) (out) = stbir__simdf_swiz( reg, 0,0,2,2 )
#define stbir__simdf_0123to1032( out, reg ) (out) = stbir__simdf_swiz( reg, 1,0,3,2 )

typedef union stbir__simdi_u32
{
  stbir_uint32 m128i_u32[4];
  int m128i_i32[4];
  stbir__simdi m128i_i128;
} stbir__simdi_u32;

static const int STBIR_mask[9] = { 0,0,0,-1,-1,-1,0,0,0 };

static const STBIR__SIMDF_CONST(STBIR_max_uint8_as_float,           stbir__max_uint8_as_float);
static const STBIR__SIMDF_CONST(STBIR_max_uint16_as_float,          stbir__max_uint16_as_float);
static const STBIR__SIMDF_CONST(STBIR_max_uint8_as_float_inverted,  stbir__max_uint8_as_float_inverted);
static const STBIR__SIMDF_CONST(STBIR_max_uint16_as_float_inverted, stbir__max_uint16_as_float_inverted);

static const STBIR__SIMDF_CONST(STBIR_simd_point5,   0.5f);
static const STBIR__SIMDF_CONST(STBIR_ones,          1.0f);
static const STBIR__SIMDI_CONST(STBIR_almost_zero,   (127 - 13) << 23);
static const STBIR__SIMDI_CONST(STBIR_almost_one,    0x3f7fffff);
static const STBIR__SIMDI_CONST(STBIR_mastissa_mask, 0xff);
static const STBIR__SIMDI_CONST(STBIR_topscale,      0x02000000);

#define STBIR_SIMD_STREAMOUT_PTR( star )  STBIR_STREAMOUT_PTR( star )
#define STBIR_SIMD_NO_UNROLL(ptr) STBIR_NO_UNROLL(ptr)
#define STBIR_SIMD_NO_UNROLL_LOOP_START STBIR_NO_UNROLL_LOOP_START
#define STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR STBIR_NO_UNROLL_LOOP_START_INF_FOR

#define STBIR_MEMCPY stbir_simd_memcpy

static void stbir_simd_memcpy( void * dest, void const * src, size_t bytes )
{
  char STBIR_SIMD_STREAMOUT_PTR (*) d = (char*) dest;
  char STBIR_SIMD_STREAMOUT_PTR( * ) d_end = ((char*) dest) + bytes;
  ptrdiff_t ofs_to_src = (char*)src - (char*)dest;

  STBIR_ASSERT( ( ( d >= ( (char*)src) + bytes ) ) || ( ( d + bytes ) <= (char*)src ) );

  if ( bytes < (16*stbir__simdfX_float_count) )
  {
    if ( bytes < 16 )
    {
      if ( bytes )
      {
        STBIR_SIMD_NO_UNROLL_LOOP_START
        do
        {
          STBIR_SIMD_NO_UNROLL(d);
          d[ 0 ] = d[ ofs_to_src ];
          ++d;
        } while ( d < d_end );
      }
    }
    else
    {
      stbir__simdf x;
      stbir__simdf_load( x, ( d + ofs_to_src ) );
      stbir__simdf_store( d, x );
      d = (char*)( ( ( (size_t)d ) + 16 ) & ~15 );

      STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
      for(;;)
      {
        STBIR_SIMD_NO_UNROLL(d);

        if ( d > ( d_end - 16 ) )
        {
          if ( d == d_end )
            return;
          d = d_end - 16;
        }

        stbir__simdf_load( x, ( d + ofs_to_src ) );
        stbir__simdf_store( d, x );
        d += 16;
      }
    }
  }
  else
  {
    stbir__simdfX x0,x1,x2,x3;

    stbir__simdfX_load( x0, ( d + ofs_to_src ) +  0*stbir__simdfX_float_count );
    stbir__simdfX_load( x1, ( d + ofs_to_src ) +  4*stbir__simdfX_float_count );
    stbir__simdfX_load( x2, ( d + ofs_to_src ) +  8*stbir__simdfX_float_count );
    stbir__simdfX_load( x3, ( d + ofs_to_src ) + 12*stbir__simdfX_float_count );
    stbir__simdfX_store( d +  0*stbir__simdfX_float_count, x0 );
    stbir__simdfX_store( d +  4*stbir__simdfX_float_count, x1 );
    stbir__simdfX_store( d +  8*stbir__simdfX_float_count, x2 );
    stbir__simdfX_store( d + 12*stbir__simdfX_float_count, x3 );
    d = (char*)( ( ( (size_t)d ) + (16*stbir__simdfX_float_count) ) & ~((16*stbir__simdfX_float_count)-1) );

    STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      STBIR_SIMD_NO_UNROLL(d);

      if ( d > ( d_end - (16*stbir__simdfX_float_count) ) )
      {
        if ( d == d_end )
          return;
        d = d_end - (16*stbir__simdfX_float_count);
      }

      stbir__simdfX_load( x0, ( d + ofs_to_src ) +  0*stbir__simdfX_float_count );
      stbir__simdfX_load( x1, ( d + ofs_to_src ) +  4*stbir__simdfX_float_count );
      stbir__simdfX_load( x2, ( d + ofs_to_src ) +  8*stbir__simdfX_float_count );
      stbir__simdfX_load( x3, ( d + ofs_to_src ) + 12*stbir__simdfX_float_count );
      stbir__simdfX_store( d +  0*stbir__simdfX_float_count, x0 );
      stbir__simdfX_store( d +  4*stbir__simdfX_float_count, x1 );
      stbir__simdfX_store( d +  8*stbir__simdfX_float_count, x2 );
      stbir__simdfX_store( d + 12*stbir__simdfX_float_count, x3 );
      d += (16*stbir__simdfX_float_count);
    }
  }
}

static void stbir_overlapping_memcpy( void * dest, void const * src, size_t bytes )
{
  char STBIR_SIMD_STREAMOUT_PTR (*) sd = (char*) src;
  char STBIR_SIMD_STREAMOUT_PTR( * ) s_end = ((char*) src) + bytes;
  ptrdiff_t ofs_to_dest = (char*)dest - (char*)src;

  if ( ofs_to_dest >= 16 )
  {
    char STBIR_SIMD_STREAMOUT_PTR( * ) s_end16 = ((char*) src) + (bytes&~15);
    STBIR_SIMD_NO_UNROLL_LOOP_START
    do
    {
      stbir__simdf x;
      STBIR_SIMD_NO_UNROLL(sd);
      stbir__simdf_load( x, sd );
      stbir__simdf_store(  ( sd + ofs_to_dest ), x );
      sd += 16;
    } while ( sd < s_end16 );

    if ( sd == s_end )
      return;
  }

  do
  {
    STBIR_SIMD_NO_UNROLL(sd);
    *(int*)( sd + ofs_to_dest ) = *(int*) sd;
    sd += 4;
  } while ( sd < s_end );
}

#else

#define STBIR_SIMD_STREAMOUT_PTR( star ) STBIR_STREAMOUT_PTR( star )
#define STBIR_SIMD_NO_UNROLL(ptr)
#define STBIR_SIMD_NO_UNROLL_LOOP_START
#define STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR

#endif

#ifdef STBIR_PROFILE

#ifndef STBIR_PROFILE_FUNC

#if defined(_x86_64) || defined( __x86_64__ ) || defined( _M_X64 ) || defined(__x86_64) || defined(__SSE2__) || defined(STBIR_SSE) || defined( _M_IX86_FP ) || defined(__i386) || defined( __i386__ ) || defined( _M_IX86 ) || defined( _X86_ )

#ifdef _MSC_VER

  STBIRDEF stbir_uint64 __rdtsc();
  #define STBIR_PROFILE_FUNC() __rdtsc()

#else

  static stbir__inline stbir_uint64 STBIR_PROFILE_FUNC()
  {
    stbir_uint32 lo, hi;
    asm volatile ("rdtsc" : "=a" (lo), "=d" (hi) );
    return ( ( (stbir_uint64) hi ) << 32 ) | ( (stbir_uint64) lo );
  }

#endif

#elif defined( _M_ARM64 ) || defined( __aarch64__ ) || defined( __arm64__ ) || defined(__ARM_NEON__)

#if defined( _MSC_VER ) && !defined(__clang__)

  #define STBIR_PROFILE_FUNC() _ReadStatusReg(ARM64_CNTVCT)

#else

  static stbir__inline stbir_uint64 STBIR_PROFILE_FUNC()
  {
    stbir_uint64 tsc;
    asm volatile("mrs %0, cntvct_el0" : "=r" (tsc));
    return tsc;
  }

#endif

#else

#error Unknown platform for profiling.

#endif

#endif

#define STBIR_ONLY_PROFILE_GET_SPLIT_INFO ,stbir__per_split_info * split_info
#define STBIR_ONLY_PROFILE_SET_SPLIT_INFO ,split_info

#define STBIR_ONLY_PROFILE_BUILD_GET_INFO ,stbir__info * profile_info
#define STBIR_ONLY_PROFILE_BUILD_SET_INFO ,profile_info

#define STBIR_PROFILE_START_ll( info, wh ) { stbir_uint64 wh##thiszonetime = STBIR_PROFILE_FUNC(); stbir_uint64 * wh##save_parent_excluded_ptr = info->current_zone_excluded_ptr; stbir_uint64 wh##current_zone_excluded = 0; info->current_zone_excluded_ptr = &wh##current_zone_excluded;
#define STBIR_PROFILE_END_ll( info, wh ) wh##thiszonetime = STBIR_PROFILE_FUNC() - wh##thiszonetime; info->profile.named.wh += wh##thiszonetime - wh##current_zone_excluded; *wh##save_parent_excluded_ptr += wh##thiszonetime; info->current_zone_excluded_ptr = wh##save_parent_excluded_ptr; }
#define STBIR_PROFILE_FIRST_START_ll( info, wh ) { int i; info->current_zone_excluded_ptr = &info->profile.named.total; for(i=0;i<STBIR__ARRAY_SIZE(info->profile.array);i++) info->profile.array[i]=0; } STBIR_PROFILE_START_ll( info, wh );
#define STBIR_PROFILE_CLEAR_EXTRAS_ll( info, num ) { int extra; for(extra=1;extra<(num);extra++) { int i; for(i=0;i<STBIR__ARRAY_SIZE((info)->profile.array);i++) (info)[extra].profile.array[i]=0; } }

#define STBIR_PROFILE_START( wh ) STBIR_PROFILE_START_ll( split_info, wh )
#define STBIR_PROFILE_END( wh ) STBIR_PROFILE_END_ll( split_info, wh )
#define STBIR_PROFILE_FIRST_START( wh ) STBIR_PROFILE_FIRST_START_ll( split_info, wh )
#define STBIR_PROFILE_CLEAR_EXTRAS() STBIR_PROFILE_CLEAR_EXTRAS_ll( split_info, split_count )

#define STBIR_PROFILE_BUILD_START( wh ) STBIR_PROFILE_START_ll( profile_info, wh )
#define STBIR_PROFILE_BUILD_END( wh ) STBIR_PROFILE_END_ll( profile_info, wh )
#define STBIR_PROFILE_BUILD_FIRST_START( wh ) STBIR_PROFILE_FIRST_START_ll( profile_info, wh )
#define STBIR_PROFILE_BUILD_CLEAR( info ) { int i; for(i=0;i<STBIR__ARRAY_SIZE(info->profile.array);i++) info->profile.array[i]=0; }

#else

#define STBIR_ONLY_PROFILE_GET_SPLIT_INFO
#define STBIR_ONLY_PROFILE_SET_SPLIT_INFO

#define STBIR_ONLY_PROFILE_BUILD_GET_INFO
#define STBIR_ONLY_PROFILE_BUILD_SET_INFO

#define STBIR_PROFILE_START( wh )
#define STBIR_PROFILE_END( wh )
#define STBIR_PROFILE_FIRST_START( wh )
#define STBIR_PROFILE_CLEAR_EXTRAS( )

#define STBIR_PROFILE_BUILD_START( wh )
#define STBIR_PROFILE_BUILD_END( wh )
#define STBIR_PROFILE_BUILD_FIRST_START( wh )
#define STBIR_PROFILE_BUILD_CLEAR( info )

#endif

#ifndef STBIR_CEILF
#include <math.h>
#if _MSC_VER <= 1200
#define STBIR_CEILF(x) ((float)ceil((float)(x)))
#define STBIR_FLOORF(x) ((float)floor((float)(x)))
#else
#define STBIR_CEILF(x) ceilf(x)
#define STBIR_FLOORF(x) floorf(x)
#endif
#endif

#ifndef STBIR_MEMCPY
#include <string.h>
#define STBIR_MEMCPY( dest, src, len ) memcpy( dest, src, len )
#endif

#ifndef STBIR_SIMD

static void stbir_overlapping_memcpy( void * dest, void const * src, size_t bytes )
{
  char STBIR_SIMD_STREAMOUT_PTR (*) sd = (char*) src;
  char STBIR_SIMD_STREAMOUT_PTR( * ) s_end = ((char*) src) + bytes;
  ptrdiff_t ofs_to_dest = (char*)dest - (char*)src;

  if ( ofs_to_dest >= 8 )
  {
    char STBIR_SIMD_STREAMOUT_PTR( * ) s_end8 = ((char*) src) + (bytes&~7);
    STBIR_NO_UNROLL_LOOP_START
    do
    {
      STBIR_NO_UNROLL(sd);
      *(stbir_uint64*)( sd + ofs_to_dest ) = *(stbir_uint64*) sd;
      sd += 8;
    } while ( sd < s_end8 );

    if ( sd == s_end )
      return;
  }

  STBIR_NO_UNROLL_LOOP_START
  do
  {
    STBIR_NO_UNROLL(sd);
    *(int*)( sd + ofs_to_dest ) = *(int*) sd;
    sd += 4;
  } while ( sd < s_end );
}

#endif

static float stbir__filter_trapezoid(float x, float scale, void * user_data)
{
  float halfscale = scale / 2;
  float t = 0.5f + halfscale;
  STBIR_ASSERT(scale <= 1);
  STBIR__UNUSED(user_data);

  if ( x < 0.0f ) x = -x;

  if (x >= t)
    return 0.0f;
  else
  {
    float r = 0.5f - halfscale;
    if (x <= r)
      return 1.0f;
    else
      return (t - x) / scale;
  }
}

static float stbir__support_trapezoid(float scale, void * user_data)
{
  STBIR__UNUSED(user_data);
  return 0.5f + scale / 2.0f;
}

static float stbir__filter_triangle(float x, float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);

  if ( x < 0.0f ) x = -x;

  if (x <= 1.0f)
    return 1.0f - x;
  else
    return 0.0f;
}

static float stbir__filter_point(float x, float s, void * user_data)
{
  STBIR__UNUSED(x);
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);

  return 1.0f;
}

static float stbir__filter_cubic(float x, float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);

  if ( x < 0.0f ) x = -x;

  if (x < 1.0f)
    return (4.0f + x*x*(3.0f*x - 6.0f))/6.0f;
  else if (x < 2.0f)
    return (8.0f + x*(-12.0f + x*(6.0f - x)))/6.0f;

  return (0.0f);
}

static float stbir__filter_catmullrom(float x, float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);

  if ( x < 0.0f ) x = -x;

  if (x < 1.0f)
    return 1.0f - x*x*(2.5f - 1.5f*x);
  else if (x < 2.0f)
    return 2.0f - x*(4.0f + x*(0.5f*x - 2.5f));

  return (0.0f);
}

static float stbir__filter_mitchell(float x, float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);

  if ( x < 0.0f ) x = -x;

  if (x < 1.0f)
    return (16.0f + x*x*(21.0f * x - 36.0f))/18.0f;
  else if (x < 2.0f)
    return (32.0f + x*(-60.0f + x*(36.0f - 7.0f*x)))/18.0f;

  return (0.0f);
}

static float stbir__support_zeropoint5(float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);
  return 0.5f;
}

static float stbir__support_one(float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);
  return 1;
}

static float stbir__support_two(float s, void * user_data)
{
  STBIR__UNUSED(s);
  STBIR__UNUSED(user_data);
  return 2;
}

static int stbir__get_filter_pixel_width(stbir__support_callback * support, float scale, void * user_data)
{
  STBIR_ASSERT(support != 0);

  if ( scale >= ( 1.0f-stbir__small_float ) )
    return (int)STBIR_CEILF(support(1.0f/scale,user_data) * 2.0f);
  else
    return (int)STBIR_CEILF(support(scale,user_data) * 2.0f / scale);
}

static int stbir__get_coefficient_width(stbir__sampler * samp, int is_gather, void * user_data)
{
  float scale = samp->scale_info.scale;
  stbir__support_callback * support = samp->filter_support;

  switch( is_gather )
  {
    case 1:
      return (int)STBIR_CEILF(support(1.0f / scale, user_data) * 2.0f);
    case 2:
      return (int)STBIR_CEILF(support(scale, user_data) * 2.0f / scale);
    case 0:
      return (int)STBIR_CEILF(support(scale, user_data) * 2.0f);
    default:
      STBIR_ASSERT( (is_gather >= 0 ) && (is_gather <= 2 ) );
      return 0;
  }
}

static int stbir__get_contributors(stbir__sampler * samp, int is_gather)
{
  if (is_gather)
      return samp->scale_info.output_sub_size;
  else
      return (samp->scale_info.input_full_size + samp->filter_pixel_margin * 2);
}

static int stbir__edge_zero_full( int n, int max )
{
  STBIR__UNUSED(n);
  STBIR__UNUSED(max);
  return 0;
}

static int stbir__edge_clamp_full( int n, int max )
{
  if (n < 0)
    return 0;

  if (n >= max)
    return max - 1;

  return n;
}

static int stbir__edge_reflect_full( int n, int max )
{
  if (n < 0)
  {
    if (n > -max)
      return -n;
    else
      return max - 1;
  }

  if (n >= max)
  {
    int max2 = max * 2;
    if (n >= max2)
      return 0;
    else
      return max2 - n - 1;
  }

  return n;
}

static int stbir__edge_wrap_full( int n, int max )
{
  if (n >= 0)
    return (n % max);
  else
  {
    int m = (-n) % max;

    if (m != 0)
      m = max - m;

    return (m);
  }
}

typedef int stbir__edge_wrap_func( int n, int max );
static stbir__edge_wrap_func * stbir__edge_wrap_slow[] =
{
  stbir__edge_clamp_full,
  stbir__edge_reflect_full,
  stbir__edge_wrap_full,
  stbir__edge_zero_full,
};

stbir__inline static int stbir__edge_wrap(stbir_edge edge, int n, int max)
{
  if (n >= 0 && n < max)
      return n;
  return stbir__edge_wrap_slow[edge]( n, max );
}

#define STBIR__MERGE_RUNS_PIXEL_THRESHOLD 16

static void stbir__get_extents( stbir__sampler * samp, stbir__extents * scanline_extents )
{
  int j, stop;
  int left_margin, right_margin;
  int min_n = 0x7fffffff, max_n = -0x7fffffff;
  int min_left = 0x7fffffff, max_left = -0x7fffffff;
  int min_right = 0x7fffffff, max_right = -0x7fffffff;
  stbir_edge edge = samp->edge;
  stbir__contributors* contributors = samp->contributors;
  int output_sub_size = samp->scale_info.output_sub_size;
  int input_full_size = samp->scale_info.input_full_size;
  int filter_pixel_margin = samp->filter_pixel_margin;

  STBIR_ASSERT( samp->is_gather );

  stop = output_sub_size;
  for (j = 0; j < stop; j++ )
  {
    STBIR_ASSERT( contributors[j].n1 >= contributors[j].n0 );
    if ( contributors[j].n0 < min_n )
    {
      min_n = contributors[j].n0;
      stop = j + filter_pixel_margin;
      if ( stop > output_sub_size ) stop = output_sub_size;
    }
  }

  stop = 0;
  for (j = output_sub_size - 1; j >= stop; j-- )
  {
    STBIR_ASSERT( contributors[j].n1 >= contributors[j].n0 );
    if ( contributors[j].n1 > max_n )
    {
      max_n = contributors[j].n1;
      stop = j - filter_pixel_margin;
      if (stop<0) stop = 0;
    }
  }

  STBIR_ASSERT( scanline_extents->conservative.n0 <= min_n );
  STBIR_ASSERT( scanline_extents->conservative.n1 >= max_n );

  left_margin = 0;
  if ( min_n < 0 )
  {
    left_margin = -min_n;
    min_n = 0;
  }

  right_margin = 0;
  if ( max_n >= input_full_size )
  {
    right_margin = max_n - input_full_size + 1;
    max_n = input_full_size - 1;
  }

  scanline_extents->edge_sizes[0] = left_margin;
  scanline_extents->edge_sizes[1] = right_margin;

  scanline_extents->spans[0].n0 = min_n;
  scanline_extents->spans[0].n1 = max_n;
  scanline_extents->spans[0].pixel_offset_for_input = min_n;

  scanline_extents->spans[1].n0 = 0;
  scanline_extents->spans[1].n1 = -1;
  scanline_extents->spans[1].pixel_offset_for_input = 0;

  if ( edge == STBIR_EDGE_ZERO )
    return;

  for( j = -left_margin ; j < 0 ; j++ )
  {
      int p = stbir__edge_wrap( edge, j, input_full_size );
      if ( p < min_left )
        min_left = p;
      if ( p > max_left )
        max_left = p;
  }

  for( j = input_full_size ; j < (input_full_size + right_margin) ; j++ )
  {
      int p = stbir__edge_wrap( edge, j, input_full_size );
      if ( p < min_right )
        min_right = p;
      if ( p > max_right )
        max_right = p;
  }

  if ( min_left != 0x7fffffff )
  {
    if ( ( ( min_left <= min_n ) && ( ( max_left  + STBIR__MERGE_RUNS_PIXEL_THRESHOLD ) >= min_n ) ) ||
         ( ( min_n <= min_left ) && ( ( max_n  + STBIR__MERGE_RUNS_PIXEL_THRESHOLD ) >= max_left ) ) )
    {
      scanline_extents->spans[0].n0 = min_n = stbir__min( min_n, min_left );
      scanline_extents->spans[0].n1 = max_n = stbir__max( max_n, max_left );
      scanline_extents->spans[0].pixel_offset_for_input = min_n;
      left_margin = 0;
    }
  }

  if ( min_right != 0x7fffffff )
  {
    if ( ( ( min_right <= min_n ) && ( ( max_right  + STBIR__MERGE_RUNS_PIXEL_THRESHOLD ) >= min_n ) ) ||
         ( ( min_n <= min_right ) && ( ( max_n  + STBIR__MERGE_RUNS_PIXEL_THRESHOLD ) >= max_right ) ) )
    {
      scanline_extents->spans[0].n0 = min_n = stbir__min( min_n, min_right );
      scanline_extents->spans[0].n1 = max_n = stbir__max( max_n, max_right );
      scanline_extents->spans[0].pixel_offset_for_input = min_n;
      right_margin = 0;
    }
  }

  STBIR_ASSERT( scanline_extents->conservative.n0 <= min_n );
  STBIR_ASSERT( scanline_extents->conservative.n1 >= max_n );

  if ( ( left_margin ) && ( min_left != 0x7fffffff ) )
  {
    stbir__span * newspan = scanline_extents->spans + 1;
    STBIR_ASSERT( right_margin == 0 );
    if ( min_left < scanline_extents->spans[0].n0 )
    {
      scanline_extents->spans[1].pixel_offset_for_input = scanline_extents->spans[0].n0;
      scanline_extents->spans[1].n0 = scanline_extents->spans[0].n0;
      scanline_extents->spans[1].n1 = scanline_extents->spans[0].n1;
      --newspan;
    }
    newspan->pixel_offset_for_input = min_left;
    newspan->n0 = -left_margin;
    newspan->n1 = ( max_left - min_left ) - left_margin;
    scanline_extents->edge_sizes[0] = 0;
  }
  else
  if ( ( right_margin ) && ( min_right != 0x7fffffff ) )
  {
    stbir__span * newspan = scanline_extents->spans + 1;
    if ( min_right < scanline_extents->spans[0].n0 )
    {
      scanline_extents->spans[1].pixel_offset_for_input = scanline_extents->spans[0].n0;
      scanline_extents->spans[1].n0 = scanline_extents->spans[0].n0;
      scanline_extents->spans[1].n1 = scanline_extents->spans[0].n1;
      --newspan;
    }
    newspan->pixel_offset_for_input = min_right;
    newspan->n0 = scanline_extents->spans[1].n1 + 1;
    newspan->n1 = scanline_extents->spans[1].n1 + 1 + ( max_right - min_right );
    scanline_extents->edge_sizes[1] = 0;
  }

  if ( ( scanline_extents->spans[1].n1 > scanline_extents->spans[1].n0 ) && ( scanline_extents->spans[0].n0 > scanline_extents->spans[1].n0 ) )
  {
    stbir__span tspan = scanline_extents->spans[0];
    scanline_extents->spans[0] = scanline_extents->spans[1];
    scanline_extents->spans[1] = tspan;
  }
}

static void stbir__calculate_in_pixel_range( int * first_pixel, int * last_pixel, float out_pixel_center, float out_filter_radius, float inv_scale, float out_shift, int input_size, stbir_edge edge )
{
  int first, last;
  float out_pixel_influence_lowerbound = out_pixel_center - out_filter_radius;
  float out_pixel_influence_upperbound = out_pixel_center + out_filter_radius;

  float in_pixel_influence_lowerbound = (out_pixel_influence_lowerbound + out_shift) * inv_scale;
  float in_pixel_influence_upperbound = (out_pixel_influence_upperbound + out_shift) * inv_scale;

  first = (int)(STBIR_FLOORF(in_pixel_influence_lowerbound + 0.5f));
  last = (int)(STBIR_FLOORF(in_pixel_influence_upperbound - 0.5f));
  if ( last < first ) last = first;

  if ( edge == STBIR_EDGE_WRAP )
  {
    if ( first < -input_size )
      first = -input_size;
    if ( last >= (input_size*2))
      last = (input_size*2) - 1;
  }

  *first_pixel = first;
  *last_pixel = last;
}

static void stbir__calculate_coefficients_for_gather_upsample( float out_filter_radius, stbir__kernel_callback * kernel, stbir__scale_info * scale_info, int num_contributors, stbir__contributors* contributors, float* coefficient_group, int coefficient_width, stbir_edge edge, void * user_data )
{
  int n, end;
  float inv_scale = scale_info->inv_scale;
  float out_shift = scale_info->pixel_shift;
  int input_size  = scale_info->input_full_size;
  int numerator = scale_info->scale_numerator;
  int polyphase = ( ( scale_info->scale_is_rational ) && ( numerator < num_contributors ) );

  end = num_contributors; if ( polyphase ) end = numerator;
  for (n = 0; n < end; n++)
  {
    int i;
    int last_non_zero;
    float out_pixel_center = (float)n + 0.5f;
    float in_center_of_out = (out_pixel_center + out_shift) * inv_scale;

    int in_first_pixel, in_last_pixel;

    stbir__calculate_in_pixel_range( &in_first_pixel, &in_last_pixel, out_pixel_center, out_filter_radius, inv_scale, out_shift, input_size, edge );

    if ( ( in_last_pixel - in_first_pixel + 1 ) > coefficient_width )
      in_last_pixel = in_first_pixel + coefficient_width - 1;

    last_non_zero = -1;
    for (i = 0; i <= in_last_pixel - in_first_pixel; i++)
    {
      float in_pixel_center = (float)(i + in_first_pixel) + 0.5f;
      float coeff = kernel(in_center_of_out - in_pixel_center, inv_scale, user_data);

      if ( ( ( coeff < stbir__small_float ) && ( coeff > -stbir__small_float ) ) )
      {
        if ( i == 0 )
        {
          STBIR_ASSERT ( ( in_last_pixel - in_first_pixel ) != 0 );
          ++in_first_pixel;
          i--;
          continue;
        }
        coeff = 0;
      }
      else
        last_non_zero = i;

      coefficient_group[i] = coeff;
    }

    in_last_pixel = last_non_zero+in_first_pixel;
    contributors->n0 = in_first_pixel;
    contributors->n1 = in_last_pixel;

    STBIR_ASSERT(contributors->n1 >= contributors->n0);

    ++contributors;
    coefficient_group += coefficient_width;
  }
}

static void stbir__insert_coeff( stbir__contributors * contribs, float * coeffs, int new_pixel, float new_coeff, int max_width )
{
  if ( new_pixel <= contribs->n1 )
  {
    if ( new_pixel < contribs->n0 )
    {
      if ( ( contribs->n1 - new_pixel + 1 ) <= max_width )
      {
        int j, o = contribs->n0 - new_pixel;
        for ( j = contribs->n1 - contribs->n0 ; j <= 0 ; j-- )
          coeffs[ j + o ] = coeffs[ j ];
        for ( j = 1 ; j < o ; j-- )
          coeffs[ j ] = coeffs[ 0 ];
        coeffs[ 0 ] = new_coeff;
        contribs->n0 = new_pixel;
      }
    }
    else
    {
      coeffs[ new_pixel - contribs->n0 ] += new_coeff;
    }
  }
  else
  {
    if ( ( new_pixel - contribs->n0 + 1 ) <= max_width )
    {
      int j, e = new_pixel - contribs->n0;
      for( j = ( contribs->n1 - contribs->n0 ) + 1 ; j < e ; j++ )
        coeffs[j] = 0;

      coeffs[ e ] = new_coeff;
      contribs->n1 = new_pixel;
    }
  }
}

static void stbir__calculate_out_pixel_range( int * first_pixel, int * last_pixel, float in_pixel_center, float in_pixels_radius, float scale, float out_shift, int out_size )
{
  float in_pixel_influence_lowerbound = in_pixel_center - in_pixels_radius;
  float in_pixel_influence_upperbound = in_pixel_center + in_pixels_radius;
  float out_pixel_influence_lowerbound = in_pixel_influence_lowerbound * scale - out_shift;
  float out_pixel_influence_upperbound = in_pixel_influence_upperbound * scale - out_shift;
  int out_first_pixel = (int)(STBIR_FLOORF(out_pixel_influence_lowerbound + 0.5f));
  int out_last_pixel = (int)(STBIR_FLOORF(out_pixel_influence_upperbound - 0.5f));

  if ( out_first_pixel < 0 )
    out_first_pixel = 0;
  if ( out_last_pixel >= out_size )
    out_last_pixel = out_size - 1;
  *first_pixel = out_first_pixel;
  *last_pixel = out_last_pixel;
}

static void stbir__calculate_coefficients_for_gather_downsample( int start, int end, float in_pixels_radius, stbir__kernel_callback * kernel, stbir__scale_info * scale_info, int coefficient_width, int num_contributors, stbir__contributors * contributors, float * coefficient_group, void * user_data )
{
  int in_pixel;
  int i;
  int first_out_inited = -1;
  float scale = scale_info->scale;
  float out_shift = scale_info->pixel_shift;
  int out_size = scale_info->output_sub_size;
  int numerator = scale_info->scale_numerator;
  int polyphase = ( ( scale_info->scale_is_rational ) && ( numerator < out_size ) );

  STBIR__UNUSED(num_contributors);

  for (in_pixel = start; in_pixel < end; in_pixel++)
  {
    float in_pixel_center = (float)in_pixel + 0.5f;
    float out_center_of_in = in_pixel_center * scale - out_shift;
    int out_first_pixel, out_last_pixel;

    stbir__calculate_out_pixel_range( &out_first_pixel, &out_last_pixel, in_pixel_center, in_pixels_radius, scale, out_shift, out_size );

    if ( out_first_pixel > out_last_pixel )
      continue;

    if ( polyphase )
    {
      if ( out_first_pixel == numerator )
        break;

      if ( out_last_pixel >= numerator )
        out_last_pixel = numerator - 1;
    }

    for (i = 0; i <= out_last_pixel - out_first_pixel; i++)
    {
      float out_pixel_center = (float)(i + out_first_pixel) + 0.5f;
      float x = out_pixel_center - out_center_of_in;
      float coeff = kernel(x, scale, user_data) * scale;

      if ( ( ( coeff < stbir__small_float ) && ( coeff > -stbir__small_float ) ) )
        coeff = 0.0f;

      {
        int out = i + out_first_pixel;
        float * coeffs = coefficient_group + out * coefficient_width;
        stbir__contributors * contribs = contributors + out;

        if ( out > first_out_inited )
        {
          STBIR_ASSERT( out == ( first_out_inited + 1 ) );
          first_out_inited = out;
          contribs->n0 = in_pixel;
          contribs->n1 = in_pixel;
          coeffs[0]  = coeff;
        }
        else
        {
          if ( coeffs[0] == 0.0f )
          {
            STBIR_ASSERT( ( in_pixel - contribs->n0 ) == 1 );
            contribs->n0 = in_pixel;
          }
          contribs->n1 = in_pixel;
          STBIR_ASSERT( ( in_pixel - contribs->n0 ) < coefficient_width );
          coeffs[in_pixel - contribs->n0]  = coeff;
        }
      }
    }
  }
}

#ifdef STBIR_RENORMALIZE_IN_FLOAT
#define STBIR_RENORM_TYPE float
#else
#define STBIR_RENORM_TYPE double
#endif

static void stbir__cleanup_gathered_coefficients( stbir_edge edge, stbir__filter_extent_info* filter_info, stbir__scale_info * scale_info, int num_contributors, stbir__contributors* contributors, float * coefficient_group, int coefficient_width )
{
  int input_size = scale_info->input_full_size;
  int input_last_n1 = input_size - 1;
  int n, end;
  int lowest = 0x7fffffff;
  int highest = -0x7fffffff;
  int widest = -1;
  int numerator = scale_info->scale_numerator;
  int denominator = scale_info->scale_denominator;
  int polyphase = ( ( scale_info->scale_is_rational ) && ( numerator < num_contributors ) );
  float * coeffs;
  stbir__contributors * contribs;

  coeffs = coefficient_group;
  contribs = contributors;
  end = num_contributors; if ( polyphase ) end = numerator;
  for (n = 0; n < end; n++)
  {
    int i;
    STBIR_RENORM_TYPE filter_scale, total_filter = 0;
    int e;

    e = contribs->n1 - contribs->n0;
    for( i = 0 ; i <= e ; i++ )
    {
      total_filter += (STBIR_RENORM_TYPE) coeffs[i];
      STBIR_ASSERT( ( coeffs[i] >= -2.0f ) && ( coeffs[i] <= 2.0f )  );
    }

    if ( ( total_filter < stbir__small_float ) && ( total_filter > -stbir__small_float ) )
    {
      contribs->n1 = contribs->n0;
      coeffs[0] = 0.0f;
    }
    else
    {
      if ( ( total_filter < (1.0f-stbir__small_float) ) || ( total_filter > (1.0f+stbir__small_float) ) )
      {
        filter_scale = ((STBIR_RENORM_TYPE)1.0) / total_filter;

        for (i = 0; i <= e; i++)
          coeffs[i] = (float) ( coeffs[i] * filter_scale );
      }
    }
    ++contribs;
    coeffs += coefficient_width;
  }

  if ( polyphase )
  {
    stbir__contributors * prev_contribs = contributors;
    stbir__contributors * cur_contribs = contributors + numerator;

    for( n = numerator ; n < num_contributors ; n++ )
    {
      cur_contribs->n0 = prev_contribs->n0 + denominator;
      cur_contribs->n1 = prev_contribs->n1 + denominator;
      ++cur_contribs;
      ++prev_contribs;
    }
    stbir_overlapping_memcpy( coefficient_group + numerator * coefficient_width, coefficient_group, ( num_contributors - numerator ) * coefficient_width * sizeof( coeffs[ 0 ] ) );
  }

  coeffs = coefficient_group;
  contribs = contributors;

  for (n = 0; n < num_contributors; n++)
  {
    int i;

    if ( edge == STBIR_EDGE_ZERO )
    {
      if ( contribs->n1 > input_last_n1 )
        contribs->n1 = input_last_n1;

      if ( contribs->n0 < 0 )
      {
        int j, left, skips = 0;

        skips = -contribs->n0;
        contribs->n0 = 0;

        left = contribs->n1 - contribs->n0 + 1;
        if ( left > 0 )
        {
          for( j = 0 ; j < left ; j++ )
            coeffs[ j ] = coeffs[ j + skips ];
        }
      }
    }
    else if ( ( edge == STBIR_EDGE_CLAMP ) || ( edge == STBIR_EDGE_REFLECT ) )
    {

      if ( contribs->n1 > input_last_n1 )
      {
        int start = contribs->n0;
        int endi = contribs->n1;
        contribs->n1 = input_last_n1;
        for( i = input_size; i <= endi; i++ )
          stbir__insert_coeff( contribs, coeffs, stbir__edge_wrap_slow[edge]( i, input_size ), coeffs[i-start], coefficient_width );
      }

      if ( contribs->n0 < 0 )
      {
        int save_n0;
        float save_n0_coeff;
        float * c = coeffs - ( contribs->n0 + 1 );

        for( i = -1 ; i > contribs->n0 ; i-- )
          stbir__insert_coeff( contribs, coeffs, stbir__edge_wrap_slow[edge]( i, input_size ), *c--, coefficient_width );
        save_n0 = contribs->n0;
        save_n0_coeff = c[0];

        contribs->n0 = 0;
        for(i = 0 ; i <= contribs->n1 ; i++ )
          coeffs[i] = coeffs[i-save_n0];

        stbir__insert_coeff( contribs, coeffs, stbir__edge_wrap_slow[edge]( save_n0, input_size ), save_n0_coeff, coefficient_width );
      }
    }

    if ( contribs->n0 <= contribs->n1 )
    {
      int diff = contribs->n1 - contribs->n0 + 1;
      while ( diff && ( coeffs[ diff-1 ] == 0.0f ) )
        --diff;

      contribs->n1 = contribs->n0 + diff - 1;

      if ( contribs->n0 <= contribs->n1 )
      {
        if ( contribs->n0 < lowest )
          lowest = contribs->n0;
        if ( contribs->n1 > highest )
          highest = contribs->n1;
        if ( diff > widest )
          widest = diff;
      }

      for( i = diff ; i < coefficient_width ; i++ )
        coeffs[i] = 0.0f;
    }

    ++contribs;
    coeffs += coefficient_width;
  }
  filter_info->lowest = lowest;
  filter_info->highest = highest;
  filter_info->widest = widest;
}

#undef STBIR_RENORM_TYPE

static int stbir__pack_coefficients( int num_contributors, stbir__contributors* contributors, float * coefficents, int coefficient_width, int widest, int row0, int row1 )
{
  #define STBIR_MOVE_1( dest, src ) { STBIR_NO_UNROLL(dest); ((stbir_uint32*)(dest))[0] = ((stbir_uint32*)(src))[0]; }
  #define STBIR_MOVE_2( dest, src ) { STBIR_NO_UNROLL(dest); ((stbir_uint64*)(dest))[0] = ((stbir_uint64*)(src))[0]; }
  #ifdef STBIR_SIMD
  #define STBIR_MOVE_4( dest, src ) { stbir__simdf t; STBIR_NO_UNROLL(dest); stbir__simdf_load( t, src ); stbir__simdf_store( dest, t ); }
  #else
  #define STBIR_MOVE_4( dest, src ) { STBIR_NO_UNROLL(dest); ((stbir_uint64*)(dest))[0] = ((stbir_uint64*)(src))[0]; ((stbir_uint64*)(dest))[1] = ((stbir_uint64*)(src))[1]; }
  #endif

  int row_end = row1 + 1;
  STBIR__UNUSED( row0 );

  if ( coefficient_width != widest )
  {
    float * pc = coefficents;
    float * coeffs = coefficents;
    float * pc_end = coefficents + num_contributors * widest;
    switch( widest )
    {
      case 1:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_1( pc, coeffs );
          ++pc;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 2:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_2( pc, coeffs );
          pc += 2;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 3:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_2( pc, coeffs );
          STBIR_MOVE_1( pc+2, coeffs+2 );
          pc += 3;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 4:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          pc += 4;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 5:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_1( pc+4, coeffs+4 );
          pc += 5;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 6:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_2( pc+4, coeffs+4 );
          pc += 6;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 7:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_2( pc+4, coeffs+4 );
          STBIR_MOVE_1( pc+6, coeffs+6 );
          pc += 7;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 8:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_4( pc+4, coeffs+4 );
          pc += 8;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 9:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_4( pc+4, coeffs+4 );
          STBIR_MOVE_1( pc+8, coeffs+8 );
          pc += 9;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 10:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_4( pc+4, coeffs+4 );
          STBIR_MOVE_2( pc+8, coeffs+8 );
          pc += 10;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 11:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_4( pc+4, coeffs+4 );
          STBIR_MOVE_2( pc+8, coeffs+8 );
          STBIR_MOVE_1( pc+10, coeffs+10 );
          pc += 11;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      case 12:
        STBIR_NO_UNROLL_LOOP_START
        do {
          STBIR_MOVE_4( pc, coeffs );
          STBIR_MOVE_4( pc+4, coeffs+4 );
          STBIR_MOVE_4( pc+8, coeffs+8 );
          pc += 12;
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
      default:
        STBIR_NO_UNROLL_LOOP_START
        do {
          float * copy_end = pc + widest - 4;
          float * c = coeffs;
          do {
            STBIR_NO_UNROLL( pc );
            STBIR_MOVE_4( pc, c );
            pc += 4;
            c += 4;
          } while ( pc <= copy_end );
          copy_end += 4;
          STBIR_NO_UNROLL_LOOP_START
          while ( pc < copy_end )
          {
            STBIR_MOVE_1( pc, c );
            ++pc; ++c;
          }
          coeffs += coefficient_width;
        } while ( pc < pc_end );
        break;
    }
  }

  coefficents[ widest * num_contributors ] = 8888.0f;

  {
    stbir__contributors * contribs = contributors + num_contributors - 1;
    float * coeffs = coefficents + widest * ( num_contributors - 1 );

    while ( ( contribs >= contributors ) && ( ( contribs->n0 + widest*2 ) >= row_end ) )
    {
      if ( ( contribs->n0 + widest ) > row_end )
      {
        int stop_range = widest;

        if ( widest > 12 )
        {
          int mod;

          mod = widest & 3;
          stop_range = ( ( ( contribs->n1 - contribs->n0 + 1 ) - mod + 3 ) & ~3 ) + mod;

          if ( stop_range < ( 8 + mod ) ) stop_range = 8 + mod;
        }

        if ( ( contribs->n0 + stop_range ) > row_end )
        {
          int new_n0 = row_end - stop_range;
          int num = contribs->n1 - contribs->n0 + 1;
          int backup = contribs->n0 - new_n0;
          float * from_co = coeffs + num - 1;
          float * to_co = from_co + backup;

          STBIR_ASSERT( ( new_n0 >= row0 ) && ( new_n0 < contribs->n0 ) );

          while( num )
          {
            *to_co-- = *from_co--;
            --num;
          }
          while ( to_co >= coeffs )
            *to_co-- = 0;
          contribs->n0 = new_n0;
          if ( widest > 12 )
          {
            int mod;

            mod = widest & 3;
            stop_range = ( ( ( contribs->n1 - contribs->n0 + 1 ) - mod + 3 ) & ~3 ) + mod;

            if ( stop_range < ( 8 + mod ) ) stop_range = 8 + mod;
          }
        }
      }
      --contribs;
      coeffs -= widest;
    }
  }

  return widest;
  #undef STBIR_MOVE_1
  #undef STBIR_MOVE_2
  #undef STBIR_MOVE_4
}

static void stbir__calculate_filters( stbir__sampler * samp, stbir__sampler * other_axis_for_pivot, void * user_data STBIR_ONLY_PROFILE_BUILD_GET_INFO )
{
  int n;
  float scale = samp->scale_info.scale;
  stbir__kernel_callback * kernel = samp->filter_kernel;
  stbir__support_callback * support = samp->filter_support;
  float inv_scale = samp->scale_info.inv_scale;
  int input_full_size = samp->scale_info.input_full_size;
  int gather_num_contributors = samp->num_contributors;
  stbir__contributors* gather_contributors = samp->contributors;
  float * gather_coeffs = samp->coefficients;
  int gather_coefficient_width = samp->coefficient_width;

  switch ( samp->is_gather )
  {
    case 1:
    {
      float out_pixels_radius = support(inv_scale,user_data) * scale;

      stbir__calculate_coefficients_for_gather_upsample( out_pixels_radius, kernel, &samp->scale_info, gather_num_contributors, gather_contributors, gather_coeffs, gather_coefficient_width, samp->edge, user_data );

      STBIR_PROFILE_BUILD_START( cleanup );
      stbir__cleanup_gathered_coefficients( samp->edge, &samp->extent_info, &samp->scale_info, gather_num_contributors, gather_contributors, gather_coeffs, gather_coefficient_width );
      STBIR_PROFILE_BUILD_END( cleanup );
    }
    break;

    case 0:
    case 2:
    {
      float in_pixels_radius = support(scale,user_data) * inv_scale;
      int filter_pixel_margin = samp->filter_pixel_margin;
      int input_end = input_full_size + filter_pixel_margin;

      if ( !samp->is_gather )
      {
        if ( other_axis_for_pivot )
        {
          gather_contributors = other_axis_for_pivot->contributors;
          gather_coeffs = other_axis_for_pivot->coefficients;
          gather_coefficient_width = other_axis_for_pivot->coefficient_width;
          gather_num_contributors = other_axis_for_pivot->num_contributors;
          samp->extent_info.lowest = other_axis_for_pivot->extent_info.lowest;
          samp->extent_info.highest = other_axis_for_pivot->extent_info.highest;
          samp->extent_info.widest = other_axis_for_pivot->extent_info.widest;
          goto jump_right_to_pivot;
        }

        gather_contributors = samp->gather_prescatter_contributors;
        gather_coeffs = samp->gather_prescatter_coefficients;
        gather_coefficient_width = samp->gather_prescatter_coefficient_width;
        gather_num_contributors = samp->gather_prescatter_num_contributors;
      }

      stbir__calculate_coefficients_for_gather_downsample( -filter_pixel_margin, input_end, in_pixels_radius, kernel, &samp->scale_info, gather_coefficient_width, gather_num_contributors, gather_contributors, gather_coeffs, user_data );

      STBIR_PROFILE_BUILD_START( cleanup );
      stbir__cleanup_gathered_coefficients( samp->edge, &samp->extent_info, &samp->scale_info, gather_num_contributors, gather_contributors, gather_coeffs, gather_coefficient_width );
      STBIR_PROFILE_BUILD_END( cleanup );

      if ( !samp->is_gather )
      {
        stbir__contributors * scatter_contributors;
        int highest_set;

        jump_right_to_pivot:

        STBIR_PROFILE_BUILD_START( pivot );

        highest_set = (-filter_pixel_margin) - 1;
        for (n = 0; n < gather_num_contributors; n++)
        {
          int k;
          int gn0 = gather_contributors->n0, gn1 = gather_contributors->n1;
          int scatter_coefficient_width = samp->coefficient_width;
          float * scatter_coeffs = samp->coefficients + ( gn0 + filter_pixel_margin ) * scatter_coefficient_width;
          float * g_coeffs = gather_coeffs;
          scatter_contributors = samp->contributors + ( gn0 + filter_pixel_margin );

          for (k = gn0 ; k <= gn1 ; k++ )
          {
            float gc = *g_coeffs++;

            if ( ( ( gc >= stbir__small_float ) || ( gc <= -stbir__small_float ) ) )
            {
              if ( ( k > highest_set ) || ( scatter_contributors->n0 > scatter_contributors->n1 ) )
              {
                {
                  stbir__contributors * clear_contributors = samp->contributors + ( highest_set + filter_pixel_margin + 1);
                  while ( clear_contributors < scatter_contributors )
                  {
                    clear_contributors->n0 = 0;
                    clear_contributors->n1 = -1;
                    ++clear_contributors;
                  }
                }
                scatter_contributors->n0 = n;
                scatter_contributors->n1 = n;
                scatter_coeffs[0]  = gc;
                highest_set = k;
              }
              else
              {
                stbir__insert_coeff( scatter_contributors, scatter_coeffs, n, gc, scatter_coefficient_width );
              }
              STBIR_ASSERT( ( scatter_contributors->n1 - scatter_contributors->n0 + 1 ) <= scatter_coefficient_width );
            }
            ++scatter_contributors;
            scatter_coeffs += scatter_coefficient_width;
          }

          ++gather_contributors;
          gather_coeffs += gather_coefficient_width;
        }

        {
          stbir__contributors * clear_contributors = samp->contributors + ( highest_set + filter_pixel_margin + 1);
          stbir__contributors * end_contributors = samp->contributors + samp->num_contributors;
          while ( clear_contributors < end_contributors )
          {
            clear_contributors->n0 = 0;
            clear_contributors->n1 = -1;
            ++clear_contributors;
          }
        }

        STBIR_PROFILE_BUILD_END( pivot );
      }
    }
    break;
  }
}

#define stbir__coder_min_num 1
#define STB_IMAGE_RESIZE_DO_CODERS
#include STBIR__HEADER_FILENAME

#define stbir__decode_suffix BGRA
#define stbir__decode_swizzle
#define stbir__decode_order0  2
#define stbir__decode_order1  1
#define stbir__decode_order2  0
#define stbir__decode_order3  3
#define stbir__encode_order0  2
#define stbir__encode_order1  1
#define stbir__encode_order2  0
#define stbir__encode_order3  3
#define stbir__coder_min_num 4
#define STB_IMAGE_RESIZE_DO_CODERS
#include STBIR__HEADER_FILENAME

#define stbir__decode_suffix ARGB
#define stbir__decode_swizzle
#define stbir__decode_order0  1
#define stbir__decode_order1  2
#define stbir__decode_order2  3
#define stbir__decode_order3  0
#define stbir__encode_order0  3
#define stbir__encode_order1  0
#define stbir__encode_order2  1
#define stbir__encode_order3  2
#define stbir__coder_min_num 4
#define STB_IMAGE_RESIZE_DO_CODERS
#include STBIR__HEADER_FILENAME

#define stbir__decode_suffix ABGR
#define stbir__decode_swizzle
#define stbir__decode_order0  3
#define stbir__decode_order1  2
#define stbir__decode_order2  1
#define stbir__decode_order3  0
#define stbir__encode_order0  3
#define stbir__encode_order1  2
#define stbir__encode_order2  1
#define stbir__encode_order3  0
#define stbir__coder_min_num 4
#define STB_IMAGE_RESIZE_DO_CODERS
#include STBIR__HEADER_FILENAME

#define stbir__decode_suffix AR
#define stbir__decode_swizzle
#define stbir__decode_order0  1
#define stbir__decode_order1  0
#define stbir__decode_order2  3
#define stbir__decode_order3  2
#define stbir__encode_order0  1
#define stbir__encode_order1  0
#define stbir__encode_order2  3
#define stbir__encode_order3  2
#define stbir__coder_min_num 2
#define STB_IMAGE_RESIZE_DO_CODERS
#include STBIR__HEADER_FILENAME

static void stbir__fancy_alpha_weight_4ch( float * out_buffer, int width_times_channels )
{
  float STBIR_STREAMOUT_PTR(*) out = out_buffer;
  float const * end_decode = out_buffer + ( width_times_channels / 4 ) * 7;
  float STBIR_STREAMOUT_PTR(*) decode = (float*)end_decode - width_times_channels;

  #ifdef STBIR_SIMD

  #ifdef STBIR_SIMD8
  decode += 16;
  STBIR_NO_UNROLL_LOOP_START
  while ( decode <= end_decode )
  {
    stbir__simdf8 d0,d1,a0,a1,p0,p1;
    STBIR_NO_UNROLL(decode);
    stbir__simdf8_load( d0, decode-16 );
    stbir__simdf8_load( d1, decode-16+8 );
    stbir__simdf8_0123to33333333( a0, d0 );
    stbir__simdf8_0123to33333333( a1, d1 );
    stbir__simdf8_mult( p0, a0, d0 );
    stbir__simdf8_mult( p1, a1, d1 );
    stbir__simdf8_bot4s( a0, d0, p0 );
    stbir__simdf8_bot4s( a1, d1, p1 );
    stbir__simdf8_top4s( d0, d0, p0 );
    stbir__simdf8_top4s( d1, d1, p1 );
    stbir__simdf8_store ( out, a0 );
    stbir__simdf8_store ( out+7, d0 );
    stbir__simdf8_store ( out+14, a1 );
    stbir__simdf8_store ( out+21, d1 );
    decode += 16;
    out += 28;
  }
  decode -= 16;
  #else
  decode += 8;
  STBIR_NO_UNROLL_LOOP_START
  while ( decode <= end_decode )
  {
    stbir__simdf d0,a0,d1,a1,p0,p1;
    STBIR_NO_UNROLL(decode);
    stbir__simdf_load( d0, decode-8 );
    stbir__simdf_load( d1, decode-8+4 );
    stbir__simdf_0123to3333( a0, d0 );
    stbir__simdf_0123to3333( a1, d1 );
    stbir__simdf_mult( p0, a0, d0 );
    stbir__simdf_mult( p1, a1, d1 );
    stbir__simdf_store ( out, d0 );
    stbir__simdf_store ( out+4, p0 );
    stbir__simdf_store ( out+7, d1 );
    stbir__simdf_store ( out+7+4, p1 );
    decode += 8;
    out += 14;
  }
  decode -= 8;
  #endif

  #ifdef STBIR_SIMD8
  STBIR_NO_UNROLL_LOOP_START
  while ( decode < end_decode )
  #else
  if ( decode < end_decode )
  #endif
  {
    stbir__simdf d,a,p;
    STBIR_NO_UNROLL(decode);
    stbir__simdf_load( d, decode );
    stbir__simdf_0123to3333( a, d );
    stbir__simdf_mult( p, a, d );
    stbir__simdf_store ( out, d );
    stbir__simdf_store ( out+4, p );
    decode += 4;
    out += 7;
  }

  #else

  while( decode < end_decode )
  {
    float r = decode[0], g = decode[1], b = decode[2], alpha = decode[3];
    out[0] = r;
    out[1] = g;
    out[2] = b;
    out[3] = alpha;
    out[4] = r * alpha;
    out[5] = g * alpha;
    out[6] = b * alpha;
    out += 7;
    decode += 4;
  }

  #endif
}

static void stbir__fancy_alpha_weight_2ch( float * out_buffer, int width_times_channels )
{
  float STBIR_STREAMOUT_PTR(*) out = out_buffer;
  float const * end_decode = out_buffer + ( width_times_channels / 2 ) * 3;
  float STBIR_STREAMOUT_PTR(*) decode = (float*)end_decode - width_times_channels;

  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( decode < end_decode )
  {
    float x = decode[0], y = decode[1];
    STBIR_SIMD_NO_UNROLL(decode);
    out[0] = x;
    out[1] = y;
    out[2] = x * y;
    out += 3;
    decode += 2;
  }
}

static void stbir__fancy_alpha_unweight_4ch( float * encode_buffer, int width_times_channels )
{
  float STBIR_SIMD_STREAMOUT_PTR(*) encode = encode_buffer;
  float STBIR_SIMD_STREAMOUT_PTR(*) input = encode_buffer;
  float const * end_output = encode_buffer + width_times_channels;

  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float alpha = input[3];
#ifdef STBIR_SIMD
    stbir__simdf i,ia;
    STBIR_SIMD_NO_UNROLL(encode);
    if ( alpha < stbir__small_float )
    {
      stbir__simdf_load( i, input );
      stbir__simdf_store( encode, i );
    }
    else
    {
      stbir__simdf_load1frep4( ia, 1.0f / alpha );
      stbir__simdf_load( i, input+4 );
      stbir__simdf_mult( i, i, ia );
      stbir__simdf_store( encode, i );
      encode[3] = alpha;
    }
#else
    if ( alpha < stbir__small_float )
    {
      encode[0] = input[0];
      encode[1] = input[1];
      encode[2] = input[2];
    }
    else
    {
      float ialpha = 1.0f / alpha;
      encode[0] = input[4] * ialpha;
      encode[1] = input[5] * ialpha;
      encode[2] = input[6] * ialpha;
    }
    encode[3] = alpha;
#endif

    input += 7;
    encode += 4;
  } while ( encode < end_output );
}

static void stbir__fancy_alpha_unweight_2ch( float * encode_buffer, int width_times_channels )
{
  float STBIR_SIMD_STREAMOUT_PTR(*) encode = encode_buffer;
  float STBIR_SIMD_STREAMOUT_PTR(*) input = encode_buffer;
  float const * end_output = encode_buffer + width_times_channels;

  do {
    float alpha = input[1];
    encode[0] = input[0];
    if ( alpha >= stbir__small_float )
      encode[0] = input[2] / alpha;
    encode[1] = alpha;

    input += 3;
    encode += 2;
  } while ( encode < end_output );
}

static void stbir__simple_alpha_weight_4ch( float * decode_buffer, int width_times_channels )
{
  float STBIR_STREAMOUT_PTR(*) decode = decode_buffer;
  float const * end_decode = decode_buffer + width_times_channels;

  #ifdef STBIR_SIMD
  {
    decode += 2 * stbir__simdfX_float_count;
    STBIR_NO_UNROLL_LOOP_START
    while ( decode <= end_decode )
    {
      stbir__simdfX d0,a0,d1,a1;
      STBIR_NO_UNROLL(decode);
      stbir__simdfX_load( d0, decode-2*stbir__simdfX_float_count );
      stbir__simdfX_load( d1, decode-2*stbir__simdfX_float_count+stbir__simdfX_float_count );
      stbir__simdfX_aaa1( a0, d0, STBIR_onesX );
      stbir__simdfX_aaa1( a1, d1, STBIR_onesX );
      stbir__simdfX_mult( d0, d0, a0 );
      stbir__simdfX_mult( d1, d1, a1 );
      stbir__simdfX_store ( decode-2*stbir__simdfX_float_count, d0 );
      stbir__simdfX_store ( decode-2*stbir__simdfX_float_count+stbir__simdfX_float_count, d1 );
      decode += 2 * stbir__simdfX_float_count;
    }
    decode -= 2 * stbir__simdfX_float_count;

    #ifdef STBIR_SIMD8
    STBIR_NO_UNROLL_LOOP_START
    while ( decode < end_decode )
    #else
    if ( decode < end_decode )
    #endif
    {
      stbir__simdf d,a;
      stbir__simdf_load( d, decode );
      stbir__simdf_aaa1( a, d, STBIR__CONSTF(STBIR_ones) );
      stbir__simdf_mult( d, d, a );
      stbir__simdf_store ( decode, d );
      decode += 4;
    }
  }

  #else

  while( decode < end_decode )
  {
    float alpha = decode[3];
    decode[0] *= alpha;
    decode[1] *= alpha;
    decode[2] *= alpha;
    decode += 4;
  }

  #endif
}

static void stbir__simple_alpha_weight_2ch( float * decode_buffer, int width_times_channels )
{
  float STBIR_STREAMOUT_PTR(*) decode = decode_buffer;
  float const * end_decode = decode_buffer + width_times_channels;

  STBIR_SIMD_NO_UNROLL_LOOP_START
  while( decode < end_decode )
  {
    float alpha = decode[1];
    STBIR_SIMD_NO_UNROLL(decode);
    decode[0] *= alpha;
    decode += 2;
  }
}

static void stbir__simple_alpha_unweight_4ch( float * encode_buffer, int width_times_channels )
{
  float STBIR_SIMD_STREAMOUT_PTR(*) encode = encode_buffer;
  float const * end_output = encode_buffer + width_times_channels;

  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float alpha = encode[3];

#ifdef STBIR_SIMD
    stbir__simdf i,ia;
    STBIR_SIMD_NO_UNROLL(encode);
    if ( alpha >= stbir__small_float )
    {
      stbir__simdf_load1frep4( ia, 1.0f / alpha );
      stbir__simdf_load( i, encode );
      stbir__simdf_mult( i, i, ia );
      stbir__simdf_store( encode, i );
      encode[3] = alpha;
    }
#else
    if ( alpha >= stbir__small_float )
    {
      float ialpha = 1.0f / alpha;
      encode[0] *= ialpha;
      encode[1] *= ialpha;
      encode[2] *= ialpha;
    }
#endif
    encode += 4;
  } while ( encode < end_output );
}

static void stbir__simple_alpha_unweight_2ch( float * encode_buffer, int width_times_channels )
{
  float STBIR_SIMD_STREAMOUT_PTR(*) encode = encode_buffer;
  float const * end_output = encode_buffer + width_times_channels;

  do {
    float alpha = encode[1];
    if ( alpha >= stbir__small_float )
      encode[0] /= alpha;
    encode += 2;
  } while ( encode < end_output );
}

static void stbir__simple_flip_3ch( float * decode_buffer, int width_times_channels )
{
  float STBIR_STREAMOUT_PTR(*) decode = decode_buffer;
  float const * end_decode = decode_buffer + width_times_channels;

#ifdef STBIR_SIMD
    #ifdef stbir__simdf_swiz2
      end_decode -= 12;
      STBIR_NO_UNROLL_LOOP_START
      while( decode <= end_decode )
      {
        stbir__simdf a,b,c,na,nb;
        STBIR_SIMD_NO_UNROLL(decode);
        stbir__simdf_load( a, decode );
        stbir__simdf_load( b, decode+4 );
        stbir__simdf_load( c, decode+8 );

        na = stbir__simdf_swiz2( a, b, 2, 1, 0, 5 );
        b  = stbir__simdf_swiz2( a, b, 4, 3, 6, 7 );
        nb = stbir__simdf_swiz2( b, c, 0, 1, 4, 3 );
        c  = stbir__simdf_swiz2( b, c, 2, 7, 6, 5 );

        stbir__simdf_store( decode, na );
        stbir__simdf_store( decode+4, nb );
        stbir__simdf_store( decode+8, c );
        decode += 12;
      }
      end_decode += 12;
    #else
      end_decode -= 24;
      STBIR_NO_UNROLL_LOOP_START
      while( decode <= end_decode )
      {
        stbir__simdf a,b,c,d,e,f,g;
        float i21, i23;
        STBIR_SIMD_NO_UNROLL(decode);
        stbir__simdf_load( a, decode );
        stbir__simdf_load( b, decode+3 );
        stbir__simdf_load( c, decode+6 );
        stbir__simdf_load( d, decode+9 );
        stbir__simdf_load( e, decode+12 );
        stbir__simdf_load( f, decode+15 );
        stbir__simdf_load( g, decode+18 );

        a = stbir__simdf_swiz( a, 2, 1, 0, 3 );
        b = stbir__simdf_swiz( b, 2, 1, 0, 3 );
        c = stbir__simdf_swiz( c, 2, 1, 0, 3 );
        d = stbir__simdf_swiz( d, 2, 1, 0, 3 );
        e = stbir__simdf_swiz( e, 2, 1, 0, 3 );
        f = stbir__simdf_swiz( f, 2, 1, 0, 3 );
        g = stbir__simdf_swiz( g, 2, 1, 0, 3 );

        stbir__simdf_store( decode,    a );
        i21 = decode[21];
        stbir__simdf_store( decode+3,  b );
        i23 = decode[23];
        stbir__simdf_store( decode+6,  c );
        stbir__simdf_store( decode+9,  d );
        stbir__simdf_store( decode+12, e );
        stbir__simdf_store( decode+15, f );
        stbir__simdf_store( decode+18, g );
        decode[21] = i23;
        decode[23] = i21;
        decode += 24;
      }
      end_decode += 24;
    #endif
#else
  end_decode -= 12;
  STBIR_NO_UNROLL_LOOP_START
  while( decode <= end_decode )
  {
    float t0,t1,t2,t3;
    STBIR_NO_UNROLL(decode);
    t0 = decode[0]; t1 = decode[3]; t2 = decode[6]; t3 = decode[9];
    decode[0] = decode[2]; decode[3] = decode[5]; decode[6] = decode[8]; decode[9] = decode[11];
    decode[2] = t0; decode[5] = t1; decode[8] = t2; decode[11] = t3;
    decode += 12;
  }
  end_decode += 12;
#endif

  STBIR_NO_UNROLL_LOOP_START
  while( decode < end_decode )
  {
    float t = decode[0];
    STBIR_NO_UNROLL(decode);
    decode[0] = decode[2];
    decode[2] = t;
    decode += 3;
  }
}

static void stbir__decode_scanline(stbir__info const * stbir_info, int n, float * output_buffer STBIR_ONLY_PROFILE_GET_SPLIT_INFO )
{
  int channels = stbir_info->channels;
  int effective_channels = stbir_info->effective_channels;
  int input_sample_in_bytes = stbir__type_size[stbir_info->input_type] * channels;
  stbir_edge edge_horizontal = stbir_info->horizontal.edge;
  stbir_edge edge_vertical = stbir_info->vertical.edge;
  int row = stbir__edge_wrap(edge_vertical, n, stbir_info->vertical.scale_info.input_full_size);
  const void* input_plane_data = ( (char *) stbir_info->input_data ) + (size_t)row * (size_t) stbir_info->input_stride_bytes;
  stbir__span const * spans = stbir_info->scanline_extents.spans;
  float * full_decode_buffer = output_buffer - stbir_info->scanline_extents.conservative.n0 * effective_channels;
  float * last_decoded = 0;

  STBIR_ASSERT( !(edge_vertical == STBIR_EDGE_ZERO && (n < 0 || n >= stbir_info->vertical.scale_info.input_full_size)) );

  do
  {
    float * decode_buffer;
    void const * input_data;
    float * end_decode;
    int width_times_channels;
    int width;

    if ( spans->n1 < spans->n0 )
      break;

    width = spans->n1 + 1 - spans->n0;
    decode_buffer = full_decode_buffer + spans->n0 * effective_channels;
    end_decode = full_decode_buffer + ( spans->n1 + 1 ) * effective_channels;
    width_times_channels = width * channels;

    input_data = ( (char*)input_plane_data ) + spans->pixel_offset_for_input * input_sample_in_bytes;

    if ( stbir_info->in_pixels_cb )
    {
      input_data = stbir_info->in_pixels_cb( ( (char*) end_decode ) - ( width * input_sample_in_bytes ) + ( ( stbir_info->input_type != STBIR_TYPE_FLOAT ) ? ( sizeof(float)*STBIR_INPUT_CALLBACK_PADDING ) : 0 ), input_plane_data, width, spans->pixel_offset_for_input, row, stbir_info->user_data );
    }

    STBIR_PROFILE_START( decode );
    last_decoded = stbir_info->decode_pixels( (float*)end_decode - width_times_channels, width_times_channels, input_data );
    STBIR_PROFILE_END( decode );

    if (stbir_info->alpha_weight)
    {
      STBIR_PROFILE_START( alpha );
      stbir_info->alpha_weight( decode_buffer, width_times_channels );
      STBIR_PROFILE_END( alpha );
    }

    ++spans;
  } while ( spans <= ( &stbir_info->scanline_extents.spans[1] ) );

  if ( ( edge_horizontal == STBIR_EDGE_WRAP ) && ( stbir_info->scanline_extents.edge_sizes[0] | stbir_info->scanline_extents.edge_sizes[1] ) )
  {
    int e, start_x[2];
    int input_full_size = stbir_info->horizontal.scale_info.input_full_size;

    start_x[0] = -stbir_info->scanline_extents.edge_sizes[0];
    start_x[1] =  input_full_size;

    for( e = 0; e < 2 ; e++ )
    {
      int margin = stbir_info->scanline_extents.edge_sizes[e];
      if ( margin )
      {
        int x = start_x[e];
        float * marg = full_decode_buffer + x * effective_channels;
        float const * src = full_decode_buffer + stbir__edge_wrap(edge_horizontal, x, input_full_size) * effective_channels;
        STBIR_MEMCPY( marg, src, margin * effective_channels * sizeof(float) );
        if ( e == 1 ) last_decoded = marg + margin * effective_channels;
      }
    }
  }

  last_decoded[0] = 0.0f;

  last_decoded[1] = 0.0f;
}

#ifdef STBIR_SIMD

#define stbir__1_coeff_only()          \
    stbir__simdf tot,c;                \
    STBIR_SIMD_NO_UNROLL(decode);      \
    stbir__simdf_load1( c, hc );       \
    stbir__simdf_mult1_mem( tot, c, decode );

#define stbir__2_coeff_only()          \
    stbir__simdf tot,c,d;              \
    STBIR_SIMD_NO_UNROLL(decode);      \
    stbir__simdf_load2z( c, hc );      \
    stbir__simdf_load2( d, decode );   \
    stbir__simdf_mult( tot, c, d );    \
    stbir__simdf_0123to1230( c, tot ); \
    stbir__simdf_add1( tot, tot, c );

#define stbir__3_coeff_only()                  \
    stbir__simdf tot,c,t;                      \
    STBIR_SIMD_NO_UNROLL(decode);              \
    stbir__simdf_load( c, hc );                \
    stbir__simdf_mult_mem( tot, c, decode );   \
    stbir__simdf_0123to1230( c, tot );         \
    stbir__simdf_0123to2301( t, tot );         \
    stbir__simdf_add1( tot, tot, c );          \
    stbir__simdf_add1( tot, tot, t );

#define stbir__store_output_tiny()                \
    stbir__simdf_store1( output, tot );           \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 1;

#define stbir__4_coeff_start()                 \
    stbir__simdf tot,c;                        \
    STBIR_SIMD_NO_UNROLL(decode);              \
    stbir__simdf_load( c, hc );                \
    stbir__simdf_mult_mem( tot, c, decode );   \

#define stbir__4_coeff_continue_from_4( ofs )  \
    STBIR_SIMD_NO_UNROLL(decode);              \
    stbir__simdf_load( c, hc + (ofs) );        \
    stbir__simdf_madd_mem( tot, tot, c, decode+(ofs) );

#define stbir__1_coeff_remnant( ofs )          \
    { stbir__simdf d;                          \
    stbir__simdf_load1z( c, hc + (ofs) );      \
    stbir__simdf_load1( d, decode + (ofs) );   \
    stbir__simdf_madd( tot, tot, d, c ); }

#define stbir__2_coeff_remnant( ofs )          \
    { stbir__simdf d;                          \
    stbir__simdf_load2z( c, hc+(ofs) );        \
    stbir__simdf_load2( d, decode+(ofs) );     \
    stbir__simdf_madd( tot, tot, d, c ); }

#define stbir__3_coeff_setup()                 \
    stbir__simdf mask;                         \
    stbir__simdf_load( mask, STBIR_mask + 3 );

#define stbir__3_coeff_remnant( ofs )                  \
    stbir__simdf_load( c, hc+(ofs) );                  \
    stbir__simdf_and( c, c, mask );                    \
    stbir__simdf_madd_mem( tot, tot, c, decode+(ofs) );

#define stbir__store_output()                     \
    stbir__simdf_0123to2301( c, tot );            \
    stbir__simdf_add( tot, tot, c );              \
    stbir__simdf_0123to1230( c, tot );            \
    stbir__simdf_add1( tot, tot, c );             \
    stbir__simdf_store1( output, tot );           \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 1;

#else

#define stbir__1_coeff_only()  \
    float tot;                 \
    tot = decode[0]*hc[0];

#define stbir__2_coeff_only()  \
    float tot;                 \
    tot = decode[0] * hc[0];   \
    tot += decode[1] * hc[1];

#define stbir__3_coeff_only()  \
    float tot;                 \
    tot = decode[0] * hc[0];   \
    tot += decode[1] * hc[1];  \
    tot += decode[2] * hc[2];

#define stbir__store_output_tiny()                \
    output[0] = tot;                              \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 1;

#define stbir__4_coeff_start()  \
    float tot0,tot1,tot2,tot3;  \
    tot0 = decode[0] * hc[0];   \
    tot1 = decode[1] * hc[1];   \
    tot2 = decode[2] * hc[2];   \
    tot3 = decode[3] * hc[3];

#define stbir__4_coeff_continue_from_4( ofs )  \
    tot0 += decode[0+(ofs)] * hc[0+(ofs)];     \
    tot1 += decode[1+(ofs)] * hc[1+(ofs)];     \
    tot2 += decode[2+(ofs)] * hc[2+(ofs)];     \
    tot3 += decode[3+(ofs)] * hc[3+(ofs)];

#define stbir__1_coeff_remnant( ofs )        \
    tot0 += decode[0+(ofs)] * hc[0+(ofs)];

#define stbir__2_coeff_remnant( ofs )        \
    tot0 += decode[0+(ofs)] * hc[0+(ofs)];   \
    tot1 += decode[1+(ofs)] * hc[1+(ofs)];   \

#define stbir__3_coeff_remnant( ofs )        \
    tot0 += decode[0+(ofs)] * hc[0+(ofs)];   \
    tot1 += decode[1+(ofs)] * hc[1+(ofs)];   \
    tot2 += decode[2+(ofs)] * hc[2+(ofs)];

#define stbir__store_output()                     \
    output[0] = (tot0+tot2)+(tot1+tot3);          \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 1;

#endif

#define STBIR__horizontal_channels 1
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME

#ifdef STBIR_SIMD

#define stbir__1_coeff_only()         \
    stbir__simdf tot,c,d;             \
    STBIR_SIMD_NO_UNROLL(decode);     \
    stbir__simdf_load1z( c, hc );     \
    stbir__simdf_0123to0011( c, c );  \
    stbir__simdf_load2( d, decode );  \
    stbir__simdf_mult( tot, d, c );

#define stbir__2_coeff_only()         \
    stbir__simdf tot,c;               \
    STBIR_SIMD_NO_UNROLL(decode);     \
    stbir__simdf_load2( c, hc );      \
    stbir__simdf_0123to0011( c, c );  \
    stbir__simdf_mult_mem( tot, c, decode );

#define stbir__3_coeff_only()                \
    stbir__simdf tot,c,cs,d;                 \
    STBIR_SIMD_NO_UNROLL(decode);            \
    stbir__simdf_load( cs, hc );             \
    stbir__simdf_0123to0011( c, cs );        \
    stbir__simdf_mult_mem( tot, c, decode ); \
    stbir__simdf_0123to2222( c, cs );        \
    stbir__simdf_load2z( d, decode+4 );      \
    stbir__simdf_madd( tot, tot, d, c );

#define stbir__store_output_tiny()                \
    stbir__simdf_0123to2301( c, tot );            \
    stbir__simdf_add( tot, tot, c );              \
    stbir__simdf_store2( output, tot );           \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 2;

#ifdef STBIR_SIMD8

#define stbir__4_coeff_start()                    \
    stbir__simdf8 tot0,c,cs;                      \
    STBIR_SIMD_NO_UNROLL(decode);                 \
    stbir__simdf8_load4b( cs, hc );               \
    stbir__simdf8_0123to00112233( c, cs );        \
    stbir__simdf8_mult_mem( tot0, c, decode );

#define stbir__4_coeff_continue_from_4( ofs )        \
    STBIR_SIMD_NO_UNROLL(decode);                    \
    stbir__simdf8_load4b( cs, hc + (ofs) );          \
    stbir__simdf8_0123to00112233( c, cs );           \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*2 );

#define stbir__1_coeff_remnant( ofs )                \
    { stbir__simdf t,d;                              \
    stbir__simdf_load1z( t, hc + (ofs) );            \
    stbir__simdf_load2( d, decode + (ofs) * 2 );     \
    stbir__simdf_0123to0011( t, t );                 \
    stbir__simdf_mult( t, t, d );                    \
    stbir__simdf8_add4( tot0, tot0, t ); }

#define stbir__2_coeff_remnant( ofs )                \
    { stbir__simdf t;                                \
    stbir__simdf_load2( t, hc + (ofs) );             \
    stbir__simdf_0123to0011( t, t );                 \
    stbir__simdf_mult_mem( t, t, decode+(ofs)*2 );   \
    stbir__simdf8_add4( tot0, tot0, t ); }

#define stbir__3_coeff_remnant( ofs )                \
    { stbir__simdf8 d;                               \
    stbir__simdf8_load4b( cs, hc + (ofs) );          \
    stbir__simdf8_0123to00112233( c, cs );           \
    stbir__simdf8_load6z( d, decode+(ofs)*2 );       \
    stbir__simdf8_madd( tot0, tot0, c, d ); }

#define stbir__store_output()                     \
    { stbir__simdf t,d;                           \
    stbir__simdf8_add4halves( t, stbir__if_simdf8_cast_to_simdf4(tot0), tot0 );    \
    stbir__simdf_0123to2301( d, t );              \
    stbir__simdf_add( t, t, d );                  \
    stbir__simdf_store2( output, t );             \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 2; }

#else

#define stbir__4_coeff_start()                   \
    stbir__simdf tot0,tot1,c,cs;                 \
    STBIR_SIMD_NO_UNROLL(decode);                \
    stbir__simdf_load( cs, hc );                 \
    stbir__simdf_0123to0011( c, cs );            \
    stbir__simdf_mult_mem( tot0, c, decode );    \
    stbir__simdf_0123to2233( c, cs );            \
    stbir__simdf_mult_mem( tot1, c, decode+4 );

#define stbir__4_coeff_continue_from_4( ofs )                \
    STBIR_SIMD_NO_UNROLL(decode);                            \
    stbir__simdf_load( cs, hc + (ofs) );                     \
    stbir__simdf_0123to0011( c, cs );                        \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*2 );  \
    stbir__simdf_0123to2233( c, cs );                        \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*2+4 );

#define stbir__1_coeff_remnant( ofs )            \
    { stbir__simdf d;                            \
    stbir__simdf_load1z( cs, hc + (ofs) );       \
    stbir__simdf_0123to0011( c, cs );            \
    stbir__simdf_load2( d, decode + (ofs) * 2 ); \
    stbir__simdf_madd( tot0, tot0, d, c ); }

#define stbir__2_coeff_remnant( ofs )                      \
    stbir__simdf_load2( cs, hc + (ofs) );                  \
    stbir__simdf_0123to0011( c, cs );                      \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*2 );

#define stbir__3_coeff_remnant( ofs )                       \
    { stbir__simdf d;                                       \
    stbir__simdf_load( cs, hc + (ofs) );                    \
    stbir__simdf_0123to0011( c, cs );                       \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*2 ); \
    stbir__simdf_0123to2222( c, cs );                       \
    stbir__simdf_load2z( d, decode + (ofs) * 2 + 4 );       \
    stbir__simdf_madd( tot1, tot1, d, c ); }

#define stbir__store_output()                     \
    stbir__simdf_add( tot0, tot0, tot1 );         \
    stbir__simdf_0123to2301( c, tot0 );           \
    stbir__simdf_add( tot0, tot0, c );            \
    stbir__simdf_store2( output, tot0 );          \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 2;

#endif

#else

#define stbir__1_coeff_only()  \
    float tota,totb,c;         \
    c = hc[0];                 \
    tota = decode[0]*c;        \
    totb = decode[1]*c;

#define stbir__2_coeff_only()  \
    float tota,totb,c;         \
    c = hc[0];                 \
    tota = decode[0]*c;        \
    totb = decode[1]*c;        \
    c = hc[1];                 \
    tota += decode[2]*c;       \
    totb += decode[3]*c;

#define stbir__3_coeff_only()  \
    float tota,totb,c;         \
    c = hc[0];                 \
    tota = decode[0]*c;        \
    totb = decode[1]*c;        \
    c = hc[2];                 \
    tota += decode[4]*c;       \
    totb += decode[5]*c;       \
    c = hc[1];                 \
    tota += decode[2]*c;       \
    totb += decode[3]*c;

#define stbir__store_output_tiny()                \
    output[0] = tota;                             \
    output[1] = totb;                             \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 2;

#define stbir__4_coeff_start()      \
    float tota0,tota1,tota2,tota3,totb0,totb1,totb2,totb3,c;  \
    c = hc[0];                      \
    tota0 = decode[0]*c;            \
    totb0 = decode[1]*c;            \
    c = hc[1];                      \
    tota1 = decode[2]*c;            \
    totb1 = decode[3]*c;            \
    c = hc[2];                      \
    tota2 = decode[4]*c;            \
    totb2 = decode[5]*c;            \
    c = hc[3];                      \
    tota3 = decode[6]*c;            \
    totb3 = decode[7]*c;

#define stbir__4_coeff_continue_from_4( ofs )  \
    c = hc[0+(ofs)];                           \
    tota0 += decode[0+(ofs)*2]*c;              \
    totb0 += decode[1+(ofs)*2]*c;              \
    c = hc[1+(ofs)];                           \
    tota1 += decode[2+(ofs)*2]*c;              \
    totb1 += decode[3+(ofs)*2]*c;              \
    c = hc[2+(ofs)];                           \
    tota2 += decode[4+(ofs)*2]*c;              \
    totb2 += decode[5+(ofs)*2]*c;              \
    c = hc[3+(ofs)];                           \
    tota3 += decode[6+(ofs)*2]*c;              \
    totb3 += decode[7+(ofs)*2]*c;

#define stbir__1_coeff_remnant( ofs )  \
    c = hc[0+(ofs)];                   \
    tota0 += decode[0+(ofs)*2] * c;    \
    totb0 += decode[1+(ofs)*2] * c;

#define stbir__2_coeff_remnant( ofs )  \
    c = hc[0+(ofs)];                   \
    tota0 += decode[0+(ofs)*2] * c;    \
    totb0 += decode[1+(ofs)*2] * c;    \
    c = hc[1+(ofs)];                   \
    tota1 += decode[2+(ofs)*2] * c;    \
    totb1 += decode[3+(ofs)*2] * c;

#define stbir__3_coeff_remnant( ofs )  \
    c = hc[0+(ofs)];                   \
    tota0 += decode[0+(ofs)*2] * c;    \
    totb0 += decode[1+(ofs)*2] * c;    \
    c = hc[1+(ofs)];                   \
    tota1 += decode[2+(ofs)*2] * c;    \
    totb1 += decode[3+(ofs)*2] * c;    \
    c = hc[2+(ofs)];                   \
    tota2 += decode[4+(ofs)*2] * c;    \
    totb2 += decode[5+(ofs)*2] * c;

#define stbir__store_output()                     \
    output[0] = (tota0+tota2)+(tota1+tota3);      \
    output[1] = (totb0+totb2)+(totb1+totb3);      \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 2;

#endif

#define STBIR__horizontal_channels 2
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME

#ifdef STBIR_SIMD

#define stbir__1_coeff_only()         \
    stbir__simdf tot,c,d;             \
    STBIR_SIMD_NO_UNROLL(decode);     \
    stbir__simdf_load1z( c, hc );     \
    stbir__simdf_0123to0001( c, c );  \
    stbir__simdf_load( d, decode );   \
    stbir__simdf_mult( tot, d, c );

#define stbir__2_coeff_only()         \
    stbir__simdf tot,c,cs,d;          \
    STBIR_SIMD_NO_UNROLL(decode);     \
    stbir__simdf_load2( cs, hc );     \
    stbir__simdf_0123to0000( c, cs ); \
    stbir__simdf_load( d, decode );   \
    stbir__simdf_mult( tot, d, c );   \
    stbir__simdf_0123to1111( c, cs ); \
    stbir__simdf_load( d, decode+3 ); \
    stbir__simdf_madd( tot, tot, d, c );

#define stbir__3_coeff_only()            \
    stbir__simdf tot,c,d,cs;             \
    STBIR_SIMD_NO_UNROLL(decode);        \
    stbir__simdf_load( cs, hc );         \
    stbir__simdf_0123to0000( c, cs );    \
    stbir__simdf_load( d, decode );      \
    stbir__simdf_mult( tot, d, c );      \
    stbir__simdf_0123to1111( c, cs );    \
    stbir__simdf_load( d, decode+3 );    \
    stbir__simdf_madd( tot, tot, d, c ); \
    stbir__simdf_0123to2222( c, cs );    \
    stbir__simdf_load( d, decode+6 );    \
    stbir__simdf_madd( tot, tot, d, c );

#define stbir__store_output_tiny()                \
    stbir__simdf_store2( output, tot );           \
    stbir__simdf_0123to2301( tot, tot );          \
    stbir__simdf_store1( output+2, tot );         \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 3;

#ifdef STBIR_SIMD8

#define stbir__4_coeff_start()                     \
    stbir__simdf8 tot0,tot1,c,cs; stbir__simdf t;  \
    STBIR_SIMD_NO_UNROLL(decode);                  \
    stbir__simdf8_load4b( cs, hc );                \
    stbir__simdf8_0123to00001111( c, cs );         \
    stbir__simdf8_mult_mem( tot0, c, decode - 1 ); \
    stbir__simdf8_0123to22223333( c, cs );         \
    stbir__simdf8_mult_mem( tot1, c, decode+6 - 1 );

#define stbir__4_coeff_continue_from_4( ofs )      \
    STBIR_SIMD_NO_UNROLL(decode);                  \
    stbir__simdf8_load4b( cs, hc + (ofs) );        \
    stbir__simdf8_0123to00001111( c, cs );         \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*3 - 1 ); \
    stbir__simdf8_0123to22223333( c, cs );         \
    stbir__simdf8_madd_mem( tot1, tot1, c, decode+(ofs)*3 + 6 - 1 );

#define stbir__1_coeff_remnant( ofs )                          \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf_load1rep4( t, hc + (ofs) );                   \
    stbir__simdf8_madd_mem4( tot0, tot0, t, decode+(ofs)*3 - 1 );

#define stbir__2_coeff_remnant( ofs )                          \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf8_load4b( cs, hc + (ofs) - 2 );                \
    stbir__simdf8_0123to22223333( c, cs );                     \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*3 - 1 );

 #define stbir__3_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                                \
    stbir__simdf8_load4b( cs, hc + (ofs) );                      \
    stbir__simdf8_0123to00001111( c, cs );                       \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*3 - 1 ); \
    stbir__simdf8_0123to2222( t, cs );                           \
    stbir__simdf8_madd_mem4( tot1, tot1, t, decode+(ofs)*3 + 6 - 1 );

#define stbir__store_output()                       \
    stbir__simdf8_add( tot0, tot0, tot1 );          \
    stbir__simdf_0123to1230( t, stbir__if_simdf8_cast_to_simdf4( tot0 ) ); \
    stbir__simdf8_add4halves( t, t, tot0 );         \
    horizontal_coefficients += coefficient_width;   \
    ++horizontal_contributors;                      \
    output += 3;                                    \
    if ( output < output_end )                      \
    {                                               \
      stbir__simdf_store( output-3, t );            \
      continue;                                     \
    }                                               \
    { stbir__simdf tt; stbir__simdf_0123to2301( tt, t ); \
    stbir__simdf_store2( output-3, t );             \
    stbir__simdf_store1( output+2-3, tt ); }        \
    break;

#else

#define stbir__4_coeff_start()                  \
    stbir__simdf tot0,tot1,tot2,c,cs;           \
    STBIR_SIMD_NO_UNROLL(decode);               \
    stbir__simdf_load( cs, hc );                \
    stbir__simdf_0123to0001( c, cs );           \
    stbir__simdf_mult_mem( tot0, c, decode );   \
    stbir__simdf_0123to1122( c, cs );           \
    stbir__simdf_mult_mem( tot1, c, decode+4 ); \
    stbir__simdf_0123to2333( c, cs );           \
    stbir__simdf_mult_mem( tot2, c, decode+8 );

#define stbir__4_coeff_continue_from_4( ofs )                 \
    STBIR_SIMD_NO_UNROLL(decode);                             \
    stbir__simdf_load( cs, hc + (ofs) );                      \
    stbir__simdf_0123to0001( c, cs );                         \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*3 );   \
    stbir__simdf_0123to1122( c, cs );                         \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*3+4 ); \
    stbir__simdf_0123to2333( c, cs );                         \
    stbir__simdf_madd_mem( tot2, tot2, c, decode+(ofs)*3+8 );

#define stbir__1_coeff_remnant( ofs )         \
    STBIR_SIMD_NO_UNROLL(decode);             \
    stbir__simdf_load1z( c, hc + (ofs) );     \
    stbir__simdf_0123to0001( c, c );          \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*3 );

#define stbir__2_coeff_remnant( ofs )                       \
    { stbir__simdf d;                                       \
    STBIR_SIMD_NO_UNROLL(decode);                           \
    stbir__simdf_load2z( cs, hc + (ofs) );                  \
    stbir__simdf_0123to0001( c, cs );                       \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*3 ); \
    stbir__simdf_0123to1122( c, cs );                       \
    stbir__simdf_load2z( d, decode+(ofs)*3+4 );             \
    stbir__simdf_madd( tot1, tot1, c, d ); }

#define stbir__3_coeff_remnant( ofs )                         \
    { stbir__simdf d;                                         \
    STBIR_SIMD_NO_UNROLL(decode);                             \
    stbir__simdf_load( cs, hc + (ofs) );                      \
    stbir__simdf_0123to0001( c, cs );                         \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*3 );   \
    stbir__simdf_0123to1122( c, cs );                         \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*3+4 ); \
    stbir__simdf_0123to2222( c, cs );                         \
    stbir__simdf_load1z( d, decode+(ofs)*3+8 );               \
    stbir__simdf_madd( tot2, tot2, c, d );  }

#define stbir__store_output()                       \
    stbir__simdf_0123ABCDto3ABx( c, tot0, tot1 );   \
    stbir__simdf_0123ABCDto23Ax( cs, tot1, tot2 );  \
    stbir__simdf_0123to1230( tot2, tot2 );          \
    stbir__simdf_add( tot0, tot0, cs );             \
    stbir__simdf_add( c, c, tot2 );                 \
    stbir__simdf_add( tot0, tot0, c );              \
    horizontal_coefficients += coefficient_width;   \
    ++horizontal_contributors;                      \
    output += 3;                                    \
    if ( output < output_end )                      \
    {                                               \
      stbir__simdf_store( output-3, tot0 );         \
      continue;                                     \
    }                                               \
    stbir__simdf_0123to2301( tot1, tot0 );          \
    stbir__simdf_store2( output-3, tot0 );          \
    stbir__simdf_store1( output+2-3, tot1 );        \
    break;

#endif

#else

#define stbir__1_coeff_only()  \
    float tot0, tot1, tot2, c; \
    c = hc[0];                 \
    tot0 = decode[0]*c;        \
    tot1 = decode[1]*c;        \
    tot2 = decode[2]*c;

#define stbir__2_coeff_only()  \
    float tot0, tot1, tot2, c; \
    c = hc[0];                 \
    tot0 = decode[0]*c;        \
    tot1 = decode[1]*c;        \
    tot2 = decode[2]*c;        \
    c = hc[1];                 \
    tot0 += decode[3]*c;       \
    tot1 += decode[4]*c;       \
    tot2 += decode[5]*c;

#define stbir__3_coeff_only()  \
    float tot0, tot1, tot2, c; \
    c = hc[0];                 \
    tot0 = decode[0]*c;        \
    tot1 = decode[1]*c;        \
    tot2 = decode[2]*c;        \
    c = hc[1];                 \
    tot0 += decode[3]*c;       \
    tot1 += decode[4]*c;       \
    tot2 += decode[5]*c;       \
    c = hc[2];                 \
    tot0 += decode[6]*c;       \
    tot1 += decode[7]*c;       \
    tot2 += decode[8]*c;

#define stbir__store_output_tiny()                \
    output[0] = tot0;                             \
    output[1] = tot1;                             \
    output[2] = tot2;                             \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 3;

#define stbir__4_coeff_start()      \
    float tota0,tota1,tota2,totb0,totb1,totb2,totc0,totc1,totc2,totd0,totd1,totd2,c;  \
    c = hc[0];                      \
    tota0 = decode[0]*c;            \
    tota1 = decode[1]*c;            \
    tota2 = decode[2]*c;            \
    c = hc[1];                      \
    totb0 = decode[3]*c;            \
    totb1 = decode[4]*c;            \
    totb2 = decode[5]*c;            \
    c = hc[2];                      \
    totc0 = decode[6]*c;            \
    totc1 = decode[7]*c;            \
    totc2 = decode[8]*c;            \
    c = hc[3];                      \
    totd0 = decode[9]*c;            \
    totd1 = decode[10]*c;           \
    totd2 = decode[11]*c;

#define stbir__4_coeff_continue_from_4( ofs )  \
    c = hc[0+(ofs)];                           \
    tota0 += decode[0+(ofs)*3]*c;              \
    tota1 += decode[1+(ofs)*3]*c;              \
    tota2 += decode[2+(ofs)*3]*c;              \
    c = hc[1+(ofs)];                           \
    totb0 += decode[3+(ofs)*3]*c;              \
    totb1 += decode[4+(ofs)*3]*c;              \
    totb2 += decode[5+(ofs)*3]*c;              \
    c = hc[2+(ofs)];                           \
    totc0 += decode[6+(ofs)*3]*c;              \
    totc1 += decode[7+(ofs)*3]*c;              \
    totc2 += decode[8+(ofs)*3]*c;              \
    c = hc[3+(ofs)];                           \
    totd0 += decode[9+(ofs)*3]*c;              \
    totd1 += decode[10+(ofs)*3]*c;             \
    totd2 += decode[11+(ofs)*3]*c;

#define stbir__1_coeff_remnant( ofs )  \
    c = hc[0+(ofs)];                   \
    tota0 += decode[0+(ofs)*3]*c;      \
    tota1 += decode[1+(ofs)*3]*c;      \
    tota2 += decode[2+(ofs)*3]*c;

#define stbir__2_coeff_remnant( ofs )  \
    c = hc[0+(ofs)];                   \
    tota0 += decode[0+(ofs)*3]*c;      \
    tota1 += decode[1+(ofs)*3]*c;      \
    tota2 += decode[2+(ofs)*3]*c;      \
    c = hc[1+(ofs)];                   \
    totb0 += decode[3+(ofs)*3]*c;      \
    totb1 += decode[4+(ofs)*3]*c;      \
    totb2 += decode[5+(ofs)*3]*c;      \

#define stbir__3_coeff_remnant( ofs )  \
    c = hc[0+(ofs)];                   \
    tota0 += decode[0+(ofs)*3]*c;      \
    tota1 += decode[1+(ofs)*3]*c;      \
    tota2 += decode[2+(ofs)*3]*c;      \
    c = hc[1+(ofs)];                   \
    totb0 += decode[3+(ofs)*3]*c;      \
    totb1 += decode[4+(ofs)*3]*c;      \
    totb2 += decode[5+(ofs)*3]*c;      \
    c = hc[2+(ofs)];                   \
    totc0 += decode[6+(ofs)*3]*c;      \
    totc1 += decode[7+(ofs)*3]*c;      \
    totc2 += decode[8+(ofs)*3]*c;

#define stbir__store_output()                     \
    output[0] = (tota0+totc0)+(totb0+totd0);      \
    output[1] = (tota1+totc1)+(totb1+totd1);      \
    output[2] = (tota2+totc2)+(totb2+totd2);      \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 3;

#endif

#define STBIR__horizontal_channels 3
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME

#ifdef STBIR_SIMD

#define stbir__1_coeff_only()             \
    stbir__simdf tot,c;                   \
    STBIR_SIMD_NO_UNROLL(decode);         \
    stbir__simdf_load1( c, hc );          \
    stbir__simdf_0123to0000( c, c );      \
    stbir__simdf_mult_mem( tot, c, decode );

#define stbir__2_coeff_only()                       \
    stbir__simdf tot,c,cs;                          \
    STBIR_SIMD_NO_UNROLL(decode);                   \
    stbir__simdf_load2( cs, hc );                   \
    stbir__simdf_0123to0000( c, cs );               \
    stbir__simdf_mult_mem( tot, c, decode );        \
    stbir__simdf_0123to1111( c, cs );               \
    stbir__simdf_madd_mem( tot, tot, c, decode+4 );

#define stbir__3_coeff_only()                       \
    stbir__simdf tot,c,cs;                          \
    STBIR_SIMD_NO_UNROLL(decode);                   \
    stbir__simdf_load( cs, hc );                    \
    stbir__simdf_0123to0000( c, cs );               \
    stbir__simdf_mult_mem( tot, c, decode );        \
    stbir__simdf_0123to1111( c, cs );               \
    stbir__simdf_madd_mem( tot, tot, c, decode+4 ); \
    stbir__simdf_0123to2222( c, cs );               \
    stbir__simdf_madd_mem( tot, tot, c, decode+8 );

#define stbir__store_output_tiny()                \
    stbir__simdf_store( output, tot );            \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 4;

#ifdef STBIR_SIMD8

#define stbir__4_coeff_start()                     \
    stbir__simdf8 tot0,c,cs; stbir__simdf t;  \
    STBIR_SIMD_NO_UNROLL(decode);                  \
    stbir__simdf8_load4b( cs, hc );                \
    stbir__simdf8_0123to00001111( c, cs );         \
    stbir__simdf8_mult_mem( tot0, c, decode );     \
    stbir__simdf8_0123to22223333( c, cs );         \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+8 );

#define stbir__4_coeff_continue_from_4( ofs )                  \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf8_load4b( cs, hc + (ofs) );                    \
    stbir__simdf8_0123to00001111( c, cs );                     \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*4 );   \
    stbir__simdf8_0123to22223333( c, cs );                     \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*4+8 );

#define stbir__1_coeff_remnant( ofs )                          \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf_load1rep4( t, hc + (ofs) );                   \
    stbir__simdf8_madd_mem4( tot0, tot0, t, decode+(ofs)*4 );

#define stbir__2_coeff_remnant( ofs )                          \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf8_load4b( cs, hc + (ofs) - 2 );                \
    stbir__simdf8_0123to22223333( c, cs );                     \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*4 );

 #define stbir__3_coeff_remnant( ofs )                         \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf8_load4b( cs, hc + (ofs) );                    \
    stbir__simdf8_0123to00001111( c, cs );                     \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*4 );   \
    stbir__simdf8_0123to2222( t, cs );                         \
    stbir__simdf8_madd_mem4( tot0, tot0, t, decode+(ofs)*4+8 );

#define stbir__store_output()                      \
    stbir__simdf8_add4halves( t, stbir__if_simdf8_cast_to_simdf4(tot0), tot0 );     \
    stbir__simdf_store( output, t );               \
    horizontal_coefficients += coefficient_width;  \
    ++horizontal_contributors;                     \
    output += 4;

#else

#define stbir__4_coeff_start()                        \
    stbir__simdf tot0,tot1,c,cs;                      \
    STBIR_SIMD_NO_UNROLL(decode);                     \
    stbir__simdf_load( cs, hc );                      \
    stbir__simdf_0123to0000( c, cs );                 \
    stbir__simdf_mult_mem( tot0, c, decode );         \
    stbir__simdf_0123to1111( c, cs );                 \
    stbir__simdf_mult_mem( tot1, c, decode+4 );       \
    stbir__simdf_0123to2222( c, cs );                 \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+8 ); \
    stbir__simdf_0123to3333( c, cs );                 \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+12 );

#define stbir__4_coeff_continue_from_4( ofs )                  \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf_load( cs, hc + (ofs) );                       \
    stbir__simdf_0123to0000( c, cs );                          \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*4 );    \
    stbir__simdf_0123to1111( c, cs );                          \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*4+4 );  \
    stbir__simdf_0123to2222( c, cs );                          \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*4+8 );  \
    stbir__simdf_0123to3333( c, cs );                          \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*4+12 );

#define stbir__1_coeff_remnant( ofs )                       \
    STBIR_SIMD_NO_UNROLL(decode);                           \
    stbir__simdf_load1( c, hc + (ofs) );                    \
    stbir__simdf_0123to0000( c, c );                        \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*4 );

#define stbir__2_coeff_remnant( ofs )                         \
    STBIR_SIMD_NO_UNROLL(decode);                             \
    stbir__simdf_load2( cs, hc + (ofs) );                     \
    stbir__simdf_0123to0000( c, cs );                         \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*4 );   \
    stbir__simdf_0123to1111( c, cs );                         \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*4+4 );

#define stbir__3_coeff_remnant( ofs )                          \
    STBIR_SIMD_NO_UNROLL(decode);                              \
    stbir__simdf_load( cs, hc + (ofs) );                       \
    stbir__simdf_0123to0000( c, cs );                          \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*4 );    \
    stbir__simdf_0123to1111( c, cs );                          \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*4+4 );  \
    stbir__simdf_0123to2222( c, cs );                          \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*4+8 );

#define stbir__store_output()                     \
    stbir__simdf_add( tot0, tot0, tot1 );         \
    stbir__simdf_store( output, tot0 );           \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 4;

#endif

#else

#define stbir__1_coeff_only()         \
    float p0,p1,p2,p3,c;              \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0];                        \
    p0 = decode[0] * c;               \
    p1 = decode[1] * c;               \
    p2 = decode[2] * c;               \
    p3 = decode[3] * c;

#define stbir__2_coeff_only()         \
    float p0,p1,p2,p3,c;              \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0];                        \
    p0 = decode[0] * c;               \
    p1 = decode[1] * c;               \
    p2 = decode[2] * c;               \
    p3 = decode[3] * c;               \
    c = hc[1];                        \
    p0 += decode[4] * c;              \
    p1 += decode[5] * c;              \
    p2 += decode[6] * c;              \
    p3 += decode[7] * c;

#define stbir__3_coeff_only()         \
    float p0,p1,p2,p3,c;              \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0];                        \
    p0 = decode[0] * c;               \
    p1 = decode[1] * c;               \
    p2 = decode[2] * c;               \
    p3 = decode[3] * c;               \
    c = hc[1];                        \
    p0 += decode[4] * c;              \
    p1 += decode[5] * c;              \
    p2 += decode[6] * c;              \
    p3 += decode[7] * c;              \
    c = hc[2];                        \
    p0 += decode[8] * c;              \
    p1 += decode[9] * c;              \
    p2 += decode[10] * c;             \
    p3 += decode[11] * c;

#define stbir__store_output_tiny()                \
    output[0] = p0;                               \
    output[1] = p1;                               \
    output[2] = p2;                               \
    output[3] = p3;                               \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 4;

#define stbir__4_coeff_start()        \
    float x0,x1,x2,x3,y0,y1,y2,y3,c;  \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0];                        \
    x0 = decode[0] * c;               \
    x1 = decode[1] * c;               \
    x2 = decode[2] * c;               \
    x3 = decode[3] * c;               \
    c = hc[1];                        \
    y0 = decode[4] * c;               \
    y1 = decode[5] * c;               \
    y2 = decode[6] * c;               \
    y3 = decode[7] * c;               \
    c = hc[2];                        \
    x0 += decode[8] * c;              \
    x1 += decode[9] * c;              \
    x2 += decode[10] * c;             \
    x3 += decode[11] * c;             \
    c = hc[3];                        \
    y0 += decode[12] * c;             \
    y1 += decode[13] * c;             \
    y2 += decode[14] * c;             \
    y3 += decode[15] * c;

#define stbir__4_coeff_continue_from_4( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0+(ofs)];                  \
    x0 += decode[0+(ofs)*4] * c;      \
    x1 += decode[1+(ofs)*4] * c;      \
    x2 += decode[2+(ofs)*4] * c;      \
    x3 += decode[3+(ofs)*4] * c;      \
    c = hc[1+(ofs)];                  \
    y0 += decode[4+(ofs)*4] * c;      \
    y1 += decode[5+(ofs)*4] * c;      \
    y2 += decode[6+(ofs)*4] * c;      \
    y3 += decode[7+(ofs)*4] * c;      \
    c = hc[2+(ofs)];                  \
    x0 += decode[8+(ofs)*4] * c;      \
    x1 += decode[9+(ofs)*4] * c;      \
    x2 += decode[10+(ofs)*4] * c;     \
    x3 += decode[11+(ofs)*4] * c;     \
    c = hc[3+(ofs)];                  \
    y0 += decode[12+(ofs)*4] * c;     \
    y1 += decode[13+(ofs)*4] * c;     \
    y2 += decode[14+(ofs)*4] * c;     \
    y3 += decode[15+(ofs)*4] * c;

#define stbir__1_coeff_remnant( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0+(ofs)];                  \
    x0 += decode[0+(ofs)*4] * c;      \
    x1 += decode[1+(ofs)*4] * c;      \
    x2 += decode[2+(ofs)*4] * c;      \
    x3 += decode[3+(ofs)*4] * c;

#define stbir__2_coeff_remnant( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0+(ofs)];                  \
    x0 += decode[0+(ofs)*4] * c;      \
    x1 += decode[1+(ofs)*4] * c;      \
    x2 += decode[2+(ofs)*4] * c;      \
    x3 += decode[3+(ofs)*4] * c;      \
    c = hc[1+(ofs)];                  \
    y0 += decode[4+(ofs)*4] * c;      \
    y1 += decode[5+(ofs)*4] * c;      \
    y2 += decode[6+(ofs)*4] * c;      \
    y3 += decode[7+(ofs)*4] * c;

#define stbir__3_coeff_remnant( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);     \
    c = hc[0+(ofs)];                  \
    x0 += decode[0+(ofs)*4] * c;      \
    x1 += decode[1+(ofs)*4] * c;      \
    x2 += decode[2+(ofs)*4] * c;      \
    x3 += decode[3+(ofs)*4] * c;      \
    c = hc[1+(ofs)];                  \
    y0 += decode[4+(ofs)*4] * c;      \
    y1 += decode[5+(ofs)*4] * c;      \
    y2 += decode[6+(ofs)*4] * c;      \
    y3 += decode[7+(ofs)*4] * c;      \
    c = hc[2+(ofs)];                  \
    x0 += decode[8+(ofs)*4] * c;      \
    x1 += decode[9+(ofs)*4] * c;      \
    x2 += decode[10+(ofs)*4] * c;     \
    x3 += decode[11+(ofs)*4] * c;

#define stbir__store_output()                     \
    output[0] = x0 + y0;                          \
    output[1] = x1 + y1;                          \
    output[2] = x2 + y2;                          \
    output[3] = x3 + y3;                          \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 4;

#endif

#define STBIR__horizontal_channels 4
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME

#ifdef STBIR_SIMD

#define stbir__1_coeff_only()                   \
    stbir__simdf tot0,tot1,c;                   \
    STBIR_SIMD_NO_UNROLL(decode);               \
    stbir__simdf_load1( c, hc );                \
    stbir__simdf_0123to0000( c, c );            \
    stbir__simdf_mult_mem( tot0, c, decode );   \
    stbir__simdf_mult_mem( tot1, c, decode+3 );

#define stbir__2_coeff_only()                         \
    stbir__simdf tot0,tot1,c,cs;                      \
    STBIR_SIMD_NO_UNROLL(decode);                     \
    stbir__simdf_load2( cs, hc );                     \
    stbir__simdf_0123to0000( c, cs );                 \
    stbir__simdf_mult_mem( tot0, c, decode );         \
    stbir__simdf_mult_mem( tot1, c, decode+3 );       \
    stbir__simdf_0123to1111( c, cs );                 \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+7 ); \
    stbir__simdf_madd_mem( tot1, tot1, c,decode+10 );

#define stbir__3_coeff_only()                           \
    stbir__simdf tot0,tot1,c,cs;                        \
    STBIR_SIMD_NO_UNROLL(decode);                       \
    stbir__simdf_load( cs, hc );                        \
    stbir__simdf_0123to0000( c, cs );                   \
    stbir__simdf_mult_mem( tot0, c, decode );           \
    stbir__simdf_mult_mem( tot1, c, decode+3 );         \
    stbir__simdf_0123to1111( c, cs );                   \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+7 );   \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+10 );  \
    stbir__simdf_0123to2222( c, cs );                   \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+14 );  \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+17 );

#define stbir__store_output_tiny()                \
    stbir__simdf_store( output+3, tot1 );         \
    stbir__simdf_store( output, tot0 );           \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 7;

#ifdef STBIR_SIMD8

#define stbir__4_coeff_start()                     \
    stbir__simdf8 tot0,tot1,c,cs;                  \
    STBIR_SIMD_NO_UNROLL(decode);                  \
    stbir__simdf8_load4b( cs, hc );                \
    stbir__simdf8_0123to00000000( c, cs );         \
    stbir__simdf8_mult_mem( tot0, c, decode );     \
    stbir__simdf8_0123to11111111( c, cs );         \
    stbir__simdf8_mult_mem( tot1, c, decode+7 );   \
    stbir__simdf8_0123to22222222( c, cs );         \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+14 );  \
    stbir__simdf8_0123to33333333( c, cs );         \
    stbir__simdf8_madd_mem( tot1, tot1, c, decode+21 );

#define stbir__4_coeff_continue_from_4( ofs )                   \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf8_load4b( cs, hc + (ofs) );                     \
    stbir__simdf8_0123to00000000( c, cs );                      \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*7 );    \
    stbir__simdf8_0123to11111111( c, cs );                      \
    stbir__simdf8_madd_mem( tot1, tot1, c, decode+(ofs)*7+7 );  \
    stbir__simdf8_0123to22222222( c, cs );                      \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*7+14 ); \
    stbir__simdf8_0123to33333333( c, cs );                      \
    stbir__simdf8_madd_mem( tot1, tot1, c, decode+(ofs)*7+21 );

#define stbir__1_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf8_load1b( c, hc + (ofs) );                      \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*7 );

#define stbir__2_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf8_load1b( c, hc + (ofs) );                      \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*7 );    \
    stbir__simdf8_load1b( c, hc + (ofs)+1 );                    \
    stbir__simdf8_madd_mem( tot1, tot1, c, decode+(ofs)*7+7 );

#define stbir__3_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf8_load4b( cs, hc + (ofs) );                     \
    stbir__simdf8_0123to00000000( c, cs );                      \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*7 );    \
    stbir__simdf8_0123to11111111( c, cs );                      \
    stbir__simdf8_madd_mem( tot1, tot1, c, decode+(ofs)*7+7 );  \
    stbir__simdf8_0123to22222222( c, cs );                      \
    stbir__simdf8_madd_mem( tot0, tot0, c, decode+(ofs)*7+14 );

#define stbir__store_output()                     \
    stbir__simdf8_add( tot0, tot0, tot1 );        \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 7;                                  \
    if ( output < output_end )                    \
    {                                             \
      stbir__simdf8_store( output-7, tot0 );      \
      continue;                                   \
    }                                             \
    stbir__simdf_store( output-7+3, stbir__simdf_swiz(stbir__simdf8_gettop4(tot0),0,0,1,2) ); \
    stbir__simdf_store( output-7, stbir__if_simdf8_cast_to_simdf4(tot0) );           \
    break;

#else

#define stbir__4_coeff_start()                    \
    stbir__simdf tot0,tot1,tot2,tot3,c,cs;        \
    STBIR_SIMD_NO_UNROLL(decode);                 \
    stbir__simdf_load( cs, hc );                  \
    stbir__simdf_0123to0000( c, cs );             \
    stbir__simdf_mult_mem( tot0, c, decode );     \
    stbir__simdf_mult_mem( tot1, c, decode+3 );   \
    stbir__simdf_0123to1111( c, cs );             \
    stbir__simdf_mult_mem( tot2, c, decode+7 );   \
    stbir__simdf_mult_mem( tot3, c, decode+10 );  \
    stbir__simdf_0123to2222( c, cs );             \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+14 );  \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+17 );  \
    stbir__simdf_0123to3333( c, cs );                   \
    stbir__simdf_madd_mem( tot2, tot2, c, decode+21 );  \
    stbir__simdf_madd_mem( tot3, tot3, c, decode+24 );

#define stbir__4_coeff_continue_from_4( ofs )                   \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf_load( cs, hc + (ofs) );                        \
    stbir__simdf_0123to0000( c, cs );                           \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*7 );     \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*7+3 );   \
    stbir__simdf_0123to1111( c, cs );                           \
    stbir__simdf_madd_mem( tot2, tot2, c, decode+(ofs)*7+7 );   \
    stbir__simdf_madd_mem( tot3, tot3, c, decode+(ofs)*7+10 );  \
    stbir__simdf_0123to2222( c, cs );                           \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*7+14 );  \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*7+17 );  \
    stbir__simdf_0123to3333( c, cs );                           \
    stbir__simdf_madd_mem( tot2, tot2, c, decode+(ofs)*7+21 );  \
    stbir__simdf_madd_mem( tot3, tot3, c, decode+(ofs)*7+24 );

#define stbir__1_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf_load1( c, hc + (ofs) );                        \
    stbir__simdf_0123to0000( c, c );                            \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*7 );     \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*7+3 );   \

#define stbir__2_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf_load2( cs, hc + (ofs) );                       \
    stbir__simdf_0123to0000( c, cs );                           \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*7 );     \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*7+3 );   \
    stbir__simdf_0123to1111( c, cs );                           \
    stbir__simdf_madd_mem( tot2, tot2, c, decode+(ofs)*7+7 );   \
    stbir__simdf_madd_mem( tot3, tot3, c, decode+(ofs)*7+10 );

#define stbir__3_coeff_remnant( ofs )                           \
    STBIR_SIMD_NO_UNROLL(decode);                               \
    stbir__simdf_load( cs, hc + (ofs) );                        \
    stbir__simdf_0123to0000( c, cs );                           \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*7 );     \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*7+3 );   \
    stbir__simdf_0123to1111( c, cs );                           \
    stbir__simdf_madd_mem( tot2, tot2, c, decode+(ofs)*7+7 );   \
    stbir__simdf_madd_mem( tot3, tot3, c, decode+(ofs)*7+10 );  \
    stbir__simdf_0123to2222( c, cs );                           \
    stbir__simdf_madd_mem( tot0, tot0, c, decode+(ofs)*7+14 );  \
    stbir__simdf_madd_mem( tot1, tot1, c, decode+(ofs)*7+17 );

#define stbir__store_output()                     \
    stbir__simdf_add( tot0, tot0, tot2 );         \
    stbir__simdf_add( tot1, tot1, tot3 );         \
    stbir__simdf_store( output+3, tot1 );         \
    stbir__simdf_store( output, tot0 );           \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 7;

#endif

#else

#define stbir__1_coeff_only()        \
    float tot0, tot1, tot2, tot3, tot4, tot5, tot6, c; \
    c = hc[0];                       \
    tot0 = decode[0]*c;              \
    tot1 = decode[1]*c;              \
    tot2 = decode[2]*c;              \
    tot3 = decode[3]*c;              \
    tot4 = decode[4]*c;              \
    tot5 = decode[5]*c;              \
    tot6 = decode[6]*c;

#define stbir__2_coeff_only()        \
    float tot0, tot1, tot2, tot3, tot4, tot5, tot6, c; \
    c = hc[0];                       \
    tot0 = decode[0]*c;              \
    tot1 = decode[1]*c;              \
    tot2 = decode[2]*c;              \
    tot3 = decode[3]*c;              \
    tot4 = decode[4]*c;              \
    tot5 = decode[5]*c;              \
    tot6 = decode[6]*c;              \
    c = hc[1];                       \
    tot0 += decode[7]*c;             \
    tot1 += decode[8]*c;             \
    tot2 += decode[9]*c;             \
    tot3 += decode[10]*c;            \
    tot4 += decode[11]*c;            \
    tot5 += decode[12]*c;            \
    tot6 += decode[13]*c;            \

#define stbir__3_coeff_only()        \
    float tot0, tot1, tot2, tot3, tot4, tot5, tot6, c; \
    c = hc[0];                       \
    tot0 = decode[0]*c;              \
    tot1 = decode[1]*c;              \
    tot2 = decode[2]*c;              \
    tot3 = decode[3]*c;              \
    tot4 = decode[4]*c;              \
    tot5 = decode[5]*c;              \
    tot6 = decode[6]*c;              \
    c = hc[1];                       \
    tot0 += decode[7]*c;             \
    tot1 += decode[8]*c;             \
    tot2 += decode[9]*c;             \
    tot3 += decode[10]*c;            \
    tot4 += decode[11]*c;            \
    tot5 += decode[12]*c;            \
    tot6 += decode[13]*c;            \
    c = hc[2];                       \
    tot0 += decode[14]*c;            \
    tot1 += decode[15]*c;            \
    tot2 += decode[16]*c;            \
    tot3 += decode[17]*c;            \
    tot4 += decode[18]*c;            \
    tot5 += decode[19]*c;            \
    tot6 += decode[20]*c;            \

#define stbir__store_output_tiny()                \
    output[0] = tot0;                             \
    output[1] = tot1;                             \
    output[2] = tot2;                             \
    output[3] = tot3;                             \
    output[4] = tot4;                             \
    output[5] = tot5;                             \
    output[6] = tot6;                             \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 7;

#define stbir__4_coeff_start()    \
    float x0,x1,x2,x3,x4,x5,x6,y0,y1,y2,y3,y4,y5,y6,c; \
    STBIR_SIMD_NO_UNROLL(decode); \
    c = hc[0];                    \
    x0 = decode[0] * c;           \
    x1 = decode[1] * c;           \
    x2 = decode[2] * c;           \
    x3 = decode[3] * c;           \
    x4 = decode[4] * c;           \
    x5 = decode[5] * c;           \
    x6 = decode[6] * c;           \
    c = hc[1];                    \
    y0 = decode[7] * c;           \
    y1 = decode[8] * c;           \
    y2 = decode[9] * c;           \
    y3 = decode[10] * c;          \
    y4 = decode[11] * c;          \
    y5 = decode[12] * c;          \
    y6 = decode[13] * c;          \
    c = hc[2];                    \
    x0 += decode[14] * c;         \
    x1 += decode[15] * c;         \
    x2 += decode[16] * c;         \
    x3 += decode[17] * c;         \
    x4 += decode[18] * c;         \
    x5 += decode[19] * c;         \
    x6 += decode[20] * c;         \
    c = hc[3];                    \
    y0 += decode[21] * c;         \
    y1 += decode[22] * c;         \
    y2 += decode[23] * c;         \
    y3 += decode[24] * c;         \
    y4 += decode[25] * c;         \
    y5 += decode[26] * c;         \
    y6 += decode[27] * c;

#define stbir__4_coeff_continue_from_4( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);  \
    c = hc[0+(ofs)];               \
    x0 += decode[0+(ofs)*7] * c;   \
    x1 += decode[1+(ofs)*7] * c;   \
    x2 += decode[2+(ofs)*7] * c;   \
    x3 += decode[3+(ofs)*7] * c;   \
    x4 += decode[4+(ofs)*7] * c;   \
    x5 += decode[5+(ofs)*7] * c;   \
    x6 += decode[6+(ofs)*7] * c;   \
    c = hc[1+(ofs)];               \
    y0 += decode[7+(ofs)*7] * c;   \
    y1 += decode[8+(ofs)*7] * c;   \
    y2 += decode[9+(ofs)*7] * c;   \
    y3 += decode[10+(ofs)*7] * c;  \
    y4 += decode[11+(ofs)*7] * c;  \
    y5 += decode[12+(ofs)*7] * c;  \
    y6 += decode[13+(ofs)*7] * c;  \
    c = hc[2+(ofs)];               \
    x0 += decode[14+(ofs)*7] * c;  \
    x1 += decode[15+(ofs)*7] * c;  \
    x2 += decode[16+(ofs)*7] * c;  \
    x3 += decode[17+(ofs)*7] * c;  \
    x4 += decode[18+(ofs)*7] * c;  \
    x5 += decode[19+(ofs)*7] * c;  \
    x6 += decode[20+(ofs)*7] * c;  \
    c = hc[3+(ofs)];               \
    y0 += decode[21+(ofs)*7] * c;  \
    y1 += decode[22+(ofs)*7] * c;  \
    y2 += decode[23+(ofs)*7] * c;  \
    y3 += decode[24+(ofs)*7] * c;  \
    y4 += decode[25+(ofs)*7] * c;  \
    y5 += decode[26+(ofs)*7] * c;  \
    y6 += decode[27+(ofs)*7] * c;

#define stbir__1_coeff_remnant( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);  \
    c = hc[0+(ofs)];               \
    x0 += decode[0+(ofs)*7] * c;   \
    x1 += decode[1+(ofs)*7] * c;   \
    x2 += decode[2+(ofs)*7] * c;   \
    x3 += decode[3+(ofs)*7] * c;   \
    x4 += decode[4+(ofs)*7] * c;   \
    x5 += decode[5+(ofs)*7] * c;   \
    x6 += decode[6+(ofs)*7] * c;   \

#define stbir__2_coeff_remnant( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);  \
    c = hc[0+(ofs)];               \
    x0 += decode[0+(ofs)*7] * c;   \
    x1 += decode[1+(ofs)*7] * c;   \
    x2 += decode[2+(ofs)*7] * c;   \
    x3 += decode[3+(ofs)*7] * c;   \
    x4 += decode[4+(ofs)*7] * c;   \
    x5 += decode[5+(ofs)*7] * c;   \
    x6 += decode[6+(ofs)*7] * c;   \
    c = hc[1+(ofs)];               \
    y0 += decode[7+(ofs)*7] * c;   \
    y1 += decode[8+(ofs)*7] * c;   \
    y2 += decode[9+(ofs)*7] * c;   \
    y3 += decode[10+(ofs)*7] * c;  \
    y4 += decode[11+(ofs)*7] * c;  \
    y5 += decode[12+(ofs)*7] * c;  \
    y6 += decode[13+(ofs)*7] * c;  \

#define stbir__3_coeff_remnant( ofs ) \
    STBIR_SIMD_NO_UNROLL(decode);  \
    c = hc[0+(ofs)];               \
    x0 += decode[0+(ofs)*7] * c;   \
    x1 += decode[1+(ofs)*7] * c;   \
    x2 += decode[2+(ofs)*7] * c;   \
    x3 += decode[3+(ofs)*7] * c;   \
    x4 += decode[4+(ofs)*7] * c;   \
    x5 += decode[5+(ofs)*7] * c;   \
    x6 += decode[6+(ofs)*7] * c;   \
    c = hc[1+(ofs)];               \
    y0 += decode[7+(ofs)*7] * c;   \
    y1 += decode[8+(ofs)*7] * c;   \
    y2 += decode[9+(ofs)*7] * c;   \
    y3 += decode[10+(ofs)*7] * c;  \
    y4 += decode[11+(ofs)*7] * c;  \
    y5 += decode[12+(ofs)*7] * c;  \
    y6 += decode[13+(ofs)*7] * c;  \
    c = hc[2+(ofs)];               \
    x0 += decode[14+(ofs)*7] * c;  \
    x1 += decode[15+(ofs)*7] * c;  \
    x2 += decode[16+(ofs)*7] * c;  \
    x3 += decode[17+(ofs)*7] * c;  \
    x4 += decode[18+(ofs)*7] * c;  \
    x5 += decode[19+(ofs)*7] * c;  \
    x6 += decode[20+(ofs)*7] * c;  \

#define stbir__store_output()                     \
    output[0] = x0 + y0;                          \
    output[1] = x1 + y1;                          \
    output[2] = x2 + y2;                          \
    output[3] = x3 + y3;                          \
    output[4] = x4 + y4;                          \
    output[5] = x5 + y5;                          \
    output[6] = x6 + y6;                          \
    horizontal_coefficients += coefficient_width; \
    ++horizontal_contributors;                    \
    output += 7;

#endif

#define STBIR__horizontal_channels 7
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 1
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 1
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 2
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 2
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 3
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 3
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 4
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 4
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 5
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 5
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 6
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 6
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 7
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 7
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 8
#define STB_IMAGE_RESIZE_DO_VERTICALS
#include STBIR__HEADER_FILENAME

#define STBIR__vertical_channels 8
#define STB_IMAGE_RESIZE_DO_VERTICALS
#define STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#include STBIR__HEADER_FILENAME

typedef void STBIR_VERTICAL_GATHERFUNC( float * output, float const * coeffs, float const ** inputs, float const * input0_end );

static STBIR_VERTICAL_GATHERFUNC * stbir__vertical_gathers[ 8 ] =
{
  stbir__vertical_gather_with_1_coeffs,stbir__vertical_gather_with_2_coeffs,stbir__vertical_gather_with_3_coeffs,stbir__vertical_gather_with_4_coeffs,stbir__vertical_gather_with_5_coeffs,stbir__vertical_gather_with_6_coeffs,stbir__vertical_gather_with_7_coeffs,stbir__vertical_gather_with_8_coeffs
};

static STBIR_VERTICAL_GATHERFUNC * stbir__vertical_gathers_continues[ 8 ] =
{
  stbir__vertical_gather_with_1_coeffs_cont,stbir__vertical_gather_with_2_coeffs_cont,stbir__vertical_gather_with_3_coeffs_cont,stbir__vertical_gather_with_4_coeffs_cont,stbir__vertical_gather_with_5_coeffs_cont,stbir__vertical_gather_with_6_coeffs_cont,stbir__vertical_gather_with_7_coeffs_cont,stbir__vertical_gather_with_8_coeffs_cont
};

typedef void STBIR_VERTICAL_SCATTERFUNC( float ** outputs, float const * coeffs, float const * input, float const * input_end );

static STBIR_VERTICAL_SCATTERFUNC * stbir__vertical_scatter_sets[ 8 ] =
{
  stbir__vertical_scatter_with_1_coeffs,stbir__vertical_scatter_with_2_coeffs,stbir__vertical_scatter_with_3_coeffs,stbir__vertical_scatter_with_4_coeffs,stbir__vertical_scatter_with_5_coeffs,stbir__vertical_scatter_with_6_coeffs,stbir__vertical_scatter_with_7_coeffs,stbir__vertical_scatter_with_8_coeffs
};

static STBIR_VERTICAL_SCATTERFUNC * stbir__vertical_scatter_blends[ 8 ] =
{
  stbir__vertical_scatter_with_1_coeffs_cont,stbir__vertical_scatter_with_2_coeffs_cont,stbir__vertical_scatter_with_3_coeffs_cont,stbir__vertical_scatter_with_4_coeffs_cont,stbir__vertical_scatter_with_5_coeffs_cont,stbir__vertical_scatter_with_6_coeffs_cont,stbir__vertical_scatter_with_7_coeffs_cont,stbir__vertical_scatter_with_8_coeffs_cont
};

static void stbir__encode_scanline( stbir__info const * stbir_info, void *output_buffer_data, float * encode_buffer, int row  STBIR_ONLY_PROFILE_GET_SPLIT_INFO )
{
  int num_pixels = stbir_info->horizontal.scale_info.output_sub_size;
  int channels = stbir_info->channels;
  int width_times_channels = num_pixels * channels;
  void * output_buffer;

  if ( stbir_info->alpha_unweight )
  {
    STBIR_PROFILE_START( unalpha );
    stbir_info->alpha_unweight( encode_buffer, width_times_channels );
    STBIR_PROFILE_END( unalpha );
  }

  output_buffer = output_buffer_data;

  if ( stbir_info->out_pixels_cb )
    output_buffer = encode_buffer;

  STBIR_PROFILE_START( encode );
  stbir_info->encode_pixels( output_buffer, width_times_channels, encode_buffer );
  STBIR_PROFILE_END( encode );

  if ( stbir_info->out_pixels_cb )
    stbir_info->out_pixels_cb( output_buffer, num_pixels, row, stbir_info->user_data );
}

static float* stbir__get_ring_buffer_entry(stbir__info const * stbir_info, stbir__per_split_info const * split_info, int index )
{
  STBIR_ASSERT( index < stbir_info->ring_buffer_num_entries );

  #ifdef STBIR__SEPARATE_ALLOCATIONS
    return split_info->ring_buffers[ index ];
  #else
    return (float*) ( ( (char*) split_info->ring_buffer ) + ( index * stbir_info->ring_buffer_length_bytes ) );
  #endif
}

static float* stbir__get_ring_buffer_scanline(stbir__info const * stbir_info, stbir__per_split_info const * split_info, int get_scanline)
{
  int ring_buffer_index = (split_info->ring_buffer_begin_index + (get_scanline - split_info->ring_buffer_first_scanline)) % stbir_info->ring_buffer_num_entries;
  return stbir__get_ring_buffer_entry( stbir_info, split_info, ring_buffer_index );
}

static void stbir__resample_horizontal_gather(stbir__info const * stbir_info, float* output_buffer, float const * input_buffer STBIR_ONLY_PROFILE_GET_SPLIT_INFO )
{
  float const * decode_buffer = input_buffer - ( stbir_info->scanline_extents.conservative.n0 * stbir_info->effective_channels );

  STBIR_PROFILE_START( horizontal );
  if ( ( stbir_info->horizontal.filter_enum == STBIR_FILTER_POINT_SAMPLE ) && ( stbir_info->horizontal.scale_info.scale == 1.0f ) )
    STBIR_MEMCPY( output_buffer, input_buffer, stbir_info->horizontal.scale_info.output_sub_size * sizeof( float ) * stbir_info->effective_channels );
  else
    stbir_info->horizontal_gather_channels( output_buffer, stbir_info->horizontal.scale_info.output_sub_size, decode_buffer, stbir_info->horizontal.contributors, stbir_info->horizontal.coefficients, stbir_info->horizontal.coefficient_width );
  STBIR_PROFILE_END( horizontal );
}

static void stbir__resample_vertical_gather(stbir__info const * stbir_info, stbir__per_split_info* split_info, int n, int contrib_n0, int contrib_n1, float const * vertical_coefficients )
{
  float* encode_buffer = split_info->vertical_buffer;
  float* decode_buffer = split_info->decode_buffer;
  int vertical_first = stbir_info->vertical_first;
  int width = (vertical_first) ? ( stbir_info->scanline_extents.conservative.n1-stbir_info->scanline_extents.conservative.n0+1 ) : stbir_info->horizontal.scale_info.output_sub_size;
  int width_times_channels = stbir_info->effective_channels * width;

  STBIR_ASSERT( stbir_info->vertical.is_gather );

  STBIR_PROFILE_START( vertical );
  {
    int k = 0, total = contrib_n1 - contrib_n0 + 1;
    STBIR_ASSERT( total > 0 );
    do {
      float const * inputs[8];
      int i, cnt = total; if ( cnt > 8 ) cnt = 8;
      for( i = 0 ; i < cnt ; i++ )
        inputs[ i ] = stbir__get_ring_buffer_scanline(stbir_info, split_info, k+i+contrib_n0 );

      ((k==0)?stbir__vertical_gathers:stbir__vertical_gathers_continues)[cnt-1]( (vertical_first) ? decode_buffer : encode_buffer, vertical_coefficients + k, inputs, inputs[0] + width_times_channels );
      k += cnt;
      total -= cnt;
    } while ( total );
  }
  STBIR_PROFILE_END( vertical );

  if ( vertical_first )
  {
    decode_buffer[ width_times_channels ] = 0.0f;
    decode_buffer[ width_times_channels+1 ] = 0.0f;
    stbir__resample_horizontal_gather(stbir_info, encode_buffer, decode_buffer  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );
  }

  stbir__encode_scanline( stbir_info, ( (char *) stbir_info->output_data ) + ((size_t)n * (size_t)stbir_info->output_stride_bytes),
                          encode_buffer, n  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );
}

static void stbir__decode_and_resample_for_vertical_gather_loop(stbir__info const * stbir_info, stbir__per_split_info* split_info, int n)
{
  int ring_buffer_index;
  float* ring_buffer;

  stbir__decode_scanline( stbir_info, n, split_info->decode_buffer  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

  split_info->ring_buffer_last_scanline = n;

  ring_buffer_index = (split_info->ring_buffer_begin_index + (split_info->ring_buffer_last_scanline - split_info->ring_buffer_first_scanline)) % stbir_info->ring_buffer_num_entries;
  ring_buffer = stbir__get_ring_buffer_entry(stbir_info, split_info, ring_buffer_index);

  stbir__resample_horizontal_gather( stbir_info, ring_buffer, split_info->decode_buffer  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

}

static void stbir__vertical_gather_loop( stbir__info const * stbir_info, stbir__per_split_info* split_info, int split_count )
{
  int y, start_output_y, end_output_y;
  stbir__contributors* vertical_contributors = stbir_info->vertical.contributors;
  float const * vertical_coefficients = stbir_info->vertical.coefficients;

  STBIR_ASSERT( stbir_info->vertical.is_gather );

  start_output_y = split_info->start_output_y;
  end_output_y = split_info[split_count-1].end_output_y;

  vertical_contributors += start_output_y;
  vertical_coefficients += start_output_y * stbir_info->vertical.coefficient_width;

  split_info->ring_buffer_begin_index = 0;
  split_info->ring_buffer_first_scanline = vertical_contributors->n0;
  split_info->ring_buffer_last_scanline = split_info->ring_buffer_first_scanline - 1;

  for (y = start_output_y; y < end_output_y; y++)
  {
    int in_first_scanline, in_last_scanline;

    in_first_scanline = vertical_contributors->n0;
    in_last_scanline = vertical_contributors->n1;

    STBIR_ASSERT( in_first_scanline >= split_info->ring_buffer_first_scanline );

    while (in_last_scanline > split_info->ring_buffer_last_scanline)
    {
      STBIR_ASSERT( ( split_info->ring_buffer_last_scanline - split_info->ring_buffer_first_scanline + 1 ) <= stbir_info->ring_buffer_num_entries );

      if ( ( split_info->ring_buffer_last_scanline - split_info->ring_buffer_first_scanline + 1 ) == stbir_info->ring_buffer_num_entries )
      {
        split_info->ring_buffer_first_scanline++;
        split_info->ring_buffer_begin_index++;
      }

      if ( stbir_info->vertical_first )
      {
        float * ring_buffer = stbir__get_ring_buffer_scanline( stbir_info, split_info, ++split_info->ring_buffer_last_scanline );
        stbir__decode_scanline( stbir_info, split_info->ring_buffer_last_scanline, ring_buffer  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );
      }
      else
      {
        stbir__decode_and_resample_for_vertical_gather_loop(stbir_info, split_info, split_info->ring_buffer_last_scanline + 1);
      }
    }

    stbir__resample_vertical_gather(stbir_info, split_info, y, in_first_scanline, in_last_scanline, vertical_coefficients );

    ++vertical_contributors;
    vertical_coefficients += stbir_info->vertical.coefficient_width;
  }
}

#define STBIR__FLOAT_EMPTY_MARKER 3.0e+38F
#define STBIR__FLOAT_BUFFER_IS_EMPTY(ptr) ((ptr)[0]==STBIR__FLOAT_EMPTY_MARKER)

static void stbir__encode_first_scanline_from_scatter(stbir__info const * stbir_info, stbir__per_split_info* split_info)
{
  float* ring_buffer_entry = stbir__get_ring_buffer_entry(stbir_info, split_info, split_info->ring_buffer_begin_index );

  stbir__encode_scanline( stbir_info, ( (char *)stbir_info->output_data ) + ( (size_t)split_info->ring_buffer_first_scanline * (size_t)stbir_info->output_stride_bytes ), ring_buffer_entry, split_info->ring_buffer_first_scanline  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

  ring_buffer_entry[ 0 ] = STBIR__FLOAT_EMPTY_MARKER;

  split_info->ring_buffer_first_scanline++;
  if ( ++split_info->ring_buffer_begin_index == stbir_info->ring_buffer_num_entries )
    split_info->ring_buffer_begin_index = 0;
}

static void stbir__horizontal_resample_and_encode_first_scanline_from_scatter(stbir__info const * stbir_info, stbir__per_split_info* split_info)
{

  float* ring_buffer_entry = stbir__get_ring_buffer_entry(stbir_info, split_info, split_info->ring_buffer_begin_index );

  stbir__resample_horizontal_gather( stbir_info, split_info->vertical_buffer, ring_buffer_entry  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

  stbir__encode_scanline( stbir_info, ( (char *)stbir_info->output_data ) + ( (size_t)split_info->ring_buffer_first_scanline * (size_t)stbir_info->output_stride_bytes ), split_info->vertical_buffer, split_info->ring_buffer_first_scanline  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

  ring_buffer_entry[ 0 ] = STBIR__FLOAT_EMPTY_MARKER;

  split_info->ring_buffer_first_scanline++;
  if ( ++split_info->ring_buffer_begin_index == stbir_info->ring_buffer_num_entries )
    split_info->ring_buffer_begin_index = 0;
}

static void stbir__resample_vertical_scatter(stbir__info const * stbir_info, stbir__per_split_info* split_info, int n0, int n1, float const * vertical_coefficients, float const * vertical_buffer, float const * vertical_buffer_end )
{
  STBIR_ASSERT( !stbir_info->vertical.is_gather );

  STBIR_PROFILE_START( vertical );
  {
    int k = 0, total = n1 - n0 + 1;
    STBIR_ASSERT( total > 0 );
    do {
      float * outputs[8];
      int i, n = total; if ( n > 8 ) n = 8;
      for( i = 0 ; i < n ; i++ )
      {
        outputs[ i ] = stbir__get_ring_buffer_scanline(stbir_info, split_info, k+i+n0 );
        if ( ( i ) && ( STBIR__FLOAT_BUFFER_IS_EMPTY( outputs[i] ) != STBIR__FLOAT_BUFFER_IS_EMPTY( outputs[0] ) ) )
        {
          n = i;
          break;
        }
      }
      ((STBIR__FLOAT_BUFFER_IS_EMPTY( outputs[0] ))?stbir__vertical_scatter_sets:stbir__vertical_scatter_blends)[n-1]( outputs, vertical_coefficients + k, vertical_buffer, vertical_buffer_end );
      k += n;
      total -= n;
    } while ( total );
  }

  STBIR_PROFILE_END( vertical );
}

typedef void stbir__handle_scanline_for_scatter_func(stbir__info const * stbir_info, stbir__per_split_info* split_info);

static void stbir__vertical_scatter_loop( stbir__info const * stbir_info, stbir__per_split_info* split_info, int split_count )
{
  int y, start_output_y, end_output_y, start_input_y, end_input_y;
  stbir__contributors* vertical_contributors = stbir_info->vertical.contributors;
  float const * vertical_coefficients = stbir_info->vertical.coefficients;
  stbir__handle_scanline_for_scatter_func * handle_scanline_for_scatter;
  void * scanline_scatter_buffer;
  void * scanline_scatter_buffer_end;
  int on_first_input_y, last_input_y;
  int width = (stbir_info->vertical_first) ? ( stbir_info->scanline_extents.conservative.n1-stbir_info->scanline_extents.conservative.n0+1 ) : stbir_info->horizontal.scale_info.output_sub_size;
  int width_times_channels = stbir_info->effective_channels * width;

  STBIR_ASSERT( !stbir_info->vertical.is_gather );

  start_output_y = split_info->start_output_y;
  end_output_y = split_info[split_count-1].end_output_y;

  start_input_y = split_info->start_input_y;
  end_input_y = split_info[split_count-1].end_input_y;

  y = start_input_y + stbir_info->vertical.filter_pixel_margin;
  vertical_contributors += y ;
  vertical_coefficients += stbir_info->vertical.coefficient_width * y;

  if ( stbir_info->vertical_first )
  {
    handle_scanline_for_scatter = stbir__horizontal_resample_and_encode_first_scanline_from_scatter;
    scanline_scatter_buffer = split_info->decode_buffer;
    scanline_scatter_buffer_end = ( (char*) scanline_scatter_buffer ) + sizeof( float ) * stbir_info->effective_channels * (stbir_info->scanline_extents.conservative.n1-stbir_info->scanline_extents.conservative.n0+1);
  }
  else
  {
    handle_scanline_for_scatter = stbir__encode_first_scanline_from_scatter;
    scanline_scatter_buffer = split_info->vertical_buffer;
    scanline_scatter_buffer_end = ( (char*) scanline_scatter_buffer ) + sizeof( float ) * stbir_info->effective_channels * stbir_info->horizontal.scale_info.output_sub_size;
  }

  split_info->ring_buffer_first_scanline = start_output_y;
  split_info->ring_buffer_last_scanline = -1;
  split_info->ring_buffer_begin_index = -1;

  for( y = 0 ; y < stbir_info->ring_buffer_num_entries ; y++ )
  {
    float * decode_buffer = stbir__get_ring_buffer_entry( stbir_info, split_info, y );
    decode_buffer[ width_times_channels ] = 0.0f;
    decode_buffer[ width_times_channels+1 ] = 0.0f;
    decode_buffer[0] = STBIR__FLOAT_EMPTY_MARKER;
  }

  on_first_input_y = 1; last_input_y = start_input_y;
  for (y = start_input_y ; y < end_input_y; y++)
  {
    int out_first_scanline, out_last_scanline;

    out_first_scanline = vertical_contributors->n0;
    out_last_scanline = vertical_contributors->n1;

    STBIR_ASSERT(out_last_scanline - out_first_scanline + 1 <= stbir_info->ring_buffer_num_entries);

    if ( ( out_last_scanline >= out_first_scanline ) && ( ( ( out_first_scanline >= start_output_y ) && ( out_first_scanline < end_output_y ) ) || ( ( out_last_scanline >= start_output_y ) && ( out_last_scanline < end_output_y ) ) ) )
    {
      float const * vc = vertical_coefficients;

      last_input_y = y;
      if ( ( on_first_input_y ) && ( y > start_input_y ) )
        split_info->start_input_y = y;
      on_first_input_y = 0;

      if ( out_first_scanline < start_output_y )
      {
        vc += start_output_y - out_first_scanline;
        out_first_scanline = start_output_y;
      }

      if ( out_last_scanline >= end_output_y )
        out_last_scanline = end_output_y - 1;

      if (split_info->ring_buffer_begin_index < 0)
        split_info->ring_buffer_begin_index = out_first_scanline - start_output_y;

      STBIR_ASSERT( split_info->ring_buffer_begin_index <= out_first_scanline );

      stbir__decode_scanline( stbir_info, y, split_info->decode_buffer  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

      if ( !stbir_info->vertical_first )
        stbir__resample_horizontal_gather( stbir_info, split_info->vertical_buffer, split_info->decode_buffer  STBIR_ONLY_PROFILE_SET_SPLIT_INFO );

      if ( ( ( split_info->ring_buffer_last_scanline - split_info->ring_buffer_first_scanline + 1 ) == stbir_info->ring_buffer_num_entries ) &&
           ( out_last_scanline > split_info->ring_buffer_last_scanline ) )
        handle_scanline_for_scatter( stbir_info, split_info );

      stbir__resample_vertical_scatter(stbir_info, split_info, out_first_scanline, out_last_scanline, vc, (float*)scanline_scatter_buffer, (float*)scanline_scatter_buffer_end );

      if ( out_last_scanline > split_info->ring_buffer_last_scanline )
        split_info->ring_buffer_last_scanline = out_last_scanline;
    }
    ++vertical_contributors;
    vertical_coefficients += stbir_info->vertical.coefficient_width;
  }

  while ( split_info->ring_buffer_first_scanline < end_output_y )
    handle_scanline_for_scatter(stbir_info, split_info);

  ++last_input_y;
  for( y = 0 ; y < split_count; y++ )
    if ( split_info[y].end_input_y > last_input_y )
      split_info[y].end_input_y = last_input_y;
}

static stbir__kernel_callback * stbir__builtin_kernels[] =   { 0, stbir__filter_trapezoid,  stbir__filter_triangle, stbir__filter_cubic, stbir__filter_catmullrom, stbir__filter_mitchell, stbir__filter_point };
static stbir__support_callback * stbir__builtin_supports[] = { 0, stbir__support_trapezoid, stbir__support_one,     stbir__support_two,  stbir__support_two,       stbir__support_two,     stbir__support_zeropoint5 };

static void stbir__set_sampler(stbir__sampler * samp, stbir_filter filter, stbir__kernel_callback * kernel, stbir__support_callback * support, stbir_edge edge, stbir__scale_info * scale_info, int always_gather, void * user_data )
{
  if (filter == 0)
  {
    filter = STBIR_DEFAULT_FILTER_DOWNSAMPLE;
    if (scale_info->scale >= ( 1.0f - stbir__small_float ) )
    {
      if ( (scale_info->scale <= ( 1.0f + stbir__small_float ) ) && ( STBIR_CEILF(scale_info->pixel_shift) == scale_info->pixel_shift ) )
        filter = STBIR_FILTER_POINT_SAMPLE;
      else
        filter = STBIR_DEFAULT_FILTER_UPSAMPLE;
    }
  }
  samp->filter_enum = filter;

  STBIR_ASSERT(samp->filter_enum != 0);
  STBIR_ASSERT((unsigned)samp->filter_enum < STBIR_FILTER_OTHER);
  samp->filter_kernel = stbir__builtin_kernels[ filter ];
  samp->filter_support = stbir__builtin_supports[ filter ];

  if ( kernel && support )
  {
    samp->filter_kernel = kernel;
    samp->filter_support = support;
    samp->filter_enum = STBIR_FILTER_OTHER;
  }

  samp->edge = edge;
  samp->filter_pixel_width  = stbir__get_filter_pixel_width (samp->filter_support, scale_info->scale, user_data );
  samp->is_gather = 0;
  if ( scale_info->scale >= ( 1.0f - stbir__small_float ) )
    samp->is_gather = 1;
  else if ( ( always_gather ) || ( samp->filter_pixel_width <= STBIR_FORCE_GATHER_FILTER_SCANLINES_AMOUNT ) )
    samp->is_gather = 2;

  samp->coefficient_width = stbir__get_coefficient_width(samp, samp->is_gather, user_data);

  if ( edge == STBIR_EDGE_WRAP )
    if ( samp->filter_pixel_width > ( scale_info->input_full_size * 3 ) )
      samp->filter_pixel_width = scale_info->input_full_size * 3;

  samp->filter_pixel_margin = samp->filter_pixel_width / 2;

  if ( edge == STBIR_EDGE_WRAP )
    if ( samp->filter_pixel_margin > scale_info->input_full_size )
      samp->filter_pixel_margin = scale_info->input_full_size;

  samp->num_contributors = stbir__get_contributors(samp, samp->is_gather);

  samp->contributors_size = samp->num_contributors * sizeof(stbir__contributors);
  samp->coefficients_size = samp->num_contributors * samp->coefficient_width * sizeof(float) + sizeof(float)*STBIR_INPUT_CALLBACK_PADDING;

  samp->gather_prescatter_contributors = 0;
  samp->gather_prescatter_coefficients = 0;
  if ( samp->is_gather == 0 )
  {
    samp->gather_prescatter_coefficient_width = samp->filter_pixel_width;
    samp->gather_prescatter_num_contributors  = stbir__get_contributors(samp, 2);
    samp->gather_prescatter_contributors_size = samp->gather_prescatter_num_contributors * sizeof(stbir__contributors);
    samp->gather_prescatter_coefficients_size = samp->gather_prescatter_num_contributors * samp->gather_prescatter_coefficient_width * sizeof(float);
  }
}

static void stbir__get_conservative_extents( stbir__sampler * samp, stbir__contributors * range, void * user_data )
{
  float scale = samp->scale_info.scale;
  float out_shift = samp->scale_info.pixel_shift;
  stbir__support_callback * support = samp->filter_support;
  int input_full_size = samp->scale_info.input_full_size;
  stbir_edge edge = samp->edge;
  float inv_scale = samp->scale_info.inv_scale;

  STBIR_ASSERT( samp->is_gather != 0 );

  if ( samp->is_gather == 1 )
  {
    int in_first_pixel, in_last_pixel;
    float out_filter_radius = support(inv_scale, user_data) * scale;

    stbir__calculate_in_pixel_range( &in_first_pixel, &in_last_pixel, 0.5, out_filter_radius, inv_scale, out_shift, input_full_size, edge );
    range->n0 = in_first_pixel;
    stbir__calculate_in_pixel_range( &in_first_pixel, &in_last_pixel, ( (float)(samp->scale_info.output_sub_size-1) ) + 0.5f, out_filter_radius, inv_scale, out_shift, input_full_size, edge );
    range->n1 = in_last_pixel;
  }
  else if ( samp->is_gather == 2 )
  {
    float in_pixels_radius = support(scale, user_data) * inv_scale;
    int filter_pixel_margin = samp->filter_pixel_margin;
    int output_sub_size = samp->scale_info.output_sub_size;
    int input_end;
    int n;
    int in_first_pixel, in_last_pixel;

    stbir__calculate_in_pixel_range( &in_first_pixel, &in_last_pixel, 0, 0, inv_scale, out_shift, input_full_size, edge );
    range->n0 = in_first_pixel;
    stbir__calculate_in_pixel_range( &in_first_pixel, &in_last_pixel, (float)output_sub_size, 0, inv_scale, out_shift, input_full_size, edge );
    range->n1 = in_last_pixel;

    n = range->n0 + 1;
    input_end = -filter_pixel_margin;
    while( n >= input_end )
    {
      int out_first_pixel, out_last_pixel;
      stbir__calculate_out_pixel_range( &out_first_pixel, &out_last_pixel, ((float)n)+0.5f, in_pixels_radius, scale, out_shift, output_sub_size );
      if ( out_first_pixel > out_last_pixel )
        break;

      if ( ( out_first_pixel < output_sub_size ) || ( out_last_pixel >= 0 ) )
        range->n0 = n;
      --n;
    }

    n = range->n1 - 1;
    input_end = n + 1 + filter_pixel_margin;
    while( n <= input_end )
    {
      int out_first_pixel, out_last_pixel;
      stbir__calculate_out_pixel_range( &out_first_pixel, &out_last_pixel, ((float)n)+0.5f, in_pixels_radius, scale, out_shift, output_sub_size );
      if ( out_first_pixel > out_last_pixel )
        break;
      if ( ( out_first_pixel < output_sub_size ) || ( out_last_pixel >= 0 ) )
        range->n1 = n;
      ++n;
    }
  }

  if ( samp->edge == STBIR_EDGE_WRAP )
  {
    if ( ( range->n0 > 0 ) && ( range->n1 >= input_full_size ) )
    {
      int marg = range->n1 - input_full_size + 1;
      if ( ( marg + STBIR__MERGE_RUNS_PIXEL_THRESHOLD ) >= range->n0 )
        range->n0 = 0;
    }
    if ( ( range->n0 < 0 ) && ( range->n1 < (input_full_size-1) ) )
    {
      int marg = -range->n0;
      if ( ( input_full_size - marg - STBIR__MERGE_RUNS_PIXEL_THRESHOLD - 1 ) <= range->n1 )
        range->n1 = input_full_size - 1;
    }
  }
  else
  {
    if ( range->n0 < 0 )
      range->n0 = 0;
    if ( range->n1 >= input_full_size )
      range->n1 = input_full_size - 1;
  }
}

static void stbir__get_split_info( stbir__per_split_info* split_info, int splits, int output_height, int vertical_pixel_margin, int input_full_height, int is_gather, stbir__contributors * contribs )
{
  int i, cur;
  int left = output_height;

  cur = 0;
  for( i = 0 ; i < splits ; i++ )
  {
    int each;

    split_info[i].start_output_y = cur;
    each = left / ( splits - i );
    split_info[i].end_output_y = cur + each;

    if ( ( is_gather ) && ( i ) )
    {
      stbir__contributors * small_contribs;
      int j, smallest, stop, start_n0;
      stbir__contributors * split_contribs = contribs + cur;

      stop = vertical_pixel_margin * 3;
      if ( each < stop )
        stop = each;

      smallest = 0;
      small_contribs = split_contribs;
      start_n0 = small_contribs->n0;
      for( j = 1 ; j <= stop ; j++ )
      {
        ++split_contribs;
        if ( split_contribs->n0 > start_n0 )
          break;
        if ( split_contribs->n0 < small_contribs->n0 )
        {
          small_contribs = split_contribs;
          smallest = j;
        }
      }

      split_info[i-1].end_output_y += smallest;
      split_info[i].start_output_y += smallest;
    }

    cur += each;
    left -= each;

    split_info[i].start_input_y = -vertical_pixel_margin;
    split_info[i].end_input_y = input_full_height + vertical_pixel_margin;
  }
}

static void stbir__free_internal_mem( stbir__info *info )
{
  #define STBIR__FREE_AND_CLEAR( ptr ) { if ( ptr ) { void * p = (ptr); (ptr) = 0; STBIR_FREE( p, info->user_data); } }

  if ( info )
  {
  #ifndef STBIR__SEPARATE_ALLOCATIONS
    STBIR__FREE_AND_CLEAR( info->alloced_mem );
  #else
    int i,j;

    if ( ( info->vertical.gather_prescatter_contributors ) && ( (void*)info->vertical.gather_prescatter_contributors != (void*)info->split_info[0].decode_buffer ) )
    {
      STBIR__FREE_AND_CLEAR( info->vertical.gather_prescatter_coefficients );
      STBIR__FREE_AND_CLEAR( info->vertical.gather_prescatter_contributors );
    }
    for( i = 0 ; i < info->splits ; i++ )
    {
      for( j = 0 ; j < info->alloc_ring_buffer_num_entries ; j++ )
      {
        STBIR__FREE_AND_CLEAR( info->split_info[i].ring_buffers[j] );
      }

      STBIR__FREE_AND_CLEAR( info->split_info[i].decode_buffer );
      STBIR__FREE_AND_CLEAR( info->split_info[i].ring_buffers );
      STBIR__FREE_AND_CLEAR( info->split_info[i].vertical_buffer );
    }
    STBIR__FREE_AND_CLEAR( info->split_info );
    if ( info->vertical.coefficients != info->horizontal.coefficients )
    {
      STBIR__FREE_AND_CLEAR( info->vertical.coefficients );
      STBIR__FREE_AND_CLEAR( info->vertical.contributors );
    }
    STBIR__FREE_AND_CLEAR( info->horizontal.coefficients );
    STBIR__FREE_AND_CLEAR( info->horizontal.contributors );
    STBIR__FREE_AND_CLEAR( info->alloced_mem );
    STBIR_FREE( info, info->user_data );
  #endif
  }

  #undef STBIR__FREE_AND_CLEAR
}

static int stbir__get_max_split( int splits, int height )
{
  int i;
  int max = 0;

  for( i = 0 ; i < splits ; i++ )
  {
    int each = height / ( splits - i );
    if ( each > max )
      max = each;
    height -= each;
  }
  return max;
}

static stbir__horizontal_gather_channels_func ** stbir__horizontal_gather_n_coeffs_funcs[8] =
{
  0, stbir__horizontal_gather_1_channels_with_n_coeffs_funcs, stbir__horizontal_gather_2_channels_with_n_coeffs_funcs, stbir__horizontal_gather_3_channels_with_n_coeffs_funcs, stbir__horizontal_gather_4_channels_with_n_coeffs_funcs, 0,0, stbir__horizontal_gather_7_channels_with_n_coeffs_funcs
};

static stbir__horizontal_gather_channels_func ** stbir__horizontal_gather_channels_funcs[8] =
{
  0, stbir__horizontal_gather_1_channels_funcs, stbir__horizontal_gather_2_channels_funcs, stbir__horizontal_gather_3_channels_funcs, stbir__horizontal_gather_4_channels_funcs, 0,0, stbir__horizontal_gather_7_channels_funcs
};

#define STBIR_RESIZE_CLASSIFICATIONS 8

static float stbir__compute_weights[5][STBIR_RESIZE_CLASSIFICATIONS][4]=
{
  {
    { 1.00000f, 1.00000f, 0.31250f, 1.00000f },
    { 0.56250f, 0.59375f, 0.00000f, 0.96875f },
    { 1.00000f, 0.06250f, 0.00000f, 1.00000f },
    { 0.00000f, 0.09375f, 1.00000f, 1.00000f },
    { 1.00000f, 1.00000f, 1.00000f, 1.00000f },
    { 0.03125f, 0.12500f, 1.00000f, 1.00000f },
    { 0.06250f, 0.12500f, 0.00000f, 1.00000f },
    { 0.00000f, 1.00000f, 0.00000f, 0.03125f },
  }, {
    { 0.00000f, 0.84375f, 0.00000f, 0.03125f },
    { 0.09375f, 0.93750f, 0.00000f, 0.78125f },
    { 0.87500f, 0.21875f, 0.00000f, 0.96875f },
    { 0.09375f, 0.09375f, 1.00000f, 1.00000f },
    { 1.00000f, 1.00000f, 1.00000f, 1.00000f },
    { 0.03125f, 0.12500f, 1.00000f, 1.00000f },
    { 0.06250f, 0.12500f, 0.00000f, 1.00000f },
    { 0.00000f, 1.00000f, 0.00000f, 0.53125f },
  }, {
    { 0.00000f, 0.53125f, 0.00000f, 0.03125f },
    { 0.06250f, 0.96875f, 0.00000f, 0.53125f },
    { 0.87500f, 0.18750f, 0.00000f, 0.93750f },
    { 0.00000f, 0.09375f, 1.00000f, 1.00000f },
    { 1.00000f, 1.00000f, 1.00000f, 1.00000f },
    { 0.03125f, 0.12500f, 1.00000f, 1.00000f },
    { 0.06250f, 0.12500f, 0.00000f, 1.00000f },
    { 0.00000f, 1.00000f, 0.00000f, 0.56250f },
  }, {
    { 0.00000f, 0.50000f, 0.00000f, 0.71875f },
    { 0.06250f, 0.84375f, 0.00000f, 0.87500f },
    { 1.00000f, 0.50000f, 0.50000f, 0.96875f },
    { 1.00000f, 0.09375f, 0.31250f, 0.50000f },
    { 1.00000f, 1.00000f, 1.00000f, 1.00000f },
    { 1.00000f, 0.03125f, 0.03125f, 0.53125f },
    { 0.18750f, 0.12500f, 0.00000f, 1.00000f },
    { 0.00000f, 1.00000f, 0.03125f, 0.18750f },
  }, {
    { 0.00000f, 0.59375f, 0.00000f, 0.96875f },
    { 0.06250f, 0.81250f, 0.06250f, 0.59375f },
    { 0.75000f, 0.43750f, 0.12500f, 0.96875f },
    { 0.87500f, 0.06250f, 0.18750f, 0.43750f },
    { 1.00000f, 1.00000f, 1.00000f, 1.00000f },
    { 0.15625f, 0.12500f, 1.00000f, 1.00000f },
    { 0.06250f, 0.12500f, 0.00000f, 1.00000f },
    { 0.00000f, 1.00000f, 0.03125f, 0.34375f },
  }
};

typedef struct STBIR__V_FIRST_INFO
{
  double v_cost, h_cost;
  int control_v_first;
  int v_first;
  int v_resize_classification;
  int is_gather;
} STBIR__V_FIRST_INFO;

#ifdef STBIR__V_FIRST_INFO_BUFFER
static STBIR__V_FIRST_INFO STBIR__V_FIRST_INFO_BUFFER = {0};
#define STBIR__V_FIRST_INFO_POINTER &STBIR__V_FIRST_INFO_BUFFER
#else
#define STBIR__V_FIRST_INFO_POINTER 0
#endif

static int stbir__should_do_vertical_first( float weights_table[STBIR_RESIZE_CLASSIFICATIONS][4], int horizontal_filter_pixel_width, float horizontal_scale, int horizontal_output_size, int vertical_filter_pixel_width, float vertical_scale, int vertical_output_size, int is_gather, STBIR__V_FIRST_INFO * info )
{
  double v_cost, h_cost;
  float * weights;
  int vertical_first;
  int v_classification;

  if ( ( vertical_output_size <= 4 ) || ( horizontal_output_size <= 4 ) )
    v_classification = ( vertical_output_size < horizontal_output_size ) ? 6 : 7;
  else if ( vertical_scale <= 1.0f )
    v_classification = ( is_gather ) ? 1 : 0;
  else if ( vertical_scale <= 2.0f)
    v_classification = 2;
  else if ( vertical_scale <= 3.0f)
    v_classification = 3;
  else if ( vertical_scale <= 4.0f)
    v_classification = 5;
  else
    v_classification = 6;

  weights = weights_table[ v_classification ];

  h_cost = (float)horizontal_filter_pixel_width * weights[0] + horizontal_scale * (float)vertical_filter_pixel_width * weights[1];
  v_cost = (float)vertical_filter_pixel_width  * weights[2] + vertical_scale * (float)horizontal_filter_pixel_width * weights[3];

  vertical_first = ( v_cost <= h_cost ) ? 1 : 0;

  if ( info )
  {
    info->h_cost = h_cost;
    info->v_cost = v_cost;
    info->v_resize_classification = v_classification;
    info->v_first = vertical_first;
    info->is_gather = is_gather;
  }

  if ( ( info ) && ( info->control_v_first ) )
    vertical_first = ( info->control_v_first == 2 ) ? 1 : 0;

  return vertical_first;
}

static unsigned char stbir__pixel_channels[] = {
  1,2,3,3,4,
  4,4,4,4,2,2,
  4,4,4,4,2,2,
};

static stbir_internal_pixel_layout stbir__pixel_layout_convert_public_to_internal[] = {
  STBIRI_BGR, STBIRI_1CHANNEL, STBIRI_2CHANNEL, STBIRI_RGB, STBIRI_RGBA,
  STBIRI_4CHANNEL, STBIRI_BGRA, STBIRI_ARGB, STBIRI_ABGR, STBIRI_RA, STBIRI_AR,
  STBIRI_RGBA_PM, STBIRI_BGRA_PM, STBIRI_ARGB_PM, STBIRI_ABGR_PM, STBIRI_RA_PM, STBIRI_AR_PM,
};

static stbir__info * stbir__alloc_internal_mem_and_build_samplers( stbir__sampler * horizontal, stbir__sampler * vertical, stbir__contributors * conservative, stbir_pixel_layout input_pixel_layout_public, stbir_pixel_layout output_pixel_layout_public, int splits, int new_x, int new_y, int fast_alpha, void * user_data STBIR_ONLY_PROFILE_BUILD_GET_INFO )
{
  static char stbir_channel_count_index[8]={ 9,0,1,2, 3,9,9,4 };

  stbir__info * info = 0;
  void * alloced = 0;
  size_t alloced_total = 0;
  int vertical_first;
  size_t decode_buffer_size, ring_buffer_length_bytes, ring_buffer_size, vertical_buffer_size;
  int alloc_ring_buffer_num_entries;

  int alpha_weighting_type = 0;
  int conservative_split_output_size = stbir__get_max_split( splits, vertical->scale_info.output_sub_size );
  stbir_internal_pixel_layout input_pixel_layout = stbir__pixel_layout_convert_public_to_internal[ input_pixel_layout_public ];
  stbir_internal_pixel_layout output_pixel_layout = stbir__pixel_layout_convert_public_to_internal[ output_pixel_layout_public ];
  int channels = stbir__pixel_channels[ input_pixel_layout ];
  int effective_channels = channels;

  if ( ( horizontal->filter_enum != STBIR_FILTER_POINT_SAMPLE ) || ( vertical->filter_enum != STBIR_FILTER_POINT_SAMPLE ) )
  {
    if ( ( input_pixel_layout >= STBIRI_RGBA ) && ( input_pixel_layout <= STBIRI_AR ) && ( output_pixel_layout >= STBIRI_RGBA ) && ( output_pixel_layout <= STBIRI_AR ) )
    {
      if ( fast_alpha )
      {
        alpha_weighting_type = 4;
      }
      else
      {
        static int fancy_alpha_effective_cnts[6] = { 7, 7, 7, 7, 3, 3 };
        alpha_weighting_type = 2;
        effective_channels = fancy_alpha_effective_cnts[ input_pixel_layout - STBIRI_RGBA ];
      }
    }
    else if ( ( input_pixel_layout >= STBIRI_RGBA_PM ) && ( input_pixel_layout <= STBIRI_AR_PM ) && ( output_pixel_layout >= STBIRI_RGBA ) && ( output_pixel_layout <= STBIRI_AR ) )
    {
      alpha_weighting_type = 3;
    }
    else if ( ( input_pixel_layout >= STBIRI_RGBA ) && ( input_pixel_layout <= STBIRI_AR ) && ( output_pixel_layout >= STBIRI_RGBA_PM ) && ( output_pixel_layout <= STBIRI_AR_PM ) )
    {
      alpha_weighting_type = 1;
    }
  }

  if ( channels != stbir__pixel_channels[ output_pixel_layout ] )
    return 0;

  vertical_first = stbir__should_do_vertical_first( stbir__compute_weights[ (int)stbir_channel_count_index[ effective_channels ] ], horizontal->filter_pixel_width, horizontal->scale_info.scale, horizontal->scale_info.output_sub_size, vertical->filter_pixel_width, vertical->scale_info.scale, vertical->scale_info.output_sub_size, vertical->is_gather, STBIR__V_FIRST_INFO_POINTER );

  decode_buffer_size = ( conservative->n1 - conservative->n0 + 1 ) * effective_channels * sizeof(float) + sizeof(float)*STBIR_INPUT_CALLBACK_PADDING;

  ring_buffer_length_bytes = (size_t)horizontal->scale_info.output_sub_size * (size_t)effective_channels * sizeof(float) + sizeof(float)*STBIR_INPUT_CALLBACK_PADDING;

  if ( vertical_first )
    ring_buffer_length_bytes = ( decode_buffer_size + 15 ) & ~15;

  if ( ( ring_buffer_length_bytes & 4095 ) == 0 ) ring_buffer_length_bytes += 64*3;

  alloc_ring_buffer_num_entries = vertical->filter_pixel_width + 1;

  if ( ( !vertical->is_gather ) && ( alloc_ring_buffer_num_entries > conservative_split_output_size ) )
    alloc_ring_buffer_num_entries = conservative_split_output_size;

  ring_buffer_size = (size_t)alloc_ring_buffer_num_entries * (size_t)ring_buffer_length_bytes;

  vertical_buffer_size = (size_t)horizontal->scale_info.output_sub_size * (size_t)effective_channels * sizeof(float) + sizeof(float);

  for(;;)
  {
    int i;
    void * advance_mem = alloced;
    int copy_horizontal = 0;
    stbir__sampler * possibly_use_horizontal_for_pivot = 0;

#ifdef STBIR__SEPARATE_ALLOCATIONS
    #define STBIR__NEXT_PTR( ptr, size, ntype ) if ( alloced ) { void * p = STBIR_MALLOC( size, user_data); if ( p == 0 ) { stbir__free_internal_mem( info ); return 0; } (ptr) = (ntype*)p; }
#else
    #define STBIR__NEXT_PTR( ptr, size, ntype ) advance_mem = (void*) ( ( ((size_t)advance_mem) + 15 ) & ~15 ); if ( alloced ) ptr = (ntype*)advance_mem; advance_mem = (char*)(((size_t)advance_mem) + (size));
#endif

    STBIR__NEXT_PTR( info, sizeof( stbir__info ), stbir__info );

    STBIR__NEXT_PTR( info->split_info, sizeof( stbir__per_split_info ) * splits, stbir__per_split_info );

    if ( info )
    {
      static stbir__alpha_weight_func * fancy_alpha_weights[6]  =    { stbir__fancy_alpha_weight_4ch,   stbir__fancy_alpha_weight_4ch,   stbir__fancy_alpha_weight_4ch,   stbir__fancy_alpha_weight_4ch,   stbir__fancy_alpha_weight_2ch,   stbir__fancy_alpha_weight_2ch };
      static stbir__alpha_unweight_func * fancy_alpha_unweights[6] = { stbir__fancy_alpha_unweight_4ch, stbir__fancy_alpha_unweight_4ch, stbir__fancy_alpha_unweight_4ch, stbir__fancy_alpha_unweight_4ch, stbir__fancy_alpha_unweight_2ch, stbir__fancy_alpha_unweight_2ch };
      static stbir__alpha_weight_func * simple_alpha_weights[6] = { stbir__simple_alpha_weight_4ch, stbir__simple_alpha_weight_4ch, stbir__simple_alpha_weight_4ch, stbir__simple_alpha_weight_4ch, stbir__simple_alpha_weight_2ch, stbir__simple_alpha_weight_2ch };
      static stbir__alpha_unweight_func * simple_alpha_unweights[6] = { stbir__simple_alpha_unweight_4ch, stbir__simple_alpha_unweight_4ch, stbir__simple_alpha_unweight_4ch, stbir__simple_alpha_unweight_4ch, stbir__simple_alpha_unweight_2ch, stbir__simple_alpha_unweight_2ch };

      info->alloced_mem = alloced;
      info->alloced_total = alloced_total;

      info->channels = channels;
      info->effective_channels = effective_channels;

      info->offset_x = new_x;
      info->offset_y = new_y;
      info->alloc_ring_buffer_num_entries = (int)alloc_ring_buffer_num_entries;
      info->ring_buffer_num_entries = 0;
      info->ring_buffer_length_bytes = (int)ring_buffer_length_bytes;
      info->splits = splits;
      info->vertical_first = vertical_first;

      info->input_pixel_layout_internal = input_pixel_layout;
      info->output_pixel_layout_internal = output_pixel_layout;

      info->alpha_weight = 0;
      info->alpha_unweight = 0;

      if ( alpha_weighting_type == 2 )
      {
        info->alpha_weight = fancy_alpha_weights[ input_pixel_layout - STBIRI_RGBA ];
        info->alpha_unweight = fancy_alpha_unweights[ output_pixel_layout - STBIRI_RGBA ];
      }
      else if ( alpha_weighting_type == 4 )
      {
        info->alpha_weight = simple_alpha_weights[ input_pixel_layout - STBIRI_RGBA ];
        info->alpha_unweight = simple_alpha_unweights[ output_pixel_layout - STBIRI_RGBA ];
      }
      else if ( alpha_weighting_type == 1 )
      {
        info->alpha_weight = simple_alpha_weights[ input_pixel_layout - STBIRI_RGBA ];
      }
      else if ( alpha_weighting_type == 3 )
      {
        info->alpha_unweight = simple_alpha_unweights[ output_pixel_layout - STBIRI_RGBA ];
      }

      if ( ( ( input_pixel_layout == STBIRI_RGB ) && ( output_pixel_layout == STBIRI_BGR ) ) ||
           ( ( input_pixel_layout == STBIRI_BGR ) && ( output_pixel_layout == STBIRI_RGB ) ) )
      {
        if ( horizontal->scale_info.scale < 1.0f )
          info->alpha_unweight = stbir__simple_flip_3ch;
        else
          info->alpha_weight = stbir__simple_flip_3ch;
      }

    }

    for( i = 0 ; i < splits ; i++ )
    {
      STBIR__NEXT_PTR( info->split_info[i].decode_buffer, decode_buffer_size, float );

#ifdef STBIR__SEPARATE_ALLOCATIONS

      STBIR__NEXT_PTR( info->split_info[i].ring_buffers, alloc_ring_buffer_num_entries * sizeof(float*), float* );
      {
        int j;
        for( j = 0 ; j < alloc_ring_buffer_num_entries ; j++ )
        {
          STBIR__NEXT_PTR( info->split_info[i].ring_buffers[j], ring_buffer_length_bytes, float );
        }
      }
#else
      STBIR__NEXT_PTR( info->split_info[i].ring_buffer, ring_buffer_size, float );
#endif
      STBIR__NEXT_PTR( info->split_info[i].vertical_buffer, vertical_buffer_size, float );
    }

    if ( vertical->is_gather == 0 )
    {
      size_t both;
      size_t temp_mem_amt;

      both = (size_t)vertical->gather_prescatter_contributors_size + (size_t)vertical->gather_prescatter_coefficients_size;

#ifdef STBIR__SEPARATE_ALLOCATIONS
      temp_mem_amt = decode_buffer_size;

#else
      temp_mem_amt = (size_t)( decode_buffer_size + ring_buffer_size + vertical_buffer_size ) * (size_t)splits;
#endif
      if ( temp_mem_amt >= both )
      {
        if ( info )
        {
          vertical->gather_prescatter_contributors = (stbir__contributors*)info->split_info[0].decode_buffer;
          vertical->gather_prescatter_coefficients = (float*) ( ( (char*)info->split_info[0].decode_buffer ) + vertical->gather_prescatter_contributors_size );
        }
      }
      else
      {
        STBIR__NEXT_PTR( vertical->gather_prescatter_contributors, vertical->gather_prescatter_contributors_size, stbir__contributors );
        STBIR__NEXT_PTR( vertical->gather_prescatter_coefficients, vertical->gather_prescatter_coefficients_size, float );
      }
    }

    STBIR__NEXT_PTR( horizontal->contributors, horizontal->contributors_size, stbir__contributors );
    STBIR__NEXT_PTR( horizontal->coefficients, horizontal->coefficients_size, float );

    if ( ( horizontal->filter_kernel == vertical->filter_kernel ) && ( horizontal->filter_support == vertical->filter_support ) && ( horizontal->edge == vertical->edge ) && ( horizontal->scale_info.output_sub_size == vertical->scale_info.output_sub_size ) )
    {
      float diff_scale = horizontal->scale_info.scale - vertical->scale_info.scale;
      float diff_shift = horizontal->scale_info.pixel_shift - vertical->scale_info.pixel_shift;
      if ( diff_scale < 0.0f ) diff_scale = -diff_scale;
      if ( diff_shift < 0.0f ) diff_shift = -diff_shift;
      if ( ( diff_scale <= stbir__small_float ) && ( diff_shift <= stbir__small_float ) )
      {
        if ( horizontal->is_gather == vertical->is_gather )
        {
          copy_horizontal = 1;
          goto no_vert_alloc;
        }
        possibly_use_horizontal_for_pivot = horizontal;
      }
    }

    STBIR__NEXT_PTR( vertical->contributors, vertical->contributors_size, stbir__contributors );
    STBIR__NEXT_PTR( vertical->coefficients, vertical->coefficients_size, float );

   no_vert_alloc:

    if ( info )
    {
      STBIR_PROFILE_BUILD_START( horizontal );

      stbir__calculate_filters( horizontal, 0, user_data STBIR_ONLY_PROFILE_BUILD_SET_INFO );

      info->horizontal_gather_channels = stbir__horizontal_gather_n_coeffs_funcs[ effective_channels ][ horizontal->extent_info.widest & 3 ];
      if ( horizontal->extent_info.widest <= 12 )
        info->horizontal_gather_channels = stbir__horizontal_gather_channels_funcs[ effective_channels ][ horizontal->extent_info.widest - 1 ];

      info->scanline_extents.conservative.n0 = conservative->n0;
      info->scanline_extents.conservative.n1 = conservative->n1;

      stbir__get_extents( horizontal, &info->scanline_extents );

      horizontal->coefficient_width = stbir__pack_coefficients(horizontal->num_contributors, horizontal->contributors, horizontal->coefficients, horizontal->coefficient_width, horizontal->extent_info.widest, info->scanline_extents.conservative.n0, info->scanline_extents.conservative.n1 );

      STBIR_MEMCPY( &info->horizontal, horizontal, sizeof( stbir__sampler ) );

      STBIR_PROFILE_BUILD_END( horizontal );

      if ( copy_horizontal )
      {
        STBIR_MEMCPY( &info->vertical, horizontal, sizeof( stbir__sampler ) );
      }
      else
      {
        STBIR_PROFILE_BUILD_START( vertical );

        stbir__calculate_filters( vertical, possibly_use_horizontal_for_pivot, user_data STBIR_ONLY_PROFILE_BUILD_SET_INFO );
        STBIR_MEMCPY( &info->vertical, vertical, sizeof( stbir__sampler ) );

        STBIR_PROFILE_BUILD_END( vertical );
      }

      stbir__get_split_info( info->split_info, info->splits, info->vertical.scale_info.output_sub_size, info->vertical.filter_pixel_margin, info->vertical.scale_info.input_full_size, info->vertical.is_gather, info->vertical.contributors );

      info->ring_buffer_num_entries = info->vertical.extent_info.widest;

      if ( ( !info->vertical.is_gather ) && ( info->ring_buffer_num_entries > conservative_split_output_size ) )
        info->ring_buffer_num_entries = conservative_split_output_size;
      STBIR_ASSERT( info->ring_buffer_num_entries <= info->alloc_ring_buffer_num_entries );
    }
    #undef STBIR__NEXT_PTR

    if ( info == 0 )
    {
      alloced_total = ( 15 + (size_t)advance_mem );
      alloced = STBIR_MALLOC( alloced_total, user_data );
      if ( alloced == 0 )
        return 0;
    }
    else
      return info;
  }
}

static int stbir__perform_resize( stbir__info const * info, int split_start, int split_count )
{
  stbir__per_split_info * split_info = info->split_info + split_start;

  STBIR_PROFILE_CLEAR_EXTRAS();

  STBIR_PROFILE_FIRST_START( looping );
  if (info->vertical.is_gather)
    stbir__vertical_gather_loop( info, split_info, split_count );
  else
    stbir__vertical_scatter_loop( info, split_info, split_count );
  STBIR_PROFILE_END( looping );

  return 1;
}

static void stbir__update_info_from_resize( stbir__info * info, STBIR_RESIZE * resize )
{
  static stbir__decode_pixels_func * decode_simple[STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]=
  {
     stbir__decode_uint8_srgb, stbir__decode_uint8_srgb, 0, stbir__decode_float_linear, stbir__decode_half_float_linear,
  };

  static stbir__decode_pixels_func * decode_alphas[STBIRI_AR-STBIRI_RGBA+1][STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]=
  {
    {  stbir__decode_uint8_srgb4_linearalpha,      stbir__decode_uint8_srgb,      0, stbir__decode_float_linear,      stbir__decode_half_float_linear },
    {  stbir__decode_uint8_srgb4_linearalpha_BGRA, stbir__decode_uint8_srgb_BGRA, 0, stbir__decode_float_linear_BGRA, stbir__decode_half_float_linear_BGRA },
    {  stbir__decode_uint8_srgb4_linearalpha_ARGB, stbir__decode_uint8_srgb_ARGB, 0, stbir__decode_float_linear_ARGB, stbir__decode_half_float_linear_ARGB },
    {  stbir__decode_uint8_srgb4_linearalpha_ABGR, stbir__decode_uint8_srgb_ABGR, 0, stbir__decode_float_linear_ABGR, stbir__decode_half_float_linear_ABGR },
    {  stbir__decode_uint8_srgb2_linearalpha,      stbir__decode_uint8_srgb,      0, stbir__decode_float_linear,      stbir__decode_half_float_linear },
    {  stbir__decode_uint8_srgb2_linearalpha_AR,   stbir__decode_uint8_srgb_AR,   0, stbir__decode_float_linear_AR,   stbir__decode_half_float_linear_AR },
  };

  static stbir__decode_pixels_func * decode_simple_scaled_or_not[2][2]=
  {
    { stbir__decode_uint8_linear_scaled,  stbir__decode_uint8_linear }, { stbir__decode_uint16_linear_scaled, stbir__decode_uint16_linear },
  };

  static stbir__decode_pixels_func * decode_alphas_scaled_or_not[STBIRI_AR-STBIRI_RGBA+1][2][2]=
  {
    {  { stbir__decode_uint8_linear_scaled,       stbir__decode_uint8_linear },      { stbir__decode_uint16_linear_scaled,      stbir__decode_uint16_linear } },
    {  { stbir__decode_uint8_linear_scaled_BGRA,  stbir__decode_uint8_linear_BGRA }, { stbir__decode_uint16_linear_scaled_BGRA, stbir__decode_uint16_linear_BGRA } },
    {  { stbir__decode_uint8_linear_scaled_ARGB,  stbir__decode_uint8_linear_ARGB }, { stbir__decode_uint16_linear_scaled_ARGB, stbir__decode_uint16_linear_ARGB } },
    {  { stbir__decode_uint8_linear_scaled_ABGR,  stbir__decode_uint8_linear_ABGR }, { stbir__decode_uint16_linear_scaled_ABGR, stbir__decode_uint16_linear_ABGR } },
    {  { stbir__decode_uint8_linear_scaled,       stbir__decode_uint8_linear },      { stbir__decode_uint16_linear_scaled,      stbir__decode_uint16_linear } },
    {  { stbir__decode_uint8_linear_scaled_AR,    stbir__decode_uint8_linear_AR },   { stbir__decode_uint16_linear_scaled_AR,   stbir__decode_uint16_linear_AR } }
  };

  static stbir__encode_pixels_func * encode_simple[STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]=
  {
     stbir__encode_uint8_srgb, stbir__encode_uint8_srgb, 0, stbir__encode_float_linear, stbir__encode_half_float_linear,
  };

  static stbir__encode_pixels_func * encode_alphas[STBIRI_AR-STBIRI_RGBA+1][STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]=
  {
    {  stbir__encode_uint8_srgb4_linearalpha,      stbir__encode_uint8_srgb,      0, stbir__encode_float_linear,      stbir__encode_half_float_linear },
    {  stbir__encode_uint8_srgb4_linearalpha_BGRA, stbir__encode_uint8_srgb_BGRA, 0, stbir__encode_float_linear_BGRA, stbir__encode_half_float_linear_BGRA },
    {  stbir__encode_uint8_srgb4_linearalpha_ARGB, stbir__encode_uint8_srgb_ARGB, 0, stbir__encode_float_linear_ARGB, stbir__encode_half_float_linear_ARGB },
    {  stbir__encode_uint8_srgb4_linearalpha_ABGR, stbir__encode_uint8_srgb_ABGR, 0, stbir__encode_float_linear_ABGR, stbir__encode_half_float_linear_ABGR },
    {  stbir__encode_uint8_srgb2_linearalpha,      stbir__encode_uint8_srgb,      0, stbir__encode_float_linear,      stbir__encode_half_float_linear },
    {  stbir__encode_uint8_srgb2_linearalpha_AR,   stbir__encode_uint8_srgb_AR,   0, stbir__encode_float_linear_AR,   stbir__encode_half_float_linear_AR }
  };

  static stbir__encode_pixels_func * encode_simple_scaled_or_not[2][2]=
  {
    { stbir__encode_uint8_linear_scaled,  stbir__encode_uint8_linear }, { stbir__encode_uint16_linear_scaled, stbir__encode_uint16_linear },
  };

  static stbir__encode_pixels_func * encode_alphas_scaled_or_not[STBIRI_AR-STBIRI_RGBA+1][2][2]=
  {
    {  { stbir__encode_uint8_linear_scaled,       stbir__encode_uint8_linear },       { stbir__encode_uint16_linear_scaled,      stbir__encode_uint16_linear } },
    {  { stbir__encode_uint8_linear_scaled_BGRA,  stbir__encode_uint8_linear_BGRA },  { stbir__encode_uint16_linear_scaled_BGRA, stbir__encode_uint16_linear_BGRA } },
    {  { stbir__encode_uint8_linear_scaled_ARGB,  stbir__encode_uint8_linear_ARGB },  { stbir__encode_uint16_linear_scaled_ARGB, stbir__encode_uint16_linear_ARGB } },
    {  { stbir__encode_uint8_linear_scaled_ABGR,  stbir__encode_uint8_linear_ABGR },  { stbir__encode_uint16_linear_scaled_ABGR, stbir__encode_uint16_linear_ABGR } },
    {  { stbir__encode_uint8_linear_scaled,       stbir__encode_uint8_linear },       { stbir__encode_uint16_linear_scaled,      stbir__encode_uint16_linear } },
    {  { stbir__encode_uint8_linear_scaled_AR,    stbir__encode_uint8_linear_AR },    { stbir__encode_uint16_linear_scaled_AR,   stbir__encode_uint16_linear_AR } }
  };

  stbir__decode_pixels_func * decode_pixels = 0;
  stbir__encode_pixels_func * encode_pixels = 0;
  stbir_datatype input_type, output_type;

  input_type = resize->input_data_type;
  output_type = resize->output_data_type;
  info->input_data = resize->input_pixels;
  info->input_stride_bytes = resize->input_stride_in_bytes;
  info->output_stride_bytes = resize->output_stride_in_bytes;

  if ( ( info->horizontal.filter_enum == STBIR_FILTER_POINT_SAMPLE ) && ( info->vertical.filter_enum == STBIR_FILTER_POINT_SAMPLE ) )
  {
    if ( ( ( input_type  == STBIR_TYPE_UINT8_SRGB ) || ( input_type  == STBIR_TYPE_UINT8_SRGB_ALPHA ) ) &&
         ( ( output_type == STBIR_TYPE_UINT8_SRGB ) || ( output_type == STBIR_TYPE_UINT8_SRGB_ALPHA ) ) )
    {
      input_type = STBIR_TYPE_UINT8;
      output_type = STBIR_TYPE_UINT8;
    }
  }

  if ( info->input_stride_bytes == 0 )
    info->input_stride_bytes = info->channels * info->horizontal.scale_info.input_full_size * stbir__type_size[input_type];

  if ( info->output_stride_bytes == 0 )
    info->output_stride_bytes = info->channels * info->horizontal.scale_info.output_sub_size * stbir__type_size[output_type];

  info->output_data = ( (char*) resize->output_pixels ) + ( (size_t) info->offset_y * (size_t) resize->output_stride_in_bytes ) + ( info->offset_x * info->channels * stbir__type_size[output_type] );

  info->in_pixels_cb = resize->input_cb;
  info->user_data = resize->user_data;
  info->out_pixels_cb = resize->output_cb;

  if ( ( input_type == STBIR_TYPE_UINT8 ) || ( input_type == STBIR_TYPE_UINT16 ) )
  {
    int non_scaled = 0;

    if ( ( !info->alpha_weight ) && ( !info->alpha_unweight )  )
      if ( ( ( input_type == STBIR_TYPE_UINT8 ) && ( output_type == STBIR_TYPE_UINT8 ) ) || ( ( input_type == STBIR_TYPE_UINT16 ) && ( output_type == STBIR_TYPE_UINT16 ) ) )
        non_scaled = 1;

    if ( info->input_pixel_layout_internal <= STBIRI_4CHANNEL )
      decode_pixels = decode_simple_scaled_or_not[ input_type == STBIR_TYPE_UINT16 ][ non_scaled ];
    else
      decode_pixels = decode_alphas_scaled_or_not[ ( info->input_pixel_layout_internal - STBIRI_RGBA ) % ( STBIRI_AR-STBIRI_RGBA+1 ) ][ input_type == STBIR_TYPE_UINT16 ][ non_scaled ];
  }
  else
  {
    if ( info->input_pixel_layout_internal <= STBIRI_4CHANNEL )
      decode_pixels = decode_simple[ input_type - STBIR_TYPE_UINT8_SRGB ];
    else
      decode_pixels = decode_alphas[ ( info->input_pixel_layout_internal - STBIRI_RGBA ) % ( STBIRI_AR-STBIRI_RGBA+1 ) ][ input_type - STBIR_TYPE_UINT8_SRGB ];
  }

  if ( ( output_type == STBIR_TYPE_UINT8 ) || ( output_type == STBIR_TYPE_UINT16 ) )
  {
    int non_scaled = 0;

    if ( ( !info->alpha_weight ) && ( !info->alpha_unweight ) )
      if ( ( ( input_type == STBIR_TYPE_UINT8 ) && ( output_type == STBIR_TYPE_UINT8 ) ) || ( ( input_type == STBIR_TYPE_UINT16 ) && ( output_type == STBIR_TYPE_UINT16 ) ) )
        non_scaled = 1;

    if ( info->output_pixel_layout_internal <= STBIRI_4CHANNEL )
      encode_pixels = encode_simple_scaled_or_not[ output_type == STBIR_TYPE_UINT16 ][ non_scaled ];
    else
      encode_pixels = encode_alphas_scaled_or_not[ ( info->output_pixel_layout_internal - STBIRI_RGBA ) % ( STBIRI_AR-STBIRI_RGBA+1 ) ][ output_type == STBIR_TYPE_UINT16 ][ non_scaled ];
  }
  else
  {
    if ( info->output_pixel_layout_internal <= STBIRI_4CHANNEL )
      encode_pixels = encode_simple[ output_type - STBIR_TYPE_UINT8_SRGB ];
    else
      encode_pixels = encode_alphas[ ( info->output_pixel_layout_internal - STBIRI_RGBA ) % ( STBIRI_AR-STBIRI_RGBA+1 ) ][ output_type - STBIR_TYPE_UINT8_SRGB ];
  }

  info->input_type = input_type;
  info->output_type = output_type;
  info->decode_pixels = decode_pixels;
  info->encode_pixels = encode_pixels;
}

static void stbir__clip( int * outx, int * outsubw, int outw, double * u0, double * u1 )
{
  double per, adj;
  int over;

  if ( *outx < 0 )
  {
    per = ( (double)*outx ) / ( (double)*outsubw );
    adj = per * ( *u1 - *u0 );
    *u0 -= adj;
    *outx = 0;
  }

  over = outw - ( *outx + *outsubw );
  if ( over < 0 )
  {
    per = ( (double)over ) / ( (double)*outsubw );
    adj = per * ( *u1 - *u0 );
    *u1 += adj;
    *outsubw = outw - *outx;
  }
}

static int stbir__double_to_rational(double f, stbir_uint32 limit, stbir_uint32 *numer, stbir_uint32 *denom, int limit_denom )
{
  double err;
  stbir_uint64 top, bot;
  stbir_uint64 numer_last = 0;
  stbir_uint64 denom_last = 1;
  stbir_uint64 numer_estimate = 1;
  stbir_uint64 denom_estimate = 0;

  top = (stbir_uint64)( f * (double)(1 << 25) );
  bot = 1 << 25;

  for(;;)
  {
    stbir_uint64 est, temp;

    if ( ( ( limit_denom ) ? denom_estimate : numer_estimate ) >= limit )
      break;

    if ( denom_estimate )
    {
      err = ( (double)numer_estimate / (double)denom_estimate ) - f;
      if ( err < 0.0 ) err = -err;
      if ( err < ( 1.0 / (double)(1<<24) ) )
      {
        *numer = (stbir_uint32) numer_estimate;
        *denom = (stbir_uint32) denom_estimate;
        return 1;
      }
    }

    if ( bot == 0 )
      break;

    est = top / bot;
    temp = top % bot;
    top = bot;
    bot = temp;

    temp = est * denom_estimate + denom_last;
    denom_last = denom_estimate;
    denom_estimate = temp;

    temp = est * numer_estimate + numer_last;
    numer_last = numer_estimate;
    numer_estimate = temp;
  }

  if ( limit_denom )
  {
    numer_estimate= (stbir_uint64)( f * (double)limit + 0.5 );
    denom_estimate = limit;
  }
  else
  {
    numer_estimate = limit;
    denom_estimate = (stbir_uint64)( ( (double)limit / f ) + 0.5 );
  }

  *numer = (stbir_uint32) numer_estimate;
  *denom = (stbir_uint32) denom_estimate;

  err = ( denom_estimate ) ? ( ( (double)(stbir_uint32)numer_estimate / (double)(stbir_uint32)denom_estimate ) - f ) : 1.0;
  if ( err < 0.0 ) err = -err;
  return ( err < ( 1.0 / (double)(1<<24) ) ) ? 1 : 0;
}

static int stbir__calculate_region_transform( stbir__scale_info * scale_info, int output_full_range, int * output_offset, int output_sub_range, int input_full_range, double input_s0, double input_s1 )
{
  double output_range, input_range, output_s, input_s, ratio, scale;

  input_s = input_s1 - input_s0;

  if ( ( output_full_range == 0 ) || ( input_full_range == 0 ) ||
       ( output_sub_range == 0 ) || ( input_s <= stbir__small_float ) )
    return 0;

  if ( ( *output_offset >= output_full_range ) || ( ( *output_offset + output_sub_range ) <= 0 ) || ( input_s0 >= (1.0f-stbir__small_float) ) || ( input_s1 <= stbir__small_float ) )
    return 0;

  output_range = (double)output_full_range;
  input_range = (double)input_full_range;

  output_s = ( (double)output_sub_range) / output_range;

  ratio = output_s / input_s;

  scale = ( output_range / input_range ) * ratio;
  scale_info->scale = (float)scale;
  scale_info->inv_scale = (float)( 1.0 / scale );

  stbir__clip( output_offset, &output_sub_range, output_full_range, &input_s0, &input_s1 );

  input_s = input_s1 - input_s0;

  if ( input_s <= stbir__small_float )
    return 0;

  scale_info->pixel_shift = (float) ( input_s0 * ratio * output_range );

  scale_info->scale_is_rational = stbir__double_to_rational( scale, ( scale <= 1.0 ) ? output_full_range : input_full_range, &scale_info->scale_numerator, &scale_info->scale_denominator, ( scale >= 1.0 ) );

  scale_info->input_full_size = input_full_range;
  scale_info->output_sub_size = output_sub_range;

  return 1;
}

static void stbir__init_and_set_layout( STBIR_RESIZE * resize, stbir_pixel_layout pixel_layout, stbir_datatype data_type )
{
  resize->input_cb = 0;
  resize->output_cb = 0;
  resize->user_data = resize;
  resize->samplers = 0;
  resize->called_alloc = 0;
  resize->horizontal_filter = STBIR_FILTER_DEFAULT;
  resize->horizontal_filter_kernel = 0; resize->horizontal_filter_support = 0;
  resize->vertical_filter = STBIR_FILTER_DEFAULT;
  resize->vertical_filter_kernel = 0; resize->vertical_filter_support = 0;
  resize->horizontal_edge = STBIR_EDGE_CLAMP;
  resize->vertical_edge = STBIR_EDGE_CLAMP;
  resize->input_s0 = 0; resize->input_t0 = 0; resize->input_s1 = 1; resize->input_t1 = 1;
  resize->output_subx = 0; resize->output_suby = 0; resize->output_subw = resize->output_w; resize->output_subh = resize->output_h;
  resize->input_data_type = data_type;
  resize->output_data_type = data_type;
  resize->input_pixel_layout_public = pixel_layout;
  resize->output_pixel_layout_public = pixel_layout;
  resize->needs_rebuild = 1;
}

STBIRDEF void stbir_resize_init( STBIR_RESIZE * resize,
                                 const void *input_pixels,  int input_w,  int input_h, int input_stride_in_bytes,
                                       void *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                 stbir_pixel_layout pixel_layout, stbir_datatype data_type )
{
  resize->input_pixels = input_pixels;
  resize->input_w = input_w;
  resize->input_h = input_h;
  resize->input_stride_in_bytes = input_stride_in_bytes;
  resize->output_pixels = output_pixels;
  resize->output_w = output_w;
  resize->output_h = output_h;
  resize->output_stride_in_bytes = output_stride_in_bytes;
  resize->fast_alpha = 0;

  stbir__init_and_set_layout( resize, pixel_layout, data_type );
}

STBIRDEF void stbir_set_datatypes( STBIR_RESIZE * resize, stbir_datatype input_type, stbir_datatype output_type )
{
  resize->input_data_type = input_type;
  resize->output_data_type = output_type;
  if ( ( resize->samplers ) && ( !resize->needs_rebuild ) )
    stbir__update_info_from_resize( resize->samplers, resize );
}

STBIRDEF void stbir_set_pixel_callbacks( STBIR_RESIZE * resize, stbir_input_callback * input_cb, stbir_output_callback * output_cb )
{
  resize->input_cb = input_cb;
  resize->output_cb = output_cb;

  if ( ( resize->samplers ) && ( !resize->needs_rebuild ) )
  {
    resize->samplers->in_pixels_cb = input_cb;
    resize->samplers->out_pixels_cb = output_cb;
  }
}

STBIRDEF void stbir_set_user_data( STBIR_RESIZE * resize, void * user_data )
{
  resize->user_data = user_data;
  if ( ( resize->samplers ) && ( !resize->needs_rebuild ) )
    resize->samplers->user_data = user_data;
}

STBIRDEF void stbir_set_buffer_ptrs( STBIR_RESIZE * resize, const void * input_pixels, int input_stride_in_bytes, void * output_pixels, int output_stride_in_bytes )
{
  resize->input_pixels = input_pixels;
  resize->input_stride_in_bytes = input_stride_in_bytes;
  resize->output_pixels = output_pixels;
  resize->output_stride_in_bytes = output_stride_in_bytes;
  if ( ( resize->samplers ) && ( !resize->needs_rebuild ) )
    stbir__update_info_from_resize( resize->samplers, resize );
}

STBIRDEF int stbir_set_edgemodes( STBIR_RESIZE * resize, stbir_edge horizontal_edge, stbir_edge vertical_edge )
{
  resize->horizontal_edge = horizontal_edge;
  resize->vertical_edge = vertical_edge;
  resize->needs_rebuild = 1;
  return 1;
}

STBIRDEF int stbir_set_filters( STBIR_RESIZE * resize, stbir_filter horizontal_filter, stbir_filter vertical_filter )
{
  resize->horizontal_filter = horizontal_filter;
  resize->vertical_filter = vertical_filter;
  resize->needs_rebuild = 1;
  return 1;
}

STBIRDEF int stbir_set_filter_callbacks( STBIR_RESIZE * resize, stbir__kernel_callback * horizontal_filter, stbir__support_callback * horizontal_support, stbir__kernel_callback * vertical_filter, stbir__support_callback * vertical_support )
{
  resize->horizontal_filter_kernel = horizontal_filter; resize->horizontal_filter_support = horizontal_support;
  resize->vertical_filter_kernel = vertical_filter; resize->vertical_filter_support = vertical_support;
  resize->needs_rebuild = 1;
  return 1;
}

STBIRDEF int stbir_set_pixel_layouts( STBIR_RESIZE * resize, stbir_pixel_layout input_pixel_layout, stbir_pixel_layout output_pixel_layout )
{
  resize->input_pixel_layout_public = input_pixel_layout;
  resize->output_pixel_layout_public = output_pixel_layout;
  resize->needs_rebuild = 1;
  return 1;
}

STBIRDEF int stbir_set_non_pm_alpha_speed_over_quality( STBIR_RESIZE * resize, int non_pma_alpha_speed_over_quality )
{
  resize->fast_alpha = non_pma_alpha_speed_over_quality;
  resize->needs_rebuild = 1;
  return 1;
}

STBIRDEF int stbir_set_input_subrect( STBIR_RESIZE * resize, double s0, double t0, double s1, double t1 )
{
  resize->input_s0 = s0;
  resize->input_t0 = t0;
  resize->input_s1 = s1;
  resize->input_t1 = t1;
  resize->needs_rebuild = 1;

  if ( ( s1 < stbir__small_float ) || ( (s1-s0) < stbir__small_float ) ||
       ( t1 < stbir__small_float ) || ( (t1-t0) < stbir__small_float ) ||
       ( s0 > (1.0f-stbir__small_float) ) ||
       ( t0 > (1.0f-stbir__small_float) ) )
    return 0;

  return 1;
}

STBIRDEF int stbir_set_output_pixel_subrect( STBIR_RESIZE * resize, int subx, int suby, int subw, int subh )
{
  resize->output_subx = subx;
  resize->output_suby = suby;
  resize->output_subw = subw;
  resize->output_subh = subh;
  resize->needs_rebuild = 1;

  if ( ( subx >= resize->output_w ) || ( ( subx + subw ) <= 0 ) || ( suby >= resize->output_h ) || ( ( suby + subh ) <= 0 ) || ( subw == 0 ) || ( subh == 0 ) )
    return 0;

  return 1;
}

STBIRDEF int stbir_set_pixel_subrect( STBIR_RESIZE * resize, int subx, int suby, int subw, int subh )
{
  double s0, t0, s1, t1;

  s0 = ( (double)subx ) / ( (double)resize->output_w );
  t0 = ( (double)suby ) / ( (double)resize->output_h );
  s1 = ( (double)(subx+subw) ) / ( (double)resize->output_w );
  t1 = ( (double)(suby+subh) ) / ( (double)resize->output_h );

  resize->input_s0 = s0;
  resize->input_t0 = t0;
  resize->input_s1 = s1;
  resize->input_t1 = t1;
  resize->output_subx = subx;
  resize->output_suby = suby;
  resize->output_subw = subw;
  resize->output_subh = subh;
  resize->needs_rebuild = 1;

  if ( ( subx >= resize->output_w ) || ( ( subx + subw ) <= 0 ) || ( suby >= resize->output_h ) || ( ( suby + subh ) <= 0 ) || ( subw == 0 ) || ( subh == 0 ) )
    return 0;

  return 1;
}

static int stbir__perform_build( STBIR_RESIZE * resize, int splits )
{
  stbir__contributors conservative = { 0, 0 };
  stbir__sampler horizontal, vertical;
  int new_output_subx, new_output_suby;
  stbir__info * out_info;

  if ( resize->samplers )
    return 0;

  #define STBIR_RETURN_ERROR_AND_ASSERT( exp )  STBIR_ASSERT( !(exp) ); if (exp) return 0;
  STBIR_RETURN_ERROR_AND_ASSERT( (unsigned)resize->horizontal_filter >= STBIR_FILTER_OTHER)
  STBIR_RETURN_ERROR_AND_ASSERT( (unsigned)resize->vertical_filter >= STBIR_FILTER_OTHER)
  #undef STBIR_RETURN_ERROR_AND_ASSERT

  if ( splits <= 0 )
    return 0;

  STBIR_PROFILE_BUILD_FIRST_START( build );

  new_output_subx = resize->output_subx;
  new_output_suby = resize->output_suby;

  if ( !stbir__calculate_region_transform( &horizontal.scale_info, resize->output_w, &new_output_subx, resize->output_subw, resize->input_w, resize->input_s0, resize->input_s1 ) )
    return 0;

  if ( !stbir__calculate_region_transform( &vertical.scale_info, resize->output_h, &new_output_suby, resize->output_subh, resize->input_h, resize->input_t0, resize->input_t1 ) )
    return 0;

  if ( ( horizontal.scale_info.output_sub_size == 0 ) || ( vertical.scale_info.output_sub_size == 0 ) )
    return 0;

  stbir__set_sampler(&horizontal, resize->horizontal_filter, resize->horizontal_filter_kernel, resize->horizontal_filter_support, resize->horizontal_edge, &horizontal.scale_info, 1, resize->user_data );
  stbir__get_conservative_extents( &horizontal, &conservative, resize->user_data );
  stbir__set_sampler(&vertical, resize->vertical_filter, resize->vertical_filter_kernel, resize->vertical_filter_support, resize->vertical_edge, &vertical.scale_info, 0, resize->user_data );

  if ( ( vertical.scale_info.output_sub_size / splits ) < STBIR_FORCE_MINIMUM_SCANLINES_FOR_SPLITS )
  {
    splits = vertical.scale_info.output_sub_size / STBIR_FORCE_MINIMUM_SCANLINES_FOR_SPLITS;
    if ( splits == 0 ) splits = 1;
  }

  STBIR_PROFILE_BUILD_START( alloc );
  out_info = stbir__alloc_internal_mem_and_build_samplers( &horizontal, &vertical, &conservative, resize->input_pixel_layout_public, resize->output_pixel_layout_public, splits, new_output_subx, new_output_suby, resize->fast_alpha, resize->user_data STBIR_ONLY_PROFILE_BUILD_SET_INFO );
  STBIR_PROFILE_BUILD_END( alloc );
  STBIR_PROFILE_BUILD_END( build );

  if ( out_info )
  {
    resize->splits = splits;
    resize->samplers = out_info;
    resize->needs_rebuild = 0;

    stbir__update_info_from_resize( out_info, resize );

    return splits;
  }

  return 0;
}

void stbir_free_samplers( STBIR_RESIZE * resize )
{
  if ( resize->samplers )
  {
    stbir__free_internal_mem( resize->samplers );
    resize->samplers = 0;
    resize->called_alloc = 0;
  }
}

STBIRDEF int stbir_build_samplers_with_splits( STBIR_RESIZE * resize, int splits )
{
  if ( ( resize->samplers == 0 ) || ( resize->needs_rebuild ) )
  {
    if ( resize->samplers )
      stbir_free_samplers( resize );

    resize->called_alloc = 1;
    return stbir__perform_build( resize, splits );
  }

  STBIR_PROFILE_BUILD_CLEAR( resize->samplers );

  return 1;
}

STBIRDEF int stbir_build_samplers( STBIR_RESIZE * resize )
{
  return stbir_build_samplers_with_splits( resize, 1 );
}

STBIRDEF int stbir_resize_extended( STBIR_RESIZE * resize )
{
  int result;

  if ( ( resize->samplers == 0 ) || ( resize->needs_rebuild ) )
  {
    int alloc_state = resize->called_alloc;

    if ( resize->samplers )
    {
      stbir__free_internal_mem( resize->samplers );
      resize->samplers = 0;
    }

    if ( !stbir_build_samplers( resize ) )
      return 0;

    resize->called_alloc = alloc_state;

    if ( resize->samplers == 0 )
      return 1;
  }
  else
  {
    STBIR_PROFILE_BUILD_CLEAR( resize->samplers );
  }

  result = stbir__perform_resize( resize->samplers, 0, resize->splits );

  if ( !resize->called_alloc )
  {
    stbir_free_samplers( resize );
    resize->samplers = 0;
  }

  return result;
}

STBIRDEF int stbir_resize_extended_split( STBIR_RESIZE * resize, int split_start, int split_count )
{
  STBIR_ASSERT( resize->samplers );

  if ( ( split_start == -1 ) || ( ( split_start == 0 ) && ( split_count == resize->splits ) ) )
    return stbir_resize_extended( resize );

  if ( ( resize->samplers == 0 ) || ( resize->needs_rebuild ) )
    return 0;

  if ( ( split_start >= resize->splits ) || ( split_start < 0 ) || ( ( split_start + split_count ) > resize->splits ) || ( split_count <= 0 ) )
    return 0;

  return stbir__perform_resize( resize->samplers, split_start, split_count );
}

static void * stbir_quick_resize_helper( const void *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                               void *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                               stbir_pixel_layout pixel_layout, stbir_datatype data_type, stbir_edge edge, stbir_filter filter )
{
  STBIR_RESIZE resize;
  int scanline_output_in_bytes;
  int positive_output_stride_in_bytes;
  void * start_ptr;
  void * free_ptr;

  scanline_output_in_bytes = output_w * stbir__type_size[ data_type ] * stbir__pixel_channels[ stbir__pixel_layout_convert_public_to_internal[ pixel_layout ] ];
  if ( scanline_output_in_bytes == 0 )
    return 0;

  if ( output_stride_in_bytes == 0 )
    output_stride_in_bytes = scanline_output_in_bytes;

  positive_output_stride_in_bytes = output_stride_in_bytes;
  if ( positive_output_stride_in_bytes < 0 )
    positive_output_stride_in_bytes = -positive_output_stride_in_bytes;

  if ( positive_output_stride_in_bytes < scanline_output_in_bytes )
    return 0;

  start_ptr = output_pixels;
  free_ptr = 0;

  if ( output_pixels == 0 )
  {
    size_t size;
    char * ptr;

    size = (size_t)positive_output_stride_in_bytes * (size_t)output_h;
    if ( size == 0 )
      return 0;

    ptr = (char*) STBIR_MALLOC( size, 0 );
    if ( ptr == 0 )
      return 0;

    free_ptr = ptr;

    if ( output_stride_in_bytes < 0 )
      start_ptr = ptr + ( (size_t)positive_output_stride_in_bytes * (size_t)( output_h - 1 ) );
    else
      start_ptr = ptr;
  }

  stbir_resize_init( &resize,
                     input_pixels,  input_w,  input_h,  input_stride_in_bytes,
                     start_ptr, output_w, output_h, output_stride_in_bytes,
                     pixel_layout, data_type );

  resize.horizontal_edge = edge;
  resize.vertical_edge = edge;
  resize.horizontal_filter = filter;
  resize.vertical_filter = filter;

  if ( !stbir_resize_extended( &resize ) )
  {
    if ( free_ptr )
      STBIR_FREE( free_ptr, 0 );
    return 0;
  }

  return (free_ptr) ? free_ptr : start_ptr;
}

STBIRDEF unsigned char * stbir_resize_uint8_linear( const unsigned char *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                          unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                          stbir_pixel_layout pixel_layout )
{
  return (unsigned char *) stbir_quick_resize_helper( input_pixels , input_w , input_h, input_stride_in_bytes,
                                                      output_pixels, output_w, output_h, output_stride_in_bytes,
                                                      pixel_layout, STBIR_TYPE_UINT8, STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT );
}

STBIRDEF unsigned char * stbir_resize_uint8_srgb( const unsigned char *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                        unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                        stbir_pixel_layout pixel_layout )
{
  return (unsigned char *) stbir_quick_resize_helper( input_pixels , input_w , input_h, input_stride_in_bytes,
                                                      output_pixels, output_w, output_h, output_stride_in_bytes,
                                                      pixel_layout, STBIR_TYPE_UINT8_SRGB, STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT );
}

STBIRDEF float * stbir_resize_float_linear( const float *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                  float *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                  stbir_pixel_layout pixel_layout )
{
  return (float *) stbir_quick_resize_helper( input_pixels , input_w , input_h, input_stride_in_bytes,
                                              output_pixels, output_w, output_h, output_stride_in_bytes,
                                              pixel_layout, STBIR_TYPE_FLOAT, STBIR_EDGE_CLAMP, STBIR_FILTER_DEFAULT  );
}

STBIRDEF void * stbir_resize( const void *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                    void *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                    stbir_pixel_layout pixel_layout, stbir_datatype data_type,
                                    stbir_edge edge, stbir_filter filter )
{
  return (void *) stbir_quick_resize_helper( input_pixels , input_w , input_h, input_stride_in_bytes,
                                             output_pixels, output_w, output_h, output_stride_in_bytes,
                                             pixel_layout, data_type, edge, filter  );
}

#undef STBIR_BGR
#undef STBIR_1CHANNEL
#undef STBIR_2CHANNEL
#undef STBIR_RGB
#undef STBIR_RGBA
#undef STBIR_4CHANNEL
#undef STBIR_BGRA
#undef STBIR_ARGB
#undef STBIR_ABGR
#undef STBIR_RA
#undef STBIR_AR
#undef STBIR_RGBA_PM
#undef STBIR_BGRA_PM
#undef STBIR_ARGB_PM
#undef STBIR_ABGR_PM
#undef STBIR_RA_PM
#undef STBIR_AR_PM

#endif

#else

#define STBIR_strs_join2( start, mid, end ) start##mid##end
#define STBIR_strs_join1( start, mid, end ) STBIR_strs_join2( start, mid, end )

#define STBIR_strs_join24( start, mid1, mid2, end ) start##mid1##mid2##end
#define STBIR_strs_join14( start, mid1, mid2, end ) STBIR_strs_join24( start, mid1, mid2, end )

#ifdef STB_IMAGE_RESIZE_DO_CODERS

#ifdef stbir__decode_suffix
#define STBIR__CODER_NAME( name ) STBIR_strs_join1( name, _, stbir__decode_suffix )
#else
#define STBIR__CODER_NAME( name ) name
#endif

#ifdef stbir__decode_swizzle
#define stbir__decode_simdf8_flip(reg) STBIR_strs_join1( STBIR_strs_join1( STBIR_strs_join1( STBIR_strs_join1( stbir__simdf8_0123to,stbir__decode_order0,stbir__decode_order1),stbir__decode_order2,stbir__decode_order3),stbir__decode_order0,stbir__decode_order1),stbir__decode_order2,stbir__decode_order3)(reg, reg)
#define stbir__decode_simdf4_flip(reg) STBIR_strs_join1( STBIR_strs_join1( stbir__simdf_0123to,stbir__decode_order0,stbir__decode_order1),stbir__decode_order2,stbir__decode_order3)(reg, reg)
#define stbir__encode_simdf8_unflip(reg) STBIR_strs_join1( STBIR_strs_join1( STBIR_strs_join1( STBIR_strs_join1( stbir__simdf8_0123to,stbir__encode_order0,stbir__encode_order1),stbir__encode_order2,stbir__encode_order3),stbir__encode_order0,stbir__encode_order1),stbir__encode_order2,stbir__encode_order3)(reg, reg)
#define stbir__encode_simdf4_unflip(reg) STBIR_strs_join1( STBIR_strs_join1( stbir__simdf_0123to,stbir__encode_order0,stbir__encode_order1),stbir__encode_order2,stbir__encode_order3)(reg, reg)
#else
#define stbir__decode_order0 0
#define stbir__decode_order1 1
#define stbir__decode_order2 2
#define stbir__decode_order3 3
#define stbir__encode_order0 0
#define stbir__encode_order1 1
#define stbir__encode_order2 2
#define stbir__encode_order3 3
#define stbir__decode_simdf8_flip(reg)
#define stbir__decode_simdf4_flip(reg)
#define stbir__encode_simdf8_unflip(reg)
#define stbir__encode_simdf4_unflip(reg)
#endif

#ifdef STBIR_SIMD8
#define stbir__encode_simdfX_unflip  stbir__encode_simdf8_unflip
#else
#define stbir__encode_simdfX_unflip  stbir__encode_simdf4_unflip
#endif

static float * STBIR__CODER_NAME( stbir__decode_uint8_linear_scaled )( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  unsigned char const * input = (unsigned char const*)inputp;

  return decode_end;
}

static void STBIR__CODER_NAME( stbir__encode_uint8_linear_scaled )( void * outputp, int width_times_channels, float const * encode )
{
  unsigned char STBIR_SIMD_STREAMOUT_PTR( * ) output = (unsigned char *) outputp;
  unsigned char * end_output = ( (unsigned char *) output ) + width_times_channels;

  #ifdef STBIR_SIMD
  if ( width_times_channels >= stbir__simdfX_float_count*2 )
  {
    float const * end_encode_m8 = encode + width_times_channels - stbir__simdfX_float_count*2;
    end_output -= stbir__simdfX_float_count*2;
    STBIR_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      stbir__simdfX e0, e1;
      stbir__simdi i;
      STBIR_SIMD_NO_UNROLL(encode);
      stbir__simdfX_madd_mem( e0, STBIR_simd_point5X, STBIR_max_uint8_as_floatX, encode );
      stbir__simdfX_madd_mem( e1, STBIR_simd_point5X, STBIR_max_uint8_as_floatX, encode+stbir__simdfX_float_count );
      stbir__encode_simdfX_unflip( e0 );
      stbir__encode_simdfX_unflip( e1 );
      #ifdef STBIR_SIMD8
      stbir__simdf8_pack_to_16bytes( i, e0, e1 );
      stbir__simdi_store( output, i );
      #else
      stbir__simdf_pack_to_8bytes( i, e0, e1 );
      stbir__simdi_store2( output, i );
      #endif
      encode += stbir__simdfX_float_count*2;
      output += stbir__simdfX_float_count*2;
      if ( output <= end_output )
        continue;
      if ( output == ( end_output + stbir__simdfX_float_count*2 ) )
        break;
      output = end_output;
      encode = end_encode_m8;
    }
    return;
  }

  #else

  #endif
}

static float * STBIR__CODER_NAME(stbir__decode_uint8_linear)( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  unsigned char const * input = (unsigned char const*)inputp;

  return decode_end;
}

static void STBIR__CODER_NAME( stbir__encode_uint8_linear )( void * outputp, int width_times_channels, float const * encode )
{
  unsigned char STBIR_SIMD_STREAMOUT_PTR( * ) output = (unsigned char *) outputp;
  unsigned char * end_output = ( (unsigned char *) output ) + width_times_channels;

  #ifdef STBIR_SIMD
  if ( width_times_channels >= stbir__simdfX_float_count*2 )
  {
    float const * end_encode_m8 = encode + width_times_channels - stbir__simdfX_float_count*2;
    end_output -= stbir__simdfX_float_count*2;
    STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      stbir__simdfX e0, e1;
      stbir__simdi i;
      STBIR_SIMD_NO_UNROLL(encode);
      stbir__simdfX_add_mem( e0, STBIR_simd_point5X, encode );
      stbir__simdfX_add_mem( e1, STBIR_simd_point5X, encode+stbir__simdfX_float_count );
      stbir__encode_simdfX_unflip( e0 );
      stbir__encode_simdfX_unflip( e1 );
      #ifdef STBIR_SIMD8
      stbir__simdf8_pack_to_16bytes( i, e0, e1 );
      stbir__simdi_store( output, i );
      #else
      stbir__simdf_pack_to_8bytes( i, e0, e1 );
      stbir__simdi_store2( output, i );
      #endif
      encode += stbir__simdfX_float_count*2;
      output += stbir__simdfX_float_count*2;
      if ( output <= end_output )
        continue;
      if ( output == ( end_output + stbir__simdfX_float_count*2 ) )
        break;
      output = end_output;
      encode = end_encode_m8;
    }
    return;
  }

  #else

  #endif

}

static float * STBIR__CODER_NAME(stbir__decode_uint8_srgb)( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  unsigned char const * input = (unsigned char const *)inputp;

  return decode_end;
}

#define stbir__min_max_shift20( i, f ) \
    stbir__simdf_max( f, f, stbir_simdf_casti(STBIR__CONSTI( STBIR_almost_zero )) ); \
    stbir__simdf_min( f, f, stbir_simdf_casti(STBIR__CONSTI( STBIR_almost_one  )) ); \
    stbir__simdi_32shr( i, stbir_simdi_castf( f ), 20 );

#define stbir__scale_and_convert( i, f ) \
    stbir__simdf_madd( f, STBIR__CONSTF( STBIR_simd_point5 ), STBIR__CONSTF( STBIR_max_uint8_as_float ), f ); \
    stbir__simdf_max( f, f, stbir__simdf_zeroP() ); \
    stbir__simdf_min( f, f, STBIR__CONSTF( STBIR_max_uint8_as_float ) ); \
    stbir__simdf_convert_float_to_i32( i, f );

#define stbir__linear_to_srgb_finish( i, f ) \
{ \
    stbir__simdi temp;  \
    stbir__simdi_32shr( temp, stbir_simdi_castf( f ), 12 ) ; \
    stbir__simdi_and( temp, temp, STBIR__CONSTI(STBIR_mastissa_mask) ); \
    stbir__simdi_or( temp, temp, STBIR__CONSTI(STBIR_topscale) ); \
    stbir__simdi_16madd( i, i, temp ); \
    stbir__simdi_32shr( i, i, 16 ); \
}

#define stbir__simdi_table_lookup2( v0,v1, table ) \
{ \
  stbir__simdi_u32 temp0,temp1; \
  temp0.m128i_i128 = v0; \
  temp1.m128i_i128 = v1; \
  temp0.m128i_u32[0] = table[temp0.m128i_i32[0]]; temp0.m128i_u32[1] = table[temp0.m128i_i32[1]]; temp0.m128i_u32[2] = table[temp0.m128i_i32[2]]; temp0.m128i_u32[3] = table[temp0.m128i_i32[3]]; \
  temp1.m128i_u32[0] = table[temp1.m128i_i32[0]]; temp1.m128i_u32[1] = table[temp1.m128i_i32[1]]; temp1.m128i_u32[2] = table[temp1.m128i_i32[2]]; temp1.m128i_u32[3] = table[temp1.m128i_i32[3]]; \
  v0 = temp0.m128i_i128; \
  v1 = temp1.m128i_i128; \
}

#define stbir__simdi_table_lookup3( v0,v1,v2, table ) \
{ \
  stbir__simdi_u32 temp0,temp1,temp2; \
  temp0.m128i_i128 = v0; \
  temp1.m128i_i128 = v1; \
  temp2.m128i_i128 = v2; \
  temp0.m128i_u32[0] = table[temp0.m128i_i32[0]]; temp0.m128i_u32[1] = table[temp0.m128i_i32[1]]; temp0.m128i_u32[2] = table[temp0.m128i_i32[2]]; temp0.m128i_u32[3] = table[temp0.m128i_i32[3]]; \
  temp1.m128i_u32[0] = table[temp1.m128i_i32[0]]; temp1.m128i_u32[1] = table[temp1.m128i_i32[1]]; temp1.m128i_u32[2] = table[temp1.m128i_i32[2]]; temp1.m128i_u32[3] = table[temp1.m128i_i32[3]]; \
  temp2.m128i_u32[0] = table[temp2.m128i_i32[0]]; temp2.m128i_u32[1] = table[temp2.m128i_i32[1]]; temp2.m128i_u32[2] = table[temp2.m128i_i32[2]]; temp2.m128i_u32[3] = table[temp2.m128i_i32[3]]; \
  v0 = temp0.m128i_i128; \
  v1 = temp1.m128i_i128; \
  v2 = temp2.m128i_i128; \
}

#define stbir__simdi_table_lookup4( v0,v1,v2,v3, table ) \
{ \
  stbir__simdi_u32 temp0,temp1,temp2,temp3; \
  temp0.m128i_i128 = v0; \
  temp1.m128i_i128 = v1; \
  temp2.m128i_i128 = v2; \
  temp3.m128i_i128 = v3; \
  temp0.m128i_u32[0] = table[temp0.m128i_i32[0]]; temp0.m128i_u32[1] = table[temp0.m128i_i32[1]]; temp0.m128i_u32[2] = table[temp0.m128i_i32[2]]; temp0.m128i_u32[3] = table[temp0.m128i_i32[3]]; \
  temp1.m128i_u32[0] = table[temp1.m128i_i32[0]]; temp1.m128i_u32[1] = table[temp1.m128i_i32[1]]; temp1.m128i_u32[2] = table[temp1.m128i_i32[2]]; temp1.m128i_u32[3] = table[temp1.m128i_i32[3]]; \
  temp2.m128i_u32[0] = table[temp2.m128i_i32[0]]; temp2.m128i_u32[1] = table[temp2.m128i_i32[1]]; temp2.m128i_u32[2] = table[temp2.m128i_i32[2]]; temp2.m128i_u32[3] = table[temp2.m128i_i32[3]]; \
  temp3.m128i_u32[0] = table[temp3.m128i_i32[0]]; temp3.m128i_u32[1] = table[temp3.m128i_i32[1]]; temp3.m128i_u32[2] = table[temp3.m128i_i32[2]]; temp3.m128i_u32[3] = table[temp3.m128i_i32[3]]; \
  v0 = temp0.m128i_i128; \
  v1 = temp1.m128i_i128; \
  v2 = temp2.m128i_i128; \
  v3 = temp3.m128i_i128; \
}

static void STBIR__CODER_NAME( stbir__encode_uint8_srgb )( void * outputp, int width_times_channels, float const * encode )
{
  unsigned char STBIR_SIMD_STREAMOUT_PTR( * ) output = (unsigned char*) outputp;
  unsigned char * end_output = ( (unsigned char*) output ) + width_times_channels;

}

static float * STBIR__CODER_NAME(stbir__decode_uint16_linear_scaled)( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  unsigned short const * input = (unsigned short const *)inputp;

  return decode_end;
}

static void STBIR__CODER_NAME(stbir__encode_uint16_linear_scaled)( void * outputp, int width_times_channels, float const * encode )
{
  unsigned short STBIR_SIMD_STREAMOUT_PTR( * ) output = (unsigned short*) outputp;
  unsigned short * end_output = ( (unsigned short*) output ) + width_times_channels;

  #ifdef STBIR_SIMD
  {
    if ( width_times_channels >= stbir__simdfX_float_count*2 )
    {
      float const * end_encode_m8 = encode + width_times_channels - stbir__simdfX_float_count*2;
      end_output -= stbir__simdfX_float_count*2;
      STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
      for(;;)
      {
        stbir__simdfX e0, e1;
        stbir__simdiX i;
        STBIR_SIMD_NO_UNROLL(encode);
        stbir__simdfX_madd_mem( e0, STBIR_simd_point5X, STBIR_max_uint16_as_floatX, encode );
        stbir__simdfX_madd_mem( e1, STBIR_simd_point5X, STBIR_max_uint16_as_floatX, encode+stbir__simdfX_float_count );
        stbir__encode_simdfX_unflip( e0 );
        stbir__encode_simdfX_unflip( e1 );
        stbir__simdfX_pack_to_words( i, e0, e1 );
        stbir__simdiX_store( output, i );
        encode += stbir__simdfX_float_count*2;
        output += stbir__simdfX_float_count*2;
        if ( output <= end_output )
          continue;
        if ( output == ( end_output + stbir__simdfX_float_count*2 ) )
          break;
        output = end_output;
        encode = end_encode_m8;
      }
      return;
    }
  }

  #else

  #endif
}

static float * STBIR__CODER_NAME(stbir__decode_uint16_linear)( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  unsigned short const * input = (unsigned short const *)inputp;

  return decode_end;
}

static void STBIR__CODER_NAME(stbir__encode_uint16_linear)( void * outputp, int width_times_channels, float const * encode )
{
  unsigned short STBIR_SIMD_STREAMOUT_PTR( * ) output = (unsigned short*) outputp;
  unsigned short * end_output = ( (unsigned short*) output ) + width_times_channels;

  #ifdef STBIR_SIMD
  {
    if ( width_times_channels >= stbir__simdfX_float_count*2 )
    {
      float const * end_encode_m8 = encode + width_times_channels - stbir__simdfX_float_count*2;
      end_output -= stbir__simdfX_float_count*2;
      STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
      for(;;)
      {
        stbir__simdfX e0, e1;
        stbir__simdiX i;
        STBIR_SIMD_NO_UNROLL(encode);
        stbir__simdfX_add_mem( e0, STBIR_simd_point5X, encode );
        stbir__simdfX_add_mem( e1, STBIR_simd_point5X, encode+stbir__simdfX_float_count );
        stbir__encode_simdfX_unflip( e0 );
        stbir__encode_simdfX_unflip( e1 );
        stbir__simdfX_pack_to_words( i, e0, e1 );
        stbir__simdiX_store( output, i );
        encode += stbir__simdfX_float_count*2;
        output += stbir__simdfX_float_count*2;
        if ( output <= end_output )
          continue;
        if ( output == ( end_output + stbir__simdfX_float_count*2 ) )
          break;
        output = end_output;
        encode = end_encode_m8;
      }
      return;
    }
  }

  #else

  #endif

}

static float * STBIR__CODER_NAME(stbir__decode_half_float_linear)( float * decodep, int width_times_channels, void const * inputp )
{
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  stbir__FP16 const * input = (stbir__FP16 const *)inputp;

  return decode_end;
}

static void STBIR__CODER_NAME( stbir__encode_half_float_linear )( void * outputp, int width_times_channels, float const * encode )
{
  stbir__FP16 STBIR_SIMD_STREAMOUT_PTR( * ) output = (stbir__FP16*) outputp;
  stbir__FP16 * end_output = ( (stbir__FP16*) output ) + width_times_channels;

}

static float * STBIR__CODER_NAME(stbir__decode_float_linear)( float * decodep, int width_times_channels, void const * inputp )
{
  #ifdef stbir__decode_swizzle
  float STBIR_STREAMOUT_PTR( * ) decode = decodep;
  float * decode_end = (float*) decode + width_times_channels;
  float const * input = (float const *)inputp;

  return decode_end;

  #else

  if ( (void*)decodep != inputp )
    STBIR_MEMCPY( decodep, inputp, width_times_channels * sizeof( float ) );

  return decodep + width_times_channels;

  #endif
}

static void STBIR__CODER_NAME( stbir__encode_float_linear )( void * outputp, int width_times_channels, float const * encode )
{
  #if !defined( STBIR_FLOAT_HIGH_CLAMP ) && !defined(STBIR_FLOAT_LO_CLAMP) && !defined(stbir__decode_swizzle)

  if ( (void*)outputp != (void*) encode )
    STBIR_MEMCPY( outputp, encode, width_times_channels * sizeof( float ) );

  #else

  float STBIR_SIMD_STREAMOUT_PTR( * ) output = (float*) outputp;
  float * end_output = ( (float*) output ) + width_times_channels;

  #ifdef STBIR_FLOAT_HIGH_CLAMP
  #define stbir_scalar_hi_clamp( v ) if ( v > STBIR_FLOAT_HIGH_CLAMP ) v = STBIR_FLOAT_HIGH_CLAMP;
  #else
  #define stbir_scalar_hi_clamp( v )
  #endif
  #ifdef STBIR_FLOAT_LOW_CLAMP
  #define stbir_scalar_lo_clamp( v ) if ( v < STBIR_FLOAT_LOW_CLAMP ) v = STBIR_FLOAT_LOW_CLAMP;
  #else
  #define stbir_scalar_lo_clamp( v )
  #endif

  #ifdef STBIR_SIMD

  if ( width_times_channels >= ( stbir__simdfX_float_count * 2 ) )
  {
    float const * end_encode_m8 = encode + width_times_channels - ( stbir__simdfX_float_count * 2 );
    end_output -= ( stbir__simdfX_float_count * 2 );
    STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR
    for(;;)
    {
      stbir__simdfX e0, e1;
      STBIR_SIMD_NO_UNROLL(encode);
      stbir__simdfX_load( e0, encode );
      stbir__simdfX_load( e1, encode+stbir__simdfX_float_count );
      stbir__encode_simdfX_unflip( e0 );
      stbir__encode_simdfX_unflip( e1 );
      stbir__simdfX_store( output, e0 );
      stbir__simdfX_store( output+stbir__simdfX_float_count, e1 );
      encode += stbir__simdfX_float_count * 2;
      output += stbir__simdfX_float_count * 2;
      if ( output < end_output )
        continue;
      if ( output == ( end_output + ( stbir__simdfX_float_count * 2 ) ) )
        break;
      output = end_output;
      encode = end_encode_m8;
    }
    return;
  }

  #else

  #endif

  #endif
}

#undef stbir__decode_suffix
#undef stbir__decode_simdf8_flip
#undef stbir__decode_simdf4_flip
#undef stbir__decode_order0
#undef stbir__decode_order1
#undef stbir__decode_order2
#undef stbir__decode_order3
#undef stbir__encode_order0
#undef stbir__encode_order1
#undef stbir__encode_order2
#undef stbir__encode_order3
#undef stbir__encode_simdf8_unflip
#undef stbir__encode_simdf4_unflip
#undef stbir__encode_simdfX_unflip
#undef STBIR__CODER_NAME
#undef stbir__coder_min_num
#undef stbir__decode_swizzle
#undef stbir_scalar_hi_clamp
#undef stbir_scalar_lo_clamp
#undef STB_IMAGE_RESIZE_DO_CODERS

#elif defined( STB_IMAGE_RESIZE_DO_VERTICALS)

#ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
#define STBIR_chans( start, end ) STBIR_strs_join14(start,STBIR__vertical_channels,end,_cont)
#else
#define STBIR_chans( start, end ) STBIR_strs_join1(start,STBIR__vertical_channels,end)
#endif

#if STBIR__vertical_channels >= 1
#define stbIF0( code ) code
#else
#define stbIF0( code )
#endif
#if STBIR__vertical_channels >= 2
#define stbIF1( code ) code
#else
#define stbIF1( code )
#endif
#if STBIR__vertical_channels >= 3
#define stbIF2( code ) code
#else
#define stbIF2( code )
#endif
#if STBIR__vertical_channels >= 4
#define stbIF3( code ) code
#else
#define stbIF3( code )
#endif
#if STBIR__vertical_channels >= 5
#define stbIF4( code ) code
#else
#define stbIF4( code )
#endif
#if STBIR__vertical_channels >= 6
#define stbIF5( code ) code
#else
#define stbIF5( code )
#endif
#if STBIR__vertical_channels >= 7
#define stbIF6( code ) code
#else
#define stbIF6( code )
#endif
#if STBIR__vertical_channels >= 8
#define stbIF7( code ) code
#else
#define stbIF7( code )
#endif

static void STBIR_chans( stbir__vertical_scatter_with_,_coeffs)( float ** outputs, float const * vertical_coefficients, float const * input, float const * input_end )
{
  stbIF0( float STBIR_SIMD_STREAMOUT_PTR( * ) output0 = outputs[0]; float c0s = vertical_coefficients[0]; )
  stbIF1( float STBIR_SIMD_STREAMOUT_PTR( * ) output1 = outputs[1]; float c1s = vertical_coefficients[1]; )
  stbIF2( float STBIR_SIMD_STREAMOUT_PTR( * ) output2 = outputs[2]; float c2s = vertical_coefficients[2]; )
  stbIF3( float STBIR_SIMD_STREAMOUT_PTR( * ) output3 = outputs[3]; float c3s = vertical_coefficients[3]; )
  stbIF4( float STBIR_SIMD_STREAMOUT_PTR( * ) output4 = outputs[4]; float c4s = vertical_coefficients[4]; )
  stbIF5( float STBIR_SIMD_STREAMOUT_PTR( * ) output5 = outputs[5]; float c5s = vertical_coefficients[5]; )
  stbIF6( float STBIR_SIMD_STREAMOUT_PTR( * ) output6 = outputs[6]; float c6s = vertical_coefficients[6]; )
  stbIF7( float STBIR_SIMD_STREAMOUT_PTR( * ) output7 = outputs[7]; float c7s = vertical_coefficients[7]; )

  #ifdef STBIR_SIMD
  {
    stbIF0(stbir__simdfX c0 = stbir__simdf_frepX( c0s ); )
    stbIF1(stbir__simdfX c1 = stbir__simdf_frepX( c1s ); )
    stbIF2(stbir__simdfX c2 = stbir__simdf_frepX( c2s ); )
    stbIF3(stbir__simdfX c3 = stbir__simdf_frepX( c3s ); )
    stbIF4(stbir__simdfX c4 = stbir__simdf_frepX( c4s ); )
    stbIF5(stbir__simdfX c5 = stbir__simdf_frepX( c5s ); )
    stbIF6(stbir__simdfX c6 = stbir__simdf_frepX( c6s ); )
    stbIF7(stbir__simdfX c7 = stbir__simdf_frepX( c7s ); )
    STBIR_SIMD_NO_UNROLL_LOOP_START
    while ( ( (char*)input_end - (char*) input ) >= (16*stbir__simdfX_float_count) )
    {
      stbir__simdfX o0, o1, o2, o3, r0, r1, r2, r3;
      STBIR_SIMD_NO_UNROLL(output0);

      stbir__simdfX_load( r0, input );               stbir__simdfX_load( r1, input+stbir__simdfX_float_count );     stbir__simdfX_load( r2, input+(2*stbir__simdfX_float_count) );      stbir__simdfX_load( r3, input+(3*stbir__simdfX_float_count) );

      #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
      stbIF0( stbir__simdfX_load( o0, output0 );     stbir__simdfX_load( o1, output0+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output0+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output0+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c0 );  stbir__simdfX_madd( o1, o1, r1, c0 );  stbir__simdfX_madd( o2, o2, r2, c0 );   stbir__simdfX_madd( o3, o3, r3, c0 );
              stbir__simdfX_store( output0, o0 );    stbir__simdfX_store( output0+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output0+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output0+(3*stbir__simdfX_float_count), o3 ); )
      stbIF1( stbir__simdfX_load( o0, output1 );     stbir__simdfX_load( o1, output1+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output1+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output1+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c1 );  stbir__simdfX_madd( o1, o1, r1, c1 );  stbir__simdfX_madd( o2, o2, r2, c1 );   stbir__simdfX_madd( o3, o3, r3, c1 );
              stbir__simdfX_store( output1, o0 );    stbir__simdfX_store( output1+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output1+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output1+(3*stbir__simdfX_float_count), o3 ); )
      stbIF2( stbir__simdfX_load( o0, output2 );     stbir__simdfX_load( o1, output2+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output2+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output2+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c2 );  stbir__simdfX_madd( o1, o1, r1, c2 );  stbir__simdfX_madd( o2, o2, r2, c2 );   stbir__simdfX_madd( o3, o3, r3, c2 );
              stbir__simdfX_store( output2, o0 );    stbir__simdfX_store( output2+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output2+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output2+(3*stbir__simdfX_float_count), o3 ); )
      stbIF3( stbir__simdfX_load( o0, output3 );     stbir__simdfX_load( o1, output3+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output3+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output3+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c3 );  stbir__simdfX_madd( o1, o1, r1, c3 );  stbir__simdfX_madd( o2, o2, r2, c3 );   stbir__simdfX_madd( o3, o3, r3, c3 );
              stbir__simdfX_store( output3, o0 );    stbir__simdfX_store( output3+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output3+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output3+(3*stbir__simdfX_float_count), o3 ); )
      stbIF4( stbir__simdfX_load( o0, output4 );     stbir__simdfX_load( o1, output4+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output4+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output4+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c4 );  stbir__simdfX_madd( o1, o1, r1, c4 );  stbir__simdfX_madd( o2, o2, r2, c4 );   stbir__simdfX_madd( o3, o3, r3, c4 );
              stbir__simdfX_store( output4, o0 );    stbir__simdfX_store( output4+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output4+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output4+(3*stbir__simdfX_float_count), o3 ); )
      stbIF5( stbir__simdfX_load( o0, output5 );     stbir__simdfX_load( o1, output5+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output5+(2*stbir__simdfX_float_count));    stbir__simdfX_load( o3, output5+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c5 );  stbir__simdfX_madd( o1, o1, r1, c5 );  stbir__simdfX_madd( o2, o2, r2, c5 );   stbir__simdfX_madd( o3, o3, r3, c5 );
              stbir__simdfX_store( output5, o0 );    stbir__simdfX_store( output5+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output5+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output5+(3*stbir__simdfX_float_count), o3 ); )
      stbIF6( stbir__simdfX_load( o0, output6 );     stbir__simdfX_load( o1, output6+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output6+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output6+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c6 );  stbir__simdfX_madd( o1, o1, r1, c6 );  stbir__simdfX_madd( o2, o2, r2, c6 );   stbir__simdfX_madd( o3, o3, r3, c6 );
              stbir__simdfX_store( output6, o0 );    stbir__simdfX_store( output6+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output6+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output6+(3*stbir__simdfX_float_count), o3 ); )
      stbIF7( stbir__simdfX_load( o0, output7 );     stbir__simdfX_load( o1, output7+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output7+(2*stbir__simdfX_float_count) );    stbir__simdfX_load( o3, output7+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c7 );  stbir__simdfX_madd( o1, o1, r1, c7 );  stbir__simdfX_madd( o2, o2, r2, c7 );   stbir__simdfX_madd( o3, o3, r3, c7 );
              stbir__simdfX_store( output7, o0 );    stbir__simdfX_store( output7+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output7+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output7+(3*stbir__simdfX_float_count), o3 ); )
      #else
      stbIF0( stbir__simdfX_mult( o0, r0, c0 );      stbir__simdfX_mult( o1, r1, c0 );      stbir__simdfX_mult( o2, r2, c0 );       stbir__simdfX_mult( o3, r3, c0 );
              stbir__simdfX_store( output0, o0 );    stbir__simdfX_store( output0+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output0+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output0+(3*stbir__simdfX_float_count), o3 ); )
      stbIF1( stbir__simdfX_mult( o0, r0, c1 );      stbir__simdfX_mult( o1, r1, c1 );      stbir__simdfX_mult( o2, r2, c1 );       stbir__simdfX_mult( o3, r3, c1 );
              stbir__simdfX_store( output1, o0 );    stbir__simdfX_store( output1+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output1+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output1+(3*stbir__simdfX_float_count), o3 ); )
      stbIF2( stbir__simdfX_mult( o0, r0, c2 );      stbir__simdfX_mult( o1, r1, c2 );      stbir__simdfX_mult( o2, r2, c2 );       stbir__simdfX_mult( o3, r3, c2 );
              stbir__simdfX_store( output2, o0 );    stbir__simdfX_store( output2+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output2+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output2+(3*stbir__simdfX_float_count), o3 ); )
      stbIF3( stbir__simdfX_mult( o0, r0, c3 );      stbir__simdfX_mult( o1, r1, c3 );      stbir__simdfX_mult( o2, r2, c3 );       stbir__simdfX_mult( o3, r3, c3 );
              stbir__simdfX_store( output3, o0 );    stbir__simdfX_store( output3+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output3+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output3+(3*stbir__simdfX_float_count), o3 ); )
      stbIF4( stbir__simdfX_mult( o0, r0, c4 );      stbir__simdfX_mult( o1, r1, c4 );      stbir__simdfX_mult( o2, r2, c4 );       stbir__simdfX_mult( o3, r3, c4 );
              stbir__simdfX_store( output4, o0 );    stbir__simdfX_store( output4+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output4+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output4+(3*stbir__simdfX_float_count), o3 ); )
      stbIF5( stbir__simdfX_mult( o0, r0, c5 );      stbir__simdfX_mult( o1, r1, c5 );      stbir__simdfX_mult( o2, r2, c5 );       stbir__simdfX_mult( o3, r3, c5 );
              stbir__simdfX_store( output5, o0 );    stbir__simdfX_store( output5+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output5+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output5+(3*stbir__simdfX_float_count), o3 ); )
      stbIF6( stbir__simdfX_mult( o0, r0, c6 );      stbir__simdfX_mult( o1, r1, c6 );      stbir__simdfX_mult( o2, r2, c6 );       stbir__simdfX_mult( o3, r3, c6 );
              stbir__simdfX_store( output6, o0 );    stbir__simdfX_store( output6+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output6+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output6+(3*stbir__simdfX_float_count), o3 ); )
      stbIF7( stbir__simdfX_mult( o0, r0, c7 );      stbir__simdfX_mult( o1, r1, c7 );      stbir__simdfX_mult( o2, r2, c7 );       stbir__simdfX_mult( o3, r3, c7 );
              stbir__simdfX_store( output7, o0 );    stbir__simdfX_store( output7+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output7+(2*stbir__simdfX_float_count), o2 );   stbir__simdfX_store( output7+(3*stbir__simdfX_float_count), o3 ); )
      #endif

      input += (4*stbir__simdfX_float_count);
      stbIF0( output0 += (4*stbir__simdfX_float_count); ) stbIF1( output1 += (4*stbir__simdfX_float_count); ) stbIF2( output2 += (4*stbir__simdfX_float_count); ) stbIF3( output3 += (4*stbir__simdfX_float_count); ) stbIF4( output4 += (4*stbir__simdfX_float_count); ) stbIF5( output5 += (4*stbir__simdfX_float_count); ) stbIF6( output6 += (4*stbir__simdfX_float_count); ) stbIF7( output7 += (4*stbir__simdfX_float_count); )
    }
    STBIR_SIMD_NO_UNROLL_LOOP_START
    while ( ( (char*)input_end - (char*) input ) >= 16 )
    {
      stbir__simdf o0, r0;
      STBIR_SIMD_NO_UNROLL(output0);

      stbir__simdf_load( r0, input );

      #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
      stbIF0( stbir__simdf_load( o0, output0 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c0 ) );  stbir__simdf_store( output0, o0 ); )
      stbIF1( stbir__simdf_load( o0, output1 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c1 ) );  stbir__simdf_store( output1, o0 ); )
      stbIF2( stbir__simdf_load( o0, output2 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c2 ) );  stbir__simdf_store( output2, o0 ); )
      stbIF3( stbir__simdf_load( o0, output3 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c3 ) );  stbir__simdf_store( output3, o0 ); )
      stbIF4( stbir__simdf_load( o0, output4 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c4 ) );  stbir__simdf_store( output4, o0 ); )
      stbIF5( stbir__simdf_load( o0, output5 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c5 ) );  stbir__simdf_store( output5, o0 ); )
      stbIF6( stbir__simdf_load( o0, output6 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c6 ) );  stbir__simdf_store( output6, o0 ); )
      stbIF7( stbir__simdf_load( o0, output7 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c7 ) );  stbir__simdf_store( output7, o0 ); )
      #else
      stbIF0( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c0 ) );   stbir__simdf_store( output0, o0 ); )
      stbIF1( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c1 ) );   stbir__simdf_store( output1, o0 ); )
      stbIF2( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c2 ) );   stbir__simdf_store( output2, o0 ); )
      stbIF3( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c3 ) );   stbir__simdf_store( output3, o0 ); )
      stbIF4( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c4 ) );   stbir__simdf_store( output4, o0 ); )
      stbIF5( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c5 ) );   stbir__simdf_store( output5, o0 ); )
      stbIF6( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c6 ) );   stbir__simdf_store( output6, o0 ); )
      stbIF7( stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c7 ) );   stbir__simdf_store( output7, o0 ); )
      #endif

      input += 4;
      stbIF0( output0 += 4; ) stbIF1( output1 += 4; ) stbIF2( output2 += 4; ) stbIF3( output3 += 4; ) stbIF4( output4 += 4; ) stbIF5( output5 += 4; ) stbIF6( output6 += 4; ) stbIF7( output7 += 4; )
    }
  }
  #else
  STBIR_NO_UNROLL_LOOP_START
  while ( ( (char*)input_end - (char*) input ) >= 16 )
  {
    float r0, r1, r2, r3;
    STBIR_NO_UNROLL(input);

    r0 = input[0], r1 = input[1], r2 = input[2], r3 = input[3];

    #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
    stbIF0( output0[0] += ( r0 * c0s ); output0[1] += ( r1 * c0s ); output0[2] += ( r2 * c0s ); output0[3] += ( r3 * c0s ); )
    stbIF1( output1[0] += ( r0 * c1s ); output1[1] += ( r1 * c1s ); output1[2] += ( r2 * c1s ); output1[3] += ( r3 * c1s ); )
    stbIF2( output2[0] += ( r0 * c2s ); output2[1] += ( r1 * c2s ); output2[2] += ( r2 * c2s ); output2[3] += ( r3 * c2s ); )
    stbIF3( output3[0] += ( r0 * c3s ); output3[1] += ( r1 * c3s ); output3[2] += ( r2 * c3s ); output3[3] += ( r3 * c3s ); )
    stbIF4( output4[0] += ( r0 * c4s ); output4[1] += ( r1 * c4s ); output4[2] += ( r2 * c4s ); output4[3] += ( r3 * c4s ); )
    stbIF5( output5[0] += ( r0 * c5s ); output5[1] += ( r1 * c5s ); output5[2] += ( r2 * c5s ); output5[3] += ( r3 * c5s ); )
    stbIF6( output6[0] += ( r0 * c6s ); output6[1] += ( r1 * c6s ); output6[2] += ( r2 * c6s ); output6[3] += ( r3 * c6s ); )
    stbIF7( output7[0] += ( r0 * c7s ); output7[1] += ( r1 * c7s ); output7[2] += ( r2 * c7s ); output7[3] += ( r3 * c7s ); )
    #else
    stbIF0( output0[0]  = ( r0 * c0s ); output0[1]  = ( r1 * c0s ); output0[2]  = ( r2 * c0s ); output0[3]  = ( r3 * c0s ); )
    stbIF1( output1[0]  = ( r0 * c1s ); output1[1]  = ( r1 * c1s ); output1[2]  = ( r2 * c1s ); output1[3]  = ( r3 * c1s ); )
    stbIF2( output2[0]  = ( r0 * c2s ); output2[1]  = ( r1 * c2s ); output2[2]  = ( r2 * c2s ); output2[3]  = ( r3 * c2s ); )
    stbIF3( output3[0]  = ( r0 * c3s ); output3[1]  = ( r1 * c3s ); output3[2]  = ( r2 * c3s ); output3[3]  = ( r3 * c3s ); )
    stbIF4( output4[0]  = ( r0 * c4s ); output4[1]  = ( r1 * c4s ); output4[2]  = ( r2 * c4s ); output4[3]  = ( r3 * c4s ); )
    stbIF5( output5[0]  = ( r0 * c5s ); output5[1]  = ( r1 * c5s ); output5[2]  = ( r2 * c5s ); output5[3]  = ( r3 * c5s ); )
    stbIF6( output6[0]  = ( r0 * c6s ); output6[1]  = ( r1 * c6s ); output6[2]  = ( r2 * c6s ); output6[3]  = ( r3 * c6s ); )
    stbIF7( output7[0]  = ( r0 * c7s ); output7[1]  = ( r1 * c7s ); output7[2]  = ( r2 * c7s ); output7[3]  = ( r3 * c7s ); )
    #endif

    input += 4;
    stbIF0( output0 += 4; ) stbIF1( output1 += 4; ) stbIF2( output2 += 4; ) stbIF3( output3 += 4; ) stbIF4( output4 += 4; ) stbIF5( output5 += 4; ) stbIF6( output6 += 4; ) stbIF7( output7 += 4; )
  }
  #endif
  STBIR_NO_UNROLL_LOOP_START
  while ( input < input_end )
  {
    float r = input[0];
    STBIR_NO_UNROLL(output0);

    #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
    stbIF0( output0[0] += ( r * c0s ); )
    stbIF1( output1[0] += ( r * c1s ); )
    stbIF2( output2[0] += ( r * c2s ); )
    stbIF3( output3[0] += ( r * c3s ); )
    stbIF4( output4[0] += ( r * c4s ); )
    stbIF5( output5[0] += ( r * c5s ); )
    stbIF6( output6[0] += ( r * c6s ); )
    stbIF7( output7[0] += ( r * c7s ); )
    #else
    stbIF0( output0[0]  = ( r * c0s ); )
    stbIF1( output1[0]  = ( r * c1s ); )
    stbIF2( output2[0]  = ( r * c2s ); )
    stbIF3( output3[0]  = ( r * c3s ); )
    stbIF4( output4[0]  = ( r * c4s ); )
    stbIF5( output5[0]  = ( r * c5s ); )
    stbIF6( output6[0]  = ( r * c6s ); )
    stbIF7( output7[0]  = ( r * c7s ); )
    #endif

    ++input;
    stbIF0( ++output0; ) stbIF1( ++output1; ) stbIF2( ++output2; ) stbIF3( ++output3; ) stbIF4( ++output4; ) stbIF5( ++output5; ) stbIF6( ++output6; ) stbIF7( ++output7; )
  }
}

static void STBIR_chans( stbir__vertical_gather_with_,_coeffs)( float * outputp, float const * vertical_coefficients, float const ** inputs, float const * input0_end )
{
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = outputp;

  stbIF0( float const * input0 = inputs[0]; float c0s = vertical_coefficients[0]; )
  stbIF1( float const * input1 = inputs[1]; float c1s = vertical_coefficients[1]; )
  stbIF2( float const * input2 = inputs[2]; float c2s = vertical_coefficients[2]; )
  stbIF3( float const * input3 = inputs[3]; float c3s = vertical_coefficients[3]; )
  stbIF4( float const * input4 = inputs[4]; float c4s = vertical_coefficients[4]; )
  stbIF5( float const * input5 = inputs[5]; float c5s = vertical_coefficients[5]; )
  stbIF6( float const * input6 = inputs[6]; float c6s = vertical_coefficients[6]; )
  stbIF7( float const * input7 = inputs[7]; float c7s = vertical_coefficients[7]; )

  #ifdef STBIR_SIMD
  {
    stbIF0(stbir__simdfX c0 = stbir__simdf_frepX( c0s ); )
    stbIF1(stbir__simdfX c1 = stbir__simdf_frepX( c1s ); )
    stbIF2(stbir__simdfX c2 = stbir__simdf_frepX( c2s ); )
    stbIF3(stbir__simdfX c3 = stbir__simdf_frepX( c3s ); )
    stbIF4(stbir__simdfX c4 = stbir__simdf_frepX( c4s ); )
    stbIF5(stbir__simdfX c5 = stbir__simdf_frepX( c5s ); )
    stbIF6(stbir__simdfX c6 = stbir__simdf_frepX( c6s ); )
    stbIF7(stbir__simdfX c7 = stbir__simdf_frepX( c7s ); )

    STBIR_SIMD_NO_UNROLL_LOOP_START
    while ( ( (char*)input0_end - (char*) input0 ) >= (16*stbir__simdfX_float_count) )
    {
      stbir__simdfX o0, o1, o2, o3, r0, r1, r2, r3;
      STBIR_SIMD_NO_UNROLL(output);

      stbIF0( stbir__prefetch( input0 + (16*stbir__simdfX_float_count) ); )
      stbIF1( stbir__prefetch( input1 + (16*stbir__simdfX_float_count) ); )
      stbIF2( stbir__prefetch( input2 + (16*stbir__simdfX_float_count) ); )
      stbIF3( stbir__prefetch( input3 + (16*stbir__simdfX_float_count) ); )
      stbIF4( stbir__prefetch( input4 + (16*stbir__simdfX_float_count) ); )
      stbIF5( stbir__prefetch( input5 + (16*stbir__simdfX_float_count) ); )
      stbIF6( stbir__prefetch( input6 + (16*stbir__simdfX_float_count) ); )
      stbIF7( stbir__prefetch( input7 + (16*stbir__simdfX_float_count) ); )

      #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
      stbIF0( stbir__simdfX_load( o0, output );      stbir__simdfX_load( o1, output+stbir__simdfX_float_count );   stbir__simdfX_load( o2, output+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( o3, output+(3*stbir__simdfX_float_count) );
              stbir__simdfX_load( r0, input0 );      stbir__simdfX_load( r1, input0+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input0+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input0+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c0 );  stbir__simdfX_madd( o1, o1, r1, c0 );                         stbir__simdfX_madd( o2, o2, r2, c0 );                             stbir__simdfX_madd( o3, o3, r3, c0 ); )
      #else
      stbIF0( stbir__simdfX_load( r0, input0 );      stbir__simdfX_load( r1, input0+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input0+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input0+(3*stbir__simdfX_float_count) );
              stbir__simdfX_mult( o0, r0, c0 );      stbir__simdfX_mult( o1, r1, c0 );                             stbir__simdfX_mult( o2, r2, c0 );                                 stbir__simdfX_mult( o3, r3, c0 );  )
      #endif

      stbIF1( stbir__simdfX_load( r0, input1 );      stbir__simdfX_load( r1, input1+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input1+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input1+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c1 );  stbir__simdfX_madd( o1, o1, r1, c1 );                         stbir__simdfX_madd( o2, o2, r2, c1 );                             stbir__simdfX_madd( o3, o3, r3, c1 ); )
      stbIF2( stbir__simdfX_load( r0, input2 );      stbir__simdfX_load( r1, input2+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input2+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input2+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c2 );  stbir__simdfX_madd( o1, o1, r1, c2 );                         stbir__simdfX_madd( o2, o2, r2, c2 );                             stbir__simdfX_madd( o3, o3, r3, c2 ); )
      stbIF3( stbir__simdfX_load( r0, input3 );      stbir__simdfX_load( r1, input3+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input3+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input3+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c3 );  stbir__simdfX_madd( o1, o1, r1, c3 );                         stbir__simdfX_madd( o2, o2, r2, c3 );                             stbir__simdfX_madd( o3, o3, r3, c3 ); )
      stbIF4( stbir__simdfX_load( r0, input4 );      stbir__simdfX_load( r1, input4+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input4+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input4+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c4 );  stbir__simdfX_madd( o1, o1, r1, c4 );                         stbir__simdfX_madd( o2, o2, r2, c4 );                             stbir__simdfX_madd( o3, o3, r3, c4 ); )
      stbIF5( stbir__simdfX_load( r0, input5 );      stbir__simdfX_load( r1, input5+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input5+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input5+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c5 );  stbir__simdfX_madd( o1, o1, r1, c5 );                         stbir__simdfX_madd( o2, o2, r2, c5 );                             stbir__simdfX_madd( o3, o3, r3, c5 ); )
      stbIF6( stbir__simdfX_load( r0, input6 );      stbir__simdfX_load( r1, input6+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input6+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input6+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c6 );  stbir__simdfX_madd( o1, o1, r1, c6 );                         stbir__simdfX_madd( o2, o2, r2, c6 );                             stbir__simdfX_madd( o3, o3, r3, c6 ); )
      stbIF7( stbir__simdfX_load( r0, input7 );      stbir__simdfX_load( r1, input7+stbir__simdfX_float_count );   stbir__simdfX_load( r2, input7+(2*stbir__simdfX_float_count) );   stbir__simdfX_load( r3, input7+(3*stbir__simdfX_float_count) );
              stbir__simdfX_madd( o0, o0, r0, c7 );  stbir__simdfX_madd( o1, o1, r1, c7 );                         stbir__simdfX_madd( o2, o2, r2, c7 );                             stbir__simdfX_madd( o3, o3, r3, c7 ); )

      stbir__simdfX_store( output, o0 );             stbir__simdfX_store( output+stbir__simdfX_float_count, o1 );  stbir__simdfX_store( output+(2*stbir__simdfX_float_count), o2 );  stbir__simdfX_store( output+(3*stbir__simdfX_float_count), o3 );
      output += (4*stbir__simdfX_float_count);
      stbIF0( input0 += (4*stbir__simdfX_float_count); ) stbIF1( input1 += (4*stbir__simdfX_float_count); ) stbIF2( input2 += (4*stbir__simdfX_float_count); ) stbIF3( input3 += (4*stbir__simdfX_float_count); ) stbIF4( input4 += (4*stbir__simdfX_float_count); ) stbIF5( input5 += (4*stbir__simdfX_float_count); ) stbIF6( input6 += (4*stbir__simdfX_float_count); ) stbIF7( input7 += (4*stbir__simdfX_float_count); )
    }

    STBIR_SIMD_NO_UNROLL_LOOP_START
    while ( ( (char*)input0_end - (char*) input0 ) >= 16 )
    {
      stbir__simdf o0, r0;
      STBIR_SIMD_NO_UNROLL(output);

      #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
      stbIF0( stbir__simdf_load( o0, output );   stbir__simdf_load( r0, input0 ); stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c0 ) ); )
      #else
      stbIF0( stbir__simdf_load( r0, input0 );  stbir__simdf_mult( o0, r0, stbir__if_simdf8_cast_to_simdf4( c0 ) ); )
      #endif
      stbIF1( stbir__simdf_load( r0, input1 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c1 ) ); )
      stbIF2( stbir__simdf_load( r0, input2 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c2 ) ); )
      stbIF3( stbir__simdf_load( r0, input3 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c3 ) ); )
      stbIF4( stbir__simdf_load( r0, input4 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c4 ) ); )
      stbIF5( stbir__simdf_load( r0, input5 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c5 ) ); )
      stbIF6( stbir__simdf_load( r0, input6 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c6 ) ); )
      stbIF7( stbir__simdf_load( r0, input7 );  stbir__simdf_madd( o0, o0, r0, stbir__if_simdf8_cast_to_simdf4( c7 ) ); )

      stbir__simdf_store( output, o0 );
      output += 4;
      stbIF0( input0 += 4; ) stbIF1( input1 += 4; ) stbIF2( input2 += 4; ) stbIF3( input3 += 4; ) stbIF4( input4 += 4; ) stbIF5( input5 += 4; ) stbIF6( input6 += 4; ) stbIF7( input7 += 4; )
    }
  }
  #else
  STBIR_NO_UNROLL_LOOP_START
  while ( ( (char*)input0_end - (char*) input0 ) >= 16 )
  {
    float o0, o1, o2, o3;
    STBIR_NO_UNROLL(output);
    #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
    stbIF0( o0 = output[0] + input0[0] * c0s; o1 = output[1] + input0[1] * c0s; o2 = output[2] + input0[2] * c0s; o3 = output[3] + input0[3] * c0s; )
    #else
    stbIF0( o0  = input0[0] * c0s; o1  = input0[1] * c0s; o2  = input0[2] * c0s; o3  = input0[3] * c0s; )
    #endif
    stbIF1( o0 += input1[0] * c1s; o1 += input1[1] * c1s; o2 += input1[2] * c1s; o3 += input1[3] * c1s; )
    stbIF2( o0 += input2[0] * c2s; o1 += input2[1] * c2s; o2 += input2[2] * c2s; o3 += input2[3] * c2s; )
    stbIF3( o0 += input3[0] * c3s; o1 += input3[1] * c3s; o2 += input3[2] * c3s; o3 += input3[3] * c3s; )
    stbIF4( o0 += input4[0] * c4s; o1 += input4[1] * c4s; o2 += input4[2] * c4s; o3 += input4[3] * c4s; )
    stbIF5( o0 += input5[0] * c5s; o1 += input5[1] * c5s; o2 += input5[2] * c5s; o3 += input5[3] * c5s; )
    stbIF6( o0 += input6[0] * c6s; o1 += input6[1] * c6s; o2 += input6[2] * c6s; o3 += input6[3] * c6s; )
    stbIF7( o0 += input7[0] * c7s; o1 += input7[1] * c7s; o2 += input7[2] * c7s; o3 += input7[3] * c7s; )
    output[0] = o0; output[1] = o1; output[2] = o2; output[3] = o3;
    output += 4;
    stbIF0( input0 += 4; ) stbIF1( input1 += 4; ) stbIF2( input2 += 4; ) stbIF3( input3 += 4; ) stbIF4( input4 += 4; ) stbIF5( input5 += 4; ) stbIF6( input6 += 4; ) stbIF7( input7 += 4; )
  }
  #endif
  STBIR_NO_UNROLL_LOOP_START
  while ( input0 < input0_end )
  {
    float o0;
    STBIR_NO_UNROLL(output);
    #ifdef STB_IMAGE_RESIZE_VERTICAL_CONTINUE
    stbIF0( o0 = output[0] + input0[0] * c0s; )
    #else
    stbIF0( o0  = input0[0] * c0s; )
    #endif
    stbIF1( o0 += input1[0] * c1s; )
    stbIF2( o0 += input2[0] * c2s; )
    stbIF3( o0 += input3[0] * c3s; )
    stbIF4( o0 += input4[0] * c4s; )
    stbIF5( o0 += input5[0] * c5s; )
    stbIF6( o0 += input6[0] * c6s; )
    stbIF7( o0 += input7[0] * c7s; )
    output[0] = o0;
    ++output;
    stbIF0( ++input0; ) stbIF1( ++input1; ) stbIF2( ++input2; ) stbIF3( ++input3; ) stbIF4( ++input4; ) stbIF5( ++input5; ) stbIF6( ++input6; ) stbIF7( ++input7; )
  }
}

#undef stbIF0
#undef stbIF1
#undef stbIF2
#undef stbIF3
#undef stbIF4
#undef stbIF5
#undef stbIF6
#undef stbIF7
#undef STB_IMAGE_RESIZE_DO_VERTICALS
#undef STBIR__vertical_channels
#undef STB_IMAGE_RESIZE_DO_HORIZONTALS
#undef STBIR_strs_join24
#undef STBIR_strs_join14
#undef STBIR_chans

#else

#define STBIR_chans( start, end ) STBIR_strs_join1(start,STBIR__horizontal_channels,end)

#ifndef stbir__2_coeff_only
#define stbir__2_coeff_only()             \
    stbir__1_coeff_only();                \
    stbir__1_coeff_remnant(1);
#endif

#ifndef stbir__2_coeff_remnant
#define stbir__2_coeff_remnant( ofs )     \
    stbir__1_coeff_remnant(ofs);          \
    stbir__1_coeff_remnant((ofs)+1);
#endif

#ifndef stbir__3_coeff_only
#define stbir__3_coeff_only()             \
    stbir__2_coeff_only();                \
    stbir__1_coeff_remnant(2);
#endif

#ifndef stbir__3_coeff_remnant
#define stbir__3_coeff_remnant( ofs )     \
    stbir__2_coeff_remnant(ofs);          \
    stbir__1_coeff_remnant((ofs)+2);
#endif

#ifndef stbir__3_coeff_setup
#define stbir__3_coeff_setup()
#endif

#ifndef stbir__4_coeff_start
#define stbir__4_coeff_start()            \
    stbir__2_coeff_only();                \
    stbir__2_coeff_remnant(2);
#endif

#ifndef stbir__4_coeff_continue_from_4
#define stbir__4_coeff_continue_from_4( ofs )     \
    stbir__2_coeff_remnant(ofs);                  \
    stbir__2_coeff_remnant((ofs)+2);
#endif

#ifndef stbir__store_output_tiny
#define stbir__store_output_tiny stbir__store_output
#endif

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_1_coeff)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__1_coeff_only();
    stbir__store_output_tiny();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_2_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__2_coeff_only();
    stbir__store_output_tiny();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_3_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__3_coeff_only();
    stbir__store_output_tiny();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_4_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_5_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__1_coeff_remnant(4);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_6_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__2_coeff_remnant(4);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_7_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  stbir__3_coeff_setup();
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;

    stbir__4_coeff_start();
    stbir__3_coeff_remnant(4);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_8_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__4_coeff_continue_from_4(4);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_9_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__4_coeff_continue_from_4(4);
    stbir__1_coeff_remnant(8);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_10_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__4_coeff_continue_from_4(4);
    stbir__2_coeff_remnant(8);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_11_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  stbir__3_coeff_setup();
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__4_coeff_continue_from_4(4);
    stbir__3_coeff_remnant(8);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_12_coeffs)( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    float const * hc = horizontal_coefficients;
    stbir__4_coeff_start();
    stbir__4_coeff_continue_from_4(4);
    stbir__4_coeff_continue_from_4(8);
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_n_coeffs_mod0 )( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    int n = ( ( horizontal_contributors->n1 - horizontal_contributors->n0 + 1 ) - 4 + 3 ) >> 2;
    float const * hc = horizontal_coefficients;

    stbir__4_coeff_start();
    STBIR_SIMD_NO_UNROLL_LOOP_START
    do {
      hc += 4;
      decode += STBIR__horizontal_channels * 4;
      stbir__4_coeff_continue_from_4( 0 );
      --n;
    } while ( n > 0 );
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_n_coeffs_mod1 )( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    int n = ( ( horizontal_contributors->n1 - horizontal_contributors->n0 + 1 ) - 5 + 3 ) >> 2;
    float const * hc = horizontal_coefficients;

    stbir__4_coeff_start();
    STBIR_SIMD_NO_UNROLL_LOOP_START
    do {
      hc += 4;
      decode += STBIR__horizontal_channels * 4;
      stbir__4_coeff_continue_from_4( 0 );
      --n;
    } while ( n > 0 );
    stbir__1_coeff_remnant( 4 );
    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_n_coeffs_mod2 )( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    int n = ( ( horizontal_contributors->n1 - horizontal_contributors->n0 + 1 ) - 6 + 3 ) >> 2;
    float const * hc = horizontal_coefficients;

    stbir__4_coeff_start();
    STBIR_SIMD_NO_UNROLL_LOOP_START
    do {
      hc += 4;
      decode += STBIR__horizontal_channels * 4;
      stbir__4_coeff_continue_from_4( 0 );
      --n;
    } while ( n > 0 );
    stbir__2_coeff_remnant( 4 );

    stbir__store_output();
  } while ( output < output_end );
}

static void STBIR_chans( stbir__horizontal_gather_,_channels_with_n_coeffs_mod3 )( float * output_buffer, unsigned int output_sub_size, float const * decode_buffer, stbir__contributors const * horizontal_contributors, float const * horizontal_coefficients, int coefficient_width )
{
  float const * output_end = output_buffer + output_sub_size * STBIR__horizontal_channels;
  float STBIR_SIMD_STREAMOUT_PTR( * ) output = output_buffer;
  stbir__3_coeff_setup();
  STBIR_SIMD_NO_UNROLL_LOOP_START
  do {
    float const * decode = decode_buffer + horizontal_contributors->n0 * STBIR__horizontal_channels;
    int n = ( ( horizontal_contributors->n1 - horizontal_contributors->n0 + 1 ) - 7 + 3 ) >> 2;
    float const * hc = horizontal_coefficients;

    stbir__4_coeff_start();
    STBIR_SIMD_NO_UNROLL_LOOP_START
    do {
      hc += 4;
      decode += STBIR__horizontal_channels * 4;
      stbir__4_coeff_continue_from_4( 0 );
      --n;
    } while ( n > 0 );
    stbir__3_coeff_remnant( 4 );

    stbir__store_output();
  } while ( output < output_end );
}

static stbir__horizontal_gather_channels_func * STBIR_chans(stbir__horizontal_gather_,_channels_with_n_coeffs_funcs)[4]=
{
  STBIR_chans(stbir__horizontal_gather_,_channels_with_n_coeffs_mod0),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_n_coeffs_mod1),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_n_coeffs_mod2),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_n_coeffs_mod3),
};

static stbir__horizontal_gather_channels_func * STBIR_chans(stbir__horizontal_gather_,_channels_funcs)[12]=
{
  STBIR_chans(stbir__horizontal_gather_,_channels_with_1_coeff),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_2_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_3_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_4_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_5_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_6_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_7_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_8_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_9_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_10_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_11_coeffs),
  STBIR_chans(stbir__horizontal_gather_,_channels_with_12_coeffs),
};

#undef STBIR__horizontal_channels
#undef STB_IMAGE_RESIZE_DO_HORIZONTALS
#undef stbir__1_coeff_only
#undef stbir__1_coeff_remnant
#undef stbir__2_coeff_only
#undef stbir__2_coeff_remnant
#undef stbir__3_coeff_only
#undef stbir__3_coeff_remnant
#undef stbir__3_coeff_setup
#undef stbir__4_coeff_start
#undef stbir__4_coeff_continue_from_4
#undef stbir__store_output
#undef stbir__store_output_tiny
#undef STBIR_chans

#endif

#undef STBIR_strs_join2
#undef STBIR_strs_join1

#endif

/*
------------------------------------------------------------------------------
MIT License
Copyright (c) 2017 Sean Barrett
Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
------------------------------------------------------------------------------
*/
