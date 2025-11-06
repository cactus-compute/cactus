
#define STBI_INCLUDE_STB_IMAGE_H

#define STBIDEF extern



typedef struct
{
   int      (*read)  (void *user,char *data,int size);   // fill 'data' with 'size' bytes.  return number of bytes actually read
   void     (*skip)  (void *user,int n);                 // skip the next 'n' bytes, or 'unget' the last -n bytes if negative
   int      (*eof)   (void *user);                       // returns nonzero if we are at end of file/data
} stbi_io_callbacks;


STBIDEF stbi_uc *stbi_load_from_memory   (stbi_uc           const *buffer, int len   , int *x, int *y, int *channels_in_file, int desired_channels);
STBIDEF stbi_uc *stbi_load_from_callbacks(stbi_io_callbacks const *clbk  , void *user, int *x, int *y, int *channels_in_file, int desired_channels);

STBIDEF stbi_uc *stbi_load            (char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);
STBIDEF stbi_uc *stbi_load_from_file  (FILE *f, int *x, int *y, int *channels_in_file, int desired_channels);

STBIDEF stbi_uc *stbi_load_gif_from_memory(stbi_uc const *buffer, int len, int **delays, int *x, int *y, int *z, int *comp, int req_comp);



STBIDEF stbi_us *stbi_load_16_from_memory   (stbi_uc const *buffer, int len, int *x, int *y, int *channels_in_file, int desired_channels);
STBIDEF stbi_us *stbi_load_16_from_callbacks(stbi_io_callbacks const *clbk, void *user, int *x, int *y, int *channels_in_file, int desired_channels);

STBIDEF stbi_us *stbi_load_16          (char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);
STBIDEF stbi_us *stbi_load_from_file_16(FILE *f, int *x, int *y, int *channels_in_file, int desired_channels);

   STBIDEF float *stbi_loadf_from_memory     (stbi_uc const *buffer, int len, int *x, int *y, int *channels_in_file, int desired_channels);
   STBIDEF float *stbi_loadf_from_callbacks  (stbi_io_callbacks const *clbk, void *user, int *x, int *y,  int *channels_in_file, int desired_channels);

   STBIDEF float *stbi_loadf            (char const *filename, int *x, int *y, int *channels_in_file, int desired_channels);
   STBIDEF float *stbi_loadf_from_file  (FILE *f, int *x, int *y, int *channels_in_file, int desired_channels);


STBIDEF int    stbi_is_hdr_from_callbacks(stbi_io_callbacks const *clbk, void *user);
STBIDEF int    stbi_is_hdr_from_memory(stbi_uc const *buffer, int len);
STBIDEF int      stbi_is_hdr          (char const *filename);
STBIDEF int      stbi_is_hdr_from_file(FILE *f);
STBIDEF void stbi_convert_iphone_png_to_rgb(int flag_true_if_should_convert);

STBIDEF void stbi_set_flip_vertically_on_load(int flag_true_if_should_flip);

STBIDEF void stbi_set_unpremultiply_on_load_thread(int flag_true_if_should_unpremultiply);
STBIDEF void stbi_convert_iphone_png_to_rgb_thread(int flag_true_if_should_convert);
STBIDEF void stbi_set_flip_vertically_on_load_thread(int flag_true_if_should_flip);


STBIDEF char *stbi_zlib_decode_malloc_guesssize(const char *buffer, int len, int initial_size, int *outlen);
STBIDEF char *stbi_zlib_decode_malloc_guesssize_headerflag(const char *buffer, int len, int initial_size, int *outlen, int parse_header);
STBIDEF char *stbi_zlib_decode_malloc(const char *buffer, int len, int *outlen);
STBIDEF int   stbi_zlib_decode_buffer(char *obuffer, int olen, const char *ibuffer, int ilen);

STBIDEF char *stbi_zlib_decode_noheader_malloc(const char *buffer, int len, int *outlen);
STBIDEF int   stbi_zlib_decode_noheader_buffer(char *obuffer, int olen, const char *ibuffer, int ilen);









#include <stdarg.h>
#include <stddef.h> // ptrdiff_t on osx
#include <stdlib.h>
#include <string.h>
#include <limits.h>

#include <math.h>  // ldexp, pow

#include <stdio.h>

#include <assert.h>
#define STBI_ASSERT(x) assert(x)

#define STBI_EXTERN extern


   #define stbi_inline



#include <stdint.h>
typedef uint16_t stbi__uint16;
typedef int16_t  stbi__int16;
typedef uint32_t stbi__uint32;
typedef int32_t  stbi__int32;

typedef unsigned char validate_uint32[sizeof(stbi__uint32)==4 ? 1 : -1];

#define STBI_NOTUSED(v)  (void)sizeof(v)


   #define stbi_lrot(x,y)  (((x) << (y)) | ((x) >> (-(y) & 31)))


#define STBI_MALLOC(sz)           malloc(sz)
#define STBI_REALLOC(p,newsz)     realloc(p,newsz)
#define STBI_FREE(p)              free(p)

#define STBI_REALLOC_SIZED(p,oldsz,newsz) STBI_REALLOC(p,newsz)




#define STBI_SSE2
#include <emmintrin.h>

#define STBI_SIMD_ALIGN(type, name) type name __attribute__((aligned(16)))

static int stbi__sse2_available(void)
{
   return 1;
}




#define STBI_SIMD_ALIGN(type, name) type name

#define STBI_MAX_DIMENSIONS (1 << 24)


typedef struct
{
   stbi__uint32 img_x, img_y;
   int img_n, img_out_n;

   stbi_io_callbacks io;
   void *io_user_data;

   int read_from_callbacks;
   int buflen;
   stbi_uc buffer_start[128];
   int callback_already_read;

   stbi_uc *img_buffer, *img_buffer_end;
   stbi_uc *img_buffer_original, *img_buffer_original_end;
} stbi__context;


static void stbi__refill_buffer(stbi__context *s);

static void stbi__start_mem(stbi__context *s, stbi_uc const *buffer, int len)
{
   s->io.read = NULL;
   s->read_from_callbacks = 0;
   s->callback_already_read = 0;
   s->img_buffer = s->img_buffer_original = (stbi_uc *) buffer;
   s->img_buffer_end = s->img_buffer_original_end = (stbi_uc *) buffer+len;
}

static void stbi__start_callbacks(stbi__context *s, stbi_io_callbacks *c, void *user)
{
   s->io = *c;
   s->io_user_data = user;
   s->buflen = sizeof(s->buffer_start);
   s->read_from_callbacks = 1;
   s->callback_already_read = 0;
   s->img_buffer = s->img_buffer_original = s->buffer_start;
   stbi__refill_buffer(s);
   s->img_buffer_original_end = s->img_buffer_end;
}


static int stbi__stdio_read(void *user, char *data, int size)
{
   return (int) fread(data,1,size,(FILE*) user);
}

static void stbi__stdio_skip(void *user, int n)
{
   int ch;
   fseek((FILE*) user, n, SEEK_CUR);
   ch = fgetc((FILE*) user);  /* have to read a byte to reset feof()'s flag */
   if (ch != EOF) {
      ungetc(ch, (FILE *) user);  /* push byte back onto stream if valid. */
   }
}

static int stbi__stdio_eof(void *user)
{
   return feof((FILE*) user) || ferror((FILE *) user);
}

static stbi_io_callbacks stbi__stdio_callbacks =
{
   stbi__stdio_read,
   stbi__stdio_skip,
   stbi__stdio_eof,
};

static void stbi__start_file(stbi__context *s, FILE *f)
{
   stbi__start_callbacks(s, &stbi__stdio_callbacks, (void *) f);
}



static void stbi__rewind(stbi__context *s)
{
   s->img_buffer = s->img_buffer_original;
   s->img_buffer_end = s->img_buffer_original_end;
}

enum
{
   STBI_ORDER_RGB,
   STBI_ORDER_BGR
};

typedef struct
{
   int bits_per_channel;
   int num_channels;
   int channel_order;
} stbi__result_info;

static int      stbi__jpeg_test(stbi__context *s);
static void    *stbi__jpeg_load(stbi__context *s, int *x, int *y, int *comp, int req_comp, stbi__result_info *ri);
static int      stbi__jpeg_info(stbi__context *s, int *x, int *y, int *comp);

static int      stbi__png_test(stbi__context *s);
static void    *stbi__png_load(stbi__context *s, int *x, int *y, int *comp, int req_comp, stbi__result_info *ri);
static int      stbi__png_info(stbi__context *s, int *x, int *y, int *comp);
static int      stbi__png_is16(stbi__context *s);

static int      stbi__gif_test(stbi__context *s);
static void    *stbi__gif_load(stbi__context *s, int *x, int *y, int *comp, int req_comp, stbi__result_info *ri);
static void    *stbi__load_gif_main(stbi__context *s, int **delays, int *x, int *y, int *z, int *comp, int req_comp);
static int      stbi__gif_info(stbi__context *s, int *x, int *y, int *comp);

static
const char *stbi__g_failure_reason;

STBIDEF const char *stbi_failure_reason(void)
{
   return stbi__g_failure_reason;
}

static int stbi__err(const char *str)
{
   stbi__g_failure_reason = str;
   return 0;
}

static void *stbi__malloc(size_t size)
{
    return STBI_MALLOC(size);
}


static int stbi__addsizes_valid(int a, int b)
{
   if (b < 0) return 0;
   return a <= INT_MAX - b;
}

static int stbi__mul2sizes_valid(int a, int b)
{
   if (a < 0 || b < 0) return 0;
   if (b == 0) return 1; // mul-by-0 is always safe
   return a <= INT_MAX/b;
}

static int stbi__mad2sizes_valid(int a, int b, int add)
{
   return stbi__mul2sizes_valid(a, b) && stbi__addsizes_valid(a*b, add);
}

static int stbi__mad3sizes_valid(int a, int b, int c, int add)
{
   return stbi__mul2sizes_valid(a, b) && stbi__mul2sizes_valid(a*b, c) &&
      stbi__addsizes_valid(a*b*c, add);
}

static int stbi__mad4sizes_valid(int a, int b, int c, int d, int add)
{
   return stbi__mul2sizes_valid(a, b) && stbi__mul2sizes_valid(a*b, c) &&
      stbi__mul2sizes_valid(a*b*c, d) && stbi__addsizes_valid(a*b*c*d, add);
}

static void *stbi__malloc_mad2(int a, int b, int add)
{
   if (!stbi__mad2sizes_valid(a, b, add)) return NULL;
   return stbi__malloc(a*b + add);
}

static void *stbi__malloc_mad3(int a, int b, int c, int add)
{
   if (!stbi__mad3sizes_valid(a, b, c, add)) return NULL;
   return stbi__malloc(a*b*c + add);
}

static void *stbi__malloc_mad4(int a, int b, int c, int d, int add)
{
   if (!stbi__mad4sizes_valid(a, b, c, d, add)) return NULL;
   return stbi__malloc(a*b*c*d + add);
}

static int stbi__addints_valid(int a, int b)
{
   if ((a >= 0) != (b >= 0)) return 1; // a and b have different signs, so no overflow
   if (a < 0 && b < 0) return a >= INT_MIN - b; // same as a + b >= INT_MIN; INT_MIN - b cannot overflow since b < 0.
   return a <= INT_MAX - b;
}

static int stbi__mul2shorts_valid(int a, int b)
{
   if (b == 0 || b == -1) return 1; // multiplication by 0 is always 0; check for -1 so SHRT_MIN/b doesn't overflow
   if ((a >= 0) == (b >= 0)) return a <= SHRT_MAX/b; // product is positive, so similar to mul2sizes_valid
   if (b < 0) return a <= SHRT_MIN / b; // same as a * b >= SHRT_MIN
   return a >= SHRT_MIN / b;
}


   #define stbi__err(x,y)  stbi__err(x)

#define stbi__errpf(x,y)   ((float *)(size_t) (stbi__err(x,y)?NULL:NULL))
#define stbi__errpuc(x,y)  ((unsigned char *)(size_t) (stbi__err(x,y)?NULL:NULL))

STBIDEF void stbi_image_free(void *retval_from_stbi_load)
{
   STBI_FREE(retval_from_stbi_load);
}

static float   *stbi__ldr_to_hdr(stbi_uc *data, int x, int y, int comp);

static int stbi__vertically_flip_on_load_global = 0;

STBIDEF void stbi_set_flip_vertically_on_load(int flag_true_if_should_flip)
{
   stbi__vertically_flip_on_load_global = flag_true_if_should_flip;
}

#define stbi__vertically_flip_on_load  stbi__vertically_flip_on_load_global

static void *stbi__load_main(stbi__context *s, int *x, int *y, int *comp, int req_comp, stbi__result_info *ri, int bpc)
{
   memset(ri, 0, sizeof(*ri)); // make sure it's initialized if we add new fields
   ri->bits_per_channel = 8; // default is 8 so most paths don't have to be changed
   ri->channel_order = STBI_ORDER_RGB; // all current input & output are this, but this is here so we can add BGR order
   ri->num_channels = 0;

   if (stbi__png_test(s))  return stbi__png_load(s,x,y,comp,req_comp, ri);
   if (stbi__gif_test(s))  return stbi__gif_load(s,x,y,comp,req_comp, ri);
   STBI_NOTUSED(bpc);

   if (stbi__jpeg_test(s)) return stbi__jpeg_load(s,x,y,comp,req_comp, ri);


   return stbi__errpuc("unknown image type", "Image not of any known type, or corrupt");
}

static stbi_uc *stbi__convert_16_to_8(stbi__uint16 *orig, int w, int h, int channels)
{
   int i;
   int img_len = w * h * channels;
   stbi_uc *reduced;

   reduced = (stbi_uc *) stbi__malloc(img_len);
   if (reduced == NULL) return stbi__errpuc("outofmem", "Out of memory");

   for (i = 0; i < img_len; ++i)
      reduced[i] = (stbi_uc)((orig[i] >> 8) & 0xFF); // top half of each byte is sufficient approx of 16->8 bit scaling

   STBI_FREE(orig);
   return reduced;
}

static stbi__uint16 *stbi__convert_8_to_16(stbi_uc *orig, int w, int h, int channels)
{
   int i;
   int img_len = w * h * channels;
   stbi__uint16 *enlarged;

   enlarged = (stbi__uint16 *) stbi__malloc(img_len*2);
   if (enlarged == NULL) return (stbi__uint16 *) stbi__errpuc("outofmem", "Out of memory");

   for (i = 0; i < img_len; ++i)
      enlarged[i] = (stbi__uint16)((orig[i] << 8) + orig[i]); // replicate to high and low byte, maps 0->0, 255->0xffff

   STBI_FREE(orig);
   return enlarged;
}

static void stbi__vertical_flip(void *image, int w, int h, int bytes_per_pixel)
{
   int row;
   size_t bytes_per_row = (size_t)w * bytes_per_pixel;
   stbi_uc temp[2048];
   stbi_uc *bytes = (stbi_uc *)image;

   for (row = 0; row < (h>>1); row++) {
      stbi_uc *row0 = bytes + row*bytes_per_row;
      stbi_uc *row1 = bytes + (h - row - 1)*bytes_per_row;
      size_t bytes_left = bytes_per_row;
      while (bytes_left) {
         size_t bytes_copy = (bytes_left < sizeof(temp)) ? bytes_left : sizeof(temp);
         memcpy(temp, row0, bytes_copy);
         memcpy(row0, row1, bytes_copy);
         memcpy(row1, temp, bytes_copy);
         row0 += bytes_copy;
         row1 += bytes_copy;
         bytes_left -= bytes_copy;
      }
   }
}

static void stbi__vertical_flip_slices(void *image, int w, int h, int z, int bytes_per_pixel)
{
   int slice;
   int slice_size = w * h * bytes_per_pixel;

   stbi_uc *bytes = (stbi_uc *)image;
   for (slice = 0; slice < z; ++slice) {
      stbi__vertical_flip(bytes, w, h, bytes_per_pixel);
      bytes += slice_size;
   }
}

static unsigned char *stbi__load_and_postprocess_8bit(stbi__context *s, int *x, int *y, int *comp, int req_comp)
{
   stbi__result_info ri;
   void *result = stbi__load_main(s, x, y, comp, req_comp, &ri, 8);

   if (result == NULL)
      return NULL;

   STBI_ASSERT(ri.bits_per_channel == 8 || ri.bits_per_channel == 16);

   if (ri.bits_per_channel != 8) {
      result = stbi__convert_16_to_8((stbi__uint16 *) result, *x, *y, req_comp == 0 ? *comp : req_comp);
      ri.bits_per_channel = 8;
   }


   if (stbi__vertically_flip_on_load) {
      int channels = req_comp ? req_comp : *comp;
      stbi__vertical_flip(result, *x, *y, channels * sizeof(stbi_uc));
   }

   return (unsigned char *) result;
}

static stbi__uint16 *stbi__load_and_postprocess_16bit(stbi__context *s, int *x, int *y, int *comp, int req_comp)
{
   stbi__result_info ri;
   void *result = stbi__load_main(s, x, y, comp, req_comp, &ri, 16);

   if (result == NULL)
      return NULL;

   STBI_ASSERT(ri.bits_per_channel == 8 || ri.bits_per_channel == 16);

   if (ri.bits_per_channel != 16) {
      result = stbi__convert_8_to_16((stbi_uc *) result, *x, *y, req_comp == 0 ? *comp : req_comp);
      ri.bits_per_channel = 16;
   }


   if (stbi__vertically_flip_on_load) {
      int channels = req_comp ? req_comp : *comp;
      stbi__vertical_flip(result, *x, *y, channels * sizeof(stbi__uint16));
   }

   return (stbi__uint16 *) result;
}





static FILE *stbi__fopen(char const *filename, char const *mode)
{
   FILE *f;
   f = fopen(filename, mode);
   return f;
}


STBIDEF stbi_uc *stbi_load(char const *filename, int *x, int *y, int *comp, int req_comp)
{
   FILE *f = stbi__fopen(filename, "rb");
   unsigned char *result;
   if (!f) return stbi__errpuc("can't fopen", "Unable to open file");
   result = stbi_load_from_file(f,x,y,comp,req_comp);
   fclose(f);
   return result;
}

STBIDEF stbi_uc *stbi_load_from_file(FILE *f, int *x, int *y, int *comp, int req_comp)
{
   unsigned char *result;
   stbi__context s;
   stbi__start_file(&s,f);
   result = stbi__load_and_postprocess_8bit(&s,x,y,comp,req_comp);
   if (result) {
      fseek(f, - (int) (s.img_buffer_end - s.img_buffer), SEEK_CUR);
   }
   return result;
}

STBIDEF stbi__uint16 *stbi_load_from_file_16(FILE *f, int *x, int *y, int *comp, int req_comp)
{
   stbi__uint16 *result;
   stbi__context s;
   stbi__start_file(&s,f);
   result = stbi__load_and_postprocess_16bit(&s,x,y,comp,req_comp);
   if (result) {
      fseek(f, - (int) (s.img_buffer_end - s.img_buffer), SEEK_CUR);
   }
   return result;
}

STBIDEF stbi_us *stbi_load_16(char const *filename, int *x, int *y, int *comp, int req_comp)
{
   FILE *f = stbi__fopen(filename, "rb");
   stbi__uint16 *result;
   if (!f) return (stbi_us *) stbi__errpuc("can't fopen", "Unable to open file");
   result = stbi_load_from_file_16(f,x,y,comp,req_comp);
   fclose(f);
   return result;
}



STBIDEF stbi_us *stbi_load_16_from_memory(stbi_uc const *buffer, int len, int *x, int *y, int *channels_in_file, int desired_channels)
{
   stbi__context s;
   stbi__start_mem(&s,buffer,len);
   return stbi__load_and_postprocess_16bit(&s,x,y,channels_in_file,desired_channels);
}

STBIDEF stbi_us *stbi_load_16_from_callbacks(stbi_io_callbacks const *clbk, void *user, int *x, int *y, int *channels_in_file, int desired_channels)
{
   stbi__context s;
   stbi__start_callbacks(&s, (stbi_io_callbacks *)clbk, user);
   return stbi__load_and_postprocess_16bit(&s,x,y,channels_in_file,desired_channels);
}

STBIDEF stbi_uc *stbi_load_from_memory(stbi_uc const *buffer, int len, int *x, int *y, int *comp, int req_comp)
{
   stbi__context s;
   stbi__start_mem(&s,buffer,len);
   return stbi__load_and_postprocess_8bit(&s,x,y,comp,req_comp);
}

STBIDEF stbi_uc *stbi_load_from_callbacks(stbi_io_callbacks const *clbk, void *user, int *x, int *y, int *comp, int req_comp)
{
   stbi__context s;
   stbi__start_callbacks(&s, (stbi_io_callbacks *) clbk, user);
   return stbi__load_and_postprocess_8bit(&s,x,y,comp,req_comp);
}

STBIDEF stbi_uc *stbi_load_gif_from_memory(stbi_uc const *buffer, int len, int **delays, int *x, int *y, int *z, int *comp, int req_comp)
{
   unsigned char *result;
   stbi__context s;
   stbi__start_mem(&s,buffer,len);

   result = (unsigned char*) stbi__load_gif_main(&s, delays, x, y, z, comp, req_comp);
   if (stbi__vertically_flip_on_load) {
      stbi__vertical_flip_slices( result, *x, *y, *z, *comp );
   }

   return result;
}

static float *stbi__loadf_main(stbi__context *s, int *x, int *y, int *comp, int req_comp)
{
   unsigned char *data;
   data = stbi__load_and_postprocess_8bit(s, x, y, comp, req_comp);
   if (data)
      return stbi__ldr_to_hdr(data, *x, *y, req_comp ? req_comp : *comp);
   return stbi__errpf("unknown image type", "Image not of any known type, or corrupt");
}

STBIDEF float *stbi_loadf_from_memory(stbi_uc const *buffer, int len, int *x, int *y, int *comp, int req_comp)
{
   stbi__context s;
   stbi__start_mem(&s,buffer,len);
   return stbi__loadf_main(&s,x,y,comp,req_comp);
}

STBIDEF float *stbi_loadf_from_callbacks(stbi_io_callbacks const *clbk, void *user, int *x, int *y, int *comp, int req_comp)
{
   stbi__context s;
   stbi__start_callbacks(&s, (stbi_io_callbacks *) clbk, user);
   return stbi__loadf_main(&s,x,y,comp,req_comp);
}

STBIDEF float *stbi_loadf(char const *filename, int *x, int *y, int *comp, int req_comp)
{
   float *result;
   FILE *f = stbi__fopen(filename, "rb");
   if (!f) return stbi__errpf("can't fopen", "Unable to open file");
   result = stbi_loadf_from_file(f,x,y,comp,req_comp);
   fclose(f);
   return result;
}

STBIDEF float *stbi_loadf_from_file(FILE *f, int *x, int *y, int *comp, int req_comp)
{
   stbi__context s;
   stbi__start_file(&s,f);
   return stbi__loadf_main(&s,x,y,comp,req_comp);
}



STBIDEF int stbi_is_hdr_from_memory(stbi_uc const *buffer, int len)
{
   STBI_NOTUSED(buffer);
   STBI_NOTUSED(len);
   return 0;
}

STBIDEF int      stbi_is_hdr          (char const *filename)
{
   FILE *f = stbi__fopen(filename, "rb");
   int result=0;
   if (f) {
      result = stbi_is_hdr_from_file(f);
      fclose(f);
   }
   return result;
}

STBIDEF int stbi_is_hdr_from_file(FILE *f)
{
   STBI_NOTUSED(f);
   return 0;
}

STBIDEF int      stbi_is_hdr_from_callbacks(stbi_io_callbacks const *clbk, void *user)
{
   STBI_NOTUSED(clbk);
   STBI_NOTUSED(user);
   return 0;
}

static float stbi__l2h_gamma=2.2f, stbi__l2h_scale=1.0f;

STBIDEF void   stbi_ldr_to_hdr_gamma(float gamma) { stbi__l2h_gamma = gamma; }
STBIDEF void   stbi_ldr_to_hdr_scale(float scale) { stbi__l2h_scale = scale; }

static float stbi__h2l_gamma_i=1.0f/2.2f, stbi__h2l_scale_i=1.0f;

STBIDEF void   stbi_hdr_to_ldr_gamma(float gamma) { stbi__h2l_gamma_i = 1/gamma; }
STBIDEF void   stbi_hdr_to_ldr_scale(float scale) { stbi__h2l_scale_i = 1/scale; }



enum
{
   STBI__SCAN_load=0,
   STBI__SCAN_type,
   STBI__SCAN_header
};

static void stbi__refill_buffer(stbi__context *s)
{
   int n = (s->io.read)(s->io_user_data,(char*)s->buffer_start,s->buflen);
   s->callback_already_read += (int) (s->img_buffer - s->img_buffer_original);
   if (n == 0) {
      s->read_from_callbacks = 0;
      s->img_buffer = s->buffer_start;
      s->img_buffer_end = s->buffer_start+1;
      *s->img_buffer = 0;
   } else {
      s->img_buffer = s->buffer_start;
      s->img_buffer_end = s->buffer_start + n;
   }
}

stbi_inline static stbi_uc stbi__get8(stbi__context *s)
{
   if (s->img_buffer < s->img_buffer_end)
      return *s->img_buffer++;
   if (s->read_from_callbacks) {
      stbi__refill_buffer(s);
      return *s->img_buffer++;
   }
   return 0;
}

stbi_inline static int stbi__at_eof(stbi__context *s)
{
   if (s->io.read) {
      if (!(s->io.eof)(s->io_user_data)) return 0;
      if (s->read_from_callbacks == 0) return 1;
   }

   return s->img_buffer >= s->img_buffer_end;
}

static void stbi__skip(stbi__context *s, int n)
{
   if (n == 0) return;  // already there!
   if (n < 0) {
      s->img_buffer = s->img_buffer_end;
      return;
   }
   if (s->io.read) {
      int blen = (int) (s->img_buffer_end - s->img_buffer);
      if (blen < n) {
         s->img_buffer = s->img_buffer_end;
         (s->io.skip)(s->io_user_data, n - blen);
         return;
      }
   }
   s->img_buffer += n;
}

static int stbi__getn(stbi__context *s, stbi_uc *buffer, int n)
{
   if (s->io.read) {
      int blen = (int) (s->img_buffer_end - s->img_buffer);
      if (blen < n) {
         int res, count;

         memcpy(buffer, s->img_buffer, blen);

         count = (s->io.read)(s->io_user_data, (char*) buffer + blen, n - blen);
         res = (count == (n-blen));
         s->img_buffer = s->img_buffer_end;
         return res;
      }
   }

   if (s->img_buffer+n <= s->img_buffer_end) {
      memcpy(buffer, s->img_buffer, n);
      s->img_buffer += n;
      return 1;
   } else
      return 0;
}

static int stbi__get16be(stbi__context *s)
{
   int z = stbi__get8(s);
   return (z << 8) + stbi__get8(s);
}
#endif


static stbi__uint32 stbi__get32be(stbi__context *s)
{
   stbi__uint32 z = stbi__get16be(s);
   return (z << 16) + stbi__get16be(s);
}
#endif


static int stbi__get16le(stbi__context *s)
{
   int z = stbi__get8(s);
   return z + (stbi__get8(s) << 8);
}
#endif

#define STBI__BYTECAST(x)  ((stbi_uc) ((x) & 255))  // truncate int to byte without warnings


static stbi_uc stbi__compute_y(int r, int g, int b)
{
   return (stbi_uc) (((r*77) + (g*150) +  (29*b)) >> 8);
}
#endif

static unsigned char *stbi__convert_format(unsigned char *data, int img_n, int req_comp, unsigned int x, unsigned int y)
{
   int i,j;
   unsigned char *good;

   if (req_comp == img_n) return data;
   STBI_ASSERT(req_comp >= 1 && req_comp <= 4);

   good = (unsigned char *) stbi__malloc_mad3(req_comp, x, y, 0);
   if (good == NULL) {
      STBI_FREE(data);
      return stbi__errpuc("outofmem", "Out of memory");
   }

   for (j=0; j < (int) y; ++j) {
      unsigned char *src  = data + j * x * img_n   ;
      unsigned char *dest = good + j * x * req_comp;

      #define STBI__COMBO(a,b)  ((a)*8+(b))
      #define STBI__CASE(a,b)   case STBI__COMBO(a,b): for(i=x-1; i >= 0; --i, src += a, dest += b)
      switch (STBI__COMBO(img_n, req_comp)) {
         STBI__CASE(1,2) { dest[0]=src[0]; dest[1]=255;                                     } break;
         STBI__CASE(1,3) { dest[0]=dest[1]=dest[2]=src[0];                                  } break;
         STBI__CASE(1,4) { dest[0]=dest[1]=dest[2]=src[0]; dest[3]=255;                     } break;
         STBI__CASE(2,1) { dest[0]=src[0];                                                  } break;
         STBI__CASE(2,3) { dest[0]=dest[1]=dest[2]=src[0];                                  } break;
         STBI__CASE(2,4) { dest[0]=dest[1]=dest[2]=src[0]; dest[3]=src[1];                  } break;
         STBI__CASE(3,4) { dest[0]=src[0];dest[1]=src[1];dest[2]=src[2];dest[3]=255;        } break;
         STBI__CASE(3,1) { dest[0]=stbi__compute_y(src[0],src[1],src[2]);                   } break;
         STBI__CASE(3,2) { dest[0]=stbi__compute_y(src[0],src[1],src[2]); dest[1] = 255;    } break;
         STBI__CASE(4,1) { dest[0]=stbi__compute_y(src[0],src[1],src[2]);                   } break;
         STBI__CASE(4,2) { dest[0]=stbi__compute_y(src[0],src[1],src[2]); dest[1] = src[3]; } break;
         STBI__CASE(4,3) { dest[0]=src[0];dest[1]=src[1];dest[2]=src[2];                    } break;
         default: STBI_ASSERT(0); STBI_FREE(data); STBI_FREE(good); return stbi__errpuc("unsupported", "Unsupported format conversion");
      }
      #undef STBI__CASE
   }

   STBI_FREE(data);
   return good;
}
#endif


static stbi__uint16 stbi__compute_y_16(int r, int g, int b)
{
   return (stbi__uint16) (((r*77) + (g*150) +  (29*b)) >> 8);
}
#endif

static stbi__uint16 *stbi__convert_format16(stbi__uint16 *data, int img_n, int req_comp, unsigned int x, unsigned int y)
{
   int i,j;
   stbi__uint16 *good;

   if (req_comp == img_n) return data;
   STBI_ASSERT(req_comp >= 1 && req_comp <= 4);

   good = (stbi__uint16 *) stbi__malloc(req_comp * x * y * 2);
   if (good == NULL) {
      STBI_FREE(data);
      return (stbi__uint16 *) stbi__errpuc("outofmem", "Out of memory");
   }

   for (j=0; j < (int) y; ++j) {
      stbi__uint16 *src  = data + j * x * img_n   ;
      stbi__uint16 *dest = good + j * x * req_comp;

      #define STBI__COMBO(a,b)  ((a)*8+(b))
      #define STBI__CASE(a,b)   case STBI__COMBO(a,b): for(i=x-1; i >= 0; --i, src += a, dest += b)
      switch (STBI__COMBO(img_n, req_comp)) {
         STBI__CASE(1,2) { dest[0]=src[0]; dest[1]=0xffff;                                     } break;
         STBI__CASE(1,3) { dest[0]=dest[1]=dest[2]=src[0];                                     } break;
         STBI__CASE(1,4) { dest[0]=dest[1]=dest[2]=src[0]; dest[3]=0xffff;                     } break;
         STBI__CASE(2,1) { dest[0]=src[0];                                                     } break;
         STBI__CASE(2,3) { dest[0]=dest[1]=dest[2]=src[0];                                     } break;
         STBI__CASE(2,4) { dest[0]=dest[1]=dest[2]=src[0]; dest[3]=src[1];                     } break;
         STBI__CASE(3,4) { dest[0]=src[0];dest[1]=src[1];dest[2]=src[2];dest[3]=0xffff;        } break;
         STBI__CASE(3,1) { dest[0]=stbi__compute_y_16(src[0],src[1],src[2]);                   } break;
         STBI__CASE(3,2) { dest[0]=stbi__compute_y_16(src[0],src[1],src[2]); dest[1] = 0xffff; } break;
         STBI__CASE(4,1) { dest[0]=stbi__compute_y_16(src[0],src[1],src[2]);                   } break;
         STBI__CASE(4,2) { dest[0]=stbi__compute_y_16(src[0],src[1],src[2]); dest[1] = src[3]; } break;
         STBI__CASE(4,3) { dest[0]=src[0];dest[1]=src[1];dest[2]=src[2];                       } break;
         default: STBI_ASSERT(0); STBI_FREE(data); STBI_FREE(good); return (stbi__uint16*) stbi__errpuc("unsupported", "Unsupported format conversion");
      }
      #undef STBI__CASE
   }

   STBI_FREE(data);
   return good;
}
#endif

static float   *stbi__ldr_to_hdr(stbi_uc *data, int x, int y, int comp)
{
   int i,k,n;
   float *output;
   if (!data) return NULL;
   output = (float *) stbi__malloc_mad4(x, y, comp, sizeof(float), 0);
   if (output == NULL) { STBI_FREE(data); return stbi__errpf("outofmem", "Out of memory"); }
   if (comp & 1) n = comp; else n = comp-1;
   for (i=0; i < x*y; ++i) {
      for (k=0; k < n; ++k) {
         output[i*comp + k] = (float) (pow(data[i*comp+k]/255.0f, stbi__l2h_gamma) * stbi__l2h_scale);
      }
   }
   if (n < comp) {
      for (i=0; i < x*y; ++i) {
         output[i*comp + n] = data[i*comp + n]/255.0f;
      }
   }
   STBI_FREE(data);
   return output;
}




#define FAST_BITS   9  // larger handles more cases; smaller stomps less cache

typedef struct
{
   stbi_uc  fast[1 << FAST_BITS];
   stbi__uint16 code[256];
   stbi_uc  values[256];
   stbi_uc  size[257];
   unsigned int maxcode[18];
   int    delta[17];   // old 'firstsymbol' - old 'firstcode'
} stbi__huffman;

typedef struct
{
   stbi__context *s;
   stbi__huffman huff_dc[4];
   stbi__huffman huff_ac[4];
   stbi__uint16 dequant[4][64];
   stbi__int16 fast_ac[4][1 << FAST_BITS];

   int img_h_max, img_v_max;
   int img_mcu_x, img_mcu_y;
   int img_mcu_w, img_mcu_h;

   struct
   {
      int id;
      int h,v;
      int tq;
      int hd,ha;
      int dc_pred;

      int x,y,w2,h2;
      stbi_uc *data;
      void *raw_data, *raw_coeff;
      stbi_uc *linebuf;
      short   *coeff;   // progressive only
      int      coeff_w, coeff_h; // number of 8x8 coefficient blocks
   } img_comp[4];

   stbi__uint32   code_buffer; // jpeg entropy-coded buffer
   int            code_bits;   // number of valid bits
   unsigned char  marker;      // marker seen while filling entropy buffer
   int            nomore;      // flag if we saw a marker so must stop

   int            progressive;
   int            spec_start;
   int            spec_end;
   int            succ_high;
   int            succ_low;
   int            eob_run;
   int            jfif;
   int            app14_color_transform; // Adobe APP14 tag
   int            rgb;

   int scan_n, order[4];
   int restart_interval, todo;

   void (*idct_block_kernel)(stbi_uc *out, int out_stride, short data[64]);
   void (*YCbCr_to_RGB_kernel)(stbi_uc *out, const stbi_uc *y, const stbi_uc *pcb, const stbi_uc *pcr, int count, int step);
   stbi_uc *(*resample_row_hv_2_kernel)(stbi_uc *out, stbi_uc *in_near, stbi_uc *in_far, int w, int hs);
} stbi__jpeg;

static int stbi__build_huffman(stbi__huffman *h, int *count)
{
   int i,j,k=0;
   unsigned int code;
   for (i=0; i < 16; ++i) {
      for (j=0; j < count[i]; ++j) {
         h->size[k++] = (stbi_uc) (i+1);
         if(k >= 257) return stbi__err("bad size list","Corrupt JPEG");
      }
   }
   h->size[k] = 0;

   code = 0;
   k = 0;
   for(j=1; j <= 16; ++j) {
      h->delta[j] = k - code;
      if (h->size[k] == j) {
         while (h->size[k] == j)
            h->code[k++] = (stbi__uint16) (code++);
         if (code-1 >= (1u << j)) return stbi__err("bad code lengths","Corrupt JPEG");
      }
      h->maxcode[j] = code << (16-j);
      code <<= 1;
   }
   h->maxcode[j] = 0xffffffff;

   memset(h->fast, 255, 1 << FAST_BITS);
   for (i=0; i < k; ++i) {
      int s = h->size[i];
      if (s <= FAST_BITS) {
         int c = h->code[i] << (FAST_BITS-s);
         int m = 1 << (FAST_BITS-s);
         for (j=0; j < m; ++j) {
            h->fast[c+j] = (stbi_uc) i;
         }
      }
   }
   return 1;
}

static void stbi__build_fast_ac(stbi__int16 *fast_ac, stbi__huffman *h)
{
   int i;
   for (i=0; i < (1 << FAST_BITS); ++i) {
      stbi_uc fast = h->fast[i];
      fast_ac[i] = 0;
      if (fast < 255) {
         int rs = h->values[fast];
         int run = (rs >> 4) & 15;
         int magbits = rs & 15;
         int len = h->size[fast];

         if (magbits && len + magbits <= FAST_BITS) {
            int k = ((i << len) & ((1 << FAST_BITS) - 1)) >> (FAST_BITS - magbits);
            int m = 1 << (magbits - 1);
            if (k < m) k += (~0U << magbits) + 1;
            if (k >= -128 && k <= 127)
               fast_ac[i] = (stbi__int16) ((k * 256) + (run * 16) + (len + magbits));
         }
      }
   }
}

static void stbi__grow_buffer_unsafe(stbi__jpeg *j)
{
   do {
      unsigned int b = j->nomore ? 0 : stbi__get8(j->s);
      if (b == 0xff) {
         int c = stbi__get8(j->s);
         while (c == 0xff) c = stbi__get8(j->s); // consume fill bytes
         if (c != 0) {
            j->marker = (unsigned char) c;
            j->nomore = 1;
            return;
         }
      }
      j->code_buffer |= b << (24 - j->code_bits);
      j->code_bits += 8;
   } while (j->code_bits <= 24);
}

static const stbi__uint32 stbi__bmask[17]={0,1,3,7,15,31,63,127,255,511,1023,2047,4095,8191,16383,32767,65535};

stbi_inline static int stbi__jpeg_huff_decode(stbi__jpeg *j, stbi__huffman *h)
{
   unsigned int temp;
   int c,k;

   if (j->code_bits < 16) stbi__grow_buffer_unsafe(j);

   c = (j->code_buffer >> (32 - FAST_BITS)) & ((1 << FAST_BITS)-1);
   k = h->fast[c];
   if (k < 255) {
      int s = h->size[k];
      if (s > j->code_bits)
         return -1;
      j->code_buffer <<= s;
      j->code_bits -= s;
      return h->values[k];
   }

   temp = j->code_buffer >> 16;
   for (k=FAST_BITS+1 ; ; ++k)
      if (temp < h->maxcode[k])
         break;
   if (k == 17) {
      j->code_bits -= 16;
      return -1;
   }

   if (k > j->code_bits)
      return -1;

   c = ((j->code_buffer >> (32 - k)) & stbi__bmask[k]) + h->delta[k];
   if(c < 0 || c >= 256) // symbol id out of bounds!
       return -1;
   STBI_ASSERT((((j->code_buffer) >> (32 - h->size[c])) & stbi__bmask[h->size[c]]) == h->code[c]);

   j->code_bits -= k;
   j->code_buffer <<= k;
   return h->values[c];
}

static const int stbi__jbias[16] = {0,-1,-3,-7,-15,-31,-63,-127,-255,-511,-1023,-2047,-4095,-8191,-16383,-32767};

stbi_inline static int stbi__extend_receive(stbi__jpeg *j, int n)
{
   unsigned int k;
   int sgn;
   if (j->code_bits < n) stbi__grow_buffer_unsafe(j);
   if (j->code_bits < n) return 0; // ran out of bits from stream, return 0s intead of continuing

   sgn = j->code_buffer >> 31; // sign bit always in MSB; 0 if MSB clear (positive), 1 if MSB set (negative)
   k = stbi_lrot(j->code_buffer, n);
   j->code_buffer = k & ~stbi__bmask[n];
   k &= stbi__bmask[n];
   j->code_bits -= n;
   return k + (stbi__jbias[n] & (sgn - 1));
}

stbi_inline static int stbi__jpeg_get_bits(stbi__jpeg *j, int n)
{
   unsigned int k;
   if (j->code_bits < n) stbi__grow_buffer_unsafe(j);
   if (j->code_bits < n) return 0; // ran out of bits from stream, return 0s intead of continuing
   k = stbi_lrot(j->code_buffer, n);
   j->code_buffer = k & ~stbi__bmask[n];
   k &= stbi__bmask[n];
   j->code_bits -= n;
   return k;
}

stbi_inline static int stbi__jpeg_get_bit(stbi__jpeg *j)
{
   unsigned int k;
   if (j->code_bits < 1) stbi__grow_buffer_unsafe(j);
   if (j->code_bits < 1) return 0; // ran out of bits from stream, return 0s intead of continuing
   k = j->code_buffer;
   j->code_buffer <<= 1;
   --j->code_bits;
   return k & 0x80000000;
}

static const stbi_uc stbi__jpeg_dezigzag[64+15] =
{
    0,  1,  8, 16,  9,  2,  3, 10,
   17, 24, 32, 25, 18, 11,  4,  5,
   12, 19, 26, 33, 40, 48, 41, 34,
   27, 20, 13,  6,  7, 14, 21, 28,
   35, 42, 49, 56, 57, 50, 43, 36,
   29, 22, 15, 23, 30, 37, 44, 51,
   58, 59, 52, 45, 38, 31, 39, 46,
   53, 60, 61, 54, 47, 55, 62, 63,
   63, 63, 63, 63, 63, 63, 63, 63,
   63, 63, 63, 63, 63, 63, 63
};

static int stbi__jpeg_decode_block(stbi__jpeg *j, short data[64], stbi__huffman *hdc, stbi__huffman *hac, stbi__int16 *fac, int b, stbi__uint16 *dequant)
{
   int diff,dc,k;
   int t;

   if (j->code_bits < 16) stbi__grow_buffer_unsafe(j);
   t = stbi__jpeg_huff_decode(j, hdc);
   if (t < 0 || t > 15) return stbi__err("bad huffman code","Corrupt JPEG");

   memset(data,0,64*sizeof(data[0]));

   diff = t ? stbi__extend_receive(j, t) : 0;
   if (!stbi__addints_valid(j->img_comp[b].dc_pred, diff)) return stbi__err("bad delta","Corrupt JPEG");
   dc = j->img_comp[b].dc_pred + diff;
   j->img_comp[b].dc_pred = dc;
   if (!stbi__mul2shorts_valid(dc, dequant[0])) return stbi__err("can't merge dc and ac", "Corrupt JPEG");
   data[0] = (short) (dc * dequant[0]);

   k = 1;
   do {
      unsigned int zig;
      int c,r,s;
      if (j->code_bits < 16) stbi__grow_buffer_unsafe(j);
      c = (j->code_buffer >> (32 - FAST_BITS)) & ((1 << FAST_BITS)-1);
      r = fac[c];
      if (r) { // fast-AC path
         k += (r >> 4) & 15; // run
         s = r & 15; // combined length
         if (s > j->code_bits) return stbi__err("bad huffman code", "Combined length longer than code bits available");
         j->code_buffer <<= s;
         j->code_bits -= s;
         zig = stbi__jpeg_dezigzag[k++];
         data[zig] = (short) ((r >> 8) * dequant[zig]);
      } else {
         int rs = stbi__jpeg_huff_decode(j, hac);
         if (rs < 0) return stbi__err("bad huffman code","Corrupt JPEG");
         s = rs & 15;
         r = rs >> 4;
         if (s == 0) {
            if (rs != 0xf0) break; // end block
            k += 16;
         } else {
            k += r;
            zig = stbi__jpeg_dezigzag[k++];
            data[zig] = (short) (stbi__extend_receive(j,s) * dequant[zig]);
         }
      }
   } while (k < 64);
   return 1;
}

static int stbi__jpeg_decode_block_prog_dc(stbi__jpeg *j, short data[64], stbi__huffman *hdc, int b)
{
   int diff,dc;
   int t;
   if (j->spec_end != 0) return stbi__err("can't merge dc and ac", "Corrupt JPEG");

   if (j->code_bits < 16) stbi__grow_buffer_unsafe(j);

   if (j->succ_high == 0) {
      memset(data,0,64*sizeof(data[0])); // 0 all the ac values now
      t = stbi__jpeg_huff_decode(j, hdc);
      if (t < 0 || t > 15) return stbi__err("can't merge dc and ac", "Corrupt JPEG");
      diff = t ? stbi__extend_receive(j, t) : 0;

      if (!stbi__addints_valid(j->img_comp[b].dc_pred, diff)) return stbi__err("bad delta", "Corrupt JPEG");
      dc = j->img_comp[b].dc_pred + diff;
      j->img_comp[b].dc_pred = dc;
      if (!stbi__mul2shorts_valid(dc, 1 << j->succ_low)) return stbi__err("can't merge dc and ac", "Corrupt JPEG");
      data[0] = (short) (dc * (1 << j->succ_low));
   } else {
      if (stbi__jpeg_get_bit(j))
         data[0] += (short) (1 << j->succ_low);
   }
   return 1;
}

static int stbi__jpeg_decode_block_prog_ac(stbi__jpeg *j, short data[64], stbi__huffman *hac, stbi__int16 *fac)
{
   int k;
   if (j->spec_start == 0) return stbi__err("can't merge dc and ac", "Corrupt JPEG");

   if (j->succ_high == 0) {
      int shift = j->succ_low;

      if (j->eob_run) {
         --j->eob_run;
         return 1;
      }

      k = j->spec_start;
      do {
         unsigned int zig;
         int c,r,s;
         if (j->code_bits < 16) stbi__grow_buffer_unsafe(j);
         c = (j->code_buffer >> (32 - FAST_BITS)) & ((1 << FAST_BITS)-1);
         r = fac[c];
         if (r) { // fast-AC path
            k += (r >> 4) & 15; // run
            s = r & 15; // combined length
            if (s > j->code_bits) return stbi__err("bad huffman code", "Combined length longer than code bits available");
            j->code_buffer <<= s;
            j->code_bits -= s;
            zig = stbi__jpeg_dezigzag[k++];
            data[zig] = (short) ((r >> 8) * (1 << shift));
         } else {
            int rs = stbi__jpeg_huff_decode(j, hac);
            if (rs < 0) return stbi__err("bad huffman code","Corrupt JPEG");
            s = rs & 15;
            r = rs >> 4;
            if (s == 0) {
               if (r < 15) {
                  j->eob_run = (1 << r);
                  if (r)
                     j->eob_run += stbi__jpeg_get_bits(j, r);
                  --j->eob_run;
                  break;
               }
               k += 16;
            } else {
               k += r;
               zig = stbi__jpeg_dezigzag[k++];
               data[zig] = (short) (stbi__extend_receive(j,s) * (1 << shift));
            }
         }
      } while (k <= j->spec_end);
   } else {

      short bit = (short) (1 << j->succ_low);

      if (j->eob_run) {
         --j->eob_run;
         for (k = j->spec_start; k <= j->spec_end; ++k) {
            short *p = &data[stbi__jpeg_dezigzag[k]];
            if (*p != 0)
               if (stbi__jpeg_get_bit(j))
                  if ((*p & bit)==0) {
                     if (*p > 0)
                        *p += bit;
                     else
                        *p -= bit;
                  }
         }
      } else {
         k = j->spec_start;
         do {
            int r,s;
            int rs = stbi__jpeg_huff_decode(j, hac); // @OPTIMIZE see if we can use the fast path here, advance-by-r is so slow, eh
            if (rs < 0) return stbi__err("bad huffman code","Corrupt JPEG");
            s = rs & 15;
            r = rs >> 4;
            if (s == 0) {
               if (r < 15) {
                  j->eob_run = (1 << r) - 1;
                  if (r)
                     j->eob_run += stbi__jpeg_get_bits(j, r);
                  r = 64; // force end of block
               } else {
               }
            } else {
               if (s != 1) return stbi__err("bad huffman code", "Corrupt JPEG");
               if (stbi__jpeg_get_bit(j))
                  s = bit;
               else
                  s = -bit;
            }

            while (k <= j->spec_end) {
               short *p = &data[stbi__jpeg_dezigzag[k++]];
               if (*p != 0) {
                  if (stbi__jpeg_get_bit(j))
                     if ((*p & bit)==0) {
                        if (*p > 0)
                           *p += bit;
                        else
                           *p -= bit;
                     }
               } else {
                  if (r == 0) {
                     *p = (short) s;
                     break;
                  }
                  --r;
               }
            }
         } while (k <= j->spec_end);
      }
   }
   return 1;
}

stbi_inline static stbi_uc stbi__clamp(int x)
{
   if ((unsigned int) x > 255) {
      if (x < 0) return 0;
      if (x > 255) return 255;
   }
   return (stbi_uc) x;
}

#define stbi__f2f(x)  ((int) (((x) * 4096 + 0.5)))
#define stbi__fsh(x)  ((x) * 4096)

#define STBI__IDCT_1D(s0,s1,s2,s3,s4,s5,s6,s7) \
   int t0,t1,t2,t3,p1,p2,p3,p4,p5,x0,x1,x2,x3; \
   p2 = s2;                                    \
   p3 = s6;                                    \
   p1 = (p2+p3) * stbi__f2f(0.5411961f);       \
   t2 = p1 + p3*stbi__f2f(-1.847759065f);      \
   t3 = p1 + p2*stbi__f2f( 0.765366865f);      \
   p2 = s0;                                    \
   p3 = s4;                                    \
   t0 = stbi__fsh(p2+p3);                      \
   t1 = stbi__fsh(p2-p3);                      \
   x0 = t0+t3;                                 \
   x3 = t0-t3;                                 \
   x1 = t1+t2;                                 \
   x2 = t1-t2;                                 \
   t0 = s7;                                    \
   t1 = s5;                                    \
   t2 = s3;                                    \
   t3 = s1;                                    \
   p3 = t0+t2;                                 \
   p4 = t1+t3;                                 \
   p1 = t0+t3;                                 \
   p2 = t1+t2;                                 \
   p5 = (p3+p4)*stbi__f2f( 1.175875602f);      \
   t0 = t0*stbi__f2f( 0.298631336f);           \
   t1 = t1*stbi__f2f( 2.053119869f);           \
   t2 = t2*stbi__f2f( 3.072711026f);           \
   t3 = t3*stbi__f2f( 1.501321110f);           \
   p1 = p5 + p1*stbi__f2f(-0.899976223f);      \
   p2 = p5 + p2*stbi__f2f(-2.562915447f);      \
   p3 = p3*stbi__f2f(-1.961570560f);           \
   p4 = p4*stbi__f2f(-0.390180644f);           \
   t3 += p1+p4;                                \
   t2 += p2+p3;                                \
   t1 += p2+p4;                                \
   t0 += p1+p3;

static void stbi__idct_block(stbi_uc *out, int out_stride, short data[64])
{
   int i,val[64],*v=val;
   stbi_uc *o;
   short *d = data;

   for (i=0; i < 8; ++i,++d, ++v) {
      if (d[ 8]==0 && d[16]==0 && d[24]==0 && d[32]==0
           && d[40]==0 && d[48]==0 && d[56]==0) {
         int dcterm = d[0]*4;
         v[0] = v[8] = v[16] = v[24] = v[32] = v[40] = v[48] = v[56] = dcterm;
      } else {
         STBI__IDCT_1D(d[ 0],d[ 8],d[16],d[24],d[32],d[40],d[48],d[56])
         x0 += 512; x1 += 512; x2 += 512; x3 += 512;
         v[ 0] = (x0+t3) >> 10;
         v[56] = (x0-t3) >> 10;
         v[ 8] = (x1+t2) >> 10;
         v[48] = (x1-t2) >> 10;
         v[16] = (x2+t1) >> 10;
         v[40] = (x2-t1) >> 10;
         v[24] = (x3+t0) >> 10;
         v[32] = (x3-t0) >> 10;
      }
   }

   for (i=0, v=val, o=out; i < 8; ++i,v+=8,o+=out_stride) {
      STBI__IDCT_1D(v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7])
      x0 += 65536 + (128<<17);
      x1 += 65536 + (128<<17);
      x2 += 65536 + (128<<17);
      x3 += 65536 + (128<<17);
      o[0] = stbi__clamp((x0+t3) >> 17);
      o[7] = stbi__clamp((x0-t3) >> 17);
      o[1] = stbi__clamp((x1+t2) >> 17);
      o[6] = stbi__clamp((x1-t2) >> 17);
      o[2] = stbi__clamp((x2+t1) >> 17);
      o[5] = stbi__clamp((x2-t1) >> 17);
      o[3] = stbi__clamp((x3+t0) >> 17);
      o[4] = stbi__clamp((x3-t0) >> 17);
   }
}



#define STBI__MARKER_none  0xff
static stbi_uc stbi__get_marker(stbi__jpeg *j)
{
   stbi_uc x;
   if (j->marker != STBI__MARKER_none) { x = j->marker; j->marker = STBI__MARKER_none; return x; }
   x = stbi__get8(j->s);
   if (x != 0xff) return STBI__MARKER_none;
   while (x == 0xff)
      x = stbi__get8(j->s); // consume repeated 0xff fill bytes
   return x;
}

#define STBI__RESTART(x)     ((x) >= 0xd0 && (x) <= 0xd7)

static void stbi__jpeg_reset(stbi__jpeg *j)
{
   j->code_bits = 0;
   j->code_buffer = 0;
   j->nomore = 0;
   j->img_comp[0].dc_pred = j->img_comp[1].dc_pred = j->img_comp[2].dc_pred = j->img_comp[3].dc_pred = 0;
   j->marker = STBI__MARKER_none;
   j->todo = j->restart_interval ? j->restart_interval : 0x7fffffff;
   j->eob_run = 0;
}

static int stbi__parse_entropy_coded_data(stbi__jpeg *z)
{
   stbi__jpeg_reset(z);
   if (!z->progressive) {
      if (z->scan_n == 1) {
         int i,j;
         STBI_SIMD_ALIGN(short, data[64]);
         int n = z->order[0];
         int w = (z->img_comp[n].x+7) >> 3;
         int h = (z->img_comp[n].y+7) >> 3;
         for (j=0; j < h; ++j) {
            for (i=0; i < w; ++i) {
               int ha = z->img_comp[n].ha;
               if (!stbi__jpeg_decode_block(z, data, z->huff_dc+z->img_comp[n].hd, z->huff_ac+ha, z->fast_ac[ha], n, z->dequant[z->img_comp[n].tq])) return 0;
               z->idct_block_kernel(z->img_comp[n].data+z->img_comp[n].w2*j*8+i*8, z->img_comp[n].w2, data);
               if (--z->todo <= 0) {
                  if (z->code_bits < 24) stbi__grow_buffer_unsafe(z);
                  if (!STBI__RESTART(z->marker)) return 1;
                  stbi__jpeg_reset(z);
               }
            }
         }
         return 1;
      } else { // interleaved
         int i,j,k,x,y;
         STBI_SIMD_ALIGN(short, data[64]);
         for (j=0; j < z->img_mcu_y; ++j) {
            for (i=0; i < z->img_mcu_x; ++i) {
               for (k=0; k < z->scan_n; ++k) {
                  int n = z->order[k];
                  for (y=0; y < z->img_comp[n].v; ++y) {
                     for (x=0; x < z->img_comp[n].h; ++x) {
                        int x2 = (i*z->img_comp[n].h + x)*8;
                        int y2 = (j*z->img_comp[n].v + y)*8;
                        int ha = z->img_comp[n].ha;
                        if (!stbi__jpeg_decode_block(z, data, z->huff_dc+z->img_comp[n].hd, z->huff_ac+ha, z->fast_ac[ha], n, z->dequant[z->img_comp[n].tq])) return 0;
                        z->idct_block_kernel(z->img_comp[n].data+z->img_comp[n].w2*y2+x2, z->img_comp[n].w2, data);
                     }
                  }
               }
               if (--z->todo <= 0) {
                  if (z->code_bits < 24) stbi__grow_buffer_unsafe(z);
                  if (!STBI__RESTART(z->marker)) return 1;
                  stbi__jpeg_reset(z);
               }
            }
         }
         return 1;
      }
   } else {
      if (z->scan_n == 1) {
         int i,j;
         int n = z->order[0];
         int w = (z->img_comp[n].x+7) >> 3;
         int h = (z->img_comp[n].y+7) >> 3;
         for (j=0; j < h; ++j) {
            for (i=0; i < w; ++i) {
               short *data = z->img_comp[n].coeff + 64 * (i + j * z->img_comp[n].coeff_w);
               if (z->spec_start == 0) {
                  if (!stbi__jpeg_decode_block_prog_dc(z, data, &z->huff_dc[z->img_comp[n].hd], n))
                     return 0;
               } else {
                  int ha = z->img_comp[n].ha;
                  if (!stbi__jpeg_decode_block_prog_ac(z, data, &z->huff_ac[ha], z->fast_ac[ha]))
                     return 0;
               }
               if (--z->todo <= 0) {
                  if (z->code_bits < 24) stbi__grow_buffer_unsafe(z);
                  if (!STBI__RESTART(z->marker)) return 1;
                  stbi__jpeg_reset(z);
               }
            }
         }
         return 1;
      } else { // interleaved
         int i,j,k,x,y;
         for (j=0; j < z->img_mcu_y; ++j) {
            for (i=0; i < z->img_mcu_x; ++i) {
               for (k=0; k < z->scan_n; ++k) {
                  int n = z->order[k];
                  for (y=0; y < z->img_comp[n].v; ++y) {
                     for (x=0; x < z->img_comp[n].h; ++x) {
                        int x2 = (i*z->img_comp[n].h + x);
                        int y2 = (j*z->img_comp[n].v + y);
                        short *data = z->img_comp[n].coeff + 64 * (x2 + y2 * z->img_comp[n].coeff_w);
                        if (!stbi__jpeg_decode_block_prog_dc(z, data, &z->huff_dc[z->img_comp[n].hd], n))
                           return 0;
                     }
                  }
               }
               if (--z->todo <= 0) {
                  if (z->code_bits < 24) stbi__grow_buffer_unsafe(z);
                  if (!STBI__RESTART(z->marker)) return 1;
                  stbi__jpeg_reset(z);
               }
            }
         }
         return 1;
      }
   }
}

static void stbi__jpeg_dequantize(short *data, stbi__uint16 *dequant)
{
   int i;
   for (i=0; i < 64; ++i)
      data[i] *= dequant[i];
}

static void stbi__jpeg_finish(stbi__jpeg *z)
{
   if (z->progressive) {
      int i,j,n;
      for (n=0; n < z->s->img_n; ++n) {
         int w = (z->img_comp[n].x+7) >> 3;
         int h = (z->img_comp[n].y+7) >> 3;
         for (j=0; j < h; ++j) {
            for (i=0; i < w; ++i) {
               short *data = z->img_comp[n].coeff + 64 * (i + j * z->img_comp[n].coeff_w);
               stbi__jpeg_dequantize(data, z->dequant[z->img_comp[n].tq]);
               z->idct_block_kernel(z->img_comp[n].data+z->img_comp[n].w2*j*8+i*8, z->img_comp[n].w2, data);
            }
         }
      }
   }
}

static int stbi__process_marker(stbi__jpeg *z, int m)
{
   int L;
   switch (m) {
      case STBI__MARKER_none: // no marker found
         return stbi__err("expected marker","Corrupt JPEG");

      case 0xDD: // DRI - specify restart interval
         if (stbi__get16be(z->s) != 4) return stbi__err("bad DRI len","Corrupt JPEG");
         z->restart_interval = stbi__get16be(z->s);
         return 1;

      case 0xDB: // DQT - define quantization table
         L = stbi__get16be(z->s)-2;
         while (L > 0) {
            int q = stbi__get8(z->s);
            int p = q >> 4, sixteen = (p != 0);
            int t = q & 15,i;
            if (p != 0 && p != 1) return stbi__err("bad DQT type","Corrupt JPEG");
            if (t > 3) return stbi__err("bad DQT table","Corrupt JPEG");

            for (i=0; i < 64; ++i)
               z->dequant[t][stbi__jpeg_dezigzag[i]] = (stbi__uint16)(sixteen ? stbi__get16be(z->s) : stbi__get8(z->s));
            L -= (sixteen ? 129 : 65);
         }
         return L==0;

      case 0xC4: // DHT - define huffman table
         L = stbi__get16be(z->s)-2;
         while (L > 0) {
            stbi_uc *v;
            int sizes[16],i,n=0;
            int q = stbi__get8(z->s);
            int tc = q >> 4;
            int th = q & 15;
            if (tc > 1 || th > 3) return stbi__err("bad DHT header","Corrupt JPEG");
            for (i=0; i < 16; ++i) {
               sizes[i] = stbi__get8(z->s);
               n += sizes[i];
            }
            if(n > 256) return stbi__err("bad DHT header","Corrupt JPEG"); // Loop over i < n would write past end of values!
            L -= 17;
            if (tc == 0) {
               if (!stbi__build_huffman(z->huff_dc+th, sizes)) return 0;
               v = z->huff_dc[th].values;
            } else {
               if (!stbi__build_huffman(z->huff_ac+th, sizes)) return 0;
               v = z->huff_ac[th].values;
            }
            for (i=0; i < n; ++i)
               v[i] = stbi__get8(z->s);
            if (tc != 0)
               stbi__build_fast_ac(z->fast_ac[th], z->huff_ac + th);
            L -= n;
         }
         return L==0;
   }

   if ((m >= 0xE0 && m <= 0xEF) || m == 0xFE) {
      L = stbi__get16be(z->s);
      if (L < 2) {
         if (m == 0xFE)
            return stbi__err("bad COM len","Corrupt JPEG");
         else
            return stbi__err("bad APP len","Corrupt JPEG");
      }
      L -= 2;

      if (m == 0xE0 && L >= 5) { // JFIF APP0 segment
         static const unsigned char tag[5] = {'J','F','I','F','\0'};
         int ok = 1;
         int i;
         for (i=0; i < 5; ++i)
            if (stbi__get8(z->s) != tag[i])
               ok = 0;
         L -= 5;
         if (ok)
            z->jfif = 1;
      } else if (m == 0xEE && L >= 12) { // Adobe APP14 segment
         static const unsigned char tag[6] = {'A','d','o','b','e','\0'};
         int ok = 1;
         int i;
         for (i=0; i < 6; ++i)
            if (stbi__get8(z->s) != tag[i])
               ok = 0;
         L -= 6;
         if (ok) {
            stbi__get8(z->s); // version
            stbi__get16be(z->s); // flags0
            stbi__get16be(z->s); // flags1
            z->app14_color_transform = stbi__get8(z->s); // color transform
            L -= 6;
         }
      }

      stbi__skip(z->s, L);
      return 1;
   }

   return stbi__err("unknown marker","Corrupt JPEG");
}

static int stbi__process_scan_header(stbi__jpeg *z)
{
   int i;
   int Ls = stbi__get16be(z->s);
   z->scan_n = stbi__get8(z->s);
   if (z->scan_n < 1 || z->scan_n > 4 || z->scan_n > (int) z->s->img_n) return stbi__err("bad SOS component count","Corrupt JPEG");
   if (Ls != 6+2*z->scan_n) return stbi__err("bad SOS len","Corrupt JPEG");
   for (i=0; i < z->scan_n; ++i) {
      int id = stbi__get8(z->s), which;
      int q = stbi__get8(z->s);
      for (which = 0; which < z->s->img_n; ++which)
         if (z->img_comp[which].id == id)
            break;
      if (which == z->s->img_n) return 0; // no match
      z->img_comp[which].hd = q >> 4;   if (z->img_comp[which].hd > 3) return stbi__err("bad DC huff","Corrupt JPEG");
      z->img_comp[which].ha = q & 15;   if (z->img_comp[which].ha > 3) return stbi__err("bad AC huff","Corrupt JPEG");
      z->order[i] = which;
   }

   {
      int aa;
      z->spec_start = stbi__get8(z->s);
      z->spec_end   = stbi__get8(z->s); // should be 63, but might be 0
      aa = stbi__get8(z->s);
      z->succ_high = (aa >> 4);
      z->succ_low  = (aa & 15);
      if (z->progressive) {
         if (z->spec_start > 63 || z->spec_end > 63  || z->spec_start > z->spec_end || z->succ_high > 13 || z->succ_low > 13)
            return stbi__err("bad SOS", "Corrupt JPEG");
      } else {
         if (z->spec_start != 0) return stbi__err("bad SOS","Corrupt JPEG");
         if (z->succ_high != 0 || z->succ_low != 0) return stbi__err("bad SOS","Corrupt JPEG");
         z->spec_end = 63;
      }
   }

   return 1;
}

static int stbi__free_jpeg_components(stbi__jpeg *z, int ncomp, int why)
{
   int i;
   for (i=0; i < ncomp; ++i) {
      if (z->img_comp[i].raw_data) {
         STBI_FREE(z->img_comp[i].raw_data);
         z->img_comp[i].raw_data = NULL;
         z->img_comp[i].data = NULL;
      }
      if (z->img_comp[i].raw_coeff) {
         STBI_FREE(z->img_comp[i].raw_coeff);
         z->img_comp[i].raw_coeff = 0;
         z->img_comp[i].coeff = 0;
      }
      if (z->img_comp[i].linebuf) {
         STBI_FREE(z->img_comp[i].linebuf);
         z->img_comp[i].linebuf = NULL;
      }
   }
   return why;
}

static int stbi__process_frame_header(stbi__jpeg *z, int scan)
{
   stbi__context *s = z->s;
   int Lf,p,i,q, h_max=1,v_max=1,c;
   Lf = stbi__get16be(s);         if (Lf < 11) return stbi__err("bad SOF len","Corrupt JPEG"); // JPEG
   p  = stbi__get8(s);            if (p != 8) return stbi__err("only 8-bit","JPEG format not supported: 8-bit only"); // JPEG baseline
   s->img_y = stbi__get16be(s);   if (s->img_y == 0) return stbi__err("no header height", "JPEG format not supported: delayed height"); // Legal, but we don't handle it--but neither does IJG
   s->img_x = stbi__get16be(s);   if (s->img_x == 0) return stbi__err("0 width","Corrupt JPEG"); // JPEG requires
   if (s->img_y > STBI_MAX_DIMENSIONS) return stbi__err("too large","Very large image (corrupt?)");
   if (s->img_x > STBI_MAX_DIMENSIONS) return stbi__err("too large","Very large image (corrupt?)");
   c = stbi__get8(s);
   if (c != 3 && c != 1 && c != 4) return stbi__err("bad component count","Corrupt JPEG");
   s->img_n = c;
   for (i=0; i < c; ++i) {
      z->img_comp[i].data = NULL;
      z->img_comp[i].linebuf = NULL;
   }

   if (Lf != 8+3*s->img_n) return stbi__err("bad SOF len","Corrupt JPEG");

   z->rgb = 0;
   for (i=0; i < s->img_n; ++i) {
      static const unsigned char rgb[3] = { 'R', 'G', 'B' };
      z->img_comp[i].id = stbi__get8(s);
      if (s->img_n == 3 && z->img_comp[i].id == rgb[i])
         ++z->rgb;
      q = stbi__get8(s);
      z->img_comp[i].h = (q >> 4);  if (!z->img_comp[i].h || z->img_comp[i].h > 4) return stbi__err("bad H","Corrupt JPEG");
      z->img_comp[i].v = q & 15;    if (!z->img_comp[i].v || z->img_comp[i].v > 4) return stbi__err("bad V","Corrupt JPEG");
      z->img_comp[i].tq = stbi__get8(s);  if (z->img_comp[i].tq > 3) return stbi__err("bad TQ","Corrupt JPEG");
   }

   if (scan != STBI__SCAN_load) return 1;

   if (!stbi__mad3sizes_valid(s->img_x, s->img_y, s->img_n, 0)) return stbi__err("too large", "Image too large to decode");

   for (i=0; i < s->img_n; ++i) {
      if (z->img_comp[i].h > h_max) h_max = z->img_comp[i].h;
      if (z->img_comp[i].v > v_max) v_max = z->img_comp[i].v;
   }

   for (i=0; i < s->img_n; ++i) {
      if (h_max % z->img_comp[i].h != 0) return stbi__err("bad H","Corrupt JPEG");
      if (v_max % z->img_comp[i].v != 0) return stbi__err("bad V","Corrupt JPEG");
   }

   z->img_h_max = h_max;
   z->img_v_max = v_max;
   z->img_mcu_w = h_max * 8;
   z->img_mcu_h = v_max * 8;
   z->img_mcu_x = (s->img_x + z->img_mcu_w-1) / z->img_mcu_w;
   z->img_mcu_y = (s->img_y + z->img_mcu_h-1) / z->img_mcu_h;

   for (i=0; i < s->img_n; ++i) {
      z->img_comp[i].x = (s->img_x * z->img_comp[i].h + h_max-1) / h_max;
      z->img_comp[i].y = (s->img_y * z->img_comp[i].v + v_max-1) / v_max;
      z->img_comp[i].w2 = z->img_mcu_x * z->img_comp[i].h * 8;
      z->img_comp[i].h2 = z->img_mcu_y * z->img_comp[i].v * 8;
      z->img_comp[i].coeff = 0;
      z->img_comp[i].raw_coeff = 0;
      z->img_comp[i].linebuf = NULL;
      z->img_comp[i].raw_data = stbi__malloc_mad2(z->img_comp[i].w2, z->img_comp[i].h2, 15);
      if (z->img_comp[i].raw_data == NULL)
         return stbi__free_jpeg_components(z, i+1, stbi__err("outofmem", "Out of memory"));
      z->img_comp[i].data = (stbi_uc*) (((size_t) z->img_comp[i].raw_data + 15) & ~15);
      if (z->progressive) {
         z->img_comp[i].coeff_w = z->img_comp[i].w2 / 8;
         z->img_comp[i].coeff_h = z->img_comp[i].h2 / 8;
         z->img_comp[i].raw_coeff = stbi__malloc_mad3(z->img_comp[i].w2, z->img_comp[i].h2, sizeof(short), 15);
         if (z->img_comp[i].raw_coeff == NULL)
            return stbi__free_jpeg_components(z, i+1, stbi__err("outofmem", "Out of memory"));
         z->img_comp[i].coeff = (short*) (((size_t) z->img_comp[i].raw_coeff + 15) & ~15);
      }
   }

   return 1;
}

#define stbi__DNL(x)         ((x) == 0xdc)
#define stbi__SOI(x)         ((x) == 0xd8)
#define stbi__EOI(x)         ((x) == 0xd9)
#define stbi__SOF(x)         ((x) == 0xc0 || (x) == 0xc1 || (x) == 0xc2)
#define stbi__SOS(x)         ((x) == 0xda)

#define stbi__SOF_progressive(x)   ((x) == 0xc2)

static int stbi__decode_jpeg_header(stbi__jpeg *z, int scan)
{
   int m;
   z->jfif = 0;
   z->app14_color_transform = -1; // valid values are 0,1,2
   z->marker = STBI__MARKER_none; // initialize cached marker to empty
   m = stbi__get_marker(z);
   if (!stbi__SOI(m)) return stbi__err("no SOI","Corrupt JPEG");
   if (scan == STBI__SCAN_type) return 1;
   m = stbi__get_marker(z);
   while (!stbi__SOF(m)) {
      if (!stbi__process_marker(z,m)) return 0;
      m = stbi__get_marker(z);
      while (m == STBI__MARKER_none) {
         if (stbi__at_eof(z->s)) return stbi__err("no SOF", "Corrupt JPEG");
         m = stbi__get_marker(z);
      }
   }
   z->progressive = stbi__SOF_progressive(m);
   if (!stbi__process_frame_header(z, scan)) return 0;
   return 1;
}

static stbi_uc stbi__skip_jpeg_junk_at_end(stbi__jpeg *j)
{
   while (!stbi__at_eof(j->s)) {
      stbi_uc x = stbi__get8(j->s);
      while (x == 0xff) { // might be a marker
         if (stbi__at_eof(j->s)) return STBI__MARKER_none;
         x = stbi__get8(j->s);
         if (x != 0x00 && x != 0xff) {
            return x;
         }
      }
   }
   return STBI__MARKER_none;
}

static int stbi__decode_jpeg_image(stbi__jpeg *j)
{
   int m;
   for (m = 0; m < 4; m++) {
      j->img_comp[m].raw_data = NULL;
      j->img_comp[m].raw_coeff = NULL;
   }
   j->restart_interval = 0;
   if (!stbi__decode_jpeg_header(j, STBI__SCAN_load)) return 0;
   m = stbi__get_marker(j);
   while (!stbi__EOI(m)) {
      if (stbi__SOS(m)) {
         if (!stbi__process_scan_header(j)) return 0;
         if (!stbi__parse_entropy_coded_data(j)) return 0;
         if (j->marker == STBI__MARKER_none ) {
         j->marker = stbi__skip_jpeg_junk_at_end(j);
         }
         m = stbi__get_marker(j);
         if (STBI__RESTART(m))
            m = stbi__get_marker(j);
      } else if (stbi__DNL(m)) {
         int Ld = stbi__get16be(j->s);
         stbi__uint32 NL = stbi__get16be(j->s);
         if (Ld != 4) return stbi__err("bad DNL len", "Corrupt JPEG");
         if (NL != j->s->img_y) return stbi__err("bad DNL height", "Corrupt JPEG");
         m = stbi__get_marker(j);
      } else {
         if (!stbi__process_marker(j, m)) return 1;
         m = stbi__get_marker(j);
      }
   }
   if (j->progressive)
      stbi__jpeg_finish(j);
   return 1;
}


typedef stbi_uc *(*resample_row_func)(stbi_uc *out, stbi_uc *in0, stbi_uc *in1,
                                    int w, int hs);

#define stbi__div4(x) ((stbi_uc) ((x) >> 2))

static stbi_uc *resample_row_1(stbi_uc *out, stbi_uc *in_near, stbi_uc *in_far, int w, int hs)
{
   STBI_NOTUSED(out);
   STBI_NOTUSED(in_far);
   STBI_NOTUSED(w);
   STBI_NOTUSED(hs);
   return in_near;
}

static stbi_uc* stbi__resample_row_v_2(stbi_uc *out, stbi_uc *in_near, stbi_uc *in_far, int w, int hs)
{
   int i;
   STBI_NOTUSED(hs);
   for (i=0; i < w; ++i)
      out[i] = stbi__div4(3*in_near[i] + in_far[i] + 2);
   return out;
}

static stbi_uc*  stbi__resample_row_h_2(stbi_uc *out, stbi_uc *in_near, stbi_uc *in_far, int w, int hs)
{
   int i;
   stbi_uc *input = in_near;

   if (w == 1) {
      out[0] = out[1] = input[0];
      return out;
   }

   out[0] = input[0];
   out[1] = stbi__div4(input[0]*3 + input[1] + 2);
   for (i=1; i < w-1; ++i) {
      int n = 3*input[i]+2;
      out[i*2+0] = stbi__div4(n+input[i-1]);
      out[i*2+1] = stbi__div4(n+input[i+1]);
   }
   out[i*2+0] = stbi__div4(input[w-2]*3 + input[w-1] + 2);
   out[i*2+1] = input[w-1];

   STBI_NOTUSED(in_far);
   STBI_NOTUSED(hs);

   return out;
}

#define stbi__div16(x) ((stbi_uc) ((x) >> 4))

static stbi_uc *stbi__resample_row_hv_2(stbi_uc *out, stbi_uc *in_near, stbi_uc *in_far, int w, int hs)
{
   int i,t0,t1;
   if (w == 1) {
      out[0] = out[1] = stbi__div4(3*in_near[0] + in_far[0] + 2);
      return out;
   }

   t1 = 3*in_near[0] + in_far[0];
   out[0] = stbi__div4(t1+2);
   for (i=1; i < w; ++i) {
      t0 = t1;
      t1 = 3*in_near[i]+in_far[i];
      out[i*2-1] = stbi__div16(3*t0 + t1 + 8);
      out[i*2  ] = stbi__div16(3*t1 + t0 + 8);
   }
   out[w*2-1] = stbi__div4(t1+2);

   STBI_NOTUSED(hs);

   return out;
}


static stbi_uc *stbi__resample_row_generic(stbi_uc *out, stbi_uc *in_near, stbi_uc *in_far, int w, int hs)
{
   int i,j;
   STBI_NOTUSED(in_far);
   for (i=0; i < w; ++i)
      for (j=0; j < hs; ++j)
         out[i*hs+j] = in_near[i];
   return out;
}

#define stbi__float2fixed(x)  (((int) ((x) * 4096.0f + 0.5f)) << 8)
static void stbi__YCbCr_to_RGB_row(stbi_uc *out, const stbi_uc *y, const stbi_uc *pcb, const stbi_uc *pcr, int count, int step)
{
   int i;
   for (i=0; i < count; ++i) {
      int y_fixed = (y[i] << 20) + (1<<19); // rounding
      int r,g,b;
      int cr = pcr[i] - 128;
      int cb = pcb[i] - 128;
      r = y_fixed +  cr* stbi__float2fixed(1.40200f);
      g = y_fixed + (cr*-stbi__float2fixed(0.71414f)) + ((cb*-stbi__float2fixed(0.34414f)) & 0xffff0000);
      b = y_fixed                                     +   cb* stbi__float2fixed(1.77200f);
      r >>= 20;
      g >>= 20;
      b >>= 20;
      if ((unsigned) r > 255) { if (r < 0) r = 0; else r = 255; }
      if ((unsigned) g > 255) { if (g < 0) g = 0; else g = 255; }
      if ((unsigned) b > 255) { if (b < 0) b = 0; else b = 255; }
      out[0] = (stbi_uc)r;
      out[1] = (stbi_uc)g;
      out[2] = (stbi_uc)b;
      out[3] = 255;
      out += step;
   }
}


static void stbi__setup_jpeg(stbi__jpeg *j)
{
   j->idct_block_kernel = stbi__idct_block;
   j->YCbCr_to_RGB_kernel = stbi__YCbCr_to_RGB_row;
   j->resample_row_hv_2_kernel = stbi__resample_row_hv_2;


}

static void stbi__cleanup_jpeg(stbi__jpeg *j)
{
   stbi__free_jpeg_components(j, j->s->img_n, 0);
}

typedef struct
{
   resample_row_func resample;
   stbi_uc *line0,*line1;
   int hs,vs;   // expansion factor in each axis
   int w_lores; // horizontal pixels pre-expansion
   int ystep;   // how far through vertical expansion we are
   int ypos;    // which pre-expansion row we're on
} stbi__resample;

static stbi_uc stbi__blinn_8x8(stbi_uc x, stbi_uc y)
{
   unsigned int t = x*y + 128;
   return (stbi_uc) ((t + (t >>8)) >> 8);
}

static stbi_uc *load_jpeg_image(stbi__jpeg *z, int *out_x, int *out_y, int *comp, int req_comp)
{
   int n, decode_n, is_rgb;
   z->s->img_n = 0; // make stbi__cleanup_jpeg safe

   if (req_comp < 0 || req_comp > 4) return stbi__errpuc("bad req_comp", "Internal error");

   if (!stbi__decode_jpeg_image(z)) { stbi__cleanup_jpeg(z); return NULL; }

   n = req_comp ? req_comp : z->s->img_n >= 3 ? 3 : 1;

   is_rgb = z->s->img_n == 3 && (z->rgb == 3 || (z->app14_color_transform == 0 && !z->jfif));

   if (z->s->img_n == 3 && n < 3 && !is_rgb)
      decode_n = 1;
   else
      decode_n = z->s->img_n;

   if (decode_n <= 0) { stbi__cleanup_jpeg(z); return NULL; }

   {
      int k;
      unsigned int i,j;
      stbi_uc *output;
      stbi_uc *coutput[4] = { NULL, NULL, NULL, NULL };

      stbi__resample res_comp[4];

      for (k=0; k < decode_n; ++k) {
         stbi__resample *r = &res_comp[k];

         z->img_comp[k].linebuf = (stbi_uc *) stbi__malloc(z->s->img_x + 3);
         if (!z->img_comp[k].linebuf) { stbi__cleanup_jpeg(z); return stbi__errpuc("outofmem", "Out of memory"); }

         r->hs      = z->img_h_max / z->img_comp[k].h;
         r->vs      = z->img_v_max / z->img_comp[k].v;
         r->ystep   = r->vs >> 1;
         r->w_lores = (z->s->img_x + r->hs-1) / r->hs;
         r->ypos    = 0;
         r->line0   = r->line1 = z->img_comp[k].data;

         if      (r->hs == 1 && r->vs == 1) r->resample = resample_row_1;
         else if (r->hs == 1 && r->vs == 2) r->resample = stbi__resample_row_v_2;
         else if (r->hs == 2 && r->vs == 1) r->resample = stbi__resample_row_h_2;
         else if (r->hs == 2 && r->vs == 2) r->resample = z->resample_row_hv_2_kernel;
         else                               r->resample = stbi__resample_row_generic;
      }

      output = (stbi_uc *) stbi__malloc_mad3(n, z->s->img_x, z->s->img_y, 1);
      if (!output) { stbi__cleanup_jpeg(z); return stbi__errpuc("outofmem", "Out of memory"); }

      for (j=0; j < z->s->img_y; ++j) {
         stbi_uc *out = output + n * z->s->img_x * j;
         for (k=0; k < decode_n; ++k) {
            stbi__resample *r = &res_comp[k];
            int y_bot = r->ystep >= (r->vs >> 1);
            coutput[k] = r->resample(z->img_comp[k].linebuf,
                                     y_bot ? r->line1 : r->line0,
                                     y_bot ? r->line0 : r->line1,
                                     r->w_lores, r->hs);
            if (++r->ystep >= r->vs) {
               r->ystep = 0;
               r->line0 = r->line1;
               if (++r->ypos < z->img_comp[k].y)
                  r->line1 += z->img_comp[k].w2;
            }
         }
         if (n >= 3) {
            stbi_uc *y = coutput[0];
            if (z->s->img_n == 3) {
               if (is_rgb) {
                  for (i=0; i < z->s->img_x; ++i) {
                     out[0] = y[i];
                     out[1] = coutput[1][i];
                     out[2] = coutput[2][i];
                     out[3] = 255;
                     out += n;
                  }
               } else {
                  z->YCbCr_to_RGB_kernel(out, y, coutput[1], coutput[2], z->s->img_x, n);
               }
            } else if (z->s->img_n == 4) {
               if (z->app14_color_transform == 0) { // CMYK
                  for (i=0; i < z->s->img_x; ++i) {
                     stbi_uc m = coutput[3][i];
                     out[0] = stbi__blinn_8x8(coutput[0][i], m);
                     out[1] = stbi__blinn_8x8(coutput[1][i], m);
                     out[2] = stbi__blinn_8x8(coutput[2][i], m);
                     out[3] = 255;
                     out += n;
                  }
               } else if (z->app14_color_transform == 2) { // YCCK
                  z->YCbCr_to_RGB_kernel(out, y, coutput[1], coutput[2], z->s->img_x, n);
                  for (i=0; i < z->s->img_x; ++i) {
                     stbi_uc m = coutput[3][i];
                     out[0] = stbi__blinn_8x8(255 - out[0], m);
                     out[1] = stbi__blinn_8x8(255 - out[1], m);
                     out[2] = stbi__blinn_8x8(255 - out[2], m);
                     out += n;
                  }
               } else { // YCbCr + alpha?  Ignore the fourth channel for now
                  z->YCbCr_to_RGB_kernel(out, y, coutput[1], coutput[2], z->s->img_x, n);
               }
            } else
               for (i=0; i < z->s->img_x; ++i) {
                  out[0] = out[1] = out[2] = y[i];
                  out[3] = 255; // not used if n==3
                  out += n;
               }
         } else {
            if (is_rgb) {
               if (n == 1)
                  for (i=0; i < z->s->img_x; ++i)
                     *out++ = stbi__compute_y(coutput[0][i], coutput[1][i], coutput[2][i]);
               else {
                  for (i=0; i < z->s->img_x; ++i, out += 2) {
                     out[0] = stbi__compute_y(coutput[0][i], coutput[1][i], coutput[2][i]);
                     out[1] = 255;
                  }
               }
            } else if (z->s->img_n == 4 && z->app14_color_transform == 0) {
               for (i=0; i < z->s->img_x; ++i) {
                  stbi_uc m = coutput[3][i];
                  stbi_uc r = stbi__blinn_8x8(coutput[0][i], m);
                  stbi_uc g = stbi__blinn_8x8(coutput[1][i], m);
                  stbi_uc b = stbi__blinn_8x8(coutput[2][i], m);
                  out[0] = stbi__compute_y(r, g, b);
                  out[1] = 255;
                  out += n;
               }
            } else if (z->s->img_n == 4 && z->app14_color_transform == 2) {
               for (i=0; i < z->s->img_x; ++i) {
                  out[0] = stbi__blinn_8x8(255 - coutput[0][i], coutput[3][i]);
                  out[1] = 255;
                  out += n;
               }
            } else {
               stbi_uc *y = coutput[0];
               if (n == 1)
                  for (i=0; i < z->s->img_x; ++i) out[i] = y[i];
               else
                  for (i=0; i < z->s->img_x; ++i) { *out++ = y[i]; *out++ = 255; }
            }
         }
      }
      stbi__cleanup_jpeg(z);
      *out_x = z->s->img_x;
      *out_y = z->s->img_y;
      if (comp) *comp = z->s->img_n >= 3 ? 3 : 1; // report original components, not output
      return output;
   }
}

static void *stbi__jpeg_load(stbi__context *s, int *x, int *y, int *comp, int req_comp, stbi__result_info *ri)
{
   unsigned char* result;
   stbi__jpeg* j = (stbi__jpeg*) stbi__malloc(sizeof(stbi__jpeg));
   if (!j) return stbi__errpuc("outofmem", "Out of memory");
   memset(j, 0, sizeof(stbi__jpeg));
   STBI_NOTUSED(ri);
   j->s = s;
   stbi__setup_jpeg(j);
   result = load_jpeg_image(j, x,y,comp,req_comp);
   STBI_FREE(j);
   return result;
}

static int stbi__jpeg_test(stbi__context *s)
{
   int r;
   stbi__jpeg* j = (stbi__jpeg*)stbi__malloc(sizeof(stbi__jpeg));
   if (!j) return stbi__err("outofmem", "Out of memory");
   memset(j, 0, sizeof(stbi__jpeg));
   j->s = s;
   stbi__setup_jpeg(j);
   r = stbi__decode_jpeg_header(j, STBI__SCAN_type);
   stbi__rewind(s);
   STBI_FREE(j);
   return r;
}

static int stbi__jpeg_info_raw(stbi__jpeg *j, int *x, int *y, int *comp)
{
   if (!stbi__decode_jpeg_header(j, STBI__SCAN_header)) {
      stbi__rewind( j->s );
      return 0;
   }
   if (x) *x = j->s->img_x;
   if (y) *y = j->s->img_y;
   if (comp) *comp = j->s->img_n >= 3 ? 3 : 1;
   return 1;
}

static int stbi__jpeg_info(stbi__context *s, int *x, int *y, int *comp)
{
   int result;
   stbi__jpeg* j = (stbi__jpeg*) (stbi__malloc(sizeof(stbi__jpeg)));
   if (!j) return stbi__err("outofmem", "Out of memory");
   memset(j, 0, sizeof(stbi__jpeg));
   j->s = s;
   result = stbi__jpeg_info_raw(j, x, y, comp);
   STBI_FREE(j);
   return result;
}



#define STBI__ZFAST_BITS  9 // accelerate all cases in default tables
#define STBI__ZFAST_MASK  ((1 << STBI__ZFAST_BITS) - 1)
#define STBI__ZNSYMS 288 // number of symbols in literal/length alphabet

typedef struct
{
   stbi__uint16 fast[1 << STBI__ZFAST_BITS];
   stbi__uint16 firstcode[16];
   int maxcode[17];
   stbi__uint16 firstsymbol[16];
   stbi_uc  size[STBI__ZNSYMS];
   stbi__uint16 value[STBI__ZNSYMS];
} stbi__zhuffman;

stbi_inline static int stbi__bitreverse16(int n)
{
  n = ((n & 0xAAAA) >>  1) | ((n & 0x5555) << 1);
  n = ((n & 0xCCCC) >>  2) | ((n & 0x3333) << 2);
  n = ((n & 0xF0F0) >>  4) | ((n & 0x0F0F) << 4);
  n = ((n & 0xFF00) >>  8) | ((n & 0x00FF) << 8);
  return n;
}

stbi_inline static int stbi__bit_reverse(int v, int bits)
{
   STBI_ASSERT(bits <= 16);
   return stbi__bitreverse16(v) >> (16-bits);
}

static int stbi__zbuild_huffman(stbi__zhuffman *z, const stbi_uc *sizelist, int num)
{
   int i,k=0;
   int code, next_code[16], sizes[17];

   memset(sizes, 0, sizeof(sizes));
   memset(z->fast, 0, sizeof(z->fast));
   for (i=0; i < num; ++i)
      ++sizes[sizelist[i]];
   sizes[0] = 0;
   for (i=1; i < 16; ++i)
      if (sizes[i] > (1 << i))
         return stbi__err("bad sizes", "Corrupt PNG");
   code = 0;
   for (i=1; i < 16; ++i) {
      next_code[i] = code;
      z->firstcode[i] = (stbi__uint16) code;
      z->firstsymbol[i] = (stbi__uint16) k;
      code = (code + sizes[i]);
      if (sizes[i])
         if (code-1 >= (1 << i)) return stbi__err("bad codelengths","Corrupt PNG");
      z->maxcode[i] = code << (16-i); // preshift for inner loop
      code <<= 1;
      k += sizes[i];
   }
   z->maxcode[16] = 0x10000; // sentinel
   for (i=0; i < num; ++i) {
      int s = sizelist[i];
      if (s) {
         int c = next_code[s] - z->firstcode[s] + z->firstsymbol[s];
         stbi__uint16 fastv = (stbi__uint16) ((s << 9) | i);
         z->size [c] = (stbi_uc     ) s;
         z->value[c] = (stbi__uint16) i;
         if (s <= STBI__ZFAST_BITS) {
            int j = stbi__bit_reverse(next_code[s],s);
            while (j < (1 << STBI__ZFAST_BITS)) {
               z->fast[j] = fastv;
               j += (1 << s);
            }
         }
         ++next_code[s];
      }
   }
   return 1;
}


typedef struct
{
   stbi_uc *zbuffer, *zbuffer_end;
   int num_bits;
   int hit_zeof_once;
   stbi__uint32 code_buffer;

   char *zout;
   char *zout_start;
   char *zout_end;
   int   z_expandable;

   stbi__zhuffman z_length, z_distance;
} stbi__zbuf;

stbi_inline static int stbi__zeof(stbi__zbuf *z)
{
   return (z->zbuffer >= z->zbuffer_end);
}

stbi_inline static stbi_uc stbi__zget8(stbi__zbuf *z)
{
   return stbi__zeof(z) ? 0 : *z->zbuffer++;
}

static void stbi__fill_bits(stbi__zbuf *z)
{
   do {
      if (z->code_buffer >= (1U << z->num_bits)) {
        z->zbuffer = z->zbuffer_end;  /* treat this as EOF so we fail. */
        return;
      }
      z->code_buffer |= (unsigned int) stbi__zget8(z) << z->num_bits;
      z->num_bits += 8;
   } while (z->num_bits <= 24);
}

stbi_inline static unsigned int stbi__zreceive(stbi__zbuf *z, int n)
{
   unsigned int k;
   if (z->num_bits < n) stbi__fill_bits(z);
   k = z->code_buffer & ((1 << n) - 1);
   z->code_buffer >>= n;
   z->num_bits -= n;
   return k;
}

static int stbi__zhuffman_decode_slowpath(stbi__zbuf *a, stbi__zhuffman *z)
{
   int b,s,k;
   k = stbi__bit_reverse(a->code_buffer, 16);
   for (s=STBI__ZFAST_BITS+1; ; ++s)
      if (k < z->maxcode[s])
         break;
   if (s >= 16) return -1; // invalid code!
   b = (k >> (16-s)) - z->firstcode[s] + z->firstsymbol[s];
   if (b >= STBI__ZNSYMS) return -1; // some data was corrupt somewhere!
   if (z->size[b] != s) return -1;  // was originally an assert, but report failure instead.
   a->code_buffer >>= s;
   a->num_bits -= s;
   return z->value[b];
}

stbi_inline static int stbi__zhuffman_decode(stbi__zbuf *a, stbi__zhuffman *z)
{
   int b,s;
   if (a->num_bits < 16) {
      if (stbi__zeof(a)) {
         if (!a->hit_zeof_once) {
            a->hit_zeof_once = 1;
            a->num_bits += 16; // add 16 implicit zero bits
         } else {
            return -1;
         }
      } else {
         stbi__fill_bits(a);
      }
   }
   b = z->fast[a->code_buffer & STBI__ZFAST_MASK];
   if (b) {
      s = b >> 9;
      a->code_buffer >>= s;
      a->num_bits -= s;
      return b & 511;
   }
   return stbi__zhuffman_decode_slowpath(a, z);
}

static int stbi__zexpand(stbi__zbuf *z, char *zout, int n)  // need to make room for n bytes
{
   char *q;
   unsigned int cur, limit, old_limit;
   z->zout = zout;
   if (!z->z_expandable) return stbi__err("output buffer limit","Corrupt PNG");
   cur   = (unsigned int) (z->zout - z->zout_start);
   limit = old_limit = (unsigned) (z->zout_end - z->zout_start);
   if (UINT_MAX - cur < (unsigned) n) return stbi__err("outofmem", "Out of memory");
   while (cur + n > limit) {
      if(limit > UINT_MAX / 2) return stbi__err("outofmem", "Out of memory");
      limit *= 2;
   }
   q = (char *) STBI_REALLOC_SIZED(z->zout_start, old_limit, limit);
   STBI_NOTUSED(old_limit);
   if (q == NULL) return stbi__err("outofmem", "Out of memory");
   z->zout_start = q;
   z->zout       = q + cur;
   z->zout_end   = q + limit;
   return 1;
}

static const int stbi__zlength_base[31] = {
   3,4,5,6,7,8,9,10,11,13,
   15,17,19,23,27,31,35,43,51,59,
   67,83,99,115,131,163,195,227,258,0,0 };

static const int stbi__zlength_extra[31]=
{ 0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0 };

static const int stbi__zdist_base[32] = { 1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,
257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577,0,0};

static const int stbi__zdist_extra[32] =
{ 0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13};

static int stbi__parse_huffman_block(stbi__zbuf *a)
{
   char *zout = a->zout;
   for(;;) {
      int z = stbi__zhuffman_decode(a, &a->z_length);
      if (z < 256) {
         if (z < 0) return stbi__err("bad huffman code","Corrupt PNG"); // error in huffman codes
         if (zout >= a->zout_end) {
            if (!stbi__zexpand(a, zout, 1)) return 0;
            zout = a->zout;
         }
         *zout++ = (char) z;
      } else {
         stbi_uc *p;
         int len,dist;
         if (z == 256) {
            a->zout = zout;
            if (a->hit_zeof_once && a->num_bits < 16) {
               return stbi__err("unexpected end","Corrupt PNG");
            }
            return 1;
         }
         if (z >= 286) return stbi__err("bad huffman code","Corrupt PNG"); // per DEFLATE, length codes 286 and 287 must not appear in compressed data
         z -= 257;
         len = stbi__zlength_base[z];
         if (stbi__zlength_extra[z]) len += stbi__zreceive(a, stbi__zlength_extra[z]);
         z = stbi__zhuffman_decode(a, &a->z_distance);
         if (z < 0 || z >= 30) return stbi__err("bad huffman code","Corrupt PNG"); // per DEFLATE, distance codes 30 and 31 must not appear in compressed data
         dist = stbi__zdist_base[z];
         if (stbi__zdist_extra[z]) dist += stbi__zreceive(a, stbi__zdist_extra[z]);
         if (zout - a->zout_start < dist) return stbi__err("bad dist","Corrupt PNG");
         if (len > a->zout_end - zout) {
            if (!stbi__zexpand(a, zout, len)) return 0;
            zout = a->zout;
         }
         p = (stbi_uc *) (zout - dist);
         if (dist == 1) { // run of one byte; common in images.
            stbi_uc v = *p;
            if (len) { do *zout++ = v; while (--len); }
         } else {
            if (len) { do *zout++ = *p++; while (--len); }
         }
      }
   }
}

static int stbi__compute_huffman_codes(stbi__zbuf *a)
{
   static const stbi_uc length_dezigzag[19] = { 16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15 };
   stbi__zhuffman z_codelength;
   stbi_uc lencodes[286+32+137];//padding for maximum single op
   stbi_uc codelength_sizes[19];
   int i,n;

   int hlit  = stbi__zreceive(a,5) + 257;
   int hdist = stbi__zreceive(a,5) + 1;
   int hclen = stbi__zreceive(a,4) + 4;
   int ntot  = hlit + hdist;

   memset(codelength_sizes, 0, sizeof(codelength_sizes));
   for (i=0; i < hclen; ++i) {
      int s = stbi__zreceive(a,3);
      codelength_sizes[length_dezigzag[i]] = (stbi_uc) s;
   }
   if (!stbi__zbuild_huffman(&z_codelength, codelength_sizes, 19)) return 0;

   n = 0;
   while (n < ntot) {
      int c = stbi__zhuffman_decode(a, &z_codelength);
      if (c < 0 || c >= 19) return stbi__err("bad codelengths", "Corrupt PNG");
      if (c < 16)
         lencodes[n++] = (stbi_uc) c;
      else {
         stbi_uc fill = 0;
         if (c == 16) {
            c = stbi__zreceive(a,2)+3;
            if (n == 0) return stbi__err("bad codelengths", "Corrupt PNG");
            fill = lencodes[n-1];
         } else if (c == 17) {
            c = stbi__zreceive(a,3)+3;
         } else if (c == 18) {
            c = stbi__zreceive(a,7)+11;
         } else {
            return stbi__err("bad codelengths", "Corrupt PNG");
         }
         if (ntot - n < c) return stbi__err("bad codelengths", "Corrupt PNG");
         memset(lencodes+n, fill, c);
         n += c;
      }
   }
   if (n != ntot) return stbi__err("bad codelengths","Corrupt PNG");
   if (!stbi__zbuild_huffman(&a->z_length, lencodes, hlit)) return 0;
   if (!stbi__zbuild_huffman(&a->z_distance, lencodes+hlit, hdist)) return 0;
   return 1;
}

static int stbi__parse_uncompressed_block(stbi__zbuf *a)
{
   stbi_uc header[4];
   int len,nlen,k;
   if (a->num_bits & 7)
      stbi__zreceive(a, a->num_bits & 7); // discard
   k = 0;
   while (a->num_bits > 0) {
      header[k++] = (stbi_uc) (a->code_buffer & 255); // suppress MSVC run-time check
      a->code_buffer >>= 8;
      a->num_bits -= 8;
   }
   if (a->num_bits < 0) return stbi__err("zlib corrupt","Corrupt PNG");
   while (k < 4)
      header[k++] = stbi__zget8(a);
   len  = header[1] * 256 + header[0];
   nlen = header[3] * 256 + header[2];
   if (nlen != (len ^ 0xffff)) return stbi__err("zlib corrupt","Corrupt PNG");
   if (a->zbuffer + len > a->zbuffer_end) return stbi__err("read past buffer","Corrupt PNG");
   if (a->zout + len > a->zout_end)
      if (!stbi__zexpand(a, a->zout, len)) return 0;
   memcpy(a->zout, a->zbuffer, len);
   a->zbuffer += len;
   a->zout += len;
   return 1;
}

static int stbi__parse_zlib_header(stbi__zbuf *a)
{
   int cmf   = stbi__zget8(a);
   int cm    = cmf & 15;
   int flg   = stbi__zget8(a);
   if (stbi__zeof(a)) return stbi__err("bad zlib header","Corrupt PNG"); // zlib spec
   if ((cmf*256+flg) % 31 != 0) return stbi__err("bad zlib header","Corrupt PNG"); // zlib spec
   if (flg & 32) return stbi__err("no preset dict","Corrupt PNG"); // preset dictionary not allowed in png
   if (cm != 8) return stbi__err("bad compression","Corrupt PNG"); // DEFLATE required for png
   return 1;
}

static const stbi_uc stbi__zdefault_length[STBI__ZNSYMS] =
{
   8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
   8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
   8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
   8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
   8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
   9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
   9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
   9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
   7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7, 7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8
};
static const stbi_uc stbi__zdefault_distance[32] =
{
   5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5
};

static int stbi__parse_zlib(stbi__zbuf *a, int parse_header)
{
   int final, type;
   if (parse_header)
      if (!stbi__parse_zlib_header(a)) return 0;
   a->num_bits = 0;
   a->code_buffer = 0;
   a->hit_zeof_once = 0;
   do {
      final = stbi__zreceive(a,1);
      type = stbi__zreceive(a,2);
      if (type == 0) {
         if (!stbi__parse_uncompressed_block(a)) return 0;
      } else if (type == 3) {
         return 0;
      } else {
         if (type == 1) {
            if (!stbi__zbuild_huffman(&a->z_length  , stbi__zdefault_length  , STBI__ZNSYMS)) return 0;
            if (!stbi__zbuild_huffman(&a->z_distance, stbi__zdefault_distance,  32)) return 0;
         } else {
            if (!stbi__compute_huffman_codes(a)) return 0;
         }
         if (!stbi__parse_huffman_block(a)) return 0;
      }
   } while (!final);
   return 1;
}

static int stbi__do_zlib(stbi__zbuf *a, char *obuf, int olen, int exp, int parse_header)
{
   a->zout_start = obuf;
   a->zout       = obuf;
   a->zout_end   = obuf + olen;
   a->z_expandable = exp;

   return stbi__parse_zlib(a, parse_header);
}

STBIDEF char *stbi_zlib_decode_malloc_guesssize(const char *buffer, int len, int initial_size, int *outlen)
{
   stbi__zbuf a;
   char *p = (char *) stbi__malloc(initial_size);
   if (p == NULL) return NULL;
   a.zbuffer = (stbi_uc *) buffer;
   a.zbuffer_end = (stbi_uc *) buffer + len;
   if (stbi__do_zlib(&a, p, initial_size, 1, 1)) {
      if (outlen) *outlen = (int) (a.zout - a.zout_start);
      return a.zout_start;
   } else {
      STBI_FREE(a.zout_start);
      return NULL;
   }
}

STBIDEF char *stbi_zlib_decode_malloc(char const *buffer, int len, int *outlen)
{
   return stbi_zlib_decode_malloc_guesssize(buffer, len, 16384, outlen);
}

STBIDEF char *stbi_zlib_decode_malloc_guesssize_headerflag(const char *buffer, int len, int initial_size, int *outlen, int parse_header)
{
   stbi__zbuf a;
   char *p = (char *) stbi__malloc(initial_size);
   if (p == NULL) return NULL;
   a.zbuffer = (stbi_uc *) buffer;
   a.zbuffer_end = (stbi_uc *) buffer + len;
   if (stbi__do_zlib(&a, p, initial_size, 1, parse_header)) {
      if (outlen) *outlen = (int) (a.zout - a.zout_start);
      return a.zout_start;
   } else {
      STBI_FREE(a.zout_start);
      return NULL;
   }
}

STBIDEF int stbi_zlib_decode_buffer(char *obuffer, int olen, char const *ibuffer, int ilen)
{
   stbi__zbuf a;
   a.zbuffer = (stbi_uc *) ibuffer;
   a.zbuffer_end = (stbi_uc *) ibuffer + ilen;
   if (stbi__do_zlib(&a, obuffer, olen, 0, 1))
      return (int) (a.zout - a.zout_start);
   else
      return -1;
}

STBIDEF char *stbi_zlib_decode_noheader_malloc(char const *buffer, int len, int *outlen)
{
   stbi__zbuf a;
   char *p = (char *) stbi__malloc(16384);
   if (p == NULL) return NULL;
   a.zbuffer = (stbi_uc *) buffer;
   a.zbuffer_end = (stbi_uc *) buffer+len;
   if (stbi__do_zlib(&a, p, 16384, 1, 0)) {
      if (outlen) *outlen = (int) (a.zout - a.zout_start);
      return a.zout_start;
   } else {
      STBI_FREE(a.zout_start);
      return NULL;
   }
}

STBIDEF int stbi_zlib_decode_noheader_buffer(char *obuffer, int olen, const char *ibuffer, int ilen)
{
   stbi__zbuf a;
   a.zbuffer = (stbi_uc *) ibuffer;
   a.zbuffer_end = (stbi_uc *) ibuffer + ilen;
   if (stbi__do_zlib(&a, obuffer, olen, 0, 0))
      return (int) (a.zout - a.zout_start);
   else
      return -1;
}


typedef struct
{
   stbi__uint32 length;
   stbi__uint32 type;
} stbi__pngchunk;

static stbi__pngchunk stbi__get_chunk_header(stbi__context *s)
{
   stbi__pngchunk c;
   c.length = stbi__get32be(s);
   c.type   = stbi__get32be(s);
   return c;
}

static int stbi__check_png_header(stbi__context *s)
{
   static const stbi_uc png_sig[8] = { 137,80,78,71,13,10,26,10 };
   int i;
   for (i=0; i < 8; ++i)
      if (stbi__get8(s) != png_sig[i]) return stbi__err("bad png sig","Not a PNG");
   return 1;
}

typedef struct
{
   stbi__context *s;
   stbi_uc *idata, *expanded, *out;
   int depth;
} stbi__png;


enum {
   STBI__F_none=0,
   STBI__F_sub=1,
   STBI__F_up=2,
   STBI__F_avg=3,
   STBI__F_paeth=4,
   STBI__F_avg_first
};

static stbi_uc first_row_filter[5] =
{
   STBI__F_none,
   STBI__F_sub,
   STBI__F_none,
   STBI__F_avg_first,
   STBI__F_sub // Paeth with b=c=0 turns out to be equivalent to sub
};

static int stbi__paeth(int a, int b, int c)
{
   int thresh = c*3 - (a + b);
   int lo = a < b ? a : b;
   int hi = a < b ? b : a;
   int t0 = (hi <= thresh) ? lo : c;
   int t1 = (thresh <= lo) ? hi : t0;
   return t1;
}

static const stbi_uc stbi__depth_scale_table[9] = { 0, 0xff, 0x55, 0, 0x11, 0,0,0, 0x01 };

static void stbi__create_png_alpha_expand8(stbi_uc *dest, stbi_uc *src, stbi__uint32 x, int img_n)
{
   int i;
   if (img_n == 1) {
      for (i=x-1; i >= 0; --i) {
         dest[i*2+1] = 255;
         dest[i*2+0] = src[i];
      }
   } else {
      STBI_ASSERT(img_n == 3);
      for (i=x-1; i >= 0; --i) {
         dest[i*4+3] = 255;
         dest[i*4+2] = src[i*3+2];
         dest[i*4+1] = src[i*3+1];
         dest[i*4+0] = src[i*3+0];
      }
   }
}

static int stbi__create_png_image_raw(stbi__png *a, stbi_uc *raw, stbi__uint32 raw_len, int out_n, stbi__uint32 x, stbi__uint32 y, int depth, int color)
{
   int bytes = (depth == 16 ? 2 : 1);
   stbi__context *s = a->s;
   stbi__uint32 i,j,stride = x*out_n*bytes;
   stbi__uint32 img_len, img_width_bytes;
   stbi_uc *filter_buf;
   int all_ok = 1;
   int k;
   int img_n = s->img_n; // copy it into a local for later

   int output_bytes = out_n*bytes;
   int filter_bytes = img_n*bytes;
   int width = x;

   STBI_ASSERT(out_n == s->img_n || out_n == s->img_n+1);
   a->out = (stbi_uc *) stbi__malloc_mad3(x, y, output_bytes, 0); // extra bytes to write off the end into
   if (!a->out) return stbi__err("outofmem", "Out of memory");

   if (!stbi__mad3sizes_valid(img_n, x, depth, 7)) return stbi__err("too large", "Corrupt PNG");
   img_width_bytes = (((img_n * x * depth) + 7) >> 3);
   if (!stbi__mad2sizes_valid(img_width_bytes, y, img_width_bytes)) return stbi__err("too large", "Corrupt PNG");
   img_len = (img_width_bytes + 1) * y;

   if (raw_len < img_len) return stbi__err("not enough pixels","Corrupt PNG");

   filter_buf = (stbi_uc *) stbi__malloc_mad2(img_width_bytes, 2, 0);
   if (!filter_buf) return stbi__err("outofmem", "Out of memory");

   if (depth < 8) {
      filter_bytes = 1;
      width = img_width_bytes;
   }

   for (j=0; j < y; ++j) {
      stbi_uc *cur = filter_buf + (j & 1)*img_width_bytes;
      stbi_uc *prior = filter_buf + (~j & 1)*img_width_bytes;
      stbi_uc *dest = a->out + stride*j;
      int nk = width * filter_bytes;
      int filter = *raw++;

      if (filter > 4) {
         all_ok = stbi__err("invalid filter","Corrupt PNG");
         break;
      }

      if (j == 0) filter = first_row_filter[filter];

      switch (filter) {
      case STBI__F_none:
         memcpy(cur, raw, nk);
         break;
      case STBI__F_sub:
         memcpy(cur, raw, filter_bytes);
         for (k = filter_bytes; k < nk; ++k)
            cur[k] = STBI__BYTECAST(raw[k] + cur[k-filter_bytes]);
         break;
      case STBI__F_up:
         for (k = 0; k < nk; ++k)
            cur[k] = STBI__BYTECAST(raw[k] + prior[k]);
         break;
      case STBI__F_avg:
         for (k = 0; k < filter_bytes; ++k)
            cur[k] = STBI__BYTECAST(raw[k] + (prior[k]>>1));
         for (k = filter_bytes; k < nk; ++k)
            cur[k] = STBI__BYTECAST(raw[k] + ((prior[k] + cur[k-filter_bytes])>>1));
         break;
      case STBI__F_paeth:
         for (k = 0; k < filter_bytes; ++k)
            cur[k] = STBI__BYTECAST(raw[k] + prior[k]); // prior[k] == stbi__paeth(0,prior[k],0)
         for (k = filter_bytes; k < nk; ++k)
            cur[k] = STBI__BYTECAST(raw[k] + stbi__paeth(cur[k-filter_bytes], prior[k], prior[k-filter_bytes]));
         break;
      case STBI__F_avg_first:
         memcpy(cur, raw, filter_bytes);
         for (k = filter_bytes; k < nk; ++k)
            cur[k] = STBI__BYTECAST(raw[k] + (cur[k-filter_bytes] >> 1));
         break;
      }

      raw += nk;

      if (depth < 8) {
         stbi_uc scale = (color == 0) ? stbi__depth_scale_table[depth] : 1; // scale grayscale values to 0..255 range
         stbi_uc *in = cur;
         stbi_uc *out = dest;
         stbi_uc inb = 0;
         stbi__uint32 nsmp = x*img_n;

         if (depth == 4) {
            for (i=0; i < nsmp; ++i) {
               if ((i & 1) == 0) inb = *in++;
               *out++ = scale * (inb >> 4);
               inb <<= 4;
            }
         } else if (depth == 2) {
            for (i=0; i < nsmp; ++i) {
               if ((i & 3) == 0) inb = *in++;
               *out++ = scale * (inb >> 6);
               inb <<= 2;
            }
         } else {
            STBI_ASSERT(depth == 1);
            for (i=0; i < nsmp; ++i) {
               if ((i & 7) == 0) inb = *in++;
               *out++ = scale * (inb >> 7);
               inb <<= 1;
            }
         }

         if (img_n != out_n)
            stbi__create_png_alpha_expand8(dest, dest, x, img_n);
      } else if (depth == 8) {
         if (img_n == out_n)
            memcpy(dest, cur, x*img_n);
         else
            stbi__create_png_alpha_expand8(dest, cur, x, img_n);
      } else if (depth == 16) {
         stbi__uint16 *dest16 = (stbi__uint16*)dest;
         stbi__uint32 nsmp = x*img_n;

         if (img_n == out_n) {
            for (i = 0; i < nsmp; ++i, ++dest16, cur += 2)
               *dest16 = (cur[0] << 8) | cur[1];
         } else {
            STBI_ASSERT(img_n+1 == out_n);
            if (img_n == 1) {
               for (i = 0; i < x; ++i, dest16 += 2, cur += 2) {
                  dest16[0] = (cur[0] << 8) | cur[1];
                  dest16[1] = 0xffff;
               }
            } else {
               STBI_ASSERT(img_n == 3);
               for (i = 0; i < x; ++i, dest16 += 4, cur += 6) {
                  dest16[0] = (cur[0] << 8) | cur[1];
                  dest16[1] = (cur[2] << 8) | cur[3];
                  dest16[2] = (cur[4] << 8) | cur[5];
                  dest16[3] = 0xffff;
               }
            }
         }
      }
   }

   STBI_FREE(filter_buf);
   if (!all_ok) return 0;

   return 1;
}

static int stbi__create_png_image(stbi__png *a, stbi_uc *image_data, stbi__uint32 image_data_len, int out_n, int depth, int color, int interlaced)
{
   int bytes = (depth == 16 ? 2 : 1);
   int out_bytes = out_n * bytes;
   stbi_uc *final;
   int p;
   if (!interlaced)
      return stbi__create_png_image_raw(a, image_data, image_data_len, out_n, a->s->img_x, a->s->img_y, depth, color);

   final = (stbi_uc *) stbi__malloc_mad3(a->s->img_x, a->s->img_y, out_bytes, 0);
   if (!final) return stbi__err("outofmem", "Out of memory");
   for (p=0; p < 7; ++p) {
      int xorig[] = { 0,4,0,2,0,1,0 };
      int yorig[] = { 0,0,4,0,2,0,1 };
      int xspc[]  = { 8,8,4,4,2,2,1 };
      int yspc[]  = { 8,8,8,4,4,2,2 };
      int i,j,x,y;
      x = (a->s->img_x - xorig[p] + xspc[p]-1) / xspc[p];
      y = (a->s->img_y - yorig[p] + yspc[p]-1) / yspc[p];
      if (x && y) {
         stbi__uint32 img_len = ((((a->s->img_n * x * depth) + 7) >> 3) + 1) * y;
         if (!stbi__create_png_image_raw(a, image_data, image_data_len, out_n, x, y, depth, color)) {
            STBI_FREE(final);
            return 0;
         }
         for (j=0; j < y; ++j) {
            for (i=0; i < x; ++i) {
               int out_y = j*yspc[p]+yorig[p];
               int out_x = i*xspc[p]+xorig[p];
               memcpy(final + out_y*a->s->img_x*out_bytes + out_x*out_bytes,
                      a->out + (j*x+i)*out_bytes, out_bytes);
            }
         }
         STBI_FREE(a->out);
         image_data += img_len;
         image_data_len -= img_len;
      }
   }
   a->out = final;

   return 1;
}

static int stbi__compute_transparency(stbi__png *z, stbi_uc tc[3], int out_n)
{
   stbi__context *s = z->s;
   stbi__uint32 i, pixel_count = s->img_x * s->img_y;
   stbi_uc *p = z->out;

   STBI_ASSERT(out_n == 2 || out_n == 4);

   if (out_n == 2) {
      for (i=0; i < pixel_count; ++i) {
         p[1] = (p[0] == tc[0] ? 0 : 255);
         p += 2;
      }
   } else {
      for (i=0; i < pixel_count; ++i) {
         if (p[0] == tc[0] && p[1] == tc[1] && p[2] == tc[2])
            p[3] = 0;
         p += 4;
      }
   }
   return 1;
}

static int stbi__compute_transparency16(stbi__png *z, stbi__uint16 tc[3], int out_n)
{
   stbi__context *s = z->s;
   stbi__uint32 i, pixel_count = s->img_x * s->img_y;
   stbi__uint16 *p = (stbi__uint16*) z->out;

   STBI_ASSERT(out_n == 2 || out_n == 4);

   if (out_n == 2) {
      for (i = 0; i < pixel_count; ++i) {
         p[1] = (p[0] == tc[0] ? 0 : 65535);
         p += 2;
      }
   } else {
      for (i = 0; i < pixel_count; ++i) {
         if (p[0] == tc[0] && p[1] == tc[1] && p[2] == tc[2])
            p[3] = 0;
         p += 4;
      }
   }
   return 1;
}

static int stbi__expand_png_palette(stbi__png *a, stbi_uc *palette, int len, int pal_img_n)
{
   stbi__uint32 i, pixel_count = a->s->img_x * a->s->img_y;
   stbi_uc *p, *temp_out, *orig = a->out;

   p = (stbi_uc *) stbi__malloc_mad2(pixel_count, pal_img_n, 0);
   if (p == NULL) return stbi__err("outofmem", "Out of memory");

   temp_out = p;

   if (pal_img_n == 3) {
      for (i=0; i < pixel_count; ++i) {
         int n = orig[i]*4;
         p[0] = palette[n  ];
         p[1] = palette[n+1];
         p[2] = palette[n+2];
         p += 3;
      }
   } else {
      for (i=0; i < pixel_count; ++i) {
         int n = orig[i]*4;
         p[0] = palette[n  ];
         p[1] = palette[n+1];
         p[2] = palette[n+2];
         p[3] = palette[n+3];
         p += 4;
      }
   }
   STBI_FREE(a->out);
   a->out = temp_out;

   STBI_NOTUSED(len);

   return 1;
}

static int stbi__unpremultiply_on_load_global = 0;
static int stbi__de_iphone_flag_global = 0;

STBIDEF void stbi_set_unpremultiply_on_load(int flag_true_if_should_unpremultiply)
{
   stbi__unpremultiply_on_load_global = flag_true_if_should_unpremultiply;
}

STBIDEF void stbi_convert_iphone_png_to_rgb(int flag_true_if_should_convert)
{
   stbi__de_iphone_flag_global = flag_true_if_should_convert;
}

#define stbi__unpremultiply_on_load  stbi__unpremultiply_on_load_global
#define stbi__de_iphone_flag  stbi__de_iphone_flag_global

static void stbi__de_iphone(stbi__png *z)
{
   stbi__context *s = z->s;
   stbi__uint32 i, pixel_count = s->img_x * s->img_y;
   stbi_uc *p = z->out;

   if (s->img_out_n == 3) {  // convert bgr to rgb
      for (i=0; i < pixel_count; ++i) {
         stbi_uc t = p[0];
         p[0] = p[2];
         p[2] = t;
         p += 3;
      }
   } else {
      STBI_ASSERT(s->img_out_n == 4);
      if (stbi__unpremultiply_on_load) {
         for (i=0; i < pixel_count; ++i) {
            stbi_uc a = p[3];
            stbi_uc t = p[0];
            if (a) {
               stbi_uc half = a / 2;
               p[0] = (p[2] * 255 + half) / a;
               p[1] = (p[1] * 255 + half) / a;
               p[2] = ( t   * 255 + half) / a;
            } else {
               p[0] = p[2];
               p[2] = t;
            }
            p += 4;
         }
      } else {
         for (i=0; i < pixel_count; ++i) {
            stbi_uc t = p[0];
            p[0] = p[2];
            p[2] = t;
            p += 4;
         }
      }
   }
}

#define STBI__PNG_TYPE(a,b,c,d)  (((unsigned) (a) << 24) + ((unsigned) (b) << 16) + ((unsigned) (c) << 8) + (unsigned) (d))

static int stbi__parse_png_file(stbi__png *z, int scan, int req_comp)
{
   stbi_uc palette[1024], pal_img_n=0;
   stbi_uc has_trans=0, tc[3]={0};
   stbi__uint16 tc16[3];
   stbi__uint32 ioff=0, idata_limit=0, i, pal_len=0;
   int first=1,k,interlace=0, color=0, is_iphone=0;
   stbi__context *s = z->s;

   z->expanded = NULL;
   z->idata = NULL;
   z->out = NULL;

   if (!stbi__check_png_header(s)) return 0;

   if (scan == STBI__SCAN_type) return 1;

   for (;;) {
      stbi__pngchunk c = stbi__get_chunk_header(s);
      switch (c.type) {
         case STBI__PNG_TYPE('C','g','B','I'):
            is_iphone = 1;
            stbi__skip(s, c.length);
            break;
         case STBI__PNG_TYPE('I','H','D','R'): {
            int comp,filter;
            if (!first) return stbi__err("multiple IHDR","Corrupt PNG");
            first = 0;
            if (c.length != 13) return stbi__err("bad IHDR len","Corrupt PNG");
            s->img_x = stbi__get32be(s);
            s->img_y = stbi__get32be(s);
            if (s->img_y > STBI_MAX_DIMENSIONS) return stbi__err("too large","Very large image (corrupt?)");
            if (s->img_x > STBI_MAX_DIMENSIONS) return stbi__err("too large","Very large image (corrupt?)");
            z->depth = stbi__get8(s);  if (z->depth != 1 && z->depth != 2 && z->depth != 4 && z->depth != 8 && z->depth != 16)  return stbi__err("1/2/4/8/16-bit only","PNG not supported: 1/2/4/8/16-bit only");
            color = stbi__get8(s);  if (color > 6)         return stbi__err("bad ctype","Corrupt PNG");
            if (color == 3 && z->depth == 16)                  return stbi__err("bad ctype","Corrupt PNG");
            if (color == 3) pal_img_n = 3; else if (color & 1) return stbi__err("bad ctype","Corrupt PNG");
            comp  = stbi__get8(s);  if (comp) return stbi__err("bad comp method","Corrupt PNG");
            filter= stbi__get8(s);  if (filter) return stbi__err("bad filter method","Corrupt PNG");
            interlace = stbi__get8(s); if (interlace>1) return stbi__err("bad interlace method","Corrupt PNG");
            if (!s->img_x || !s->img_y) return stbi__err("0-pixel image","Corrupt PNG");
            if (!pal_img_n) {
               s->img_n = (color & 2 ? 3 : 1) + (color & 4 ? 1 : 0);
               if ((1 << 30) / s->img_x / s->img_n < s->img_y) return stbi__err("too large", "Image too large to decode");
            } else {
               s->img_n = 1;
               if ((1 << 30) / s->img_x / 4 < s->img_y) return stbi__err("too large","Corrupt PNG");
            }
            break;
         }

         case STBI__PNG_TYPE('P','L','T','E'):  {
            if (first) return stbi__err("first not IHDR", "Corrupt PNG");
            if (c.length > 256*3) return stbi__err("invalid PLTE","Corrupt PNG");
            pal_len = c.length / 3;
            if (pal_len * 3 != c.length) return stbi__err("invalid PLTE","Corrupt PNG");
            for (i=0; i < pal_len; ++i) {
               palette[i*4+0] = stbi__get8(s);
               palette[i*4+1] = stbi__get8(s);
               palette[i*4+2] = stbi__get8(s);
               palette[i*4+3] = 255;
            }
            break;
         }

         case STBI__PNG_TYPE('t','R','N','S'): {
            if (first) return stbi__err("first not IHDR", "Corrupt PNG");
            if (z->idata) return stbi__err("tRNS after IDAT","Corrupt PNG");
            if (pal_img_n) {
               if (scan == STBI__SCAN_header) { s->img_n = 4; return 1; }
               if (pal_len == 0) return stbi__err("tRNS before PLTE","Corrupt PNG");
               if (c.length > pal_len) return stbi__err("bad tRNS len","Corrupt PNG");
               pal_img_n = 4;
               for (i=0; i < c.length; ++i)
                  palette[i*4+3] = stbi__get8(s);
            } else {
               if (!(s->img_n & 1)) return stbi__err("tRNS with alpha","Corrupt PNG");
               if (c.length != (stbi__uint32) s->img_n*2) return stbi__err("bad tRNS len","Corrupt PNG");
               has_trans = 1;
               if (scan == STBI__SCAN_header) { ++s->img_n; return 1; }
               if (z->depth == 16) {
                  for (k = 0; k < s->img_n && k < 3; ++k) // extra loop test to suppress false GCC warning
                     tc16[k] = (stbi__uint16)stbi__get16be(s); // copy the values as-is
               } else {
                  for (k = 0; k < s->img_n && k < 3; ++k)
                     tc[k] = (stbi_uc)(stbi__get16be(s) & 255) * stbi__depth_scale_table[z->depth]; // non 8-bit images will be larger
               }
            }
            break;
         }

         case STBI__PNG_TYPE('I','D','A','T'): {
            if (first) return stbi__err("first not IHDR", "Corrupt PNG");
            if (pal_img_n && !pal_len) return stbi__err("no PLTE","Corrupt PNG");
            if (scan == STBI__SCAN_header) {
               if (pal_img_n)
                  s->img_n = pal_img_n;
               return 1;
            }
            if (c.length > (1u << 30)) return stbi__err("IDAT size limit", "IDAT section larger than 2^30 bytes");
            if ((int)(ioff + c.length) < (int)ioff) return 0;
            if (ioff + c.length > idata_limit) {
               stbi__uint32 idata_limit_old = idata_limit;
               stbi_uc *p;
               if (idata_limit == 0) idata_limit = c.length > 4096 ? c.length : 4096;
               while (ioff + c.length > idata_limit)
                  idata_limit *= 2;
               STBI_NOTUSED(idata_limit_old);
               p = (stbi_uc *) STBI_REALLOC_SIZED(z->idata, idata_limit_old, idata_limit); if (p == NULL) return stbi__err("outofmem", "Out of memory");
               z->idata = p;
            }
            if (!stbi__getn(s, z->idata+ioff,c.length)) return stbi__err("outofdata","Corrupt PNG");
            ioff += c.length;
            break;
         }

         case STBI__PNG_TYPE('I','E','N','D'): {
            stbi__uint32 raw_len, bpl;
            if (first) return stbi__err("first not IHDR", "Corrupt PNG");
            if (scan != STBI__SCAN_load) return 1;
            if (z->idata == NULL) return stbi__err("no IDAT","Corrupt PNG");
            bpl = (s->img_x * z->depth + 7) / 8; // bytes per line, per component
            raw_len = bpl * s->img_y * s->img_n /* pixels */ + s->img_y /* filter mode per row */;
            z->expanded = (stbi_uc *) stbi_zlib_decode_malloc_guesssize_headerflag((char *) z->idata, ioff, raw_len, (int *) &raw_len, !is_iphone);
            if (z->expanded == NULL) return 0; // zlib should set error
            STBI_FREE(z->idata); z->idata = NULL;
            if ((req_comp == s->img_n+1 && req_comp != 3 && !pal_img_n) || has_trans)
               s->img_out_n = s->img_n+1;
            else
               s->img_out_n = s->img_n;
            if (!stbi__create_png_image(z, z->expanded, raw_len, s->img_out_n, z->depth, color, interlace)) return 0;
            if (has_trans) {
               if (z->depth == 16) {
                  if (!stbi__compute_transparency16(z, tc16, s->img_out_n)) return 0;
               } else {
                  if (!stbi__compute_transparency(z, tc, s->img_out_n)) return 0;
               }
            }
            if (is_iphone && stbi__de_iphone_flag && s->img_out_n > 2)
               stbi__de_iphone(z);
            if (pal_img_n) {
               s->img_n = pal_img_n; // record the actual colors we had
               s->img_out_n = pal_img_n;
               if (req_comp >= 3) s->img_out_n = req_comp;
               if (!stbi__expand_png_palette(z, palette, pal_len, s->img_out_n))
                  return 0;
            } else if (has_trans) {
               ++s->img_n;
            }
            STBI_FREE(z->expanded); z->expanded = NULL;
            stbi__get32be(s);
            return 1;
         }

         default:
            if (first) return stbi__err("first not IHDR", "Corrupt PNG");
            if ((c.type & (1 << 29)) == 0) {
               static char invalid_chunk[] = "XXXX PNG chunk not known";
               invalid_chunk[0] = STBI__BYTECAST(c.type >> 24);
               invalid_chunk[1] = STBI__BYTECAST(c.type >> 16);
               invalid_chunk[2] = STBI__BYTECAST(c.type >>  8);
               invalid_chunk[3] = STBI__BYTECAST(c.type >>  0);
               return stbi__err(invalid_chunk, "PNG not supported: unknown PNG chunk type");
            }
            stbi__skip(s, c.length);
            break;
      }
      stbi__get32be(s);
   }
}

static void *stbi__do_png(stbi__png *p, int *x, int *y, int *n, int req_comp, stbi__result_info *ri)
{
   void *result=NULL;
   if (req_comp < 0 || req_comp > 4) return stbi__errpuc("bad req_comp", "Internal error");
   if (stbi__parse_png_file(p, STBI__SCAN_load, req_comp)) {
      if (p->depth <= 8)
         ri->bits_per_channel = 8;
      else if (p->depth == 16)
         ri->bits_per_channel = 16;
      else
         return stbi__errpuc("bad bits_per_channel", "PNG not supported: unsupported color depth");
      result = p->out;
      p->out = NULL;
      if (req_comp && req_comp != p->s->img_out_n) {
         if (ri->bits_per_channel == 8)
            result = stbi__convert_format((unsigned char *) result, p->s->img_out_n, req_comp, p->s->img_x, p->s->img_y);
         else
            result = stbi__convert_format16((stbi__uint16 *) result, p->s->img_out_n, req_comp, p->s->img_x, p->s->img_y);
         p->s->img_out_n = req_comp;
         if (result == NULL) return result;
      }
      *x = p->s->img_x;
      *y = p->s->img_y;
      if (n) *n = p->s->img_n;
   }
   STBI_FREE(p->out);      p->out      = NULL;
   STBI_FREE(p->expanded); p->expanded = NULL;
   STBI_FREE(p->idata);    p->idata    = NULL;

   return result;
}

static void *stbi__png_load(stbi__context *s, int *x, int *y, int *comp, int req_comp, stbi__result_info *ri)
{
   stbi__png p;
   p.s = s;
   return stbi__do_png(&p, x,y,comp,req_comp, ri);
}

static int stbi__png_test(stbi__context *s)
{
   int r;
   r = stbi__check_png_header(s);
   stbi__rewind(s);
   return r;
}

static int stbi__png_info_raw(stbi__png *p, int *x, int *y, int *comp)
{
   if (!stbi__parse_png_file(p, STBI__SCAN_header, 0)) {
      stbi__rewind( p->s );
      return 0;
   }
   if (x) *x = p->s->img_x;
   if (y) *y = p->s->img_y;
   if (comp) *comp = p->s->img_n;
   return 1;
}

static int stbi__png_info(stbi__context *s, int *x, int *y, int *comp)
{
   stbi__png p;
   p.s = s;
   return stbi__png_info_raw(&p, x, y, comp);
}

static int stbi__png_is16(stbi__context *s)
{
   stbi__png p;
   p.s = s;
   if (!stbi__png_info_raw(&p, NULL, NULL, NULL))
	   return 0;
   if (p.depth != 16) {
      stbi__rewind(p.s);
      return 0;
   }
   return 1;
}



typedef struct
{
   stbi__int16 prefix;
   stbi_uc first;
   stbi_uc suffix;
} stbi__gif_lzw;

typedef struct
{
   int w,h;
   stbi_uc *out;                 // output buffer (always 4 components)
   stbi_uc *background;          // The current "background" as far as a gif is concerned
   stbi_uc *history;
   int flags, bgindex, ratio, transparent, eflags;
   stbi_uc  pal[256][4];
   stbi_uc lpal[256][4];
   stbi__gif_lzw codes[8192];
   stbi_uc *color_table;
   int parse, step;
   int lflags;
   int start_x, start_y;
   int max_x, max_y;
   int cur_x, cur_y;
   int line_size;
   int delay;
} stbi__gif;

static int stbi__gif_test_raw(stbi__context *s)
{
   int sz;
   if (stbi__get8(s) != 'G' || stbi__get8(s) != 'I' || stbi__get8(s) != 'F' || stbi__get8(s) != '8') return 0;
   sz = stbi__get8(s);
   if (sz != '9' && sz != '7') return 0;
   if (stbi__get8(s) != 'a') return 0;
   return 1;
}

static int stbi__gif_test(stbi__context *s)
{
   int r = stbi__gif_test_raw(s);
   stbi__rewind(s);
   return r;
}

static void stbi__gif_parse_colortable(stbi__context *s, stbi_uc pal[256][4], int num_entries, int transp)
{
   int i;
   for (i=0; i < num_entries; ++i) {
      pal[i][2] = stbi__get8(s);
      pal[i][1] = stbi__get8(s);
      pal[i][0] = stbi__get8(s);
      pal[i][3] = transp == i ? 0 : 255;
   }
}

static int stbi__gif_header(stbi__context *s, stbi__gif *g, int *comp, int is_info)
{
   stbi_uc version;
   if (stbi__get8(s) != 'G' || stbi__get8(s) != 'I' || stbi__get8(s) != 'F' || stbi__get8(s) != '8')
      return stbi__err("not GIF", "Corrupt GIF");

   version = stbi__get8(s);
   if (version != '7' && version != '9')    return stbi__err("not GIF", "Corrupt GIF");
   if (stbi__get8(s) != 'a')                return stbi__err("not GIF", "Corrupt GIF");

   stbi__g_failure_reason = "";
   g->w = stbi__get16le(s);
   g->h = stbi__get16le(s);
   g->flags = stbi__get8(s);
   g->bgindex = stbi__get8(s);
   g->ratio = stbi__get8(s);
   g->transparent = -1;

   if (g->w > STBI_MAX_DIMENSIONS) return stbi__err("too large","Very large image (corrupt?)");
   if (g->h > STBI_MAX_DIMENSIONS) return stbi__err("too large","Very large image (corrupt?)");

   if (comp != 0) *comp = 4;  // can't actually tell whether it's 3 or 4 until we parse the comments

   if (is_info) return 1;

   if (g->flags & 0x80)
      stbi__gif_parse_colortable(s,g->pal, 2 << (g->flags & 7), -1);

   return 1;
}

static int stbi__gif_info_raw(stbi__context *s, int *x, int *y, int *comp)
{
   stbi__gif* g = (stbi__gif*) stbi__malloc(sizeof(stbi__gif));
   if (!g) return stbi__err("outofmem", "Out of memory");
   if (!stbi__gif_header(s, g, comp, 1)) {
      STBI_FREE(g);
      stbi__rewind( s );
      return 0;
   }
   if (x) *x = g->w;
   if (y) *y = g->h;
   STBI_FREE(g);
   return 1;
}

static void stbi__out_gif_code(stbi__gif *g, stbi__uint16 code)
{
   stbi_uc *p, *c;
   int idx;

   if (g->codes[code].prefix >= 0)
      stbi__out_gif_code(g, g->codes[code].prefix);

   if (g->cur_y >= g->max_y) return;

   idx = g->cur_x + g->cur_y;
   p = &g->out[idx];
   g->history[idx / 4] = 1;

   c = &g->color_table[g->codes[code].suffix * 4];
   if (c[3] > 128) { // don't render transparent pixels;
      p[0] = c[2];
      p[1] = c[1];
      p[2] = c[0];
      p[3] = c[3];
   }
   g->cur_x += 4;

   if (g->cur_x >= g->max_x) {
      g->cur_x = g->start_x;
      g->cur_y += g->step;

      while (g->cur_y >= g->max_y && g->parse > 0) {
         g->step = (1 << g->parse) * g->line_size;
         g->cur_y = g->start_y + (g->step >> 1);
         --g->parse;
      }
   }
}

static stbi_uc *stbi__process_gif_raster(stbi__context *s, stbi__gif *g)
{
   stbi_uc lzw_cs;
   stbi__int32 len, init_code;
   stbi__uint32 first;
   stbi__int32 codesize, codemask, avail, oldcode, bits, valid_bits, clear;
   stbi__gif_lzw *p;

   lzw_cs = stbi__get8(s);
   if (lzw_cs > 12) return NULL;
   clear = 1 << lzw_cs;
   first = 1;
   codesize = lzw_cs + 1;
   codemask = (1 << codesize) - 1;
   bits = 0;
   valid_bits = 0;
   for (init_code = 0; init_code < clear; init_code++) {
      g->codes[init_code].prefix = -1;
      g->codes[init_code].first = (stbi_uc) init_code;
      g->codes[init_code].suffix = (stbi_uc) init_code;
   }

   avail = clear+2;
   oldcode = -1;

   len = 0;
   for(;;) {
      if (valid_bits < codesize) {
         if (len == 0) {
            len = stbi__get8(s); // start new block
            if (len == 0)
               return g->out;
         }
         --len;
         bits |= (stbi__int32) stbi__get8(s) << valid_bits;
         valid_bits += 8;
      } else {
         stbi__int32 code = bits & codemask;
         bits >>= codesize;
         valid_bits -= codesize;
         if (code == clear) {  // clear code
            codesize = lzw_cs + 1;
            codemask = (1 << codesize) - 1;
            avail = clear + 2;
            oldcode = -1;
            first = 0;
         } else if (code == clear + 1) { // end of stream code
            stbi__skip(s, len);
            while ((len = stbi__get8(s)) > 0)
               stbi__skip(s,len);
            return g->out;
         } else if (code <= avail) {
            if (first) {
               return stbi__errpuc("no clear code", "Corrupt GIF");
            }

            if (oldcode >= 0) {
               p = &g->codes[avail++];
               if (avail > 8192) {
                  return stbi__errpuc("too many codes", "Corrupt GIF");
               }

               p->prefix = (stbi__int16) oldcode;
               p->first = g->codes[oldcode].first;
               p->suffix = (code == avail) ? p->first : g->codes[code].first;
            } else if (code == avail)
               return stbi__errpuc("illegal code in raster", "Corrupt GIF");

            stbi__out_gif_code(g, (stbi__uint16) code);

            if ((avail & codemask) == 0 && avail <= 0x0FFF) {
               codesize++;
               codemask = (1 << codesize) - 1;
            }

            oldcode = code;
         } else {
            return stbi__errpuc("illegal code in raster", "Corrupt GIF");
         }
      }
   }
}

static stbi_uc *stbi__gif_load_next(stbi__context *s, stbi__gif *g, int *comp, int req_comp, stbi_uc *two_back)
{
   int dispose;
   int first_frame;
   int pi;
   int pcount;
   STBI_NOTUSED(req_comp);

   first_frame = 0;
   if (g->out == 0) {
      if (!stbi__gif_header(s, g, comp,0)) return 0; // stbi__g_failure_reason set by stbi__gif_header
      if (!stbi__mad3sizes_valid(4, g->w, g->h, 0))
         return stbi__errpuc("too large", "GIF image is too large");
      pcount = g->w * g->h;
      g->out = (stbi_uc *) stbi__malloc(4 * pcount);
      g->background = (stbi_uc *) stbi__malloc(4 * pcount);
      g->history = (stbi_uc *) stbi__malloc(pcount);
      if (!g->out || !g->background || !g->history)
         return stbi__errpuc("outofmem", "Out of memory");

      memset(g->out, 0x00, 4 * pcount);
      memset(g->background, 0x00, 4 * pcount); // state of the background (starts transparent)
      memset(g->history, 0x00, pcount);        // pixels that were affected previous frame
      first_frame = 1;
   } else {
      dispose = (g->eflags & 0x1C) >> 2;
      pcount = g->w * g->h;

      if ((dispose == 3) && (two_back == 0)) {
         dispose = 2; // if I don't have an image to revert back to, default to the old background
      }

      if (dispose == 3) { // use previous graphic
         for (pi = 0; pi < pcount; ++pi) {
            if (g->history[pi]) {
               memcpy( &g->out[pi * 4], &two_back[pi * 4], 4 );
            }
         }
      } else if (dispose == 2) {
         for (pi = 0; pi < pcount; ++pi) {
            if (g->history[pi]) {
               memcpy( &g->out[pi * 4], &g->background[pi * 4], 4 );
            }
         }
      } else {
      }

      memcpy( g->background, g->out, 4 * g->w * g->h );
   }

   memset( g->history, 0x00, g->w * g->h );        // pixels that were affected previous frame

   for (;;) {
      int tag = stbi__get8(s);
      switch (tag) {
         case 0x2C: /* Image Descriptor */
         {
            stbi__int32 x, y, w, h;
            stbi_uc *o;

            x = stbi__get16le(s);
            y = stbi__get16le(s);
            w = stbi__get16le(s);
            h = stbi__get16le(s);
            if (((x + w) > (g->w)) || ((y + h) > (g->h)))
               return stbi__errpuc("bad Image Descriptor", "Corrupt GIF");

            g->line_size = g->w * 4;
            g->start_x = x * 4;
            g->start_y = y * g->line_size;
            g->max_x   = g->start_x + w * 4;
            g->max_y   = g->start_y + h * g->line_size;
            g->cur_x   = g->start_x;
            g->cur_y   = g->start_y;

            if (w == 0)
               g->cur_y = g->max_y;

            g->lflags = stbi__get8(s);

            if (g->lflags & 0x40) {
               g->step = 8 * g->line_size; // first interlaced spacing
               g->parse = 3;
            } else {
               g->step = g->line_size;
               g->parse = 0;
            }

            if (g->lflags & 0x80) {
               stbi__gif_parse_colortable(s,g->lpal, 2 << (g->lflags & 7), g->eflags & 0x01 ? g->transparent : -1);
               g->color_table = (stbi_uc *) g->lpal;
            } else if (g->flags & 0x80) {
               g->color_table = (stbi_uc *) g->pal;
            } else
               return stbi__errpuc("missing color table", "Corrupt GIF");

            o = stbi__process_gif_raster(s, g);
            if (!o) return NULL;

            pcount = g->w * g->h;
            if (first_frame && (g->bgindex > 0)) {
               for (pi = 0; pi < pcount; ++pi) {
                  if (g->history[pi] == 0) {
                     g->pal[g->bgindex][3] = 255; // just in case it was made transparent, undo that; It will be reset next frame if need be;
                     memcpy( &g->out[pi * 4], &g->pal[g->bgindex], 4 );
                  }
               }
            }

            return o;
         }

         case 0x21: // Comment Extension.
         {
            int len;
            int ext = stbi__get8(s);
            if (ext == 0xF9) { // Graphic Control Extension.
               len = stbi__get8(s);
               if (len == 4) {
                  g->eflags = stbi__get8(s);
                  g->delay = 10 * stbi__get16le(s); // delay - 1/100th of a second, saving as 1/1000ths.

                  if (g->transparent >= 0) {
                     g->pal[g->transparent][3] = 255;
                  }
                  if (g->eflags & 0x01) {
                     g->transparent = stbi__get8(s);
                     if (g->transparent >= 0) {
                        g->pal[g->transparent][3] = 0;
                     }
                  } else {
                     stbi__skip(s, 1);
                     g->transparent = -1;
                  }
               } else {
                  stbi__skip(s, len);
                  break;
               }
            }
            while ((len = stbi__get8(s)) != 0) {
               stbi__skip(s, len);
            }
            break;
         }

         case 0x3B: // gif stream termination code
            return (stbi_uc *) s; // using '1' causes warning on some compilers

         default:
            return stbi__errpuc("unknown code", "Corrupt GIF");
      }
   }
}

static void *stbi__load_gif_main_outofmem(stbi__gif *g, stbi_uc *out, int **delays)
{
   STBI_FREE(g->out);
   STBI_FREE(g->history);
   STBI_FREE(g->background);

   if (out) STBI_FREE(out);
   if (delays && *delays) STBI_FREE(*delays);
   return stbi__errpuc("outofmem", "Out of memory");
}

static void *stbi__load_gif_main(stbi__context *s, int **delays, int *x, int *y, int *z, int *comp, int req_comp)
{
   if (stbi__gif_test(s)) {
      int layers = 0;
      stbi_uc *u = 0;
      stbi_uc *out = 0;
      stbi_uc *two_back = 0;
      stbi__gif g;
      int stride;
      int out_size = 0;
      int delays_size = 0;

      STBI_NOTUSED(out_size);
      STBI_NOTUSED(delays_size);

      memset(&g, 0, sizeof(g));
      if (delays) {
         *delays = 0;
      }

      do {
         u = stbi__gif_load_next(s, &g, comp, req_comp, two_back);
         if (u == (stbi_uc *) s) u = 0;  // end of animated gif marker

         if (u) {
            *x = g.w;
            *y = g.h;
            ++layers;
            stride = g.w * g.h * 4;

            if (out) {
               void *tmp = (stbi_uc*) STBI_REALLOC_SIZED( out, out_size, layers * stride );
               if (!tmp)
                  return stbi__load_gif_main_outofmem(&g, out, delays);
               else {
                   out = (stbi_uc*) tmp;
                   out_size = layers * stride;
               }

               if (delays) {
                  int *new_delays = (int*) STBI_REALLOC_SIZED( *delays, delays_size, sizeof(int) * layers );
                  if (!new_delays)
                     return stbi__load_gif_main_outofmem(&g, out, delays);
                  *delays = new_delays;
                  delays_size = layers * sizeof(int);
               }
            } else {
               out = (stbi_uc*)stbi__malloc( layers * stride );
               if (!out)
                  return stbi__load_gif_main_outofmem(&g, out, delays);
               out_size = layers * stride;
               if (delays) {
                  *delays = (int*) stbi__malloc( layers * sizeof(int) );
                  if (!*delays)
                     return stbi__load_gif_main_outofmem(&g, out, delays);
                  delays_size = layers * sizeof(int);
               }
            }
            memcpy( out + ((layers - 1) * stride), u, stride );
            if (layers >= 2) {
               two_back = out - 2 * stride;
            }

            if (delays) {
               (*delays)[layers - 1U] = g.delay;
            }
         }
      } while (u != 0);

      STBI_FREE(g.out);
      STBI_FREE(g.history);
      STBI_FREE(g.background);

      if (req_comp && req_comp != 4)
         out = stbi__convert_format(out, 4, req_comp, layers * g.w, g.h);

      *z = layers;
      return out;
   } else {
      return stbi__errpuc("not GIF", "Image was not as a gif type.");
   }
}

static void *stbi__gif_load(stbi__context *s, int *x, int *y, int *comp, int req_comp, stbi__result_info *ri)
{
   stbi_uc *u = 0;
   stbi__gif g;
   memset(&g, 0, sizeof(g));
   STBI_NOTUSED(ri);

   u = stbi__gif_load_next(s, &g, comp, req_comp, 0);
   if (u == (stbi_uc *) s) u = 0;  // end of animated gif marker
   if (u) {
      *x = g.w;
      *y = g.h;

      if (req_comp && req_comp != 4)
         u = stbi__convert_format(u, 4, req_comp, g.w, g.h);
   } else if (g.out) {
      STBI_FREE(g.out);
   }

   STBI_FREE(g.history);
   STBI_FREE(g.background);

   return u;
}

static int stbi__gif_info(stbi__context *s, int *x, int *y, int *comp)
{
   return stbi__gif_info_raw(s,x,y,comp);
}




static int stbi__info_main(stbi__context *s, int *x, int *y, int *comp)
{
   if (stbi__jpeg_info(s, x, y, comp)) return 1;

   if (stbi__png_info(s, x, y, comp))  return 1;

   if (stbi__gif_info(s, x, y, comp))  return 1;




   return stbi__err("unknown image type", "Image not of any known type, or corrupt");
}

static int stbi__is_16_main(stbi__context *s)
{
   if (stbi__png_is16(s))  return 1;

   return 0;
}

STBIDEF int stbi_info(char const *filename, int *x, int *y, int *comp)
{
    FILE *f = stbi__fopen(filename, "rb");
    int result;
    if (!f) return stbi__err("can't fopen", "Unable to open file");
    result = stbi_info_from_file(f, x, y, comp);
    fclose(f);
    return result;
}

STBIDEF int stbi_info_from_file(FILE *f, int *x, int *y, int *comp)
{
   int r;
   stbi__context s;
   long pos = ftell(f);
   stbi__start_file(&s, f);
   r = stbi__info_main(&s,x,y,comp);
   fseek(f,pos,SEEK_SET);
   return r;
}

STBIDEF int stbi_is_16_bit(char const *filename)
{
    FILE *f = stbi__fopen(filename, "rb");
    int result;
    if (!f) return stbi__err("can't fopen", "Unable to open file");
    result = stbi_is_16_bit_from_file(f);
    fclose(f);
    return result;
}

STBIDEF int stbi_is_16_bit_from_file(FILE *f)
{
   int r;
   stbi__context s;
   long pos = ftell(f);
   stbi__start_file(&s, f);
   r = stbi__is_16_main(&s);
   fseek(f,pos,SEEK_SET);
   return r;
}

STBIDEF int stbi_info_from_memory(stbi_uc const *buffer, int len, int *x, int *y, int *comp)
{
   stbi__context s;
   stbi__start_mem(&s,buffer,len);
   return stbi__info_main(&s,x,y,comp);
}

STBIDEF int stbi_info_from_callbacks(stbi_io_callbacks const *c, void *user, int *x, int *y, int *comp)
{
   stbi__context s;
   stbi__start_callbacks(&s, (stbi_io_callbacks *) c, user);
   return stbi__info_main(&s,x,y,comp);
}

STBIDEF int stbi_is_16_bit_from_memory(stbi_uc const *buffer, int len)
{
   stbi__context s;
   stbi__start_mem(&s,buffer,len);
   return stbi__is_16_main(&s);
}

STBIDEF int stbi_is_16_bit_from_callbacks(stbi_io_callbacks const *c, void *user)
{
   stbi__context s;
   stbi__start_callbacks(&s, (stbi_io_callbacks *) c, user);
   return stbi__is_16_main(&s);
}

#endif // STB_IMAGE_IMPLEMENTATION



#define STBIR_INCLUDE_STB_IMAGE_RESIZE2_H

#include <stddef.h>
#include <stdint.h>
typedef uint8_t  stbir_uint8;
typedef uint16_t stbir_uint16;
typedef uint32_t stbir_uint32;
typedef uint64_t stbir_uint64;

#define STBIRDEF extern



typedef enum
{
  STBIR_1CHANNEL = 1,
  STBIR_2CHANNEL = 2,
  STBIR_RGB      = 3,               // 3-chan, with order specified (for channel flipping)
  STBIR_BGR      = 0,               // 3-chan, with order specified (for channel flipping)
  STBIR_4CHANNEL = 5,

  STBIR_RGBA = 4,                   // alpha formats, where alpha is NOT premultiplied into color channels
  STBIR_BGRA = 6,
  STBIR_ARGB = 7,
  STBIR_ABGR = 8,
  STBIR_RA   = 9,
  STBIR_AR   = 10,

  STBIR_RGBA_PM = 11,               // alpha formats, where alpha is premultiplied into color channels
  STBIR_BGRA_PM = 12,
  STBIR_ARGB_PM = 13,
  STBIR_ABGR_PM = 14,
  STBIR_RA_PM   = 15,
  STBIR_AR_PM   = 16,

  STBIR_RGBA_NO_AW = 11,            // alpha formats, where NO alpha weighting is applied at all!
  STBIR_BGRA_NO_AW = 12,            //   these are just synonyms for the _PM flags (which also do
  STBIR_ARGB_NO_AW = 13,            //   no alpha weighting). These names just make it more clear
  STBIR_ABGR_NO_AW = 14,            //   for some folks).
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
  STBIR_EDGE_WRAP    = 2,  // this edge mode is slower and uses more memory
  STBIR_EDGE_ZERO    = 3,
} stbir_edge;

typedef enum
{
  STBIR_FILTER_DEFAULT      = 0,  // use same filter type that easy-to-use API chooses
  STBIR_FILTER_BOX          = 1,  // A trapezoid w/1-pixel wide ramps, same result as box for integer scale ratios
  STBIR_FILTER_TRIANGLE     = 2,  // On upsampling, produces same results as bilinear texture filtering
  STBIR_FILTER_CUBICBSPLINE = 3,  // The cubic b-spline (aka Mitchell-Netrevalli with B=1,C=0), gaussian-esque
  STBIR_FILTER_CATMULLROM   = 4,  // An interpolating cubic spline
  STBIR_FILTER_MITCHELL     = 5,  // Mitchell-Netrevalli filter with B=1/3, C=1/3
  STBIR_FILTER_POINT_SAMPLE = 6,  // Simple point sampling
  STBIR_FILTER_OTHER        = 7,  // User callback specified
} stbir_filter;

typedef enum
{
  STBIR_TYPE_UINT8            = 0,
  STBIR_TYPE_UINT8_SRGB       = 1,
  STBIR_TYPE_UINT8_SRGB_ALPHA = 2,  // alpha channel, when present, should also be SRGB (this is very unusual)
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

typedef float stbir__kernel_callback( float x, float scale, void * user_data ); // centered at zero
typedef float stbir__support_callback( float scale, void * user_data );

typedef struct stbir__info stbir__info;

typedef struct STBIR_RESIZE  // use the stbir_resize_init and stbir_override functions to set these values for future compatibility
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
                                 const void *input_pixels,  int input_w,  int input_h, int input_stride_in_bytes, // stride can be zero
                                       void *output_pixels, int output_w, int output_h, int output_stride_in_bytes, // stride can be zero
                                 stbir_pixel_layout pixel_layout, stbir_datatype data_type );


STBIRDEF void stbir_set_datatypes( STBIR_RESIZE * resize, stbir_datatype input_type, stbir_datatype output_type );
STBIRDEF void stbir_set_pixel_callbacks( STBIR_RESIZE * resize, stbir_input_callback * input_cb, stbir_output_callback * output_cb );   // no callbacks by default
STBIRDEF void stbir_set_user_data( STBIR_RESIZE * resize, void * user_data );                                               // pass back STBIR_RESIZE* by default
STBIRDEF void stbir_set_buffer_ptrs( STBIR_RESIZE * resize, const void * input_pixels, int input_stride_in_bytes, void * output_pixels, int output_stride_in_bytes );




STBIRDEF int stbir_set_pixel_layouts( STBIR_RESIZE * resize, stbir_pixel_layout input_pixel_layout, stbir_pixel_layout output_pixel_layout );  // sets new buffer layouts
STBIRDEF int stbir_set_edgemodes( STBIR_RESIZE * resize, stbir_edge horizontal_edge, stbir_edge vertical_edge );       // CLAMP by default

STBIRDEF int stbir_set_filters( STBIR_RESIZE * resize, stbir_filter horizontal_filter, stbir_filter vertical_filter ); // STBIR_DEFAULT_FILTER_UPSAMPLE/DOWNSAMPLE by default
STBIRDEF int stbir_set_filter_callbacks( STBIR_RESIZE * resize, stbir__kernel_callback * horizontal_filter, stbir__support_callback * horizontal_support, stbir__kernel_callback * vertical_filter, stbir__support_callback * vertical_support );

STBIRDEF int stbir_set_pixel_subrect( STBIR_RESIZE * resize, int subx, int suby, int subw, int subh );        // sets both sub-regions (full regions by default)
STBIRDEF int stbir_set_input_subrect( STBIR_RESIZE * resize, double s0, double t0, double s1, double t1 );    // sets input sub-region (full region by default)
STBIRDEF int stbir_set_output_pixel_subrect( STBIR_RESIZE * resize, int subx, int suby, int subw, int subh ); // sets output sub-region (full region by default)

STBIRDEF int stbir_set_non_pm_alpha_speed_over_quality( STBIR_RESIZE * resize, int non_pma_alpha_speed_over_quality );



STBIRDEF int stbir_build_samplers( STBIR_RESIZE * resize );

STBIRDEF void stbir_free_samplers( STBIR_RESIZE * resize );


STBIRDEF int stbir_resize_extended( STBIR_RESIZE * resize );




STBIRDEF int stbir_build_samplers_with_splits( STBIR_RESIZE * resize, int try_splits );




STBIRDEF int stbir_resize_extended_split( STBIR_RESIZE * resize, int split_start, int split_count );












#include <assert.h>
#define STBIR_ASSERT(x) assert(x)

#include <stdlib.h>
#define STBIR_MALLOC(size,user_data) ((void)(user_data), malloc(size))
#define STBIR_FREE(ptr,user_data)    ((void)(user_data), free(ptr))


#define stbir__inline __inline__




#pragma STDC FP_CONTRACT OFF

#define STBIR__UNUSED(v)  (void)sizeof(v)

#define STBIR__ARRAY_SIZE(a) (sizeof((a))/sizeof((a)[0]))


#define STBIR_DEFAULT_FILTER_UPSAMPLE    STBIR_FILTER_CATMULLROM

#define STBIR_DEFAULT_FILTER_DOWNSAMPLE  STBIR_FILTER_MITCHELL


#define STBIR__HEADER_FILENAME "stb_image_resize2.h"

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
  1,1,1,2,4,2 // STBIR_TYPE_UINT8,STBIR_TYPE_UINT8_SRGB,STBIR_TYPE_UINT8_SRGB_ALPHA,STBIR_TYPE_UINT16,STBIR_TYPE_FLOAT,STBIR_TYPE_HALF_FLOAT
};

typedef struct
{
  int n0; // First contributing pixel
  int n1; // Last contributing pixel
} stbir__contributors;

typedef struct
{
  int lowest;    // First sample index for whole filter
  int highest;   // Last sample index for whole filter
  int widest;    // widest single set of samples for an output
} stbir__filter_extent_info;

typedef struct
{
  int n0; // First pixel of decode buffer to write to
  int n1; // Last pixel of decode that will be written to
  int pixel_offset_for_input;  // Pixel offset into input_scanline
} stbir__span;

typedef struct stbir__scale_info
{
  int input_full_size;
  int output_sub_size;
  float scale;
  float inv_scale;
  float pixel_shift; // starting shift in output pixel space (in pixels)
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
  int is_gather;  // 0 = scatter, 1 = gather with scale >= 1, 2 = gather with scale < 1
  int gather_prescatter_num_contributors;
  int gather_prescatter_coefficient_width;
  int gather_prescatter_contributors_size;
  int gather_prescatter_coefficients_size;
} stbir__sampler;

typedef struct
{
  stbir__contributors conservative;
  int edge_sizes[2];    // this can be less than filter_pixel_margin, if the filter and scaling falls off
  stbir__span spans[2]; // can be two spans, if doing input subrect with clamp mode WRAP
} stbir__extents;

typedef struct
{
  float* decode_buffer;

  int ring_buffer_first_scanline;
  int ring_buffer_last_scanline;
  int ring_buffer_begin_index;    // first_scanline is at this index in the ring buffer
  int start_output_y, end_output_y;
  int start_input_y, end_input_y;  // used in scatter only

    float* ring_buffer;  // one big buffer that we index into

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
  int ring_buffer_length_bytes;   // The length of an individual entry in the ring buffer. The total number of ring buffers is stbir__get_filter_pixel_width(filter)
  int ring_buffer_num_entries;    // Total number of entries in the ring buffer.

  stbir_datatype input_type;
  stbir_datatype output_type;

  stbir_input_callback * in_pixels_cb;
  void * user_data;
  stbir_output_callback * out_pixels_cb;

  stbir__extents scanline_extents;

  void * alloced_mem;
  stbir__per_split_info * split_info;  // by default 1, but there will be N of these allocated based on the thread init you did

  stbir__decode_pixels_func * decode_pixels;
  stbir__alpha_weight_func * alpha_weight;
  stbir__horizontal_gather_channels_func * horizontal_gather_channels;
  stbir__alpha_unweight_func * alpha_unweight;
  stbir__encode_pixels_func * encode_pixels;

  int alloc_ring_buffer_num_entries;    // Number of entries in the ring buffer that will be allocated
  int splits; // count of splits

  stbir_internal_pixel_layout input_pixel_layout_internal;
  stbir_internal_pixel_layout output_pixel_layout_internal;

  int input_color_and_type;
  int offset_x, offset_y; // offset within output_data
  int vertical_first;
  int channels;
  int effective_channels; // same as channels, except on RGBA/ARGB (7), or XA/AX (3)
  size_t alloced_total;
};


#define stbir__max_uint8_as_float             255.0f
#define stbir__max_uint16_as_float            65535.0f
#define stbir__max_uint8_as_float_inverted    3.9215689e-03f     // (1.0f/255.0f)
#define stbir__max_uint16_as_float_inverted   1.5259022e-05f     // (1.0f/65535.0f)
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
  static const stbir__FP32 almostone = { 0x3f7fffff }; // 1-eps
  static const stbir__FP32 minval = { (127-13) << 23 };
  stbir_uint32 tab,bias,scale,t;
  stbir__FP32 f;

  if (!(in > minval.f)) // written this way to catch NaNs
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

#define STBIR_FORCE_GATHER_FILTER_SCANLINES_AMOUNT 32 // when downsampling and <= 32 scanlines of buffering, use gather. gather used down to 1/8th scaling for 25% win.

#define STBIR_FORCE_MINIMUM_SCANLINES_FOR_SPLITS 4 // when threading, what is the minimum number of scanlines for a split?

#define STBIR_INPUT_CALLBACK_PADDING 3







  #define STBIR_STREAMOUT_PTR( star ) star
  #define STBIR_NO_UNROLL( ptr )
  #define STBIR_NO_UNROLL_LOOP_START

#define STBIR_NO_UNROLL_LOOP_START_INF_FOR STBIR_NO_UNROLL_LOOP_START




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



  typedef union stbir__FP16
  {
    unsigned short u;
  } stbir__FP16;







#define STBIR_SIMD_STREAMOUT_PTR( star ) STBIR_STREAMOUT_PTR( star )
#define STBIR_SIMD_NO_UNROLL(ptr)
#define STBIR_SIMD_NO_UNROLL_LOOP_START
#define STBIR_SIMD_NO_UNROLL_LOOP_START_INF_FOR




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


#include <math.h>
#define STBIR_CEILF(x) ceilf(x)
#define STBIR_FLOORF(x) floorf(x)

#include <string.h>
#define STBIR_MEMCPY( dest, src, len ) memcpy( dest, src, len )


static void stbir_overlapping_memcpy( void * dest, void const * src, size_t bytes )
{
  char STBIR_SIMD_STREAMOUT_PTR (*) sd = (char*) src;
  char STBIR_SIMD_STREAMOUT_PTR( * ) s_end = ((char*) src) + bytes;
  ptrdiff_t ofs_to_dest = (char*)dest - (char*)src;

  if ( ofs_to_dest >= 8 ) // is the overlap more than 8 away?
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

  if ( scale >= ( 1.0f-stbir__small_float ) ) // upscale
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
  return 0; // NOTREACHED
}

static int stbir__edge_clamp_full( int n, int max )
{
  if (n < 0)
    return 0;

  if (n >= max)
    return max - 1;

  return n; // NOTREACHED
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

  return n; // NOTREACHED
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
  stbir__edge_clamp_full,    // STBIR_EDGE_CLAMP
  stbir__edge_reflect_full,  // STBIR_EDGE_REFLECT
  stbir__edge_wrap_full,     // STBIR_EDGE_WRAP
  stbir__edge_zero_full,     // STBIR_EDGE_ZERO
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
      stop = j + filter_pixel_margin;  // if we find a new min, only scan another filter width
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
      stop = j - filter_pixel_margin;  // if we find a new max, only scan another filter width
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
    scanline_extents->edge_sizes[0] = 0;  // don't need to copy the left margin, since we are directly decoding into the margin
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
    scanline_extents->edge_sizes[1] = 0;  // don't need to copy the right margin, since we are directly decoding into the margin
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
  if ( last < first ) last = first; // point sample mode can span a value *right* at 0.5, and cause these to cross

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
        if ( i == 0 )  // if we're at the front, just eat zero contributors
        {
          STBIR_ASSERT ( ( in_last_pixel - in_first_pixel ) != 0 ); // there should be at least one contrib
          ++in_first_pixel;
          i--;
          continue;
        }
        coeff = 0;  // make sure is fully zero (should keep denormals away)
      }
      else
        last_non_zero = i;

      coefficient_group[i] = coeff;
    }

    in_last_pixel = last_non_zero+in_first_pixel; // kills trailing zeros
    contributors->n0 = in_first_pixel;
    contributors->n1 = in_last_pixel;

    STBIR_ASSERT(contributors->n1 >= contributors->n0);

    ++contributors;
    coefficient_group += coefficient_width;
  }
}

static void stbir__insert_coeff( stbir__contributors * contribs, float * coeffs, int new_pixel, float new_coeff, int max_width )
{
  if ( new_pixel <= contribs->n1 )  // before the end
  {
    if ( new_pixel < contribs->n0 ) // before the front?
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
      for( j = ( contribs->n1 - contribs->n0 ) + 1 ; j < e ; j++ ) // clear in-betweens coeffs if there are any
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
          STBIR_ASSERT( out == ( first_out_inited + 1 ) ); // ensure we have only advanced one at time
          first_out_inited = out;
          contribs->n0 = in_pixel;
          contribs->n1 = in_pixel;
          coeffs[0]  = coeff;
        }
        else
        {
          if ( coeffs[0] == 0.0f )  // if the first coefficent is zero, then zap it for this coeffs
          {
            STBIR_ASSERT( ( in_pixel - contribs->n0 ) == 1 ); // ensure that when we zap, we're at the 2nd pos
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

#define STBIR_RENORM_TYPE double

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
      STBIR_ASSERT( ( coeffs[i] >= -2.0f ) && ( coeffs[i] <= 2.0f )  ); // check for wonky weights
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
        save_n0_coeff = c[0]; // save it, since we didn't do the final one (i==n0), because there might be too many coeffs to hold (before we resize)!

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
  #define STBIR_MOVE_4( dest, src ) { STBIR_NO_UNROLL(dest); ((stbir_uint64*)(dest))[0] = ((stbir_uint64*)(src))[0]; ((stbir_uint64*)(dest))[1] = ((stbir_uint64*)(src))[1]; }

  int row_end = row1 + 1;
  STBIR__UNUSED( row0 ); // only used in an assert

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
    case 1: // gather upsample
    {
      float out_pixels_radius = support(inv_scale,user_data) * scale;

      stbir__calculate_coefficients_for_gather_upsample( out_pixels_radius, kernel, &samp->scale_info, gather_num_contributors, gather_contributors, gather_coeffs, gather_coefficient_width, samp->edge, user_data );

      STBIR_PROFILE_BUILD_START( cleanup );
      stbir__cleanup_gathered_coefficients( samp->edge, &samp->extent_info, &samp->scale_info, gather_num_contributors, gather_contributors, gather_coeffs, gather_coefficient_width );
      STBIR_PROFILE_BUILD_END( cleanup );
    }
    break;

    case 0: // scatter downsample (only on vertical)
    case 2: // gather downsample
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
  float const * end_decode = out_buffer + ( width_times_channels / 4 ) * 7;  // decode buffer aligned to end of out_buffer
  float STBIR_STREAMOUT_PTR(*) decode = (float*)end_decode - width_times_channels;



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


  while( decode < end_decode )
  {
    float alpha = decode[3];
    decode[0] *= alpha;
    decode[1] *= alpha;
    decode[2] *= alpha;
    decode += 4;
  }

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

    if ( alpha >= stbir__small_float )
    {
      float ialpha = 1.0f / alpha;
      encode[0] *= ialpha;
      encode[1] *= ialpha;
      encode[2] *= ialpha;
    }
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

    start_x[0] = -stbir_info->scanline_extents.edge_sizes[0];  // left edge start x
    start_x[1] =  input_full_size;                             // right edge

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


#define STBIR__horizontal_channels 1
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME




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


#define STBIR__horizontal_channels 2
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME




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


#define STBIR__horizontal_channels 3
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME



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


#define STBIR__horizontal_channels 4
#define STB_IMAGE_RESIZE_DO_HORIZONTALS
#include STBIR__HEADER_FILENAME





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

    return (float*) ( ( (char*) split_info->ring_buffer ) + ( index * stbir_info->ring_buffer_length_bytes ) );
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
    decode_buffer[ width_times_channels ] = 0.0f; // clear two over for horizontals with a remnant of 3
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
  split_info->ring_buffer_last_scanline = split_info->ring_buffer_first_scanline - 1; // means "empty"

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
        if ( ( i ) && ( STBIR__FLOAT_BUFFER_IS_EMPTY( outputs[i] ) != STBIR__FLOAT_BUFFER_IS_EMPTY( outputs[0] ) ) ) // make sure runs are of the same type
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
  end_output_y = split_info[split_count-1].end_output_y;  // may do multiple split counts

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
    decode_buffer[ width_times_channels ] = 0.0f; // clear two over for horizontals with a remnant of 3
    decode_buffer[ width_times_channels+1 ] = 0.0f; 
    decode_buffer[0] = STBIR__FLOAT_EMPTY_MARKER; // only used on scatter
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
    filter = STBIR_DEFAULT_FILTER_DOWNSAMPLE; // default to downsample
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
  samp->coefficients_size = samp->num_contributors * samp->coefficient_width * sizeof(float) + sizeof(float)*STBIR_INPUT_CALLBACK_PADDING; // extra sizeof(float) is padding

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
  else if ( samp->is_gather == 2 ) // downsample gather, refine
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

static void stbir__get_split_info( stbir__per_split_info* split_info, int splits, int output_height, int vertical_pixel_margin, int input_full_height )
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
    STBIR__FREE_AND_CLEAR( info->alloced_mem );
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

static float stbir__compute_weights[5][STBIR_RESIZE_CLASSIFICATIONS][4]=  // 5 = 0=1chan, 1=2chan, 2=3chan, 3=4chan, 4=7chan
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
  int control_v_first; // 0 = no control, 1 = force hori, 2 = force vert
  int v_first;
  int v_resize_classification;
  int is_gather;
} STBIR__V_FIRST_INFO;

#define STBIR__V_FIRST_INFO_POINTER 0


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
  1,2,3,3,4,   // 1ch, 2ch, rgb, bgr, 4ch
  4,4,4,4,2,2, // RGBA,BGRA,ARGB,ABGR,RA,AR
  4,4,4,4,2,2, // RGBA_PM,BGRA_PM,ARGB_PM,ABGR_PM,RA_PM,AR_PM
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

  int alpha_weighting_type = 0; // 0=none, 1=simple, 2=fancy
  int conservative_split_output_size = stbir__get_max_split( splits, vertical->scale_info.output_sub_size );
  stbir_internal_pixel_layout input_pixel_layout = stbir__pixel_layout_convert_public_to_internal[ input_pixel_layout_public ];
  stbir_internal_pixel_layout output_pixel_layout = stbir__pixel_layout_convert_public_to_internal[ output_pixel_layout_public ];
  int channels = stbir__pixel_channels[ input_pixel_layout ];
  int effective_channels = channels;

  if ( ( horizontal->filter_enum != STBIR_FILTER_POINT_SAMPLE ) || ( vertical->filter_enum != STBIR_FILTER_POINT_SAMPLE ) ) // no alpha weighting on point sampling
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

  decode_buffer_size = ( conservative->n1 - conservative->n0 + 1 ) * effective_channels * sizeof(float) + sizeof(float)*STBIR_INPUT_CALLBACK_PADDING; // extra floats for input callback stagger


  ring_buffer_length_bytes = (size_t)horizontal->scale_info.output_sub_size * (size_t)effective_channels * sizeof(float) + sizeof(float)*STBIR_INPUT_CALLBACK_PADDING; // extra floats for padding

  if ( vertical_first )
    ring_buffer_length_bytes = ( decode_buffer_size + 15 ) & ~15;

  if ( ( ring_buffer_length_bytes & 4095 ) == 0 ) ring_buffer_length_bytes += 64*3; // avoid 4k alias

  alloc_ring_buffer_num_entries = vertical->filter_pixel_width + 1;

  if ( ( !vertical->is_gather ) && ( alloc_ring_buffer_num_entries > conservative_split_output_size ) )
    alloc_ring_buffer_num_entries = conservative_split_output_size;

  ring_buffer_size = (size_t)alloc_ring_buffer_num_entries * (size_t)ring_buffer_length_bytes;

  vertical_buffer_size = (size_t)horizontal->scale_info.output_sub_size * (size_t)effective_channels * sizeof(float) + sizeof(float);  // extra float for padding

  for(;;)
  {
    int i;
    void * advance_mem = alloced;
    int copy_horizontal = 0;
    stbir__sampler * possibly_use_horizontal_for_pivot = 0;

    #define STBIR__NEXT_PTR( ptr, size, ntype ) advance_mem = (void*) ( ( ((size_t)advance_mem) + 15 ) & ~15 ); if ( alloced ) ptr = (ntype*)advance_mem; advance_mem = (char*)(((size_t)advance_mem) + (size));

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

      STBIR__NEXT_PTR( info->split_info[i].ring_buffer, ring_buffer_size, float );
      STBIR__NEXT_PTR( info->split_info[i].vertical_buffer, vertical_buffer_size, float );
    }

    if ( vertical->is_gather == 0 )
    {
      size_t both;
      size_t temp_mem_amt;


      both = (size_t)vertical->gather_prescatter_contributors_size + (size_t)vertical->gather_prescatter_coefficients_size;

      temp_mem_amt = (size_t)( decode_buffer_size + ring_buffer_size + vertical_buffer_size ) * (size_t)splits;
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

      stbir__get_split_info( info->split_info, info->splits, info->vertical.scale_info.output_sub_size, info->vertical.filter_pixel_margin, info->vertical.scale_info.input_full_size );

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
      return info;  // success
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
    /* 1ch-4ch */ stbir__decode_uint8_srgb, stbir__decode_uint8_srgb, 0, stbir__decode_float_linear, stbir__decode_half_float_linear,
  };

  static stbir__decode_pixels_func * decode_alphas[STBIRI_AR-STBIRI_RGBA+1][STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]=
  {
    { /* RGBA */ stbir__decode_uint8_srgb4_linearalpha,      stbir__decode_uint8_srgb,      0, stbir__decode_float_linear,      stbir__decode_half_float_linear },
    { /* BGRA */ stbir__decode_uint8_srgb4_linearalpha_BGRA, stbir__decode_uint8_srgb_BGRA, 0, stbir__decode_float_linear_BGRA, stbir__decode_half_float_linear_BGRA },
    { /* ARGB */ stbir__decode_uint8_srgb4_linearalpha_ARGB, stbir__decode_uint8_srgb_ARGB, 0, stbir__decode_float_linear_ARGB, stbir__decode_half_float_linear_ARGB },
    { /* ABGR */ stbir__decode_uint8_srgb4_linearalpha_ABGR, stbir__decode_uint8_srgb_ABGR, 0, stbir__decode_float_linear_ABGR, stbir__decode_half_float_linear_ABGR },
    { /* RA   */ stbir__decode_uint8_srgb2_linearalpha,      stbir__decode_uint8_srgb,      0, stbir__decode_float_linear,      stbir__decode_half_float_linear },
    { /* AR   */ stbir__decode_uint8_srgb2_linearalpha_AR,   stbir__decode_uint8_srgb_AR,   0, stbir__decode_float_linear_AR,   stbir__decode_half_float_linear_AR },
  };

  static stbir__decode_pixels_func * decode_simple_scaled_or_not[2][2]=
  {
    { stbir__decode_uint8_linear_scaled,  stbir__decode_uint8_linear }, { stbir__decode_uint16_linear_scaled, stbir__decode_uint16_linear },
  };

  static stbir__decode_pixels_func * decode_alphas_scaled_or_not[STBIRI_AR-STBIRI_RGBA+1][2][2]=
  {
    { /* RGBA */ { stbir__decode_uint8_linear_scaled,       stbir__decode_uint8_linear },      { stbir__decode_uint16_linear_scaled,      stbir__decode_uint16_linear } },
    { /* BGRA */ { stbir__decode_uint8_linear_scaled_BGRA,  stbir__decode_uint8_linear_BGRA }, { stbir__decode_uint16_linear_scaled_BGRA, stbir__decode_uint16_linear_BGRA } },
    { /* ARGB */ { stbir__decode_uint8_linear_scaled_ARGB,  stbir__decode_uint8_linear_ARGB }, { stbir__decode_uint16_linear_scaled_ARGB, stbir__decode_uint16_linear_ARGB } },
    { /* ABGR */ { stbir__decode_uint8_linear_scaled_ABGR,  stbir__decode_uint8_linear_ABGR }, { stbir__decode_uint16_linear_scaled_ABGR, stbir__decode_uint16_linear_ABGR } },
    { /* RA   */ { stbir__decode_uint8_linear_scaled,       stbir__decode_uint8_linear },      { stbir__decode_uint16_linear_scaled,      stbir__decode_uint16_linear } },
    { /* AR   */ { stbir__decode_uint8_linear_scaled_AR,    stbir__decode_uint8_linear_AR },   { stbir__decode_uint16_linear_scaled_AR,   stbir__decode_uint16_linear_AR } }
  };

  static stbir__encode_pixels_func * encode_simple[STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]=
  {
    /* 1ch-4ch */ stbir__encode_uint8_srgb, stbir__encode_uint8_srgb, 0, stbir__encode_float_linear, stbir__encode_half_float_linear,
  };

  static stbir__encode_pixels_func * encode_alphas[STBIRI_AR-STBIRI_RGBA+1][STBIR_TYPE_HALF_FLOAT-STBIR_TYPE_UINT8_SRGB+1]=
  {
    { /* RGBA */ stbir__encode_uint8_srgb4_linearalpha,      stbir__encode_uint8_srgb,      0, stbir__encode_float_linear,      stbir__encode_half_float_linear },
    { /* BGRA */ stbir__encode_uint8_srgb4_linearalpha_BGRA, stbir__encode_uint8_srgb_BGRA, 0, stbir__encode_float_linear_BGRA, stbir__encode_half_float_linear_BGRA },
    { /* ARGB */ stbir__encode_uint8_srgb4_linearalpha_ARGB, stbir__encode_uint8_srgb_ARGB, 0, stbir__encode_float_linear_ARGB, stbir__encode_half_float_linear_ARGB },
    { /* ABGR */ stbir__encode_uint8_srgb4_linearalpha_ABGR, stbir__encode_uint8_srgb_ABGR, 0, stbir__encode_float_linear_ABGR, stbir__encode_half_float_linear_ABGR },
    { /* RA   */ stbir__encode_uint8_srgb2_linearalpha,      stbir__encode_uint8_srgb,      0, stbir__encode_float_linear,      stbir__encode_half_float_linear },
    { /* AR   */ stbir__encode_uint8_srgb2_linearalpha_AR,   stbir__encode_uint8_srgb_AR,   0, stbir__encode_float_linear_AR,   stbir__encode_half_float_linear_AR }
  };

  static stbir__encode_pixels_func * encode_simple_scaled_or_not[2][2]=
  {
    { stbir__encode_uint8_linear_scaled,  stbir__encode_uint8_linear }, { stbir__encode_uint16_linear_scaled, stbir__encode_uint16_linear },
  };

  static stbir__encode_pixels_func * encode_alphas_scaled_or_not[STBIRI_AR-STBIRI_RGBA+1][2][2]=
  {
    { /* RGBA */ { stbir__encode_uint8_linear_scaled,       stbir__encode_uint8_linear },       { stbir__encode_uint16_linear_scaled,      stbir__encode_uint16_linear } },
    { /* BGRA */ { stbir__encode_uint8_linear_scaled_BGRA,  stbir__encode_uint8_linear_BGRA },  { stbir__encode_uint16_linear_scaled_BGRA, stbir__encode_uint16_linear_BGRA } },
    { /* ARGB */ { stbir__encode_uint8_linear_scaled_ARGB,  stbir__encode_uint8_linear_ARGB },  { stbir__encode_uint16_linear_scaled_ARGB, stbir__encode_uint16_linear_ARGB } },
    { /* ABGR */ { stbir__encode_uint8_linear_scaled_ABGR,  stbir__encode_uint8_linear_ABGR },  { stbir__encode_uint16_linear_scaled_ABGR, stbir__encode_uint16_linear_ABGR } },
    { /* RA   */ { stbir__encode_uint8_linear_scaled,       stbir__encode_uint8_linear },       { stbir__encode_uint16_linear_scaled,      stbir__encode_uint16_linear } },
    { /* AR   */ { stbir__encode_uint8_linear_scaled_AR,    stbir__encode_uint8_linear_AR },    { stbir__encode_uint16_linear_scaled_AR,   stbir__encode_uint16_linear_AR } }
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

    if ( ( !info->alpha_weight ) && ( !info->alpha_unweight )  ) // don't short circuit when alpha weighting (get everything to 0-1.0 as usual)
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

    if ( ( !info->alpha_weight ) && ( !info->alpha_unweight ) ) // don't short circuit when alpha weighting (get everything to 0-1.0 as usual)
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
    per = ( (double)*outx ) / ( (double)*outsubw ); // is negative
    adj = per * ( *u1 - *u0 );
    *u0 -= adj; // increases u0
    *outx = 0;
  }

  over = outw - ( *outx + *outsubw );
  if ( over < 0 )
  {
    per = ( (double)over ) / ( (double)*outsubw ); // is negative
    adj = per * ( *u1 - *u0 );
    *u1 += adj; // decrease u1
    *outsubw = outw - *outx;
  }
}

static int stbir__double_to_rational(double f, stbir_uint32 limit, stbir_uint32 *numer, stbir_uint32 *denom, int limit_denom ) // limit_denom (1) or limit numer (0)
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
                                 const void *input_pixels,  int input_w,  int input_h, int input_stride_in_bytes, // stride can be zero
                                       void *output_pixels, int output_w, int output_h, int output_stride_in_bytes, // stride can be zero
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

STBIRDEF void stbir_set_datatypes( STBIR_RESIZE * resize, stbir_datatype input_type, stbir_datatype output_type )  // by default, datatype from resize_init
{
  resize->input_data_type = input_type;
  resize->output_data_type = output_type;
  if ( ( resize->samplers ) && ( !resize->needs_rebuild ) )
    stbir__update_info_from_resize( resize->samplers, resize );
}

STBIRDEF void stbir_set_pixel_callbacks( STBIR_RESIZE * resize, stbir_input_callback * input_cb, stbir_output_callback * output_cb )   // no callbacks by default
{
  resize->input_cb = input_cb;
  resize->output_cb = output_cb;

  if ( ( resize->samplers ) && ( !resize->needs_rebuild ) )
  {
    resize->samplers->in_pixels_cb = input_cb;
    resize->samplers->out_pixels_cb = output_cb;
  }
}

STBIRDEF void stbir_set_user_data( STBIR_RESIZE * resize, void * user_data )                                     // pass back STBIR_RESIZE* by default
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


STBIRDEF int stbir_set_edgemodes( STBIR_RESIZE * resize, stbir_edge horizontal_edge, stbir_edge vertical_edge )       // CLAMP by default
{
  resize->horizontal_edge = horizontal_edge;
  resize->vertical_edge = vertical_edge;
  resize->needs_rebuild = 1;
  return 1;
}

STBIRDEF int stbir_set_filters( STBIR_RESIZE * resize, stbir_filter horizontal_filter, stbir_filter vertical_filter ) // STBIR_DEFAULT_FILTER_UPSAMPLE/DOWNSAMPLE by default
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

STBIRDEF int stbir_set_pixel_layouts( STBIR_RESIZE * resize, stbir_pixel_layout input_pixel_layout, stbir_pixel_layout output_pixel_layout )   // sets new pixel layouts
{
  resize->input_pixel_layout_public = input_pixel_layout;
  resize->output_pixel_layout_public = output_pixel_layout;
  resize->needs_rebuild = 1;
  return 1;
}


STBIRDEF int stbir_set_non_pm_alpha_speed_over_quality( STBIR_RESIZE * resize, int non_pma_alpha_speed_over_quality )   // sets alpha speed
{
  resize->fast_alpha = non_pma_alpha_speed_over_quality;
  resize->needs_rebuild = 1;
  return 1;
}

STBIRDEF int stbir_set_input_subrect( STBIR_RESIZE * resize, double s0, double t0, double s1, double t1 )                 // sets input region (full region by default)
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

STBIRDEF int stbir_set_output_pixel_subrect( STBIR_RESIZE * resize, int subx, int suby, int subw, int subh )          // sets input region (full region by default)
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

STBIRDEF int stbir_set_pixel_subrect( STBIR_RESIZE * resize, int subx, int suby, int subw, int subh )                 // sets both regions (full regions by default)
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
  stbir__set_sampler(&vertical, resize->vertical_filter, resize->horizontal_filter_kernel, resize->vertical_filter_support, resize->vertical_edge, &vertical.scale_info, 0, resize->user_data );

  if ( ( vertical.scale_info.output_sub_size / splits ) < STBIR_FORCE_MINIMUM_SCANLINES_FOR_SPLITS ) // each split should be a minimum of 4 scanlines (handwavey choice)
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
    int alloc_state = resize->called_alloc;  // remember allocated state

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

static int stbir__check_output_stuff( void ** ret_ptr, int * ret_pitch, void * output_pixels, int type_size, int output_w, int output_h, int output_stride_in_bytes, stbir_internal_pixel_layout pixel_layout )
{
  size_t size;
  int pitch;
  void * ptr;

  pitch = output_w * type_size * stbir__pixel_channels[ pixel_layout ];
  if ( pitch == 0 )
    return 0;

  if ( output_stride_in_bytes == 0 )
    output_stride_in_bytes = pitch;

  if ( output_stride_in_bytes < pitch )
    return 0;

  size = (size_t)output_stride_in_bytes * (size_t)output_h;
  if ( size == 0 )
    return 0;

  *ret_ptr = 0;
  *ret_pitch = output_stride_in_bytes;

  if ( output_pixels == 0 )
  {
    ptr = STBIR_MALLOC( size, 0 );
    if ( ptr == 0 )
      return 0;

    *ret_ptr = ptr;
    *ret_pitch = pitch;
  }

  return 1;
}


STBIRDEF unsigned char * stbir_resize_uint8_linear( const unsigned char *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                          unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                          stbir_pixel_layout pixel_layout )
{
  STBIR_RESIZE resize;
  unsigned char * optr;
  int opitch;

  if ( !stbir__check_output_stuff( (void**)&optr, &opitch, output_pixels, sizeof( unsigned char ), output_w, output_h, output_stride_in_bytes, stbir__pixel_layout_convert_public_to_internal[ pixel_layout ] ) )
    return 0;

  stbir_resize_init( &resize,
                     input_pixels,  input_w,  input_h,  input_stride_in_bytes,
                     (optr) ? optr : output_pixels, output_w, output_h, opitch,
                     pixel_layout, STBIR_TYPE_UINT8 );

  if ( !stbir_resize_extended( &resize ) )
  {
    if ( optr )
      STBIR_FREE( optr, 0 );
    return 0;
  }

  return (optr) ? optr : output_pixels;
}

STBIRDEF unsigned char * stbir_resize_uint8_srgb( const unsigned char *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                        unsigned char *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                        stbir_pixel_layout pixel_layout )
{
  STBIR_RESIZE resize;
  unsigned char * optr;
  int opitch;

  if ( !stbir__check_output_stuff( (void**)&optr, &opitch, output_pixels, sizeof( unsigned char ), output_w, output_h, output_stride_in_bytes, stbir__pixel_layout_convert_public_to_internal[ pixel_layout ] ) )
    return 0;

  stbir_resize_init( &resize,
                     input_pixels,  input_w,  input_h,  input_stride_in_bytes,
                     (optr) ? optr : output_pixels, output_w, output_h, opitch,
                     pixel_layout, STBIR_TYPE_UINT8_SRGB );

  if ( !stbir_resize_extended( &resize ) )
  {
    if ( optr )
      STBIR_FREE( optr, 0 );
    return 0;
  }

  return (optr) ? optr : output_pixels;
}


STBIRDEF float * stbir_resize_float_linear( const float *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                                  float *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                                                  stbir_pixel_layout pixel_layout )
{
  STBIR_RESIZE resize;
  float * optr;
  int opitch;

  if ( !stbir__check_output_stuff( (void**)&optr, &opitch, output_pixels, sizeof( float ), output_w, output_h, output_stride_in_bytes, stbir__pixel_layout_convert_public_to_internal[ pixel_layout ] ) )
    return 0;

  stbir_resize_init( &resize,
                     input_pixels,  input_w,  input_h,  input_stride_in_bytes,
                     (optr) ? optr : output_pixels, output_w, output_h, opitch,
                     pixel_layout, STBIR_TYPE_FLOAT );

  if ( !stbir_resize_extended( &resize ) )
  {
    if ( optr )
      STBIR_FREE( optr, 0 );
    return 0;
  }

  return (optr) ? optr : output_pixels;
}


STBIRDEF void * stbir_resize( const void *input_pixels , int input_w , int input_h, int input_stride_in_bytes,
                                    void *output_pixels, int output_w, int output_h, int output_stride_in_bytes,
                              stbir_pixel_layout pixel_layout, stbir_datatype data_type,
                              stbir_edge edge, stbir_filter filter )
{
  STBIR_RESIZE resize;
  float * optr;
  int opitch;

  if ( !stbir__check_output_stuff( (void**)&optr, &opitch, output_pixels, stbir__type_size[data_type], output_w, output_h, output_stride_in_bytes, stbir__pixel_layout_convert_public_to_internal[ pixel_layout ] ) )
    return 0;

  stbir_resize_init( &resize,
                     input_pixels,  input_w,  input_h,  input_stride_in_bytes,
                     (optr) ? optr : output_pixels, output_w, output_h, output_stride_in_bytes,
                     pixel_layout, data_type );

  resize.horizontal_edge = edge;
  resize.vertical_edge = edge;
  resize.horizontal_filter = filter;
  resize.vertical_filter = filter;

  if ( !stbir_resize_extended( &resize ) )
  {
    if ( optr )
      STBIR_FREE( optr, 0 );
    return 0;
  }

  return (optr) ? optr : output_pixels;
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



/*
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
*/
