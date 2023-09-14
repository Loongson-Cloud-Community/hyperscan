/*
 * Copyright (c) 2015-2017, Intel Corporation
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of Intel Corporation nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/** \file
 * \brief SIMD types and primitive operations.
 */

#ifndef SIMD_LSX
#define SIMD_LSX

#include "config.h"
#include "ue2common.h"
#include "simd_types.h"
#include "unaligned.h"
#include "util/arch.h"
#include "util/intrinsics.h"
#include <lsxintrin.h>

#include <string.h> // for memcpy

// Define a common assume_aligned using an appropriate compiler built-in, if
// it's available. Note that we need to handle C or C++ compilation.
#ifdef __cplusplus
#ifdef HAVE_CXX_BUILTIN_ASSUME_ALIGNED
#define assume_aligned(x, y) __builtin_assume_aligned((x), (y))
#endif
#else
#ifdef HAVE_CC_BUILTIN_ASSUME_ALIGNED
#define assume_aligned(x, y) __builtin_assume_aligned((x), (y))
#endif
#endif

// Fallback to identity case.
#ifndef assume_aligned
#define assume_aligned(x, y) (x)
#endif

#ifdef __cplusplus
extern "C" {
#endif
extern const char vbs_mask_data[];
#ifdef __cplusplus
}
#endif

static really_inline m128 ones128(void) {
    /* gcc gets this right */
   return  __lsx_vldi(0xFF);
}

static really_inline m128 zeroes128(void) {
   return  __lsx_vldi(0);
}

/** \brief Bitwise not for m128*/
static really_inline m128 not128(m128 a) {
    return  __lsx_vxor_v(a,ones128());
}

/** \brief Return 1 if a and b are different otherwise 0 */
static really_inline int diff128(m128 a, m128 b) {
    return (__lsx_vpickve2gr_hu(__lsx_vmskltz_b(__lsx_vseq_b(a, b)), 0) ^ 0xffff);
}

static really_inline int isnonzero128(m128 a) {
    return !!diff128(a, zeroes128());
}

/**
 * "Rich" version of diff128(). Takes two vectors a and b and returns a 4-bit
 * mask indicating which 32-bit words contain differences.
 */
static really_inline u32 diffrich128(m128 a, m128 b) {
     a = __lsx_vseq_w(a, b);
     return ~( __lsx_vpickve2gr_hu(__lsx_vmskltz_w(a),0)) & 0xf;
}
/**
 * "Rich" version of diff128(), 64-bit variant. Takes two vectors a and b and
 * returns a 4-bit mask indicating which 64-bit words contain differences.
 */
static really_inline u32 diffrich64_128(m128 a, m128 b) {
    u32 d = diffrich128(a, b);
    return (d | (d >> 1)) & 0x5;
}

static really_really_inline
m128 lshift64_m128(m128 a, unsigned b) {
    m128 tmp = __lsx_vinsgr2vr_w(zeroes128(), b, 0);

    m128 x = __lsx_vinsgr2vr_w(tmp, b, 2);
    return  __lsx_vsll_d(a, x);
}

#define rshift64_m128(a, b) __lsx_vsrli_d((a), (b))
#define eq128(a, b)      __lsx_vseq_b((a), (b))
#define movemask128(a)  __lsx_vpickve2gr_hu(__lsx_vmskltz_b(a), 0)

static really_inline m128 set16x8(u8 c) {
    return __lsx_vreplgr2vr_b(c);
}

static really_inline m128 set4x32(u32 c) {
    return  __lsx_vreplgr2vr_w(c);
}

static really_inline u32 movd(const m128 in) {
    return __lsx_vpickve2gr_w(in, 0);
}

static really_inline u64a movq(const m128 in) {
    u32 lo = movd(in);
    u32 hi = movd(__lsx_vsrli_d(in, 32));
    return (u64a)hi << 32 | lo;
}

/* another form of movq */
static really_inline
m128 load_m128_from_u64a(const u64a *p) {
    return __lsx_vinsgr2vr_d(__lsx_vreplgr2vr_d(0LL), *p, 0);
}

#define L(a)  __lsx_vand_v(a, __lsx_vinsgr2vr_d(__lsx_vldi(0), 0xFFFFFFFFFFFFFFFF, 0))
#define M(a) __lsx_vand_v(a, __lsx_vinsgr2vr_d(__lsx_vldi(0xFF), 0x0000000000000000, 0))
#define N(a) __lsx_vpickod_d(__lsx_vldi(0),a)
#define U(a) __lsx_vpickev_d(a, __lsx_vldi(0))
#define rshiftbyte_m128(a, count_immed) \
        (((count_immed) < 8) ? (__lsx_vor_v(__lsx_vsrli_d(M(a), (8*count_immed)), __lsx_vor_v(__lsx_vsrli_d(L(a), (8*count_immed)), __lsx_vslli_d(N(a), (64-(8*count_immed)))))) : (__lsx_vsrli_d(N(a),((8*count_immed)-64))))

#define lshiftbyte_m128(a, count_immed) \
        (((count_immed) < 8) ? (__lsx_vor_v(__lsx_vslli_d(L(a), (8*count_immed)), __lsx_vor_v(__lsx_vslli_d(M(a), (8*count_immed)), __lsx_vsrli_d(U(a), (64-(8*count_immed)))))) : (__lsx_vslli_d(U(a),((8*count_immed)-64))))

#define extract32from128(a, imm) \
        (((imm) < 2) ? (__lsx_vor_v(__lsx_vsrli_d(M(a), (32*imm)), __lsx_vor_v(__lsx_vsrli_d(L(a), (32*imm)), __lsx_vslli_d(N(a), (64-(32*imm)))))) : (__lsx_vsrli_d(N(a),((32*imm)-64))))
#define extract64from128(a, imm) \
        (((imm) < 1) ? (__lsx_vor_v(__lsx_vsrli_d(M(a), (64*imm)), __lsx_vor_v(__lsx_vsrli_d(L(a), (64*imm)), __lsx_vslli_d(N(a), (64-(64*imm)))))) : (__lsx_vsrli_d(N(a),((64*imm)-64))))

#define extractlow64from256(a) movq(a.lo)
#define extractlow32from256(a) movd(a.lo)

static really_inline m128 and128(m128 a, m128 b) {
    return __lsx_vand_v(a,b);
}

static really_inline m128 xor128(m128 a, m128 b) {
    return __lsx_vxor_v(a,b);
}

static really_inline m128 or128(m128 a, m128 b) {
    return __lsx_vor_v(a,b);
}

static really_inline m128 andnot128(m128 a, m128 b) {
    return __lsx_vandn_v(a,b);
}

// aligned load
static really_inline m128 load128(const void *ptr) {
    assert(ISALIGNED_N(ptr, alignof(m128)));
    ptr = assume_aligned(ptr, 16);
    return __lsx_vldx((const m128 *)ptr,0);
}

// aligned store
static really_inline void store128(void *ptr, m128 a) {
    assert(ISALIGNED_N(ptr, alignof(m128)));
    ptr = assume_aligned(ptr, 16);
    *(m128 *)ptr = a;
}

// unaligned load
static really_inline m128 loadu128(const void *ptr) {
    return __lsx_vldx((const m128 *)ptr,0);
}

// unaligned store
static really_inline void storeu128(void *ptr, m128 a) {
    __lsx_vst(a,(m128 *)ptr,0);
}

// packed unaligned store of first N bytes
static really_inline
void storebytes128(void *ptr, m128 a, unsigned int n) {
    assert(n <= sizeof(a));
    memcpy(ptr, &a, n);
}

// packed unaligned load of first N bytes, pad with zero
static really_inline
m128 loadbytes128(const void *ptr, unsigned int n) {
    m128 a = zeroes128();
    assert(n <= sizeof(a));
    memcpy(&a, ptr, n);
    return a;
}

#ifdef __cplusplus
extern "C" {
#endif
extern const u8 simd_onebit_masks[];
#ifdef __cplusplus
}
#endif

static really_inline
m128 mask1bit128(unsigned int n) {
    assert(n < sizeof(m128) * 8);
    u32 mask_idx = ((n % 8) * 64) + 95;
    mask_idx -= n / 8;
    return loadu128(&simd_onebit_masks[mask_idx]);
}

// switches on bit N in the given vector.
static really_inline
void setbit128(m128 *ptr, unsigned int n) {
    *ptr = or128(mask1bit128(n), *ptr);
}

// switches off bit N in the given vector.
static really_inline
void clearbit128(m128 *ptr, unsigned int n) {
    *ptr = andnot128(mask1bit128(n), *ptr);
}

// tests bit N in the given vector.
static really_inline
char testbit128(m128 val, unsigned int n) {
    const m128 mask = mask1bit128(n);
    return isnonzero128(and128(mask, val));
}

#define palignr(r, l, offset) \
       (((offset) < 8) ? __lsx_vor_v(rshiftbyte_m128(l,(offset)),lshiftbyte_m128(U(r),(8-(offset)))) : __lsx_vor_v(rshiftbyte_m128(l,(offset)), lshiftbyte_m128(r,(16-(offset)))))

static really_inline
m128 shuffle_epi8(m128 a, m128 b) {
	m128 tmp1,tmp2,tmp3,dst;
        tmp1 = ~(__lsx_vslt_b(b,__lsx_vldi(0)));
        tmp2 = __lsx_vand_v(b,tmp1);
        tmp3 = __lsx_vand_v(tmp2, __lsx_vldi(0x0F));
        unsigned char* p = (unsigned char*)&tmp3;
        unsigned char* pa = (unsigned char*)&a;
        for (int i = 0; i < 16; i++) {
                unsigned char value = p[i];
                switch(i){
                        case 0:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 0);}else{dst = tmp3;}break;
                        case 1:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 1);}else{dst = tmp3;}break;
                        case 2:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 2);}else{dst = tmp3;}break;
                        case 3:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 3);}else{dst = tmp3;}break;
                        case 4:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 4);}else{dst = tmp3;}break;
                        case 5:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 5);}else{dst = tmp3;}break;
                        case 6:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 6);}else{dst = tmp3;}break;
                        case 7:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 7);}else{dst = tmp3;}break;
                        case 8:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 8);}else{dst = tmp3;}break;
                        case 9:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 9);}else{dst = tmp3;}break;
                        case 10:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 10);}else{dst = tmp3;}break;
                        case 11:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 11);}else{dst = tmp3;}break;
                        case 12:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 12);}else{dst = tmp3;}break;
                        case 13:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 13);}else{dst = tmp3;}break;
                        case 14:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 14);}else{dst = tmp3;}break;
                        case 15:if(value > 0){dst = __lsx_vinsgr2vr_b(tmp3, pa[value], 15);}else{dst = tmp3;}break;
                        default:break;
                }
                tmp3 = dst;
        }
    return dst;
}

static really_inline
m128 pshufb_m128(m128 a, m128 b) {
    m128 result;
    result = shuffle_epi8(a, b);
    return result;
}

static really_inline
m256 pshufb_m256(m256 a, m256 b) {
    m256 rv;
    rv.lo = pshufb_m128(a.lo, b.lo);
    rv.hi = pshufb_m128(a.hi, b.hi);
    return rv;
}

static really_inline
m128 variable_byte_shift_m128(m128 in, s32 amount) {
    assert(amount >= -16 && amount <= 16);
    m128 shift_mask = loadu128(vbs_mask_data + 16 - amount);
    return pshufb_m128(in, shift_mask);
}

static really_inline
m128 max_u8_m128(m128 a, m128 b) {
    return __lsx_vmax_bu(a, b);
}

static really_inline
m128 min_u8_m128(m128 a, m128 b) {
    return __lsx_vmin_bu(a, b);
}

static really_inline
m128 sadd_u8_m128(m128 a, m128 b) {
    return __lsx_vsadd_bu(a, b);
}

static really_inline
m128 sub_u8_m128(m128 a, m128 b) {
    return __lsx_vsub_b(a, b);
}

static really_inline
m128 set64x2(u64a hi, u64a lo) {
    return __lsx_vinsgr2vr_d(__lsx_vreplgr2vr_d(hi),lo,0);
}

/****
 **** 256-bit Primitives
 ****/

static really_really_inline
m256 lshift64_m256(m256 a, int b) {
    m256 rv = a;
    rv.lo = lshift64_m128(rv.lo, b);
    rv.hi = lshift64_m128(rv.hi, b);
    return rv;
}

static really_inline
m256 rshift64_m256(m256 a, int b) {
    m256 rv = a;
    rv.lo = rshift64_m128(rv.lo, b);
    rv.hi = rshift64_m128(rv.hi, b);
    return rv;
}
static really_inline
m256 set32x8(u32 in) {
    m256 rv;
    rv.lo = set16x8((u8) in);
    rv.hi = rv.lo;
    return rv;
}

static really_inline
m256 eq256(m256 a, m256 b) {
    m256 rv;
    rv.lo = eq128(a.lo, b.lo);
    rv.hi = eq128(a.hi, b.hi);
    return rv;
}

static really_inline
u32 movemask256(m256 a) {
    u32 lo_mask = movemask128(a.lo);
    u32 hi_mask = movemask128(a.hi);
    return lo_mask | (hi_mask << 16);
}

static really_inline
m256 set2x128(m128 a) {
    m256 rv = {a, a};
    return rv;
}

static really_inline m256 zeroes256(void) {
    m256 rv = {zeroes128(), zeroes128()};
    return rv;
}

static really_inline m256 ones256(void) {
    m256 rv = {ones128(), ones128()};
    return rv;
}

static really_inline m256 and256(m256 a, m256 b) {
    m256 rv;
    rv.lo = and128(a.lo, b.lo);
    rv.hi = and128(a.hi, b.hi);
    return rv;
}

static really_inline m256 or256(m256 a, m256 b) {
    m256 rv;
    rv.lo = or128(a.lo, b.lo);
    rv.hi = or128(a.hi, b.hi);
    return rv;
}

static really_inline m256 xor256(m256 a, m256 b) {
    m256 rv;
    rv.lo = xor128(a.lo, b.lo);
    rv.hi = xor128(a.hi, b.hi);
    return rv;
}

static really_inline m256 not256(m256 a) {
    m256 rv;
    rv.lo = not128(a.lo);
    rv.hi = not128(a.hi);
    return rv;
}

static really_inline m256 andnot256(m256 a, m256 b) {
    m256 rv;
    rv.lo = andnot128(a.lo, b.lo);
    rv.hi = andnot128(a.hi, b.hi);
    return rv;
}

static really_inline int diff256(m256 a, m256 b) {
    return diff128(a.lo, b.lo) || diff128(a.hi, b.hi);
}

static really_inline int isnonzero256(m256 a) {
    return isnonzero128(or128(a.lo, a.hi));
}

/**
 * "Rich" version of diff256(). Takes two vectors a and b and returns an 8-bit
 * mask indicating which 32-bit words contain differences.
 */
static really_inline u32 diffrich256(m256 a, m256 b) {
    m128 z = zeroes128();
    m128 tmp0,tmp1,tmp2,tmp3;
    a.lo = __lsx_vseq_w(a.lo, b.lo);
    a.hi = __lsx_vseq_w(a.hi, b.hi);

    tmp0 =__lsx_vsat_w(a.lo, 15);
    tmp1 =__lsx_vsat_w(b.hi, 15);
    tmp2 =__lsx_vsat_h(__lsx_vpickev_h(tmp1, tmp0), 7);
    tmp3 =__lsx_vsat_h(z, 7);
    m128 packed = __lsx_vpickev_b(tmp3, tmp2);

    return ~(__lsx_vpickve2gr_hu(__lsx_vmskltz_b(packed), 0)) & 0xff;
}

/**
 * "Rich" version of diff256(), 64-bit variant. Takes two vectors a and b and
 * returns an 8-bit mask indicating which 64-bit words contain differences.
 */
static really_inline u32 diffrich64_256(m256 a, m256 b) {
    u32 d = diffrich256(a, b);
    return (d | (d >> 1)) & 0x55555555;
}

// aligned load
static really_inline m256 load256(const void *ptr) {
    assert(ISALIGNED_N(ptr, alignof(m256)));
    m256 rv = {load128(ptr), load128((const char *)ptr + 16)};
    return rv;
}

// aligned load  of 128-bit value to low and high part of 256-bit value
static really_inline m256 load2x128(const void *ptr) {
    assert(ISALIGNED_N(ptr, alignof(m128)));
    m256 rv;
    rv.hi = rv.lo = load128(ptr);
    return rv;
}

static really_inline m256 loadu2x128(const void *ptr) {
    return set2x128(loadu128(ptr));
}

// aligned store
static really_inline void store256(void *ptr, m256 a) {
    assert(ISALIGNED_N(ptr, alignof(m256)));
    ptr = assume_aligned(ptr, 16);
    *(m256 *)ptr = a;
}

// unaligned load
static really_inline m256 loadu256(const void *ptr) {
    m256 rv = {loadu128(ptr), loadu128((const char *)ptr + 16)};
    return rv;
}

// unaligned store
static really_inline void storeu256(void *ptr, m256 a) {
    storeu128(ptr, a.lo);
    storeu128((char *)ptr + 16, a.hi);
}

// packed unaligned store of first N bytes
static really_inline void storebytes256(void *ptr, m256 a, unsigned int n) {
    assert(n <= sizeof(a));
    memcpy(ptr, &a, n);
}

// packed unaligned load of first N bytes, pad with zero
static really_inline m256 loadbytes256(const void *ptr, unsigned int n) {
    m256 a = zeroes256();
    assert(n <= sizeof(a));
    memcpy(&a, ptr, n);
    return a;
}

static really_inline m256 mask1bit256(unsigned int n) {
    assert(n < sizeof(m256) * 8);
    u32 mask_idx = ((n % 8) * 64) + 95;
    mask_idx -= n / 8;
    return loadu256(&simd_onebit_masks[mask_idx]);
}

static really_inline m256 set64x4(u64a hi_1, u64a hi_0, u64a lo_1, u64a lo_0) {
    m256 rv;
    rv.hi = set64x2(hi_1, hi_0);
    rv.lo = set64x2(lo_1, lo_0);
    return rv;
}

// switches on bit N in the given vector.
static really_inline void setbit256(m256 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo;
    } else {
        sub = &ptr->hi;
        n -= 128;
    }
    setbit128(sub, n);
}

// switches off bit N in the given vector.
static really_inline void clearbit256(m256 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo;
    } else {
        sub = &ptr->hi;
        n -= 128;
    }
    clearbit128(sub, n);
}

// tests bit N in the given vector.
static really_inline char testbit256(m256 val, unsigned int n) {
    assert(n < sizeof(val) * 8);
    m128 sub;
    if (n < 128) {
        sub = val.lo;
    } else {
        sub = val.hi;
        n -= 128;
    }
    return testbit128(sub, n);
}

static really_really_inline
m128 movdq_hi(m256 x) {
    return x.hi;
}

static really_really_inline m128 movdq_lo(m256 x) { return x.lo;}

static really_inline m256 combine2x128(m128 hi, m128 lo) {
    m256 rv = {lo, hi};
    return rv;
}

/****
 **** 384-bit Primitives
 ****/

static really_inline m384 and384(m384 a, m384 b) {
    m384 rv;
    rv.lo = and128(a.lo, b.lo);
    rv.mid = and128(a.mid, b.mid);
    rv.hi = and128(a.hi, b.hi);
    return rv;
}

static really_inline m384 or384(m384 a, m384 b) {
    m384 rv;
    rv.lo = or128(a.lo, b.lo);
    rv.mid = or128(a.mid, b.mid);
    rv.hi = or128(a.hi, b.hi);
    return rv;
}

static really_inline m384 xor384(m384 a, m384 b) {
    m384 rv;
    rv.lo = xor128(a.lo, b.lo);
    rv.mid = xor128(a.mid, b.mid);
    rv.hi = xor128(a.hi, b.hi);
    return rv;
}
static really_inline m384 not384(m384 a) {
    m384 rv;
    rv.lo = not128(a.lo);
    rv.mid = not128(a.mid);
    rv.hi = not128(a.hi);
    return rv;
}
static really_inline m384 andnot384(m384 a, m384 b) {
    m384 rv;
    rv.lo = andnot128(a.lo, b.lo);
    rv.mid = andnot128(a.mid, b.mid);
    rv.hi = andnot128(a.hi, b.hi);
    return rv;
}

static really_really_inline m384 lshift64_m384(m384 a, unsigned b) {
    m384 rv;
    rv.lo = lshift64_m128(a.lo, b);
    rv.mid = lshift64_m128(a.mid, b);
    rv.hi = lshift64_m128(a.hi, b);
    return rv;
}

static really_inline m384 zeroes384(void) {
    m384 rv = {zeroes128(), zeroes128(), zeroes128()};
    return rv;
}

static really_inline m384 ones384(void) {
    m384 rv = {ones128(), ones128(), ones128()};
    return rv;
}

static really_inline int diff384(m384 a, m384 b) {
    return diff128(a.lo, b.lo) || diff128(a.mid, b.mid) || diff128(a.hi, b.hi);
}

static really_inline int isnonzero384(m384 a) {
    return isnonzero128(or128(or128(a.lo, a.mid), a.hi));
}

/**
 * "Rich" version of diff384(). Takes two vectors a and b and returns a 12-bit
 * mask indicating which 32-bit words contain differences.
 */
static really_inline u32 diffrich384(m384 a, m384 b) {
    m128 z = zeroes128();
    m128 tmp0,tmp1,tmp2,tmp3,tmp4,tmp5;
    a.lo = __lsx_vseq_w(a.lo, b.lo);
    a.mid = __lsx_vseq_w(a.mid, b.mid);
    a.hi = __lsx_vseq_w(a.hi, b.hi);

    tmp0 = __lsx_vsat_w(a.lo, 15);
    tmp1 = __lsx_vsat_w(b.mid, 15);

    tmp2 = __lsx_vsat_w(b.hi, 15);
    tmp3 = __lsx_vsat_w(z, 15);

    tmp4 = __lsx_vsat_h(__lsx_vpickev_h(tmp1, tmp0),7);
    tmp5 = __lsx_vsat_h(__lsx_vpickev_h(tmp3, tmp2),7);

    m128 packed = __lsx_vpickev_b(tmp5,tmp4);

    return ~(__lsx_vpickve2gr_hu(__lsx_vmskltz_b(packed), 0)) & 0xfff;
}

/**
 * "Rich" version of diff384(), 64-bit variant. Takes two vectors a and b and
 * returns a 12-bit mask indicating which 64-bit words contain differences.
 */
static really_inline u32 diffrich64_384(m384 a, m384 b) {
    u32 d = diffrich384(a, b);
    return (d | (d >> 1)) & 0x55555555;
}

// aligned load
static really_inline m384 load384(const void *ptr) {
    assert(ISALIGNED_16(ptr));
    m384 rv = {load128(ptr), load128((const char *)ptr + 16),
               load128((const char *)ptr + 32)};
    return rv;
}

// aligned store
static really_inline void store384(void *ptr, m384 a) {
    assert(ISALIGNED_16(ptr));
    ptr = assume_aligned(ptr, 16);
    *(m384 *)ptr = a;
}

// unaligned load
static really_inline m384 loadu384(const void *ptr) {
    m384 rv = {loadu128(ptr), loadu128((const char *)ptr + 16),
               loadu128((const char *)ptr + 32)};
    return rv;
}

// packed unaligned store of first N bytes
static really_inline void storebytes384(void *ptr, m384 a, unsigned int n) {
    assert(n <= sizeof(a));
    memcpy(ptr, &a, n);
}

// packed unaligned load of first N bytes, pad with zero
static really_inline m384 loadbytes384(const void *ptr, unsigned int n) {
    m384 a = zeroes384();
    assert(n <= sizeof(a));
    memcpy(&a, ptr, n);
    return a;
}

// switches on bit N in the given vector.
static really_inline void setbit384(m384 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo;
    } else if (n < 256) {
        sub = &ptr->mid;
    } else {
        sub = &ptr->hi;
    }
    setbit128(sub, n % 128);
}

// switches off bit N in the given vector.
static really_inline void clearbit384(m384 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo;
    } else if (n < 256) {
        sub = &ptr->mid;
    } else {
        sub = &ptr->hi;
    }
    clearbit128(sub, n % 128);
}

// tests bit N in the given vector.
static really_inline char testbit384(m384 val, unsigned int n) {
    assert(n < sizeof(val) * 8);
    m128 sub;
    if (n < 128) {
        sub = val.lo;
    } else if (n < 256) {
        sub = val.mid;
    } else {
        sub = val.hi;
    }
    return testbit128(sub, n % 128);
}

/****
 **** 512-bit Primitives
 ****/

static really_inline m512 zeroes512(void) {
    m512 rv = {zeroes256(), zeroes256()};
    return rv;
}

static really_inline m512 ones512(void) {
    m512 rv = {ones256(), ones256()};
    return rv;
}

static really_inline m512 and512(m512 a, m512 b) {
    m512 rv;
    rv.lo = and256(a.lo, b.lo);
    rv.hi = and256(a.hi, b.hi);
    return rv;
}

static really_inline m512 or512(m512 a, m512 b) {
    m512 rv;
    rv.lo = or256(a.lo, b.lo);
    rv.hi = or256(a.hi, b.hi);
    return rv;
}

static really_inline m512 xor512(m512 a, m512 b) {
    m512 rv;
    rv.lo = xor256(a.lo, b.lo);
    rv.hi = xor256(a.hi, b.hi);
    return rv;
}

static really_inline m512 not512(m512 a) {
    m512 rv;
    rv.lo = not256(a.lo);
    rv.hi = not256(a.hi);
    return rv;
}

static really_inline m512 andnot512(m512 a, m512 b) {
    m512 rv;
    rv.lo = andnot256(a.lo, b.lo);
    rv.hi = andnot256(a.hi, b.hi);
    return rv;
}

static really_really_inline m512 lshift64_m512(m512 a, unsigned b) {
    m512 rv;
    rv.lo = lshift64_m256(a.lo, b);
    rv.hi = lshift64_m256(a.hi, b);
    return rv;
}

static really_inline int diff512(m512 a, m512 b) {
    return diff256(a.lo, b.lo) || diff256(a.hi, b.hi);
}

static really_inline int isnonzero512(m512 a) {
    m128 x = or128(a.lo.lo, a.lo.hi);
    m128 y = or128(a.hi.lo, a.hi.hi);
    return isnonzero128(or128(x, y));
}

/**
 * "Rich" version of diff512(). Takes two vectors a and b and returns a 16-bit
 * mask indicating which 32-bit words contain differences.
 */
static really_inline u32 diffrich512(m512 a, m512 b) {
    m128 tmp0,tmp1,tmp2,tmp3,tmp4,tmp5,tmp6,tmp7;
    a.lo.lo = __lsx_vseq_w(a.lo.lo, b.lo.lo);
    a.lo.hi = __lsx_vseq_w(a.lo.hi, b.lo.hi);
    a.hi.lo = __lsx_vseq_w(a.hi.lo, b.hi.lo);
    a.hi.hi = __lsx_vseq_w(a.hi.hi, b.hi.hi);
   
    tmp0 =__lsx_vsat_w(a.lo.lo, 15);
    tmp1 =__lsx_vsat_w(a.lo.hi, 15);
    tmp2 =__lsx_vpickev_h(tmp1, tmp0);

    tmp3 =__lsx_vsat_w(a.hi.lo, 15);
    tmp4 =__lsx_vsat_w(a.hi.hi, 15);
    tmp5 =__lsx_vpickev_h(tmp4, tmp3);

    tmp6 =__lsx_vsat_h(tmp2, 7);
    tmp7 =__lsx_vsat_h(tmp5, 7);
    m128 packed = __lsx_vpickev_b(tmp7, tmp6);

    return ~(__lsx_vpickve2gr_hu(__lsx_vmskltz_b(packed), 0)) & 0xffff;   // ok
}

/**
 * "Rich" version of diffrich(), 64-bit variant. Takes two vectors a and b and
 * returns a 16-bit mask indicating which 64-bit words contain differences.
 */
static really_inline u32 diffrich64_512(m512 a, m512 b) {
    u32 d = diffrich512(a, b);
    return (d | (d >> 1)) & 0x55555555;
}

// aligned load
static really_inline m512 load512(const void *ptr) {
    assert(ISALIGNED_N(ptr, alignof(m256)));
    m512 rv = {load256(ptr), load256((const char *)ptr + 32)};
    return rv;
}

// aligned store
static really_inline void store512(void *ptr, m512 a) {
    assert(ISALIGNED_N(ptr, alignof(m512)));
    ptr = assume_aligned(ptr, 16);
    *(m512 *)ptr = a;
}

// unaligned load
static really_inline m512 loadu512(const void *ptr) {
    m512 rv = {loadu256(ptr), loadu256((const char *)ptr + 32)};
    return rv;
}

// packed unaligned store of first N bytes
static really_inline void storebytes512(void *ptr, m512 a, unsigned int n) {
    assert(n <= sizeof(a));
    memcpy(ptr, &a, n);
}

// packed unaligned load of first N bytes, pad with zero
static really_inline m512 loadbytes512(const void *ptr, unsigned int n) {
    m512 a = zeroes512();
    assert(n <= sizeof(a));
    memcpy(&a, ptr, n);
    return a;
}

static really_inline m512 mask1bit512(unsigned int n) {
    assert(n < sizeof(m512) * 8);
    u32 mask_idx = ((n % 8) * 64) + 95;
    mask_idx -= n / 8;
    return loadu512(&simd_onebit_masks[mask_idx]);
}

// switches on bit N in the given vector.
static really_inline void setbit512(m512 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo.lo;
    } else if (n < 256) {
        sub = &ptr->lo.hi;
    } else if (n < 384) {
        sub = &ptr->hi.lo;
    } else {
        sub = &ptr->hi.hi;
    }
    setbit128(sub, n % 128);
}

// switches off bit N in the given vector.
static really_inline void clearbit512(m512 *ptr, unsigned int n) {
    assert(n < sizeof(*ptr) * 8);
    m128 *sub;
    if (n < 128) {
        sub = &ptr->lo.lo;
    } else if (n < 256) {
        sub = &ptr->lo.hi;
    } else if (n < 384) {
        sub = &ptr->hi.lo;
    } else {
        sub = &ptr->hi.hi;
    }
    clearbit128(sub, n % 128);
}

// tests bit N in the given vector.
static really_inline char testbit512(m512 val, unsigned int n) {
    assert(n < sizeof(val) * 8);
    m128 sub;
    if (n < 128) {
        sub = val.lo.lo;
    } else if (n < 256) {
        sub = val.lo.hi;
    } else if (n < 384) {
        sub = val.hi.lo;
    } else {
        sub = val.hi.hi;
    }
    return testbit128(sub, n % 128);
}

#endif
