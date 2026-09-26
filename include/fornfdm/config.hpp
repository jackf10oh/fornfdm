// config.hpp
//
// contains all macro configuration of the library
//
// JAF 8/4/2026

#ifndef FORNFDM_CONFIG_H
#define FORNFDM_CONFIG_H

// Real + Scalar types
#ifndef FORNFDM_CUSTOM_SCALAR
#define FORNFDM_CUSTOM_SCALAR double
#endif 

#ifndef FORNFDM_CUSTOM_REAL
#define FORNFDM_CUSTOM_REAL Eigen::NumTraits<FORNFDM_CUSTOM_SCALAR>::Real 
#endif

// Modulo can be faster if all axes are powers of 2
#ifdef FORNFDM_USE_BITWISE_OPS
#define FORNFDM_MODULO(x,y) ( (x) & (y - 1) )
#else
#define FORNFDM_MODULO(x,y) ( (x) % (y) )
#endif 

// Division can be faster if all axes are powers of 2
#ifdef FORNFDM_USE_BITWISE_OPS
  #if defined(__GNUC__) || defined(__clang__)
    #define FORNFDM_DIVISION(x,y) ( (x) >> (__builtin_ctz(y)) )
  #elif defined(_MSC_VER)
    #include<intrin.h>
    inline std::size_t __msvc_ctz(std::size_t mask){
      unsigned long native_idx; 
      if constexpr(sizeof(std::size_t)==8){ // targeting 64 bit  
        #if defined(_M_X64) || defined(_M_ARM64)
          _BitScanForward64(&native_idx, mask);
        #else
          std::size_t fallback_idx = 0;
          while((mask & 1)==0) { mask >>= 1; ++fallback_idx; }
          return fallback_idx;
        #endif 
      }
      else{ // targeting 32 bit
        _BitScanForward(&native_idx, mask);
      }
      return static_cast<std::size_t>(native_idx);
    }
    #define FORNFDM_DIVISION(x,y) ( (x) >> (__msvc_ctz(y)))
  #endif
#else
#define FORNFDM_DIVISION(x,y) ( (x) / (y) )
#endif

#endif // config.hpp
