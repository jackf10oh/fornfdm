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
#ifdef FORNFDM_USE_BITWISE_MODULO
#define FORNFDM_MODULO(x,y) ( (x) & (y - 1) )
#else
#define FORNFDM_MODULO(x,y) ( (x) % (y) )
#endif 

#endif // config.hpp
