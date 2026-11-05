// traits.hpp
//
// compile time traits for TExpr types 
//
// JAF 4/4/2026 

#ifndef FORNFDM_TEXPR_TRAITS_H
#define FORNFDM_TEXPR_TRAITS_H

#include<cstdint>
#include<tuple>
#include<type_traits>
#include "../diffops/traits.hpp" // max_order
#include "../types.hpp"

namespace fornfdm{
namespace texprs{

// // Forward Declarations 
// template<typename Derived>
// class TimeDerivBase; 

// template<std::size_t N>
// class NthTimeDeriv; 

// template<typename Coeff, typename TimeDeriv>
// class CoeffMultExpr; 

// Given size_t i,j,... get the maximum -------------------------------------------- 
namespace internal{
  
template<typename T>
struct traits_impl
{
  // static constexpr bool is_texpr = false;
  // static constexpr bool is_unarop = false; 
  // static constexpr bool is_binop = false; 
  // static constexpr bool is_ternop = false; 
  // // TODO what all is needed here? 
  // static constexpr bool is_timedep = false;
  // static constexpr std::size_t max_arity = 0;  
  // // unique orders contained in the expression
  // static constexpr std::size_t order = 1; 
  // static constexpr std::size_t orders_seq = std::index_sequence<1>{}; 
};

template<class T>
using traits = traits_impl<std::remove_cv_t<std::remove_reference_t<T>>>;

template<typename T>
struct is_tuple_impl : std::false_type{};

template<typename... Ts>
struct is_tuple_impl<std::tuple<Ts...>> : std::true_type{};

template<typename T>
using is_tuple = is_tuple_impl<std::remove_cv_t<std::remove_reference_t<T>>>;

// Determine what type to use to nest any texpr 
template<class T, typename = void>
struct NestedStorage
{
  typedef typename std::conditional<
    (std::is_lvalue_reference<T>::value),
    T,
    typename std::remove_reference<T>::type
  >::type type; 
};

template<class T>
struct NestedStorage<T, std::void_t<decltype(fornfdm::texprs::internal::traits<T>::is_texpr)>>
{ 
  using traits_t = typename fornfdm::texprs::internal::traits<T>; 
  typedef typename std::conditional<
    (!traits_t::is_unarop && !traits_t::is_binop && !traits_t::is_ternop) && (std::is_lvalue_reference<T>::value),
    T,
    typename std::remove_reference<T>::type
  >::type type; 
};

// the gist of calculating inverse coefficient / RHS expression. 

// if none of the expressions use coefficient/ max_arity == 0.
// can just outright add the terms together and (-1.0) * the whole thing to RHS
// same for inversion coefficient. 

// if there's any coefficient with max_arity > 0.
// detect if there are coefficients?  
// later could use the flags for Uxx * Ut?

// Given a tuple <A,B,C,...> return a subset where PRED<X>::value returns true.   
template<template<typename...> class PRED, typename TUP_T>
auto filter_tup(TUP_T&& tup)
{
  auto filter_lam = [](auto&& elem){
    using T = decltype(elem); 
    using CLEAN_T = std::remove_reference_t<std::remove_cv_t<T>>;
    if constexpr(PRED<CLEAN_T>::value){
      // singleton tuple ( elem )
      if constexpr(std::is_lvalue_reference<T>::value){
        return std::tie(elem); 
      }
      else{
        return std::make_tuple(std::move(elem)); 
      }
    }
    else{
      // empty tuple ( _ )
      return std::tuple<>{}; 
    }
  }; 

  auto singletons = std::apply(
    [&](auto&&... elems){
      return std::make_tuple(filter_lam(std::forward<decltype(elems)>(elems))...); 
    }, 
    tup
  ); 

  return std::apply(
    [](auto&&... elems){
      return std::tuple_cat(std::forward<decltype(elems)>(elems)...); 
    }, 
    singletons
  ); 
}

} // end namespace internal
} // end namespace texprs 
} // end namespace fornfdm 

#endif 