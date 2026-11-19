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

// Forward Declarations 
template<typename Derived>
class TExprBase; 

namespace internal{

// Given a type T detect if it is derived from TExprBase<T> or is the same as TExprBase<T>
template<class T>
struct is_texpr_crtp_helper : std::false_type{}; 

template<class T>
struct is_texpr_crtp_helper<TExprBase<T>> : std::true_type{}; 

template<class T>
struct is_texpr_crtp_impl : std::disjunction<std::is_base_of<TExprBase<T>,T>, is_texpr_crtp_helper<T>>{}; 

template<class T>
using is_texpr_crtp = is_texpr_crtp_impl<std::remove_cv_t<std::remove_reference_t<T>>>; 

template<typename T>
struct traits_impl{};

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
struct NestedStorage<T, std::enable_if_t<is_texpr_crtp<T>::value>>
{ 
  using traits_t = typename fornfdm::texprs::internal::traits<T>; 
  typedef typename std::conditional<
    (!traits_t::is_unarop && !traits_t::is_binop && !traits_t::is_ternop) && (std::is_lvalue_reference<T>::value),
    T,
    std::remove_reference_t<T>
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