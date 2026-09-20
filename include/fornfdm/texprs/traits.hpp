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
#include "../types.hpp"

namespace fornfdm{
  namespace texprs{

template<typename Derived>
class TexprBase;

template<std::size_t nthOrder>
class NthTimeDeriv;

template<typename L, typename R, typename BinOp>
class BinaryTexpr;

template<typename T, typename UnarOp>
class UnaryTexpr;

template<typename L, typename R>
class CoeffProductTexpr;

namespace internal{

template<typename T>
struct traits_impl{};

template<typename T>
using traits = traits_impl<std::remove_cv_t<std::remove_reference_t<T>>>;

template<typename T>
NestedStorage
{
  using type = std::conditional_t<
    traits<T>::nest_by_ref && std::is_lvalue_reference_v<T>,
    T,
    std::remove_reference_t<T>
  >;
}; 

template<typename Derived>
class traits_impl<TexprBase<Derived>> : traits<Derived>{};

template<std::size_t nthOrder>
class traits_impl<NthTimeDeriv<nthOrder>>
{
  static constexpr bool nest_by_ref = true;
  static constexpr std::size_t max_order = nthOrder;
  static constexpr max_arity = 0;
};

template<typename L, typename R, typename BinOp>
class traits_impl<BinaryTexpr<L,R,BinOp>>
{
  static constexpr bool nest_by_ref = false;
  static constexpr std::size_t max_order = std::max(traits<L>::max_order, traits<R>::max_order);
  static constexpr std::size_t max_arity = std::max(traits<L>::max_arity, traits<R>::max_arity);
};

template<typename T, typename UnarOp>
class traits_impl<UnaryTexpr>
{
  static constexpr bool nest_by_ref = false;
  static constexpr std::size_t max_order = traits<T>::max_order;
  static constexpr std::size_t max_arity = traits<T>::max_arity;
};

} // end namespace internal

// how granular do I need to be when setting an inverse coefficient? 
// texprs Executor has 2 main functionalities from wrapped expression 
// in any given FDM a*U(n-2) b*U(n-1) c*U(n) d*U(n+1) = ... 

// 1.) move all terms U(i) i<=n to the R.H.S. with an eigen expression 

/* we are working to accomodate various different number of arity.
  each of the arity is from a CoeffProductTexpr. we have to load each into eigen's packets internally 
  old way was to simply construct any eigen expression regardless. 
  probably can't optimize this without lots of eigen internal packet math or simd. 
  but we could build Eigen::Map sequentially for every product_after / product_before 
  that results from kronecker producting the diagonals of every matrix. 
  
  example 1.) 
  If every coefficient is of arity 1, we can build expressions using the vector directly. 
  and then use a CyclicWrapper as before. 

  example 2.) 
  What if there is a mixture of arity? 
*/

// 2.) build an expression for the inverse of "d" so that we obtain U(n+1) = ...

template<typename Derived, std::size_t N>
class TimeDerivBase; 

template<std::size_t N>
class NthTimeDeriv; 

template<typename Coeff, typename TimeDeriv>
class CoeffMultExpr; 

// traits =======================================================================================
// See if a given type T derives from TimeDerivBase ---------------------------------------------
namespace internal{
template<typename T, typename = void> 
struct is_timederiv_crtp_impl : public std::false_type{}; 

template<typename T> 
struct is_timederiv_crtp_impl<T, std::void_t<typename T::is_timederiv_tag>>: public std::true_type{}; 
} // end namespace internal 

namespace traits{
template<typename T>
using is_timederiv_crtp = texprs::internal::is_timederiv_crtp_impl<std::remove_cv_t<std::remove_reference_t<T>>>; 
} // end namespace traits

// See if a given type T derives from CoeffMultExpr ------------------------------
namespace internal{
template<typename T> 
struct is_coeffmult_crtp_impl : public std::false_type{}; 

template<typename L, typename R> 
struct is_coeffmult_crtp_impl<CoeffMultExpr<L,R>>: public std::true_type{}; 
} // end namespace internal 

namespace traits{
template<typename T>
using is_coeffmult_crtp = texprs::internal::is_coeffmult_crtp_impl<std::remove_cv_t<std::remove_reference_t<T>>>; 
} // end namespace traits

// Given a type T, determine how to store it in CoeffMultExpr ------------------------------------ 
namespace traits{ 

template<typename T, typename = void>
struct Storage
{ 
  using type = T; 
}; 

template<typename T>
struct Storage<T, std::enable_if_t<is_timederiv_crtp<T>::value>>
{
  using type = std::conditional_t<
    std::is_lvalue_reference<T>::value && !(is_coeffmult_crtp<T>::value), 
    T,
    std::remove_reference_t<T>
  >; 
}; 

} // end namespace traits 

// Given size_t i,j,... get the maximum -------------------------------------------- 
namespace internal{
  
template<std::size_t... is>
struct variadicFoldMaximum{/* no value for empty args*/}; 

template<std::size_t i>
struct variadicFoldMaximum<i>
{
  static constexpr std::size_t value = i; 
}; 

template<std::size_t i, std::size_t... trailing>
struct variadicFoldMaximum<i,trailing...>
{
  static constexpr std::size_t value = std::max(i, variadicFoldMaximum<trailing...>::value); 
};

} // end namespace internal 

// given a type T. detect if it returns a fornfdm::Real from coeffAt<>() or some other matrix expression. 
namespace traits{

template<typename TIMEDERIV_T>
struct TimeDerivTraits
{
  using _coeffatreturntype_unclean_ = decltype(std::declval<TIMEDERIV_T>().template coeffAt<0,0>(std::declval<std::array<fornfdm::Real,0>>()));  
  using CoeffAtReturnType = std::remove_cv_t<std::remove_reference_t<_coeffatreturntype_unclean_>>;
}; 

template<typename T>
struct coeffat_returns_real_impl : std::false_type{};

template<std::size_t nthOrder>
struct coeffat_returns_real_impl<texprs::NthTimeDeriv<nthOrder>> : std::true_type{}; 

template<class Lhs, class Rhs>
struct coeffat_returns_real_impl<texprs::CoeffMultExpr<Lhs,Rhs>> : std::conjunction<
  std::is_convertible<decltype(std::declval<Lhs>() * fornfdm::Real()),fornfdm::Real>, 
  coeffat_returns_real_impl<std::remove_cv_t<std::remove_reference_t<Rhs>>>
>{}; 

template<class T>
using coeffat_returns_real = coeffat_returns_real_impl<std::remove_cv_t<std::remove_reference_t<T>>>; 

template<typename T>
struct coeffat_returns_other : public std::negation<coeffat_returns_real<T>>{}; 

} // end namespace traits

// Given a tuple <A,B,C,...> return a subset where PRED<X>::value returns true.   
namespace traits{ 

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

} // end namespace traits 

  } // end namespace texprs 
} // end namespace fornfdm 

#endif 