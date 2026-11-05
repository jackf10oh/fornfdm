// functors.hpp
//
// the function objects behind expressions of TExprs subdir. 
// not the same as functors for diffops since TExprs 
// has to handle tuples where each entry is a term in a summation
// a + b + ... + z
//
// JAF 11/4/2026

#ifndef FORNFDM_TEXPRS_FUNCTORS_H
#define FORNFDM_TEXPRS_FUNCTORS_H 

#include "traits.hpp"
#include "UnaryTExpr.hpp"

namespace fornfdm{
namespace texprs{
namespace internal{

struct BinaryAdditionFO
{
  template<typename X, typename Y, std::enable_if_t<(!is_tuple<X>::value || !is_tuple<X>::value),void>>
  auto operator()(const X& x, const Y& y) const { return x + y; }

  template<typename... Ts, typename... Us>
  auto operator()(std::tuple<Ts...> left_tup, std::tuple<Us...> right_tup) const
  {
    // just combines terms into 1 big summation
    return std::tuple_cat(left_tup, right_tup);
  }
}; 

struct UnaryNegateFO
{
  template<typename X, std::enable_if_t<!is_tuple<X>::value,void>>
  auto operator()(const X& x) const { return -x; } 

  template<typename... Ts>
  auto operator()(std::tuple<Ts...> tup) const 
  {
    // take every entry T,U,V,... of the tuple. return a single 
    // texprs::UnaryTExpr of each entry. 
    auto negate_lam = [](auto&& xpr)
    { 
      return UnaryTExpr<UnaryNegateFO,decltype(xpr)>(
        std::forward<decltype(xpr)>(xpr),
        UnaryNegateFO{}
      );
    };
    return std::apply(
      [&](auto&&... elems){ return std::make_tuple(negate_lam(std::forward<decltype(elems)>(elems))...); },
      tup
    );
  }
}; 

template<typename C>
struct UnaryScalarMultiplyFO
{
  using ScalarTypeNested =
    std::conditional_t<
      std::is_lvalue_reference_v<C>,
      C,
      std::remove_reference_t<C>
    >;

  ScalarTypeNested m_scalar;

  // Constructors
  UnaryScalarMultiplyFO()=delete;
  
  UnaryScalarMultiplyFO(C&& c)
    : m_scalar(std::forward<C>(c))
  {}
  
  UnaryScalarMultiplyFO(const UnaryScalarMultiplyFO& other)=default;
  
  ~UnaryScalarMultiplyFO()=default;

  // Member Funcs
  template<typename X, std::enable_if_t<(!is_tuple<X>::value),void>>
  auto operator()(const X& x) const
  {
    return m_scalar * x;
  }

  template<typename... Ts>
  auto operator()(std::tuple<Ts...> tup)
  {
    auto singleton_lam = [this](auto&& xpr)
    {
      return UnaryTExpr<UnaryScalarMultiplyFO<C>,decltype(xpr)>(
        std::forward<decltype(xpr)>(xpr),
        *this
      );
    };
    return std::apply(
      [&](auto&&... elems){ return std::make_tuple(singleton_lam(std::forward<decltype(elems)>(elems))...); },
      tup
    );
  }
};

} // end namespace internal
} // end namespace texprs
} // end namespace fornfdm 

#endif