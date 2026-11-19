// UnaryTExpr.hpp
//
// Class holding a sinlge texpr and a unary functor
//
// JAF 10/16/2026

#ifndef FORNFDM_TEXPRS_UNARYTEXPR_H
#define FORNFDM_TEXPRS_UNARYTEXPR_H

#include<cstddef>
#include "../types.hpp"
#include "traits.hpp"
#include "TExprBase.hpp"
#include "../diffops/functors.hpp"
#include "Queue.hpp"

namespace fornfdm{
namespace texprs{

// forward declaration
template<class UnaryOp, class XprType>
struct UnaryTExpr;

namespace internal{

template<class UnaryOp, class XprType>
struct traits_impl<UnaryTExpr<UnaryOp,XprType>>
{
  static constexpr bool is_unarop = true; 
  static constexpr bool is_binop = false; 
  static constexpr bool is_ternop = false; 
  static constexpr bool is_timedep = traits<XprType>::is_timedep; 
  static constexpr std::size_t max_arity = traits<XprType>::max_arity;  
  static constexpr std::size_t max_order = traits<XprType>::max_order;  
};

} // end namespace internal

template<class UnaryOp, class XprType>
class UnaryTExpr : public TExprBase<UnaryTExpr<UnaryOp,XprType>>
{
  public: 
    // Type Defs 
    typedef typename fornfdm::texprs::internal::NestedStorage<XprType>::type XprTypeNested;
    typedef typename std::remove_cv_t<std::remove_reference<XprType>>::type NestedExpression;

  protected:
    // Member data 
    XprTypeNested m_xpr; 
    UnaryOp m_functor; 
  
  public:
    // Constructors + Destructor
    UnaryTExpr()=delete; 

    UnaryTExpr(XprType&& xpr, UnaryOp func = UnaryOp())
      : m_xpr(std::forward<XprType>(xpr)), m_functor(func) 
    {}

    UnaryTExpr(const UnaryTExpr& other)=default;

    ~UnaryTExpr()=default;

    // Member Funcs
    const auto& functor() const { return m_functor; }
    const auto& nestedExpression() const { return m_xpr; }
    auto& nestedExpression(){ return m_xpr; }
};

template<typename Derived>
auto TExprBase<Derived>::operator-() const &
{
  using fornfdm::linops::internal::UnaryNegateFO;
  return UnaryTExpr<UnaryNegateFO, const Derived&>(derived(),UnaryNegateFO{});
}

template<typename Derived>
auto TExprBase<Derived>::operator-() &&
{
  using fornfdm::linops::internal::UnaryNegateFO;
  return UnaryTExpr<UnaryNegateFO, Derived>(std::move(derived()),UnaryNegateFO{});
}

template<
  typename C, 
  typename XprType, 
  typename = std::enable_if_t<
    (internal::is_texpr_crtp<XprType>::value && 
     std::is_convertible_v<C,fornfdm::Scalar>)
  >
>
auto operator*(C&& c, XprType&& xpr)
{
  using fornfdm::linops::internal::UnaryScalarMultiplyFO;
  return UnaryTExpr<UnaryScalarMultiplyFO<C>,XprType>(std::forward<XprType>(xpr), UnaryScalarMultiplyFO<C>(std::forward<C>(c)));
}

template<std::size_t Ith, typename Q, typename UnaryOp, typename XprType>
inline auto coeff(const Q& queue, const UnaryTExpr<UnaryOp,XprType>& xpr, const fornfdm::Mesh* m, fornfdm::Real t)
{
  auto c = coeff<Ith>(queue, xpr.nestedExpression(), m, t);
  if constexpr(internal::traits<XprType>::max_arity == 0)
  {
    // c is a scalar. return f( c ).
    return xpr.functor()(c);
  }
  else
  {
    // c is some lambda that accepts Coordinate<N> as argument. 
    return [c, &f = xpr.functor()](const auto& coord){ return f(c(coord)); }; 
  }
}

template<typename Q, typename UnaryOp, typename XprType>
inline auto rhsTerm(const Q& queue, const UnaryTExpr<UnaryOp,XprType>& xpr, const fornfdm::Mesh* m, fornfdm::Real t)
{
  // since UnaryTExpr simply multiplies weights/coeffs linearl by scalar,
  // safe to pull out the operation onto the large RHS expression.
  return xpr.functor(rhsTerm(queue, xpr.nestedExpression(), m, t));
}

template<typename Q, typename UnaryOp, typename XprType>
inline auto inverseCoeff(const Q& queue, const UnaryTExpr<UnaryOp,XprType>& xpr, const fornfdm::Mesh* m, fornfdm::Real t)
{
  auto c = coeff<queue.numTimes - 1>(queue, xpr, m, t);
  if constexpr(internal::traits<XprType>::max_arity == 0)
  {
    // c is a Real number. 
    return (fornfdm::Real(1.0) / c);
  }
  else
  {
    // c is a lambda accepting Coordinate<N> as input 
    // turn it into an eigen expression
    auto f = [c, m](std::size_t idx)
    {
      return (fornfdm::Real(1.0) / c(fornfdm::Coordinate<internal::traits<XprType>::max_arity>(m, idx))); 
    };
    using F = std::remove_reference_t<std::remove_cv_t<decltype(f)>>;
    std::size_t n = m->sizesProduct();
    // dangling reference? possibly need our own wrapper? use diagonal matrix as the template param!!!
    return Eigen::CwiseNullaryOp<F, fornfdm::Vector>(n, 1, f).asDiagonal();
  }
}

} // end namespace texprs
} // end namespace fornfdm

#endif