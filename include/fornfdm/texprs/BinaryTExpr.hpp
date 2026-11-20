// BinaryTExpr.hpp
//
// Class holding two texprs and a binary functor.
// currently only useful to sums A + B 
//
// JAF 10/16/2026

#ifndef FORNFDM_TEXPRS_BINARYTEXPR_H
#define FORNFDM_TEXPRS_BINARYTEXPR_H

#include<cstddef>
#include "../Mesh.hpp"
#include "../diffops/traits.hpp"
#include "traits.hpp"
#include "TExprBase.hpp"

namespace fornfdm{
namespace texprs{

// Forward Declaration
template<class BinaryOp, class LhsType, class RhsType>
struct BinaryTExpr; 

namespace internal{

template<class BinaryOp, class LhsType, class RhsType>
struct traits_impl<BinaryTExpr<BinaryOp, LhsType, RhsType>>
{
  static constexpr bool is_unarop = false; 
  static constexpr bool is_binop = true; 
  static constexpr bool is_ternop = false; 
  static constexpr bool is_timedep = traits<LhsType>::is_timedep || traits<RhsType>::is_timedep; 
  static constexpr std::size_t max_arity = std::max(traits<LhsType>::max_arity,traits<RhsType>::max_arity);  
  static constexpr std::size_t max_order = std::max(traits<LhsType>::max_order,traits<RhsType>::max_order);  
};

} // end namespace internal

template<class BinaryOp, class LhsType, class RhsType>
class BinaryTExpr : public TExprBase<BinaryTExpr<BinaryOp, LhsType, RhsType>>
{
  public:
    // Type Defs 
    typedef typename std::remove_cv_t<std::remove_reference_t<LhsType>> Lhs; 
    typedef typename std::remove_cv_t<std::remove_reference_t<RhsType>> Rhs; 
    typedef typename fornfdm::texprs::internal::NestedStorage<LhsType>::type LhsNested;
    typedef typename fornfdm::texprs::internal::NestedStorage<RhsType>::type RhsNested;
  
  private:
    // Member Data  
    LhsNested m_lhs;
    RhsNested m_rhs;
    BinaryOp m_functor; 

  public:
    // Constructors + Destructor
    BinaryTExpr()=delete;

    BinaryTExpr(LhsType&& lhs, RhsType&& rhs, BinaryOp func = BinaryOp())
      : m_lhs(std::forward<LhsType>(lhs)), m_rhs(std::forward<RhsType>(rhs)), m_functor(func) 
    {}

    BinaryTExpr(const BinaryTExpr& other)=default;

    ~BinaryTExpr()=default;

    // Member Funcs 
    const auto& functor() const { return m_functor; }
    const auto& lhs() const { return m_lhs; }
    auto& lhs(){ return m_lhs; }
    const auto& rhs() const { return m_rhs; }
    auto& rhs(){ return m_rhs; }
};

template<typename Derived>
template<typename Rhs>
inline auto TExprBase<Derived>::operator+(Rhs&& rhs) const &
{
  static_assert(internal::is_texpr_crtp<Rhs>::value, "only supports binary + between texprs");
  return BinaryTExpr<fornfdm::linops::internal::BinaryAdditionFO, const Derived&, Rhs>(
    derived(),
    std::forward<Rhs>(rhs),
    fornfdm::linops::internal::BinaryAdditionFO{}
  );
}

template<typename Derived>
template<typename Rhs>
inline auto TExprBase<Derived>::operator+(Rhs&& rhs) &&
{
  static_assert(internal::is_texpr_crtp<Rhs>::value, "only supports binary + between texprs");
  return BinaryTExpr<fornfdm::linops::internal::BinaryAdditionFO, Derived, Rhs>(
    std::move(derived()),
    std::forward<Rhs>(rhs),
    fornfdm::linops::internal::BinaryAdditionFO{}
  );
}

template<typename Derived>
template<typename Rhs>
inline auto TExprBase<Derived>::operator-(Rhs&& rhs) const &
{
  static_assert(internal::is_texpr_crtp<Rhs>::value, "only supports binary - between texprs");
  // a + (-b)
  return BinaryTExpr<fornfdm::linops::internal::BinaryAdditionFO, const Derived&, UnaryTExpr<fornfdm::linops::internal::UnaryNegateFO,Rhs>>(
    derived(),
    UnaryTExpr<fornfdm::linops::internal::UnaryNegateFO,Rhs>(
      std::forward<Rhs>(rhs),
      fornfdm::linops::internal::UnaryNegateFO{}
    ),
    fornfdm::linops::internal::BinaryAdditionFO{}
  );
}

template<typename Derived>
template<typename Rhs>
inline auto TExprBase<Derived>::operator-(Rhs&& rhs) &&
{
  static_assert(internal::is_texpr_crtp<Rhs>::value, "only supports binary - between texprs");
  // a + (-b)
  return BinaryTExpr<fornfdm::linops::internal::BinaryAdditionFO, Derived, UnaryTExpr<fornfdm::linops::internal::UnaryNegateFO,Rhs>>(
    std::move(derived()),
    UnaryTExpr<fornfdm::linops::internal::UnaryNegateFO,Rhs>(
      std::forward<Rhs>(rhs),
      fornfdm::linops::internal::UnaryNegateFO{}
    ),
    fornfdm::linops::internal::BinaryAdditionFO{}
  );
}

template<std::size_t Ith, typename Q, typename BinaryOp, typename LhsType, typename RhsType>
auto coeff(const Q& queue, const BinaryTExpr<BinaryOp,LhsType,RhsType>& xpr, const fornfdm::Mesh* m, fornfdm::Real t)
{
  constexpr bool lhs_is_scalar = internal::traits<LhsType>::max_arity == 0;
  constexpr bool rhs_is_scalar = internal::traits<RhsType>::max_arity == 0;

  if constexpr(lhs_is_scalar && rhs_is_scalar)
  {
    // both give scalars
    return xpr.functor()(
      coeff<Ith>(queue, xpr.lhs(), m, t),
      coeff<Ith>(queue, xpr.rhs(), m, t)
    );
  }
  else if constexpr(lhs_is_scalar && !rhs_is_scalar)
  {
    // 1 returns a scalar, the other a lambda
    fornfdm::Real c = coeff<Ith>(queue, xpr.lhs(), m, t);
    auto lam_func = coeff<Ith>(queue, xpr.rhs(), m, t); 
    return [c, lam_func, &f = xpr.functor()](const auto& coord){ return f(c, lam_func(coord)); }; 
  }
  else if constexpr(!lhs_is_scalar && rhs_is_scalar)
  {
    auto lam_func = coeff<Ith>(queue, xpr.lhs(), m, t); 
    fornfdm::Real c = coeff<Ith>(queue, xpr.rhs(), m, t);
    return [c, lam_func, &f = xpr.functor()](const auto& coord){ return f(lam_func(coord), c); };    
  }
  else // both have arity > 0
  {
    // both return a lambda.
    auto lhs_func = coeff<Ith>(queue, xpr.lhs(), m, t); 
    auto rhs_func = coeff<Ith>(queue, xpr.rhs(), m, t); 
    return [lhs_func, rhs_func, &f = xpr.functor()](const auto& coord){ return f(lhs_func(coord), rhs_func(coord)); };
  }
}

template<typename Q, typename BinaryOp, typename LhsType, typename RhsType>
auto rhsTerm(const Q& queue, const BinaryTExpr<BinaryOp,LhsType,RhsType>& xpr, const fornfdm::Mesh* m, fornfdm::Real t)
{
  // since BinaryTExpr simply adds/subtracts weights/coeffs,
  // safe to pull out the operation onto the large RHS expression.
  return xpr.functor(
    rhsTerm(queue, xpr.lhs(), m, t),
    rhsTerm(queue, xpr.rhs(), m, t)
  );
}

template<typename Q, typename BinaryOp, typename LhsType, typename RhsType>
auto inverseCoeff(const Q& queue, const BinaryTExpr<BinaryOp,LhsType,RhsType>& xpr, const fornfdm::Mesh* m, fornfdm::Real t)
{
  auto c = coeff<queue.numTimes-1>(queue,xpr,m,t);
  constexpr std::size_t necessary_arity = internal::traits<BinaryTExpr<BinaryOp,LhsType,RhsType>>::max_arity; 
  if constexpr(necessary_arity == 0)
  {
    return (fornfdm::Real(1.0) / c);
  }
  else
  {
    auto f = [c, m](std::size_t i)
    { 
      return (fornfdm::Real(1.0) / c(fornfdm::Coordinate<necessary_arity>(m, i))); 
    };
    using F = std::remove_reference_t<std::remove_cv_t<decltype(f)>>;
    std::size_t n = m->sizesProduct();
    return Eigen::CwiseNullaryOp<F,fornfdm::RealVector>(n,1,f).asDiagonal();
  }
}

} // end namespace texprs
} // end namespace fornfdm

#endif