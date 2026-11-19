// ProductTExpr.hpp
//
// TODO 
//
// JAF 11/5/2026 

#ifndef FORNFDM_TEXPRS_PRODUCTTEXPR_H
#define FORNFDM_TEXPRS_PRODUCTTEXPR_H

#include "../diffops/traits.hpp"
#include "../coeffs/CoeffBase.hpp"
#include "traits.hpp"
#include "TExprBase.hpp"

namespace fornfdm{
namespace texprs{

// forward declarations
template<typename LeftCoeff, typename RightTExpr>
class ProductTExpr;

namespace internal{

template<class LeftCoeff, class RightTExpr>
struct traits_impl<ProductTExpr<LeftCoeff,RightTExpr>>
{
  private:
  using lhs_traits_t = typename fornfdm::linops::internal::traits<LeftCoeff>;
  public:
  static constexpr bool is_unarop = false; 
  static constexpr bool is_binop = true; 
  static constexpr bool is_ternop = false;
  static constexpr bool is_timedep = lhs_traits_t::is_timedep || traits<RightTExpr>::is_timedep; 
  static constexpr std::size_t max_arity = std::max(lhs_traits_t::max_arity,traits<RightTExpr>::max_arity);  
  static constexpr std::size_t max_order = traits<RightTExpr>::max_order;  
};

} // end namespace internal

template<class LeftCoeff, class RightTExpr>
class ProductTExpr : public TExprBase<ProductTExpr<LeftCoeff,RightTExpr>>
{
  public:
    // Type Defs
    typedef typename std::remove_cv_t<std::remove_reference_t<LeftCoeff>> Lhs; 
    typedef typename std::remove_cv_t<std::remove_reference_t<RightTExpr>> Rhs; 
    typedef typename fornfdm::linops::internal::NestedStorage<LeftCoeff>::type LhsNested;
    typedef typename internal::NestedStorage<RightTExpr>::type RhsNested;

  private:
    // Member Data
    typename fornfdm::linops::internal::BinaryMultiplyFO m_functor;
    LhsNested m_lhs; 
    RhsNested m_rhs;

  public:
    // Constructors + Destructor
    ProductTExpr(LeftCoeff&& lhs, RightTExpr&& rhs)
      : m_lhs(lhs), m_rhs(rhs), m_functor() 
    {}

    // Member Funcs 
    const auto& functor() const{ return m_functor; }
    const auto& lhs() const { return m_lhs; }
    auto& lhs(){ return m_lhs; }
    const auto& rhs() const { return m_rhs; }
    auto& rhs(){ return m_rhs; }
};

template<std::size_t Ith, typename Q, typename LeftCoeff, typename RightTExpr>
auto coeff(const Q& queue, const ProductTExpr<LeftCoeff,RightTExpr>& xpr, const SharedConstMesh& m, fornfdm::Real t)
{
  using lhs_traits_t = typename fornfdm::linops::internal::traits<LeftCoeff>;
  constexpr bool lhs_is_scalar = lhs_traits_t::max_arity == 0;
  constexpr bool rhs_is_scalar = internal::traits<RightTExpr>::max_arity == 0;

  if constexpr(lhs_is_scalar && rhs_is_scalar)
  {
    // both give scalars
    auto c = coeff<Ith>(queue, xpr.rhs(), m, t);
    if constexpr(lhs_traits_t::is_timedep)
    {
      return xpr.lhs().callable()(t) * c;
    }
    else
    {
      return xpr.lhs().callable()() * c;
    }
  }
  else if constexpr(lhs_is_scalar && !rhs_is_scalar)
  {
    // 1 returns a scalar, the other a lambda
    auto lam_func = coeff<Ith>(queue, xpr.rhs(), m, t); 
    if constexpr(lhs_traits_t::is_timedep)
    {
      auto c = xpr.lhs().callable()(t);
      return [c, lam_func](const auto& coord){ return c*lam_func(coord); }; 
    }
    else
    {
      auto c = xpr.lhs().callable()();
      return [c, lam_func](const auto& coord){ return c*lam_func(coord); }; 
    }
  }
  else if constexpr(!lhs_is_scalar && rhs_is_scalar)
  {
    auto lam_func = [&g = xpr.lhs().callable(), t](const auto& coord)
    {
      if constexpr(lhs_traits_t::is_timedep)
      {
        return coord.applyBindFirst(g,t); 
      }
      else
      {
        return coord.apply(g);
      }
    };
    fornfdm::Real c = coeff<Ith>(queue, xpr.rhs(), m, t);
    return [c, lam_func](const auto& coord){ return lam_func(coord) * c; };    
  }
  else // both have arity > 0
  {
    // both return a lambda.
    auto lhs_func = [&g = xpr.lhs().callable(), t](const auto& coord)
    {
      if constexpr(lhs_traits_t::is_timedep)
      {
        return coord.applyBindFirst(g,t); 
      }
      else
      {
        return coord.apply(g);
      }
    };
    auto rhs_func = coeff<Ith>(queue, xpr.rhs(), m, t); 
    return [lhs_func, rhs_func](const auto& coord){ return lhs_func(coord)*rhs_func(coord); };
  }
}

template<typename Q, typename LeftCoeff, typename RightTExpr>
auto rhsTerm(const Q& queue, const ProductTExpr<LeftCoeff,RightTExpr>& xpr, const fornfdm::Mesh* m, fornfdm::Real t)
{
  using lhs_traits_t = fornfdm::linops::internal::traits<LeftCoeff>;
  if constexpr(lhs_traits_t::max_arity == 0)
  {
    if constexpr(lhs_traits_t::is_timedep)
    {
      return xpr.lhs().callable()(t) * rhsTerm(queue, xpr.rhs(), m, t);
    }
    else
    {
      return xpr.lhs().callable() * rhsTerm(queue, xpr.rhs(), m, t);
    }
  }
  else
  {
    auto lam_func = [&g = xpr.lhs().callable(), t, m](std::size_t idx)
    {
      fornfdm::Coordinate<lhs_traits_t::max_arity> coord(m,idx);
      if constexpr(lhs_traits_t::is_timedep)
      {
        return coord.applyBindFirst(g,t);
      }
      else
      {
        return coord.apply(g);
      }
    };
    auto vec = rhsTerm(queue,xpr.rhs(), m, t);
    auto final_lam = [lam_func, vec](std::size_t idx){ return lam_func(idx) * vec[idx]; };
    using F = std::remove_reference_t<std::remove_cv_t<decltype(final_lam)>>;
    return Eigen::CwiseNullaryOp<F,fornfdm::Vector>(vec.rows(), 1, final_lam);
  }
}

template<typename Q, typename LeftCoeff, typename RightTExpr>
auto inverseCoeff(const Q& queue, const ProductTExpr<LeftCoeff,RightTExpr>& xpr, const fornfdm::Mesh* m, fornfdm::Real t)
{
  auto c = coeff<queue.numTimes-1>(queue,xpr,m,t);
  constexpr std::size_t necessary_arity = internal::traits<ProductTExpr<LeftCoeff,RightTExpr>>::max_arity; 
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