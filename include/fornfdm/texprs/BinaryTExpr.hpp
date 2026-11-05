// BinaryTExpr.hpp
//
// Class holding two texprs and a binary functor.
// currently only useful to sums A + B 
//
// JAF 10/16/2026

#ifndef FORNFDM_TEXPRS_BINARYTEXPR_H
#define FORNFDM_TEXPRS_BINARYTEXPR_H

#include<cstddef>
#include "../diffops/traits.hpp"
#include "traits.hpp"
#include "functors.hpp"
#include "UnaryTExpr.hpp"

namespace fornfdm{
namespace texprs{

// Forward Declaration
template<class BinaryOp, class LhsType, class RhsType>
struct BinaryTExpr; 

namespace internal{

template<class BinaryOp, class LhsType, class RhsType>
struct traits_impl<BinaryTExpr<BinaryOp, LhsType, RhsType>>
{
  static constexpr bool is_texpr = true;
  static constexpr bool is_unarop = false; 
  static constexpr bool is_binop = true; 
  static constexpr bool is_ternop = false; 
  static constexpr bool is_timedep = traits<LhsType>::is_timedep || traits<RhsType>::is_timedep; 
  static constexpr std::size_t max_arity = std::max(traits<LhsType>::max_arity,traits<RhsType>::max_arity);  
  static constexpr std::size_t max_order = std::max(traits<LhsType>::max_order,traits<RhsType>::max_order);  
  using orders_seq = typename fornfdm::linops::internal::merge_orders<typename traits<LhsType>::orders_seq,typename traits<RhsType>::orders_seq>::type;
};

// ExecutorImpl 

// every executor needs an 
// - assign to. 
// - add to. 
// which are hooked based on the arity 
// so scalars are created inplace, mem copied into first block,
// since its entry wise, we're ok with just producing a scalar result for each. 
// and using fold expression + ... + to get each entry

} // end namespace internal

template<class BinaryOp, class LhsType, class RhsType>
class BinaryTExpr
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
    auto makeSummationTerms()
    {
      return m_functor(m_lhs.makeSummationTerms(), m_rhs.makeSummationTerms());
    }

    const auto& functor() const { return m_functor; }
    const auto& lhs() const { return m_lhs; }
    auto& lhs(){ return m_lhs; }
    const auto& rhs() const { return m_rhs; }
    auto& rhs(){ return m_rhs; }
};

template<
  typename LeftArg,
  typename RightArg, 
  typename = std::enable_if_t<(fornfdm::texprs::internal::traits<LeftArg>::is_texpr && fornfdm::texprs::internal::traits<RightArg>::is_texpr), void>
>
auto operator+(LeftArg&& lhs, RightArg&& rhs)
{
  return BinaryTExpr<internal::BinaryAdditionFO, LeftArg, RightArg>(
    std::forward<LeftArg>(lhs),
    std::forward<RightArg>(rhs),
    internal::BinaryAdditionFO{}
  );
}

template<
  typename LeftArg,
  typename RightArg, 
  typename = std::enable_if_t<internal::traits<LeftArg>::is_texpr && internal::traits<RightArg>::is_texpr, void>
>
auto operator-(LeftArg&& lhs, RightArg&& rhs)
{
  // lhs + (-rhs)
  return BinaryTExpr<internal::BinaryAdditionFO, LeftArg, UnaryTExpr<internal::UnaryNegateFO,RightArg>>(
    std::forward<LeftArg>(lhs),
    UnaryTExpr<internal::UnaryNegateFO,RightArg>(
      std::forward<RightArg>(rhs),
      internal::UnaryNegateFO{}
    ),
    internal::BinaryAdditionFO{}
  );
}

} // end namespace texprs
} // end namespace fornfdm

#endif