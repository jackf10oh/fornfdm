// UnaryTExpr.hpp
//
// Class holding a sinlge texpr and a unary functor. 
//
// JAF 10/16/2026

#ifndef FORNFDM_TEXPRS_UNARYTEXPR_H
#define FORNFDM_TEXPRS_UNARYTEXPR_H

#include<cstddef>
#include "../types.hpp"
#include "traits.hpp"

namespace fornfdm{
namespace texprs{

// forward declaration
template<class UnaryOp, class XprType>
struct UnaryTExpr;

namespace internal{

template<class UnaryOp, class XprType>
struct traits_impl<UnaryTExpr<UnaryOp,XprType>>
{
  static constexpr bool is_texpr = true;
  static constexpr bool is_unarop = true; 
  static constexpr bool is_binop = false; 
  static constexpr bool is_ternop = false; 
  static constexpr bool is_timedep = traits<XprType>::is_timedep; 
  static constexpr std::size_t max_arity = traits<XprType>::max_arity;  
  static constexpr std::size_t max_order = traits<XprType>::max_order;  
  using orders_seq = typename traits<XprType>::orders_seq;
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

template<class UnaryOp, class XprType>
class UnaryTExpr
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
    // Constructors 
    UnaryTExpr()=delete; 

    UnaryTExpr(XprType&& xpr, UnaryOp func = UnaryOp())
      : m_xpr(std::forward<XprType>(xpr)), m_functor(func) 
    {}

    UnaryTExpr(const UnaryTExpr& other)=default;

    ~UnaryTExpr()=default;

    // Member Funcs
    auto makeSummationTerms()
    {
      return (m_functor(m_xpr.makeSummationTerms()));
    }
    const auto& functor() const { return m_functor; }
    const auto& nestedExpression() const { return m_xpr; }
    auto& nestedExpression(){ return m_xpr; }
};

} // end namespace texprs 
} // end namespace fornfdm

#include "functors.hpp"

namespace fornfdm{
namespace texprs{

template<typename XprType, typename = std::enable_if_t<internal::traits<XprType>::is_texpr, void>>
auto operator-(XprType&& xpr)
{
  return UnaryTExpr<internal::UnaryNegateFO, XprType>(std::forward<XprType>(xpr),internal::UnaryNegateFO{});
}

template<typename C, typename XprType, typename = std::enable_if_t<std::is_convertible<C,fornfdm::Scalar>::value && internal::traits<XprType>::is_texpr>>
auto operator*(C&& c, XprType&& xpr)
{
  return UnaryTExpr<internal::UnaryScalarMultiplyFO<C>,XprType>(std::forward<XprType>(xpr), internal::UnaryScalarMultiplyFO<C>(std::forward<C>(c)));
}

// template<class UnaryOp, class XprType>
// class Executor<UnaryTExpr<UnaryOp,XprType>, std::size_t numTimesMin = internal::traits<UnaryTExpr<UnaryOp,XprType>>::max_order+1> 
//   : public ExecutorBase<UnaryTExpr<UnaryOp,XprType>, numTimesMin>
// {
//     // anything else? 
//   public:
//     // Constructors + Destructor ? 
//     Executor()
//     // Member Functions 

//     // Member Data 
// }

} // end namespace texprs
} // end namespace fornfdm

#endif