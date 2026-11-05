// NthTimeDeriv.hpp
//
// Concrete class derived from TimeDerivBase 
// That represent operator like Ut, Utt, etc 
// for time derivative template expressions 
//
// JAF 1/15/2026 

#ifndef FORNFDM_TEXPRS_NTHTIMEDERIV_H
#define FORNFDM_TEXPRS_NTHTIMEDERIV_H

#include<cstdint>
#include "traits.hpp"

namespace fornfdm{
namespace texprs{

// forward declaration
template<std::size_t N>
class NthTimeDeriv;

namespace internal{

template<std::size_t N>
struct traits_impl<NthTimeDeriv<N>>
{
  static constexpr bool is_texpr = true;
  static constexpr bool is_unarop = false; 
  static constexpr bool is_binop = false; 
  static constexpr bool is_ternop = false; 
  static constexpr bool is_timedep = false; 
  static constexpr std::size_t max_arity = N;  
  static constexpr std::size_t max_order = N;  
  using orders_seq = typename std::index_sequence<N>;
};

} // end namespace internal 

template<std::size_t N>
class NthTimeDeriv
{
  public:
    // Memeber Data 
    static constexpr std::size_t order = N; 

    // Constructors + Destructor
    NthTimeDeriv()=default;
    NthTimeDeriv(const NthTimeDeriv& other)=default;
    ~NthTimeDeriv()=default; 

    // Member Funcs
    auto makeSummationTerms() &
    {
      return std::tuple<NthTimeDeriv<N>&>(*this);
    }

    auto makeSummationTerms() &&
    {
      return std::tuple<NthTimeDeriv>(*this);
    }
}; 

// template<std::size_t N, std::size_t M>
// class ExecutorImpl<Executor<NthTimeDeriv<N>,M>>
// ExecutorImpl
// {
//   public:
//     // Constructors + Destructor 
//     ExecutorImpl()=default;
//     ExecutorImpl(const NthTimeDeriv<N>& other)=delete;
//     ~ExecutorImpl()=default;

//     // Member Functions 
  
//     // non op.
//     void setMesh(const std::shared_ptr<const fornfdm::Mesh>& m){}

//     // non op.
//     void setTime(fornfdm::Real t){}

//     // 
//     void calculate(fornfdm::Real t)
//     {
//       // update internal fornberg weights 
//       utils::fornberg(
//         this->m_stored_times.cbegin(), this->m_stored_times.cend(),
//         t, N,
//         this->m_weights_arr.begin()
//       );
//       m_inv_coeff = (1.0) / coeffAt<derived().numStoredTimes-1, derived().numStoredTimes>(derived().getWeights().data());
//     }

//     const fornfdm::Real& getInvCoeff() const { return m_inv_coeff; } 
//     auto getRhsExpression() const 
//     {
//       // some disgusting fold expression of T_(n-3) * U(n-3) + T_(n-2) * U(n-2) + T_(n-1) * U(n-1)
//       return -m_inv_coeff * getRhsExpression_impl(std::make_index_sequence<derived().numStoredSols>{});
//     }

//   private:
//     // some implementations...
//     auto& derived(){ return *static_cast<Executor<NthTimeDeriv<N>,M>*>(this);}
//     const auto& derived() const { return *static_cast<const Executor<NthTimeDeriv<N>,M>*>(this);}

//     template<std::size_t ithCol, std::size_t numCols>
//     fornfdm::Real coeffAt(const fornfdm::Real* v) const 
//     {
//       return v[numCols * N + ithCol]; 
//     }

//     template<std::size_t... Idxs>
//     auto getRhsExpression_impl(std::index_sequence<Idxs...> idxs) const 
//     {
//       // for example, using k=4 times, from the expression 
//       // c(n-2)*U(n-2) + c(n-1)*U(n-1) + c(n)*U(n) + c(n+1)*U(n+1)
//       // we need to construct the expression that is all terms except the (n+1) in the future
//       // c(n-2)*U(n-2) + c(n-1)*U(n-1) + c(n)*U(n)
//       return ( ( coeffAt<Idxs,derived().numStoredTimes>(derived().getWeights().data()) * derived().getStoredSolutions()[Idxs] ) + ...);
//     }
//     // Member Data 
//     fornfdm::Real m_inv_coeff;
// };

} // end namespace texprs 
} // end namespace fornfdm 

#endif // NthTimeDeriv.hpp