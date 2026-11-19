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
#include "TExprBase.hpp"
#include "Queue.hpp"

namespace fornfdm{
namespace texprs{

// forward declaration
template<std::size_t N>
class NthTimeDeriv;

namespace internal{

template<std::size_t N>
struct traits_impl<NthTimeDeriv<N>>
{
  static constexpr bool is_unarop = false; 
  static constexpr bool is_binop = false; 
  static constexpr bool is_ternop = false; 
  static constexpr bool is_timedep = false; 
  static constexpr std::size_t max_arity = 0;  
  static constexpr std::size_t max_order = N;  
};

} // end namespace internal 

template<std::size_t N>
class NthTimeDeriv : public TExprBase<NthTimeDeriv<N>>
{
  public:
    // Memeber Data 
    static constexpr std::size_t order = N; 

    // Constructors + Destructor
    NthTimeDeriv()=default;
    NthTimeDeriv(const NthTimeDeriv& other)=default;
    ~NthTimeDeriv()=default; 
}; 

template<std::size_t Ith, typename Q, std::size_t N>
inline auto coeff(const Q& queue, const NthTimeDeriv<N>& xpr, const SharedConstMesh& m, fornfdm::Real t)
{
  return queue.getWeights()[N * queue.numStoredTimes + Ith];
}

template<typename Q, std::size_t N>
inline auto rhsTerm(const Q& queue, const NthTimeDeriv<N>& xpr, const SharedConstMesh& m, fornfdm::Real t)
{
  auto lam = [&queue, &xpr,&m,t](auto... Idxs)
  {
    return ((coeff<Idxs>(queue,xpr,m,t) * queue.getStoredSolutions()[Idxs]) + ...);
  }(std::make_index_sequence<queue.numStoredTimes-1>{});
}

template<typename Q, std::size_t N>
inline auto inverseCoeff(const Q& queue, const NthTimeDeriv<N>& xpr, const SharedConstMesh& m, fornfdm::Real t)
{
  return ( fornfdm::Real(1.0) / coeff<queue.numStoredTimes-1>(queue, xpr, m, t) );
}

} // end namespace texprs 
} // end namespace fornfdm 

#endif // NthTimeDeriv.hpp