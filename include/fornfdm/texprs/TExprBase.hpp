// TexprBase.hpp
//
// base class define interface for ALL time expressions
//
// JAF 9/19/2026 

#ifndef FORNFDM_TEXPRS_TEXPRBASE_H
#define FORNFDM_TEXPRS_TEXPRBASE_H

namespace fornfdm{
namespace texprs{

template<typename Derived>
class TExprBase;

namespace internal{

template<typename Derived>
struct traits_impl<TExprBase<Derived>> : traits<Derived>{};

} // end namespace internal 

template<typename Derived>
class TExprBase
{
  public:
    // Member Functions 
    Derived& derived(){ return *static_cast<Derived*>(this); }
    const Derived& derived() const{ return *static_cast<const Derived*>(this); }

    // Operators

    // Binary Addition (L val)
    template<typename Rhs>
    auto operator+(Rhs&& rhs) const&;
    // (R val)
    template<typename Rhs>
    auto operator+(Rhs&& rhs) &&;

    // Binary Subtraction (L val)
    template<typename Rhs>
    auto operator-(Rhs&& rhs) const &;
    // (R val)
    template<typename Rhs>
    auto operator-(Rhs&& rhs) &&;

    // Unary Negation (L val)
    auto operator-() const &;
    // (R val)
    auto operator-() &&;
};

// retrieve the fornberg fdm weight at a index
template<std::size_t Ith, typename Q, typename Derived>
inline auto coeff(const Q& queue, const TExprBase<Derived>& xpr, const fornfdm::Mesh* m, fornfdm::Real t)
{
  return coeff<Ith>(queue, xpr.derived(), m, t); 
}

// combine a*U(n-1)+...+ c*U(n) so that U(n+1) can be solved for
template<typename Q, typename Derived>
inline auto rhsTerm(const Q& queue, const TExprBase<Derived>& xpr, const fornfdm::Mesh* m, fornfdm::Real t)
{
  rhsTerm(queue, xpr.derived(), m, t);
}

template<typename Q, typename Derived>
inline auto inverseCoeff(const Q& queue, const TExprBase<Derived>& xpr, const fornfdm::Mesh* m, fornfdm::Real t)
{
  return inverseCoeff(queue, xpr.derived(), m, t);
}

} // end namespace texprs
} // end namespace fornfdm

#endif // TexprBase.hpp