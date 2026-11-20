// StepContexts.hpp
//
// 1. ) POD struct for 
// current time, 
// immediate next time, 
// and (optionally) container of times.
// 
// 2. ) Wrapper class for 
// shared_ptr<> to mesh domain in space
// raw pointer to TExpr
// raw pointer to RHS linear operator expression
// raw pointer to Queue of times + solutions
// raw pointer to entire solver  
// 
// JAF 3/21/2026 

#ifndef FORNFDM_OSTEPS_STEPCONTEXTS_H
#define FORNFDM_OSTEPS_STEPCONTEXTS_H

#include<cstddef>
#include<memory>
#include "../types.hpp"
#include "../Mesh.hpp"

namespace fornfdm{
  namespace osteps{

template<typename C = void>
struct TimeContext
{
  fornfdm::Real now; 
  fornfdm::Real next; 
  std::shared_ptr<const C> container; 
  TimeContext(fornfdm::Real t1=0.0, fornfdm::Real t2=0.0, std::shared_ptr<const C> c=nullptr)
    : now(t1), next(t2), container(c)
  {}
  TimeContext(const TimeContext& other)=delete; 
}; 

template<typename M = fornfdm::Mesh, typename L=void, typename R=void, typename Q=void, typename S=void>
class Context
{
  private:
    using MCleaned = std::remove_cv_t<std::remove_reference_t<M>>; 
    static_assert(std::is_same<fornfdm::Mesh, MCleaned>::value, "shared_ptr<> passed to osteps::Context must point to Mesh or const Mesh");

    // Member Data --------------------------------------------
    const std::shared_ptr<M> m_mesh;
    const L* const m_texpr; 
    R* const m_rhs_expr;
    Q* const m_queue;
    S* const m_solver; 
  public:
    // Contstructor ======================================================= 
    Context(std::shared_ptr<M> m=nullptr, const L* tx=nullptr, R* r=nullptr, Q* q=nullptr, S* s=nullptr)
      : m_mesh(std::move(m)), 
      m_texpr(tx), 
      m_rhs_expr(r),
      m_queue(q), 
      m_solver(s)
    {}

    Context(const Context& other) = default; 

    ~Context()=default; 
    
    // Member Funcs ==================================================
    // const / non const getters. 
    M* getMesh(){ return m_mesh.get(); }
    const M* getMesh() const { return m_mesh.get(); }
    std::shared_ptr<M> getMeshShared(){ return m_mesh; } 
    std::shared_ptr<const M> getMeshShared() const { return m_mesh; }; 
    const L* getTExpr() const { return m_texpr; }
    R* getRhsExpr(){ return m_rhs_expr; }
    const R* getRhsExpr() const { return m_rhs_expr; }
    Q* getQueue(){ return m_queue; } 
    const Q* getQueue() const { return m_queue; } 
    S* getSolver(){ return m_solver; }
    const S* getSolver() const { return m_solver; }  
}; 

// entry point functions
template<typename C=void>
auto make_time(
  fornfdm::Real t1=0.0, 
  fornfdm::Real t2=0.0, 
  std::shared_ptr<const C> c=nullptr
)
{
  return TimeContext(t1,t2,c); 
}

template<
  typename M = fornfdm::Mesh, 
  typename L=void, 
  typename R=void, 
  typename Q=void, 
  typename S=void
>
auto make_context(
  std::shared_ptr<M> m=nullptr, 
  const L* tx=nullptr, 
  R* r=nullptr, 
  Q* q=nullptr, 
  S* s=nullptr
)
{
  return Context(m,tx,r,q,s); 
}

  } // end namespace osteps 
} // end namespace fornfdm 

#endif // StepContexts.hpp 