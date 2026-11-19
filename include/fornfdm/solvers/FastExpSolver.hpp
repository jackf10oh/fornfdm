// FastExpSolver.hpp
//
// class that ducks some of the main contracts in Solvers interface. 
// notably there is no beforeLinAlgebra, applyBeforeMat, or applyBeforeVec
// the full CSRMatrix is never calculated, so RHS operator is possibly time dependent.
//
// JAF 8/4/2026

#ifndef FORNFDM_SOLVERS_FastExpSolver_H
#define FORNFDM_SOLVERS_FastExpSolver_H

#include<cstdint>
#include<Eigen/Core>
#include<Eigen/SparseCore>
#include "../diffops/traits.hpp" // check RHS is_timedep
#include "../texprs/traits.hpp" // check LHS is time derivatives 
#include "../texprs/Queue.hpp"
#include "../outside_steps/StepContexts.hpp"  // feed to outside steps tuple 
#include "../outside_steps/OStepBase.hpp" // StepType scoped enumeration 
#include "SolverBase.hpp"
#include "SolverArgs.hpp"
#include "SavePolicies.hpp"

namespace fornfdm{
  namespace solvers{ 

template<typename LhsType, typename RhsType, typename OStepTup>
class FastExpSolver : public SolverBase<FastExpSolver<LhsType, RhsType, OStepTup>, LhsType, RhsType, OStepTup> 
{
  private:
    // Type Defs -------
    using Base = SolverBase<FastExpSolver<LhsType, RhsType, OStepTup>, LhsType, RhsType, OStepTup>; 
  
    public:
    // Constructors + Destructor ==============================================

    FastExpSolver()=delete; 

    FastExpSolver(const LhsType& l_init, RhsType& r_init, OStepTup ostep_init)
      : Base(l_init, r_init, std::move(ostep_init))
    {}

    // not copyable! 
    FastExpSolver(const FastExpSolver& other)=delete; 

    // moveable
    FastExpSolver(FastExpSolver&& other)=default;  

    // destructor  
    ~FastExpSolver()=default; 

    // Member Functions ======================================================
    template<class M, class C, class Pred = LastSaver>
    auto calculate(SolverArgs<M, C> args, Pred save_policy = {}) 
    {
      // Fast Solvers only accept a SharedConstMesh and const TimeArgs for arguments!
      static_assert(std::is_same<M, const fornfdm::Mesh>::value, "FastImpSolver only takes const Mesh for domain.");
      static_assert(std::is_same<C, const solvers::TimeArg>::value, "FastImpSolver only takes TimeArg for times.");

      fornfdm::Real t_start = args.times->getStart();
      fornfdm::Real t_stop = args.times->getStop();
      fornfdm::Real t_stepsize = args.times->getStepSize();
      std::size_t end = args.times->getNumSteps();

      // setup time context 
      std::size_t step_n = args.initialConditions.size()-1;
      auto time_ctx = fornfdm::osteps::make_time(t_start + step_n*t_stepsize, 0.0, std::move(args.times));
      ++step_n; 

      // setup queue 
      constexpr std::size_t m = fornfdm::texprs::internal::traits<LhsType>::max_order;
      fornfdm::texprs::Queue<m+1, m> queue{};

      auto n = step_n - queue.numTimes + 1;
      for(auto i=1; i<queue.numTimes; ++i){
        queue.getTimes()[i] = t_start + t_stepsize * n;
        ++n;
      };

      auto sol_end = queue.pushSolutionRange(args.initialConditions.begin(), args.initialConditions.end()); 
      for(auto sol_it=args.initialConditions.begin(); sol_it != sol_end; ++sol_it)
      {
        save_policy.saveSolution(std::move(*sol_it)); 
      }

      // set up operators. 
      this->m_rhs.setMesh(args.mesh); 

      // set up context 
      auto ctx = fornfdm::osteps::make_context(
        std::move(args.mesh), 
        &(this->m_lhs), 
        &(this->m_rhs), 
        &queue,
        this
      ); 

      fornfdm::Vector solution_u;  
      // hot loop through times
      for(; step_n != end; ++step_n)
      { 
        time_ctx.next = t_start + step_n * t_stepsize; 
        queue.pushTime(time_ctx.next);
        queue.calculateWeights(time_ctx.next); // TODO no need to repeat weight calculation?
        this->m_rhs.setTime(time_ctx.now);

        // no beforeLinAlgebra!
        // no applyBeforeMat!
        // no applyBeforeVec! 

        // Explicit Step 
        auto c = fornfdm::texprs::inverseCoeff(queue, this->m_lhs, ctx.getMesh(), time_ctx.now);
        auto rhs = fornfdm::texprs::rhsTerm(queue, this->m_lhs, ctx.getMesh(), time_ctx.now);
        solution_u = c * ((this->m_rhs.toEigen() * queue.getCurrentSolution()) - rhs); 
        
        // outside steps solution after step(next_sol) 
        this->template tupleApplyAfterVec<fornfdm::osteps::StepType::Explicit>(solution_u, time_ctx, ctx); 

        // save oldest solution before it goes 
        save_policy.saveSolution(queue.getExpiringSolution()); 

        // push solution into queue
        queue.rotateSolutions(1); 
        queue.getSolutions().back().swap(solution_u); 

        // advance time_ctx 1 step in time 
        time_ctx.now = time_ctx.next; 
      }        
      // Give remaining solutions to save_policy 
      for(auto i=0; i < queue.numSolutions-1; ++i) save_policy.saveSolution( std::move(queue.getSolutions()[i])); 

      // save_policy also determines return type / how to handle last solution
      return save_policy.saveLastSolution(std::move( queue.getCurrentSolution() )); 
    } 
}; 

  } // end namespace solvers
} // end namespace fornfdm 

#endif // FastExpSolver.hpp 