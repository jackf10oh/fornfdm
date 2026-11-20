// ImplicitSolver.hpp
//
// Mediator class .. 
// in order to solver PDEs with finite difference methods 
// namely implicit steps from t(n-1) to t(n) 
//
// JAF 3/4/2026 

#ifndef FORNFDM_SOLVERS_IMPLICITSOLVER_H
#define FORNFDM_SOLVERS_IMPLICITSOLVER_H

#include<cstdint>
#include<Eigen/Core>
#include<Eigen/SparseCore>
#include<Eigen/IterativeLinearSolvers> // BiCGSTAB sparse iterative solver  
#include "../diffops/traits.hpp" // check RHS is_timedep
#include "../texprs/traits.hpp" // check LHS is time derivatives 
#include "../texprs/Queue.hpp"
#include "../outside_steps/StepContexts.hpp"  // feed to outside steps tuple 
#include "../outside_steps/OStepBase.hpp" // StepType scoped enumeration 
#include "../utilities/Identity.hpp"
#include "SolverBase.hpp"
#include "SolverArgs.hpp"
#include "SavePolicies.hpp"

namespace fornfdm{
  namespace solvers{ 

// template<typename LhsType, typename RhsType, typename OStepTup>
template<
  typename LhsType, 
  typename RhsType, 
  typename OStepTup, 
  class SparseIterativeSolver=Eigen::BiCGSTAB<fornfdm::CSRMatrix>
>
class ImplicitSolver : public SolverBase<ImplicitSolver<LhsType, RhsType, OStepTup,SparseIterativeSolver>, LhsType, RhsType, OStepTup> 
{
  private:
    // Type Defs -------------------------------------- 
    using Base = SolverBase<ImplicitSolver<LhsType, RhsType, OStepTup,SparseIterativeSolver>, LhsType, RhsType, OStepTup>;   
    
  private:
    // Member Data -------------------------------------
    std::unique_ptr<SparseIterativeSolver> m_iterative_solver;

  public:
    // Constructors + Destructor ===========================

    ImplicitSolver()=delete; 

    ImplicitSolver(
      const LhsType& l_init, 
      RhsType& r_init, 
      OStepTup ostep_init, 
      std::unique_ptr<SparseIterativeSolver> s_init = std::make_unique<SparseIterativeSolver>()
    )
      : Base(l_init, r_init, std::move(ostep_init)), m_iterative_solver(std::move(s_init))
    {
      static_assert(std::is_same_v<typename SparseIterativeSolver::MatrixType, fornfdm::CSRMatrix>, "must use iterative solver on fornfdm::CSRMatrix"); 
    }

    // not copyable!
    ImplicitSolver(const ImplicitSolver& other)=delete; 

    // moveable 
    ImplicitSolver(ImplicitSolver&& other)=default; 

    // destructor 
    ~ImplicitSolver()=default; 

    // Member Functions ===========================================================
    auto& getIterativeSolver(){ return *m_iterative_solver; }
    const auto& getIterativeSolver() const { return *m_iterative_solver; }

    template<typename M, typename C, typename Pred = LastSaver>
    auto calculate(SolverArgs<M,C> args, Pred save_policy = {}) 
    {
      // setup time context 
      auto it = std::next(args.times->cbegin(), args.initialConditions.size()-1); 
      auto time_ctx = fornfdm::osteps::make_time(*it, 0.0, std::move(args.times));
      ++it; 

      // setup queue 
      constexpr std::size_t m = fornfdm::texprs::internal::traits<LhsType>::max_order;
      fornfdm::texprs::Queue<m+1,(m+1)*(m+1)> queue{};
      queue.pushTimeRange(time_ctx.container->cbegin(), it); 
      auto sol_end = queue.pushSolutionRange(args.initialConditions.begin(), args.initialConditions.end()); 
      for(auto sol_it=args.initialConditions.begin(); sol_it != sol_end; ++sol_it)
      {
        // Eventually want save policies to take by (solution, time) 
        save_policy.saveSolution(std::move(*sol_it)); 
      }

      // set up operators. 
      this->m_rhs.setMesh(args.mesh); 

      // set up context 
      auto ctx = fornfdm::osteps::make_context(std::move(args.mesh), &(this->m_lhs), &(this->m_rhs), &queue, this); 

      // store allocated memory between steps in solver hot loop  
      fornfdm::CSRMatrix stencil; 
      fornfdm::Vector rhs_vector;  
      fornfdm::Vector solution_u;  

      // hot loop through times
      auto end = time_ctx.container->cend();
      for(; it!= end; ++it)
      { 
        time_ctx.next = *it; 

        // using right side of [t(n), t(n+1)] time step 
        queue.pushTime(time_ctx.next); 
        queue.calculateWeights(time_ctx.next, fornfdm::texprs::internal::traits<LhsType>::max_order);
        
        // outside steps before any type of linear algebra is performed... 
        this->template tupleBeforeLinAlgebra<fornfdm::osteps::StepType::Implicit>(time_ctx,ctx);

        // store the matrix into stencil 
        if constexpr(fornfdm::linops::internal::traits<RhsType>::is_timedep)
        {
          this->m_rhs.setTime(time_ctx.next);
        }
        auto c = fornfdm::texprs::inverseCoeff(queue, this->m_lhs, ctx.getMesh(), time_ctx.next);
        std::size_t s = this->m_rhs.rows(); 
        stencil = fornfdm::utils::Identity(s,s) - (c * (this->m_rhs.toEigen()));
        
        // outside steps matrix before step
        this->template tupleApplyBeforeMat<fornfdm::osteps::StepType::Implicit>(stencil, time_ctx, ctx); 

        // store the expression into a vector 
        rhs_vector = -c * fornfdm::texprs::rhsTerm(queue,this->m_lhs, ctx.getMesh(), time_ctx.next);

        // outside steps vector before step 
        this->template tupleApplyBeforeVec<fornfdm::osteps::StepType::Implicit>(rhs_vector, time_ctx, ctx); 

        // Implicit Step (I - D(t+1))*U(n+1) = rhs 
        m_iterative_solver->compute(stencil); 
        solution_u = m_iterative_solver->solveWithGuess(rhs_vector,rhs_vector);
        
        // outside steps solution after step(next_sol) 
        this->template tupleApplyAfterVec<fornfdm::osteps::StepType::Implicit>(solution_u, time_ctx, ctx); 

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

#endif // ExplicitSolver.hpp 