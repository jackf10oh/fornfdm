// ExecutorBase.hpp
//
// class to manage circular buffer of time + solutions. 
// Specialized for each kind of TExpr. 
//
// specialization will have to implement getRhsExpression and getInvCoeff. 
//
// JAF 1/16/2025 

#ifndef FORNFDM_TEXPRS_EXECUTORBASE_H
#define FORNFDM_TEXPRS_EXECUTORBASE_H

#include<algorithm>
#include<array>
#include<cstdint>
#include<Eigen/Core>
#include "../types.hpp" // Scalar, DiagMatrix
#include "../utilities/fornberg.hpp" // not using stack calc. we need a std::array<Real> not std::array<Scalar> 
#include "traits.hpp" 

namespace fornfdm{
  namespace texprs{

// specialized according to each expression type.
// TODO nothing goes to this class on construction? 
template<typename Derived>
class ExecutorImpl{};

template<typename TimeDeriv, std::size_t numTimesMin = 0>
class Executor : public ExecutorImpl<Executor<TimeDeriv, numTimesMin>> 
{
  public:
    // Type Defs ------------------------------------------- 
    using WrappedExpression = TimeDeriv; 
    // number of nodes used in Fornberg algorithm 
    static constexpr std::size_t numStoredTimes = std::max(numTimesMin, internal::traits<TimeDeriv>::max_order+1); 
    static constexpr std::size_t numStoredSols = numStoredTimes-1;

  protected:
    // Member Data ---------------------------------

    // reference to the expression the executor is working on. 
    TimeDeriv& m_wrapped;

    // list of t0, t1, ..., tn 
    std::array<fornfdm::Real, numStoredTimes> m_stored_times; 

    // list of solutions u0, u1, ..., un-1 at times t0, t1, ..., tn-1 
    std::array<fornfdm::Vector, numStoredSols> m_stored_sols; 

    // forberg weights calculator 
    std::array<fornfdm::Real, numStoredTimes * (internal::traits<TimeDeriv>::max_order+1)> m_weights_arr; 

  public:
    // Constructors + Destructor =======================================
    
    // default 
    ExecutorBase()=delete; 

    // from expressions of Time derivatives
    ExecutorBase(TimeDeriv& expr_init) 
      : m_wrapped(expr_init), m_stored_times(), m_stored_sols(), m_weights_arr()
    {}

    // copy 
    ExecutorBase(const ExecutorBase& other)=delete;
    
    // move
    ExecutorBase(ExecutorBase&& other)=default; 

    // destructor 
    ~ExecutorBase()=default; 

    // Member Funcs =======================================

    // Getters to wrapped expression
    TimeDeriv& getWrappedExpression(){ return m_wrapped; } 
    const TimeDeriv& getWrappedExpression() const { return m_wrapped; } 

    // Getters to stored times. 
    auto& getStoredTimes(){ return m_stored_times; }; 
    const auto& getStoredTimes() const { return m_stored_times; }; 

    // getters to stored solutions 
    auto& getStoredSolutions(){ return m_stored_sols; }
    const auto& getStoredSolutions() const { return m_stored_sols; }

    // returns ref to newest solution 
    fornfdm::Vector& getCurrentSolution(){ return m_stored_sols.back(); }
    const fornfdm::Vector& getCurrentSolution() const { return m_stored_sols.back(); }

    // return ref to first elem in m_stored_sols. Gives an opportunity to move it elsewhere before overwritten in ConsumeSolution  
    fornfdm::Vector& getExpiringSolution(){ return m_stored_sols.front(); }
    const fornfdm::Vector& getExpiringSolution() const { return m_stored_sols.front(); }

    // getter to internal weights array 
    const auto& getWeights() const { return m_weights_arr; }
    
    // consume a time. push back all previous
    void pushTime(fornfdm::Real t)
    {
      // all values in m_stored_time have been left shifted by 1
      std::move(std::next(m_stored_times.begin()), m_stored_times.end(), m_stored_times.begin()); 
      // most recent time set to t 
      m_stored_times.back() = t;
    }

    // sets current times from start,end iterators. returns iterator such that [start, iterator) isn't pushed into times. 
    template<typename Iter>
    Iter pushTimeRange(Iter start, Iter end){ 
      std::size_t d = std::distance(start,end);        
      if(d >= numStoredTimes) 
      {
        // throw std::runtime_error("Executor setStoredTimes(it,it) error: distance(start,end) > numStoredTimes");
        // only copy from rightmost times 
        // std::copy(std::next(m_stored_times.begin(),d-numStoredTimes), m_stored_times.end(), m_stored_times.begin()); 
        auto it = std::next(start, d - numStoredTimes); 
        std::copy(it,end,m_stored_times.begin()); 
        return it; 
      } 
      else
      {
        // only copy into rightmost times 
        std::copy(std::next(m_stored_times.begin(),d), m_stored_times.end(), m_stored_times.begin()); 
        std::copy(start,end,std::next(m_stored_times.begin(),numStoredTimes-d)); 
        return start; 
      }
    }

    // rotates stored solutions. meaning the put to back of array. useful to keep memory allocated in hot loops
    void rotateStoredSolutions(std::size_t idx=1)
    {
      std::rotate(m_stored_sols.begin(), std::next(m_stored_sols.begin(),idx), m_stored_sols.end()); 
    }
    
    // consume a solution. push back all previous 
    void pushSolution(fornfdm::Vector sol)
    { 
      /* same idea as pushTime(). with move semantics*/
      std::move(std::next(m_stored_sols.begin()), m_stored_sols.end(), m_stored_sols.begin()); 
      m_stored_sols[numStoredSols-1] = std::move(sol); 
    }

    // ConsumeSolution but copies full list into m_stored_sols. returns iterator such that [start, iterator) isn't pushed into times.
    template<typename Iter>
    Iter pushSolutionRange(Iter start, Iter end)
    {
      std::size_t d = std::distance(start,end); 
      if(d >= numStoredSols) 
      {
        // move [start+d-numStoredSols,end) to last entries of m_stored_sols
        auto it = std::next(start, d - numStoredSols); 
        std::move(it, end, m_stored_sols.begin());
        return it; 
      }
      else
      {
        // push numStoredSols - d solutions to front of m_stored_sols
        std::move( 
            std::prev(m_stored_sols.end(), d),
            m_stored_sols.end(),
            m_stored_sols.begin()
        ); 
        // move [start,end) to last entries of m_stored_sols
        std::move(
            start,
            end,
            std::next(m_stored_sols.begin(), numStoredSols - d)
        ); 
        return start;
      }
    }
};

// entry point for giving specific # of nodes, CTAD time derivate
template<std::size_t numStoredTimes, typename TimeDeriv>
auto make_Executor(TimeDeriv& tderiv)
{
  return Executor<TimeDeriv,numStoredTimes>(tderiv); 
}

// entry point  without nodes being specified 
template<typename TimeDeriv>
auto make_Executor(TimeDeriv& tderiv)
{
  return Executor<TimeDeriv, TimeDeriv::maxOrder+1>(tderiv); 
}

  } // end namespace texprs 
} // end namespace fornfdm 

#endif // Executor.hpp

/** Ramblings on a new texprs refactor. 
 * move away from the old tuple based expressions to a traditional binary/unary expressions
 * 
 * Consider 3 cases of expressions 
 *  - all scalars from reading fornberg weights 
 *  - a couple of AutonomousCoeffs/TimeDepCoeffs left multiplying the TExpr c*Ut 
 *  - a hairy case where a Diffop multiplies a TExpr? Ux * Ut  
 * dispatch by a type tag inside of internal::traits
 * 
 * Break apart into an evaluator and an executor?
 * Evaluator class 2 responsibilities. building a RhsExpression and a InvCoeff expression. 
 * 
 * simple case. all scalars from reading weights
 * the rhs expression is just the sum of all weights < T_n 
 * the inverse coefficient is just the sum(all reads of T_n) ^ -1
 * 
 * 2nd case. a couple of AutonomousCoeff / TimeDepCoeff sitting in the expressions 
 * inverse coefficient has to be some type of diagonal matrix. 
 * rhs expression is (-1.0) * (expressions setTime) * (sum of all weights < T_n)
 * 
 * 3rd case. scary stuff like Uxx * Ut in the expression
 * inverse coefficient is? some type of dense matrix. potentially a block diagonal? not sure... 
 * rhs expression is?
 */