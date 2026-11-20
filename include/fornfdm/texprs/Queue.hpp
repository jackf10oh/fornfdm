// Queue.hpp
//
// class to manage circular buffer of time + solutions. 
// And a (order+1)*(# of times) sized array of fornberg weights
// Strictly manages storage only. Executor is specialized 
// based off time expression. any calculations will have to 
// use the public interfaces...
//
// JAF 11/13/2025 

#ifndef FORNFDM_TEXPRS_QUEUE_H
#define FORNFDM_TEXPRS_QUEUE_H

#include<algorithm>
#include<array>
#include<cstdint>
#include<Eigen/Core>
#include "../types.hpp" // Scalar, DiagMatrix
#include "../utilities/fornberg.hpp" // not using stack calc. we need a std::array<Real> not std::array<Scalar> 

namespace fornfdm{
  namespace texprs{

template<std::size_t nt, std::size_t nw, std::size_t ns = nt-1>
class Queue
{
  public:
    // Type Defs ------------------------------------------- 
    // number of nodes used in Fornberg algorithm 
    static constexpr std::size_t numTimes = nt; 
    static constexpr std::size_t numWeights = nw;
    static constexpr std::size_t numSolutions = ns;

  protected:
    // Member Data ---------------------------------

    // list of t0, t1, ..., tn 
    std::array<fornfdm::Real, numTimes> m_stored_times; 

    // list of solutions u0, u1, ..., un-1 at times t0, t1, ..., tn-1 
    std::array<fornfdm::Vector, numSolutions> m_stored_sols; 

    // forberg weights calculator 
    std::array<fornfdm::Real, numWeights> m_weights_arr; 

  public:
    // Constructors + Destructor 
    Queue()=default; 
    Queue(const Queue& other)=default;
    ~Queue()=default; 

    // Member Funcs ----- 

    // Getters to stored times. 
    auto& getTimes(){ return m_stored_times; }; 
    const auto& getTimes() const { return m_stored_times; }; 

    // getters to stored solutions 
    auto& getSolutions(){ return m_stored_sols; }
    const auto& getSolutions() const { return m_stored_sols; }

    // returns ref to newest solution 
    fornfdm::Vector& getCurrentSolution(){ return m_stored_sols.back(); }
    const fornfdm::Vector& getCurrentSolution() const { return m_stored_sols.back(); }

    // return ref to first elem in m_stored_sols. Gives an opportunity to move it elsewhere before overwritten in ConsumeSolution  
    fornfdm::Vector& getExpiringSolution(){ return m_stored_sols.front(); }
    const fornfdm::Vector& getExpiringSolution() const { return m_stored_sols.front(); }

    // getter to internal weights array 
    auto& getWeights(){ return m_weights_arr; }
    const auto& getWeights() const { return m_weights_arr; }

    void calculateWeights(fornfdm::Real t, std::size_t maxOrder)
    {
      utils::fornberg(
        m_stored_times.data(), m_stored_times.data()+m_stored_times.size(),
        t, maxOrder,
        m_weights_arr.data()
      );
    }
    
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
      if(d >= numTimes) 
      {
        // throw std::runtime_error("Queue setStoredTimes(it,it) error: distance(start,end) > numTimes");
        // only copy from rightmost times 
        // std::copy(std::next(m_stored_times.begin(),d-numTimes), m_stored_times.end(), m_stored_times.begin()); 
        auto it = std::next(start, d - numTimes); 
        std::copy(it,end,m_stored_times.begin()); 
        return it; 
      } 
      else
      {
        // only copy into rightmost times 
        std::copy(std::next(m_stored_times.begin(),d), m_stored_times.end(), m_stored_times.begin()); 
        std::copy(start,end,std::next(m_stored_times.begin(),numTimes-d)); 
        return start; 
      }
    }

    // rotates stored solutions. meaning the put to back of array. useful to keep memory allocated in hot loops
    void rotateSolutions(std::size_t idx=1)
    {
      std::rotate(m_stored_sols.begin(), std::next(m_stored_sols.begin(),idx), m_stored_sols.end()); 
    }
    
    // consume a solution. push back all previous 
    void pushSolution(fornfdm::Vector sol)
    { 
      /* same idea as pushTime(). with move semantics*/
      std::move(std::next(m_stored_sols.begin()), m_stored_sols.end(), m_stored_sols.begin()); 
      m_stored_sols[numSolutions-1] = std::move(sol); 
    }

    // ConsumeSolution but copies full list into m_stored_sols. returns iterator such that [start, iterator) isn't pushed into times.
    template<typename Iter>
    Iter pushSolutionRange(Iter start, Iter end)
    {
      std::size_t d = std::distance(start,end); 
      if(d >= numSolutions) 
      {
        // move [start+d-numSolutions,end) to last entries of m_stored_sols
        auto it = std::next(start, d - numSolutions); 
        std::move(it, end, m_stored_sols.begin());
        return it; 
      }
      else
      {
        // push numSolutions - d solutions to front of m_stored_sols
        std::move( 
            std::prev(m_stored_sols.end(), d),
            m_stored_sols.end(),
            m_stored_sols.begin()
        ); 
        // move [start,end) to last entries of m_stored_sols
        std::move(
            start,
            end,
            std::next(m_stored_sols.begin(), numSolutions - d)
        ); 
        return start;
      }
    }
};

} // end namespace texprs 
} // end namespace fornfdm 

#endif // Queue.hpp

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