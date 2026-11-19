// tests_texprs.cpp 
// 
// test texprs + coeffs subdirectories 
// through their basic interface, operators,
// and traits
// 
// JAF 11/18/2026 

// #define FORNFDM_USE_BITWISE_OPS
// #define FORNFDM_CUSTOM_SCALAR float
// #include<fornfdm/plugin.hpp>
#include<fornfdm/texprs/all.hpp>
#include<fornfdm/coeffs/all.hpp>
#include<Eigen/Core>
#include<Eigen/Sparse>
#include<gtest/gtest.h>
#include<gmock/gmock.h>

using namespace fornfdm; 
using namespace fornfdm::texprs; 
using namespace fornfdm::linops; 

struct TExprsSuite : public ::testing::Test
{
  TExprsSuite()
  {
    queue_01.getWeights() = {0,1,2,3,   4,5,6,7,    8,9,10,11};
    queue_01.getTimes() = {0.0, 1.0, 2.0, 3.0}; 

    queue_02.getWeights() = {1,1,1,1,   2,2,2,2,   3,3,3,3,   4,4,4,4};
    queue_02.getTimes() = {0.0, 0.1, 0.2, 0.3};

    queue_03.getWeights() = {4,4,4,   5,5,5,   6,6,6};
    queue_03.getTimes() = {0.0, 10.0, 20.0};
  }

  texprs::Queue<4,2> queue_01;  
  texprs::Queue<4,3> queue_02; 
  texprs::Queue<3,2> queue_03; 
};

// TExprs Suite ------------------------------------------------- 
TEST_F(TExprsSuite, QueueSizes){
  ASSERT_EQ(queue_01.numStoredTimes, 4);
  ASSERT_EQ(queue_01.numStoredSols, 3);
  ASSERT_EQ(queue_01.maxOrder, 2);

  ASSERT_EQ(queue_02.numStoredTimes, 4);
  ASSERT_EQ(queue_02.numStoredSols, 3);
  ASSERT_EQ(queue_02.maxOrder, 3);

  ASSERT_EQ(queue_03.numStoredTimes, 3);
  ASSERT_EQ(queue_03.numStoredSols, 2);
  ASSERT_EQ(queue_03.maxOrder, 2);
}

TEST_F(TExprsSuite, BasicNthTimeDeriv)
{
  auto test_lam = [this](auto xpr)
  {
    if(xpr.order <= queue_01.maxOrder)
    {
      ASSERT_EQ(coeff<0>(queue_01, xpr, nullptr, 0.0), queue_01.getWeights()[xpr.order*queue_01.numStoredTimes + 0]);
      ASSERT_EQ(coeff<1>(queue_01, xpr, nullptr, 0.0), queue_01.getWeights()[xpr.order*queue_01.numStoredTimes + 1]);
      ASSERT_EQ(coeff<2>(queue_01, xpr, nullptr, 0.0), queue_01.getWeights()[xpr.order*queue_01.numStoredTimes + 2]);
      ASSERT_EQ(coeff<3>(queue_01, xpr, nullptr, 0.0), queue_01.getWeights()[xpr.order*queue_01.numStoredTimes + 3]);
    }
    if(xpr.order <= queue_02.maxOrder)
    {
      ASSERT_EQ(coeff<0>(queue_02, xpr, nullptr, 0.0), queue_02.getWeights()[xpr.order*queue_02.numStoredTimes + 0]);
      ASSERT_EQ(coeff<1>(queue_02, xpr, nullptr, 0.0), queue_02.getWeights()[xpr.order*queue_02.numStoredTimes + 1]);
      ASSERT_EQ(coeff<2>(queue_02, xpr, nullptr, 0.0), queue_02.getWeights()[xpr.order*queue_02.numStoredTimes + 2]);
      ASSERT_EQ(coeff<3>(queue_02, xpr, nullptr, 0.0), queue_02.getWeights()[xpr.order*queue_02.numStoredTimes + 3]);
    }
    if(xpr.order <= queue_03.maxOrder)
    {
      ASSERT_EQ(coeff<0>(queue_03, xpr, nullptr, 0.0), queue_03.getWeights()[xpr.order*queue_03.numStoredTimes + 0]);
      ASSERT_EQ(coeff<1>(queue_03, xpr, nullptr, 0.0), queue_03.getWeights()[xpr.order*queue_03.numStoredTimes + 1]);
      ASSERT_EQ(coeff<2>(queue_03, xpr, nullptr, 0.0), queue_03.getWeights()[xpr.order*queue_03.numStoredTimes + 2]);
    }
  };
  test_lam(NthTimeDeriv<0>{});
  test_lam(NthTimeDeriv<1>{});
  test_lam(NthTimeDeriv<2>{});
  test_lam(NthTimeDeriv<3>{});
}