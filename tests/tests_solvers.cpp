// tests_solvers.cpp 
// 
// tests all 3 solvers against diffusion equation with exact solution 
// interpolate() free function 
// in 1D, 2D, and 3D linear solutions 
// 
// JAF 5/18/2026 

// #define FORNFDM_USE_BITWISE_OPS
// #define FORNFDM_CUSTOM_SCALAR float
#include<fornfdm/plugin.hpp>
#include<fornfdm/all.hpp>
#include<Eigen/Core>
#include<Eigen/Sparse>
#include<gtest/gtest.h>
#include<gmock/gmock.h>

using namespace fornfdm; 

// Solver Suite ------------------------------------------------- 
struct SolverSuite : public ::testing::Test
{
  // domain in space 
  static constexpr fornfdm::Scalar pi = 3.14159; 
  fornfdm::SharedConstMesh mesh = Mesh::create(1)->setAxis(0, fornfdm::linspaced(32,0.0, pi));

  // in time 
  std::shared_ptr<const fornfdm::RealVector> times = std::make_shared<const fornfdm::RealVector>(fornfdm::linspaced(300, 0.0, 0.5));
  std::shared_ptr<const solvers::TimeArg> fast_times = solvers::TimeArg::builder().setStart(0.0).setStop(0.5).setNumSteps(300).build();

  // complete args. initial cond is sin(x) 
  fornfdm::solvers::SolverArgs<const fornfdm::Mesh, const fornfdm::RealVector> args = {
    mesh, 
    times,
    {fornfdm::discretize(mesh, [](fornfdm::Scalar x){ return std::sin(x);})}
  };

  fornfdm::solvers::SolverArgs<const fornfdm::Mesh, const fornfdm::solvers::TimeArg> fast_args = {
    mesh, 
    fast_times,
    {fornfdm::discretize(mesh, [](fornfdm::Scalar x){ return std::sin(x);})}
  };

  // L.H.S. in time 
  fornfdm::texprs::NthTimeDeriv<1> Ut; 

  // R.H.S. in space 
  fornfdm::linops::NthPartialDeriv<2,0> Uxx;

  // Dirichlet boundaries clamped at 0 
  fornfdm::osteps::BCPair<fornfdm::osteps::Dirichlet,fornfdm::osteps::Dirichlet> bcs = {fornfdm::osteps::Dirichlet(0.0), fornfdm::osteps::Dirichlet(0.0)};

  void checkAnswer(const fornfdm::Vector& vec)
  {
    for(std::size_t idx=0; idx<vec.size(); ++idx)
    {
      fornfdm::Real t_final = args.times->data()[args.times->size()-1];
      fornfdm::Scalar exact_val = std::sin(mesh->getAxis(0)[idx]) * std::exp(-t_final);
      ASSERT_NEAR(vec[idx], exact_val, 0.025);
    }
  }
};

TEST_F(SolverSuite, TimeArgsBuilder){
  auto test_lam_01 = [](fornfdm::Real t0, fornfdm::Real t1, fornfdm::Real dt)
  {
    auto arg = solvers::TimeArg::builder().setStart(t0).setStop(t1).setStepSize(dt).build();
    ASSERT_EQ(arg->getStart(), t0);
    ASSERT_EQ(arg->getStop(), t1);
    ASSERT_EQ(arg->getStepSize(), dt);
  };
  test_lam_01(0.0, 10.0, 2.0);
  test_lam_01(0.0, 7.0, 2.0);
  test_lam_01(2.0, 10.0, 2.0);
  test_lam_01(0.0, 10.0, 0.2);
  test_lam_01(20.0, 21.0, 0.01);

  auto test_lam_02 = [](fornfdm::Real t0, fornfdm::Real t1, std::size_t n)
  {
    auto arg = solvers::TimeArg::builder().setStart(t0).setStop(t1).setNumSteps(n).build();
    ASSERT_EQ(arg->getStart(), t0);
    ASSERT_EQ(arg->getStop(), t1);
    ASSERT_EQ(arg->getStepSize(), (t1-t0)/n);
  };
  test_lam_01(0.0, 10.0, 2);
  test_lam_01(0.0, 7.0, 3);
  test_lam_01(2.0, 10.0, 5);
  test_lam_01(0.0, 10.0, 8);
  test_lam_01(20.0, 21.0, 13);
}

TEST_F(SolverSuite, OneDimExplicitSolver){
  solvers::ExplicitSolver exp_solver(Ut,Uxx,std::tie(bcs)); 
  auto sol = exp_solver.calculate(args, solvers::LastSaver{});
  checkAnswer(sol); 
};

TEST_F(SolverSuite, OneDimImplicitSolver){
  solvers::ImplicitSolver imp_solver(Ut,Uxx,std::tie(bcs)); 
  auto sol = imp_solver.calculate(args, solvers::LastSaver{}); 
  checkAnswer(sol);
};

TEST_F(SolverSuite, OneDimCrankNicolsonSolver){
  solvers::CrankNicolsonSolver cn_solver(Ut,Uxx,std::tie(bcs)); 
  auto sol = cn_solver.calculate(args, solvers::LastSaver{}); 
  checkAnswer(sol);
};

TEST_F(SolverSuite, OneDimFastExpSolver){
  solvers::FastExpSolver fast_solver(Ut,Uxx,std::tie(bcs)); 
  auto sol = fast_solver.calculate(fast_args, solvers::LastSaver{}); 
  checkAnswer(sol);
};

TEST_F(SolverSuite, OneDimFastImpSolver){
  solvers::FastImpSolver fast_solver(Ut,Uxx,std::tie(bcs)); 
  auto sol = fast_solver.calculate(fast_args, solvers::LastSaver{}); 
  checkAnswer(sol);
};

// todo higher dimensions

// todo irregular mesh/time

// Interpolation Suite ---------------------------------------- 
TEST(InterpolationSuite, HighDimLinearInterpolation){

  // 1D test 
  auto mesh_1d = Mesh::create(1)->setAxis(0,fornfdm::linspaced(16,-10.0,10.0)); 
  auto mesh_1d_fine = Mesh::create(1)->setAxis(0,fornfdm::linspaced(128,-10.0,10.0)); 
  auto lam_1d = [](fornfdm::Scalar x){ return 10.54 + 32.7 * x; }; 
  auto vec_1d = discretize(mesh_1d, lam_1d); 
  for(auto i = 0; i < mesh_1d_fine->sizeOfDim(0); ++i)
  {
    fornfdm::Coordinate<1> coord{mesh_1d_fine->getAxis(0)[i]};
    auto exact_val = coord.apply(lam_1d); 
    auto interp_val = fornfdm::solvers::interpolate(coord, vec_1d,mesh_1d.get());
    ASSERT_NEAR(interp_val, exact_val, 1e-4);
  }

  // 2D test 
  auto mesh_2d = Mesh::create(2)
    ->setAxis(0,fornfdm::linspaced(16,-10.0,10.0))
    ->setAxis(1,fornfdm::linspaced(16,-10.0,10.0));

  auto mesh_2d_fine = Mesh::create(2)
    ->setAxis(0,fornfdm::linspaced(128,-10.0,10.0))
    ->setAxis(1,fornfdm::linspaced(128,-10.0,10.0));

  auto lam_2d = [](fornfdm::Scalar x, fornfdm::Scalar y){ return 10.54 + 32.7 * x + 17.2 * y; }; 
  auto vec_2d = discretize(mesh_2d, lam_2d); 
  for(auto i = 0; i < mesh_2d_fine->sizeOfDim(0); ++i)
  {
    for(auto j = 0; j < mesh_2d_fine->sizeOfDim(1); ++j)
    {
      fornfdm::Coordinate<2> coord{mesh_2d_fine->getAxis(0)[i], mesh_2d_fine->getAxis(1)[j]};
      auto exact_val = coord.apply(lam_2d); 
      auto interp_val = fornfdm::solvers::interpolate(coord, vec_2d,mesh_2d.get());
      ASSERT_NEAR(interp_val, exact_val, 1e-4);
    }
  }


  // 3D test
  auto mesh_3d = Mesh::create(3)
    ->setAxis(0, fornfdm::linspaced(8,-10.0,10.0))
    ->setAxis(1, fornfdm::linspaced(8,-10.0,10.0))
    ->setAxis(2, fornfdm::linspaced(8,-10.0,10.0));

  auto mesh_3d_fine = Mesh::create(3)
    ->setAxis(0, fornfdm::linspaced(32,-10.0,10.0))
    ->setAxis(1, fornfdm::linspaced(32,-10.0,10.0))
    ->setAxis(2, fornfdm::linspaced(32,-10.0,10.0));

  auto lam_3d = [](fornfdm::Scalar x, fornfdm::Scalar y, fornfdm::Scalar z){ return 10.54 + 32.7 * x + 17.2 * y - 26.49 * z; }; 
  auto vec_3d = discretize(mesh_3d, lam_3d); 
  for(auto i = 0; i < mesh_3d_fine->sizeOfDim(0); ++i)
  {
    for(auto j = 0; j < mesh_3d_fine->sizeOfDim(1); ++j)
    {
      for(auto k = 0; k < mesh_3d_fine->sizeOfDim(2); ++k)
      {
        fornfdm::Coordinate<3> coord{mesh_3d_fine->getAxis(0)[i], mesh_3d_fine->getAxis(1)[j], mesh_3d_fine->getAxis(2)[k]};
        auto exact_val = coord.apply(lam_3d); 
        auto interp_val = fornfdm::solvers::interpolate(coord, vec_3d,mesh_3d.get());
        ASSERT_NEAR(interp_val, exact_val, 1e-4);
      }
    }
  }

};

TEST(InterpolationSuite, OneDimInterpolator){

  constexpr fornfdm::Scalar pi = 3.14159265385; 
  // Domain + Time  
  SharedConstMesh m = Mesh::create(1)->setAxis(0,fornfdm::linspaced(32,0.0,pi)); 
  fornfdm::solvers::SolverArgs args{
    /*.mesh=*/m, 
    /*.times=*/std::make_shared<const fornfdm::RealVector>(fornfdm::linspaced(300,0.0,0.5))
  }; 

  // Initial Conditions  
  auto v = discretize(args.mesh, [](fornfdm::Scalar x){ return std::sin(x); }); 
  args.initialConditions = { std::move(v) }; 

  // LHS in time 
  auto Ut = texprs::NthTimeDeriv<1>{}; 

  // RHS in space 
  auto Uxx = linops::NthPartialDeriv<2,0,fornfdm::linops::Centered<5>>{}; 

  // Boundary Conditions 
  auto left = osteps::Dirichlet(0.0); 
  auto right = left;
  osteps::BCPair bcs(left,right); 

  solvers::CrankNicolsonSolver cn_solver(Ut,Uxx,std::tie(bcs)); 
  solvers::Interpolator interp(std::move(cn_solver), args);  
  const auto& axis = args.mesh->getAxis(0); 
  auto mesh_redone = Mesh::create(1)->setAxis(0,fornfdm::linspaced(64,axis[0], axis[axis.size()-1])); 

  const auto& t_axis = *args.times; 
  auto times_redone = std::make_shared<const fornfdm::Vector>(fornfdm::linspaced(64,t_axis[0],t_axis[t_axis.size()-1])); 

  for(auto i = 0; i < mesh_redone->sizeOfDim(0); ++i)
  {
    for(auto j = 0; j < times_redone->size(); ++j)
    {
      fornfdm::Scalar x = mesh_redone->getAxis(0)[i]; 
      fornfdm::Real t = (*times_redone)[j]; 
      fornfdm::Scalar exact_val = std::sin(x) * std::exp(-t); 
      fornfdm::Scalar interpolated_val = interp.solAt<1>(t, {x}); 
      ASSERT_NEAR(interpolated_val, exact_val, 0.025);
    }
  }
};