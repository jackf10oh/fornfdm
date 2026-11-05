// examples.cpp
//
// JAF 12/8/2025

// #define FORNFDM_STORE_FULL_KRONECKER
// #define FORNFDM_CUSTOM_SCALAR float
// #define FORNFDM_CUSTOM_REAL float

#include<iostream>
#include<iomanip>
#include<fornfdm/texprs/NthTimeDeriv.hpp>
#include<fornfdm/texprs/UnaryTExpr.hpp>
#include<fornfdm/texprs/BinaryTExpr.hpp>
#include<Eigen/SparseCore>

using namespace fornfdm; 

using std::endl, std::cout; 

template<typename... Ts>
void tup_print(std::tuple<Ts...> tup)
{
  std::apply(
    [](auto&&... elem){ ((std::cout << typeid(decltype(elem)).name() << "\n"), ...); },
    tup
  );
}

int main()
{
  // IO manip
  std::cout << "hello world!\n"; 

  auto xpr = texprs::NthTimeDeriv<1>{} + 4.0 * texprs::NthTimeDeriv<2>{} + texprs::NthTimeDeriv<3>{} - texprs::NthTimeDeriv<4>{}; 

  auto summation = xpr.makeSummationTerms();

  // tup_print(std::make_tuple(xpr));
  tup_print(summation);
  std::cout << "-----------\n\n";

  auto left_mess = texprs::NthTimeDeriv<1>{} + 4.0 * texprs::NthTimeDeriv<2>{};
  auto right_mess = -texprs::NthTimeDeriv<3>{} + texprs::NthTimeDeriv<4>{};

  auto gross = -(left_mess + right_mess) + 2.0 * right_mess;
  auto tup_gross = gross.makeSummationTerms();
  tup_print(tup_gross);
};