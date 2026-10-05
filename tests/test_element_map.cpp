#include "so3lr_element_map.h"

#include <iostream>

int main()
{
  using so3lr_lammps::atomic_number_from_token;
  if (atomic_number_from_token("H") != 1 ||
      atomic_number_from_token("o") != 8 ||
      atomic_number_from_token("CL") != 17 ||
      atomic_number_from_token("C") != 6 ||
      atomic_number_from_token("99") != 99 ||
      atomic_number_from_token("0") != 0 ||
      atomic_number_from_token("100") != 0 ||
      atomic_number_from_token("NULL") != 0 ||
      atomic_number_from_token("8x") != 0) {
    std::cerr << "SO3LR element mapping test failed\n";
    return 1;
  }
  std::cout << "SO3LR_ELEMENT_MAP=PASS\n";
  return 0;
}
