#ifndef LMP_SO3LR_ELEMENT_MAP_H
#define LMP_SO3LR_ELEMENT_MAP_H

#include <array>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <string>

namespace so3lr_lammps {

// SO3LR native format v1 carries physical tables for Z=1..99.  Keep the
// LAMMPS-facing mapping here so the numerical runtime remains independent of
// atom-type ordering and users can write either "H O" or "1 8".
inline int atomic_number_from_token(const char *raw)
{
  if (raw == nullptr || raw[0] == '\0') return 0;

  char *end = nullptr;
  errno = 0;
  const long numeric = std::strtol(raw, &end, 10);
  if (errno == 0 && end != raw && *end == '\0') {
    return numeric >= 1 && numeric <= 99 ? static_cast<int>(numeric) : 0;
  }

  std::string symbol(raw);
  symbol[0] = static_cast<char>(
      std::toupper(static_cast<unsigned char>(symbol[0])));
  for (std::size_t index = 1; index < symbol.size(); ++index) {
    symbol[index] = static_cast<char>(
        std::tolower(static_cast<unsigned char>(symbol[index])));
  }

  static constexpr std::array<const char *, 100> symbols = {
      "",   "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",
      "Ne", "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",
      "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
      "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
      "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In",
      "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
      "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm",
      "Yb", "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
      "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
      "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es"};
  for (int z = 1; z <= 99; ++z) {
    if (symbol == symbols[static_cast<std::size_t>(z)]) return z;
  }
  return 0;
}

}  // namespace so3lr_lammps

#endif
