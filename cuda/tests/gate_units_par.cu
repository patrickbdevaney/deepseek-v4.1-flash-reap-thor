// gate_units_par: the parallel expert grouping (k_units_par) reproduces the serial k_units exactly
#include <cstdio>
#include "v41_kernels.h"
int main() {
    const int bad = v41k::units_selftest();
    printf("gate_units_par: %d mismatching routings -> %s\n", bad, bad == 0 ? "PASS" : "FAIL");
    return bad == 0 ? 0 : 1;
}
