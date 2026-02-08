#include "performance.h"
#include "csr.h"
#include "csr_registers.h"

void init_csr_counters(){
    // activar contadores
    CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);

    // seleccionar eventos
    CSR_WRITE(CSR_REG_MHPMEVENT3, 3);   // loads
    CSR_WRITE(CSR_REG_MHPMEVENT4, 4);   // stores
    CSR_WRITE(CSR_REG_MHPMEVENT5, 10);  // load stall
    CSR_WRITE(CSR_REG_MHPMEVENT6, 13);  // data miss
}

void reset_csr_counters(){
    CSR_WRITE(CSR_REG_MHPMCOUNTER3, 0);
    CSR_WRITE(CSR_REG_MHPMCOUNTER4, 0);
    CSR_WRITE(CSR_REG_MHPMCOUNTER5, 0);
    CSR_WRITE(CSR_REG_MHPMCOUNTER6, 0);
    CSR_WRITE(CSR_REG_MCYCLE, 0);
    CSR_WRITE(CSR_REG_MINSTRET, 0);
}

void read_csr_counters(){
    // leer resultados
    uint32_t cycles, inst, loads, stores, ldstall, dmiss;

    CSR_READ(CSR_REG_MCYCLE, &cycles);
    CSR_READ(CSR_REG_MINSTRET, &inst);
    CSR_READ(CSR_REG_MHPMCOUNTER3, &loads);
    CSR_READ(CSR_REG_MHPMCOUNTER4, &stores);
    CSR_READ(CSR_REG_MHPMCOUNTER5, &ldstall);
    CSR_READ(CSR_REG_MHPMCOUNTER6, &dmiss);

    printf("Cc: %lu\n", cycles);
    printf("Instr: %lu\n", inst);
    printf("Lds: %lu\n", loads);
    printf("Str: %lu\n", stores);
    printf("Ld Stalls: %lu\n", ldstall);
    printf("Data Miss: %lu\n", dmiss);
}