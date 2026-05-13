#include "performance.h"
#include "csr.h"
#include "csr_registers.h"

void init_csr_counters(){
    // activar contadores
    CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);

    // seleccionar eventos
    CSR_WRITE(CSR_REG_MHPMEVENT3, 5);   // loads
    CSR_WRITE(CSR_REG_MHPMEVENT4, 6);   // stores
    CSR_WRITE(CSR_REG_MHPMEVENT5, 2);   // load stall due to hazard
    CSR_WRITE(CSR_REG_MHPMEVENT6, 11);  // pipe stall
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
    uint32_t cycles, inst, loads, stores, ldstall, pipestall;

    CSR_READ(CSR_REG_MCYCLE, &cycles);
    CSR_READ(CSR_REG_MINSTRET, &inst);
    CSR_READ(CSR_REG_MHPMCOUNTER3, &loads);
    CSR_READ(CSR_REG_MHPMCOUNTER4, &stores);
    CSR_READ(CSR_REG_MHPMCOUNTER5, &ldstall);
    CSR_READ(CSR_REG_MHPMCOUNTER6, &pipestall);
    printf("-----------------------\n");
    printf("Cc: %lu\n", cycles);
    printf("Instr: %lu\n", inst);
}