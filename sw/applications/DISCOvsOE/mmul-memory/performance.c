#include "performance.h"
#include "csr.h"
#include "csr_registers.h"

/*
void init_csr_counters(){
    // activar contadores
    CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0);

    // seleccionar eventos
    CSR_WRITE(CSR_REG_MHPMEVENT3, 5);   // loads
    CSR_WRITE(CSR_REG_MHPMEVENT4, 6);   // stores
}
*/

void init_csr_counters() {
    // 1. Select events for the counters
    CSR_WRITE(CSR_REG_MHPMEVENT3, 5);   // Event 5: Number of loads
    CSR_WRITE(CSR_REG_MHPMEVENT4, 6);   // Event 6: Number of stores

    // 2. Clear counters (High and Low) to ensure a clean start
    CSR_WRITE(CSR_REG_MHPMCOUNTER3,  0);
    CSR_WRITE(CSR_REG_MHPMCOUNTER3H, 0);
    CSR_WRITE(CSR_REG_MHPMCOUNTER4,  0);
    CSR_WRITE(CSR_REG_MHPMCOUNTER4H, 0);

    // 3. Start counting by clearing the inhibit bits
    // Bits 3 and 4 correspond to counters 3 and 4
    CSR_WRITE(CSR_REG_MCOUNTINHIBIT, 0); 
}

void reset_csr_counters(){
    CSR_WRITE(CSR_REG_MHPMCOUNTER3, 0);
    CSR_WRITE(CSR_REG_MHPMCOUNTER3H, 0);
    CSR_WRITE(CSR_REG_MHPMCOUNTER4, 0);
    CSR_WRITE(CSR_REG_MHPMCOUNTER4H, 0);
    CSR_WRITE(CSR_REG_MCYCLE, 0);
    CSR_WRITE(CSR_REG_MCYCLEH, 0);
    CSR_WRITE(CSR_REG_MINSTRET, 0);
    CSR_WRITE(CSR_REG_MINSTRETH, 0);
}

uint64_t read_64bit_csr(int reg_low, int reg_high) {
    uint32_t hi, lo, check_hi;
    do {
        CSR_READ(reg_high, &hi);
        CSR_READ(reg_low,  &lo);
        CSR_READ(reg_high, &check_hi);
    } while (hi != check_hi); // Repeat if a carry occurred during read
    
    return (((uint64_t)hi) << 32) | lo;
}

void read_csr_counters() {
    // Pipeline flush: Ensure all memory ops (loads/stores) are finished 
    // before we grab the final counter values.
    asm volatile ("fence"); 

    uint64_t loads  = read_64bit_csr(CSR_REG_MHPMCOUNTER3, CSR_REG_MHPMCOUNTER3H);
    uint64_t stores = read_64bit_csr(CSR_REG_MHPMCOUNTER4, CSR_REG_MHPMCOUNTER4H);
    uint64_t cycles = read_64bit_csr(CSR_REG_MCYCLE, CSR_REG_MCYCLEH);
    uint64_t inst   = read_64bit_csr(CSR_REG_MINSTRET, CSR_REG_MINSTRETH);

    printf("\n--- Performance Stats ---\n");
    printf("Total Cycles: %llu\n", cycles);
    printf("Load Instrs:  %llu\n", loads);
    printf("Store Instrs: %llu\n", stores);
    printf("Total Instrs: %llu\n", inst);
    printf("-------------------------\n");
}

/*
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
    printf("Lds: %lu\n", loads);
    printf("Str: %lu\n", stores);
    printf("Ld Stalls: %lu\n", ldstall);
    printf("Pipe stalls: %lu\n", pipestall);
    printf("-----------------------\n");
}
*/