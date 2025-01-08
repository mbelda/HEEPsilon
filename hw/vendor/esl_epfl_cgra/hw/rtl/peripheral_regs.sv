// Copyright 2022 EPFL
// Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1

module peripheral_regs
  import cgra_pkg::*;
  import reg_pkg::*;
(
  input  logic                               clk_i,
  input  logic                               rst_ni,
  input  logic                               acc_ack_i,
  input  logic     [              N_COL-1:0] col_stall_i,
  input  reg_req_t                           reg_req_i,
  output reg_rsp_t                           reg_rsp_o,
  input  logic     [              N_COL-1:0] acc_req_i,
  input  logic     [              N_COL-1:0] acc_end_i,
  output logic     [              N_COL-1:0] col_status_o,
  output logic     [KER_CONF_N_REG_LOG2-1:0] ker_id_o,
  output logic     [           DP_WIDTH-1:0] rd_ptr_o [0:MAX_COL_REQ-1],
  output logic     [           DP_WIDTH-1:0] wr_ptr_o [0:MAX_COL_REQ-1]
);

  import cgra_reg_pkg::*;

  cgra_reg2hw_t reg2hw;
  cgra_hw2reg_t hw2reg;

  logic [N_COL-1:0] col_status;
  logic perf_cnt_en;
  logic perf_cnt_reset;

  assign col_status_o = col_status;

  assign perf_cnt_en    = reg2hw.perf_cnt_enable.q;
  assign perf_cnt_reset = reg2hw.perf_cnt_reset.q;


  // Columns' read and write pointers
  always_comb
  begin
    ker_id_o = reg2hw.kernel_id.q[KER_CONF_N_REG_LOG2-1:0];

    rd_ptr_o[0] = reg2hw.ptr_in_col_0.q;
    wr_ptr_o[0] = reg2hw.ptr_out_col_0.q;
    rd_ptr_o[1] = reg2hw.ptr_in_col_1.q;
    wr_ptr_o[1] = reg2hw.ptr_out_col_1.q;
    rd_ptr_o[2] = reg2hw.ptr_in_col_2.q;
    wr_ptr_o[2] = reg2hw.ptr_out_col_2.q;
    rd_ptr_o[3] = reg2hw.ptr_in_col_3.q;
    wr_ptr_o[3] = reg2hw.ptr_out_col_3.q;
  end

  // Reset/update performance counters
  assign hw2reg.perf_cnt_total_kernels.de = perf_cnt_reset | (perf_cnt_en & acc_ack_i);
  assign hw2reg.perf_cnt_total_kernels.d  = perf_cnt_reset == 1'b1 ? '0 : reg2hw.perf_cnt_total_kernels.q + 1;

  assign hw2reg.perf_cnt_col_0_active_cycles.de = perf_cnt_reset | (perf_cnt_en & (col_status[0] | acc_req_i[0]));
  assign hw2reg.perf_cnt_col_0_active_cycles.d  = perf_cnt_reset == 1'b1 ? '0 : reg2hw.perf_cnt_col_0_active_cycles.q + 1;
  assign hw2reg.perf_cnt_col_0_stall_cycles.de  = perf_cnt_reset | (perf_cnt_en & col_stall_i[0]);
  assign hw2reg.perf_cnt_col_0_stall_cycles.d   = perf_cnt_reset == 1'b1 ? '0 : reg2hw.perf_cnt_col_0_stall_cycles.q + 1;
  assign hw2reg.perf_cnt_col_1_active_cycles.de = perf_cnt_reset | (perf_cnt_en & (col_status[1] | acc_req_i[1]));
  assign hw2reg.perf_cnt_col_1_active_cycles.d  = perf_cnt_reset == 1'b1 ? '0 : reg2hw.perf_cnt_col_1_active_cycles.q + 1;
  assign hw2reg.perf_cnt_col_1_stall_cycles.de  = perf_cnt_reset | (perf_cnt_en & col_stall_i[1]);
  assign hw2reg.perf_cnt_col_1_stall_cycles.d   = perf_cnt_reset == 1'b1 ? '0 : reg2hw.perf_cnt_col_1_stall_cycles.q + 1;
  assign hw2reg.perf_cnt_col_2_active_cycles.de = perf_cnt_reset | (perf_cnt_en & (col_status[2] | acc_req_i[2]));
  assign hw2reg.perf_cnt_col_2_active_cycles.d  = perf_cnt_reset == 1'b1 ? '0 : reg2hw.perf_cnt_col_2_active_cycles.q + 1;
  assign hw2reg.perf_cnt_col_2_stall_cycles.de  = perf_cnt_reset | (perf_cnt_en & col_stall_i[2]);
  assign hw2reg.perf_cnt_col_2_stall_cycles.d   = perf_cnt_reset == 1'b1 ? '0 : reg2hw.perf_cnt_col_2_stall_cycles.q + 1;
  assign hw2reg.perf_cnt_col_3_active_cycles.de = perf_cnt_reset | (perf_cnt_en & (col_status[3] | acc_req_i[3]));
  assign hw2reg.perf_cnt_col_3_active_cycles.d  = perf_cnt_reset == 1'b1 ? '0 : reg2hw.perf_cnt_col_3_active_cycles.q + 1;
  assign hw2reg.perf_cnt_col_3_stall_cycles.de  = perf_cnt_reset | (perf_cnt_en & col_stall_i[3]);
  assign hw2reg.perf_cnt_col_3_stall_cycles.d   = perf_cnt_reset == 1'b1 ? '0 : reg2hw.perf_cnt_col_3_stall_cycles.q + 1;

  // Disable reset
  assign hw2reg.perf_cnt_reset.de = perf_cnt_reset;
  assign hw2reg.perf_cnt_reset.d  = '0;

  // Clear kernel ID request
  assign hw2reg.kernel_id.de = acc_ack_i;
  assign hw2reg.kernel_id.d  = '0;

  // Update CGRA column status:
  // 0 - Column is free
  // 1 - Column is used
  assign col_status = reg2hw.col_status.q;
  assign hw2reg.col_status.de = ((|acc_req_i) & acc_ack_i) | (|acc_end_i);
  assign hw2reg.col_status.d  = ((|acc_req_i) & acc_ack_i) == 1'b1 ? (col_status | acc_req_i) : (col_status & ~acc_end_i);

  cgra_reg_top #(
      .reg_req_t(reg_req_t),
      .reg_rsp_t(reg_rsp_t)
  ) cgra_reg_top_i (
      .clk_i,
      .rst_ni,
      .reg_req_i,
      .reg_rsp_o,
      .reg2hw,
      .hw2reg,
      .devmode_i(1'b1)
  );

endmodule
