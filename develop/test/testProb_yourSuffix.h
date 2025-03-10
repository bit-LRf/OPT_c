/* This file was automatically generated by CasADi 3.6.7.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

extern "C" int testProb_f(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
extern "C" int testProb_f_alloc_mem(void);
extern "C" int testProb_f_init_mem(int mem);
extern "C" void testProb_f_free_mem(int mem);
extern "C" int testProb_f_checkout(void);
extern "C" void testProb_f_release(int mem);
extern "C" void testProb_f_incref(void);
extern "C" void testProb_f_decref(void);
extern "C" casadi_int testProb_f_n_in(void);
extern "C" casadi_int testProb_f_n_out(void);
extern "C" casadi_real testProb_f_default_in(casadi_int i);
extern "C" const char* testProb_f_name_in(casadi_int i);
extern "C" const char* testProb_f_name_out(casadi_int i);
extern "C" const casadi_int* testProb_f_sparsity_in(casadi_int i);
extern "C" const casadi_int* testProb_f_sparsity_out(casadi_int i);
extern "C" int testProb_f_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
extern "C" int testProb_f_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define testProb_f_SZ_ARG 2
#define testProb_f_SZ_RES 1
#define testProb_f_SZ_IW 0
#define testProb_f_SZ_W 6
extern "C" int testProb_df(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
extern "C" int testProb_df_alloc_mem(void);
extern "C" int testProb_df_init_mem(int mem);
extern "C" void testProb_df_free_mem(int mem);
extern "C" int testProb_df_checkout(void);
extern "C" void testProb_df_release(int mem);
extern "C" void testProb_df_incref(void);
extern "C" void testProb_df_decref(void);
extern "C" casadi_int testProb_df_n_in(void);
extern "C" casadi_int testProb_df_n_out(void);
extern "C" casadi_real testProb_df_default_in(casadi_int i);
extern "C" const char* testProb_df_name_in(casadi_int i);
extern "C" const char* testProb_df_name_out(casadi_int i);
extern "C" const casadi_int* testProb_df_sparsity_in(casadi_int i);
extern "C" const casadi_int* testProb_df_sparsity_out(casadi_int i);
extern "C" int testProb_df_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
extern "C" int testProb_df_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define testProb_df_SZ_ARG 2
#define testProb_df_SZ_RES 1
#define testProb_df_SZ_IW 0
#define testProb_df_SZ_W 8
extern "C" int testProb_eq(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
extern "C" int testProb_eq_alloc_mem(void);
extern "C" int testProb_eq_init_mem(int mem);
extern "C" void testProb_eq_free_mem(int mem);
extern "C" int testProb_eq_checkout(void);
extern "C" void testProb_eq_release(int mem);
extern "C" void testProb_eq_incref(void);
extern "C" void testProb_eq_decref(void);
extern "C" casadi_int testProb_eq_n_in(void);
extern "C" casadi_int testProb_eq_n_out(void);
extern "C" casadi_real testProb_eq_default_in(casadi_int i);
extern "C" const char* testProb_eq_name_in(casadi_int i);
extern "C" const char* testProb_eq_name_out(casadi_int i);
extern "C" const casadi_int* testProb_eq_sparsity_in(casadi_int i);
extern "C" const casadi_int* testProb_eq_sparsity_out(casadi_int i);
extern "C" int testProb_eq_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
extern "C" int testProb_eq_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define testProb_eq_SZ_ARG 2
#define testProb_eq_SZ_RES 1
#define testProb_eq_SZ_IW 0
#define testProb_eq_SZ_W 4
extern "C" int testProb_deq(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
extern "C" int testProb_deq_alloc_mem(void);
extern "C" int testProb_deq_init_mem(int mem);
extern "C" void testProb_deq_free_mem(int mem);
extern "C" int testProb_deq_checkout(void);
extern "C" void testProb_deq_release(int mem);
extern "C" void testProb_deq_incref(void);
extern "C" void testProb_deq_decref(void);
extern "C" casadi_int testProb_deq_n_in(void);
extern "C" casadi_int testProb_deq_n_out(void);
extern "C" casadi_real testProb_deq_default_in(casadi_int i);
extern "C" const char* testProb_deq_name_in(casadi_int i);
extern "C" const char* testProb_deq_name_out(casadi_int i);
extern "C" const casadi_int* testProb_deq_sparsity_in(casadi_int i);
extern "C" const casadi_int* testProb_deq_sparsity_out(casadi_int i);
extern "C" int testProb_deq_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
extern "C" int testProb_deq_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define testProb_deq_SZ_ARG 2
#define testProb_deq_SZ_RES 1
#define testProb_deq_SZ_IW 0
#define testProb_deq_SZ_W 2
extern "C" int testProb_h_lower(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem);
extern "C" int testProb_h_lower_alloc_mem(void);
extern "C" int testProb_h_lower_init_mem(int mem);
extern "C" void testProb_h_lower_free_mem(int mem);
extern "C" int testProb_h_lower_checkout(void);
extern "C" void testProb_h_lower_release(int mem);
extern "C" void testProb_h_lower_incref(void);
extern "C" void testProb_h_lower_decref(void);
extern "C" casadi_int testProb_h_lower_n_in(void);
extern "C" casadi_int testProb_h_lower_n_out(void);
extern "C" casadi_real testProb_h_lower_default_in(casadi_int i);
extern "C" const char* testProb_h_lower_name_in(casadi_int i);
extern "C" const char* testProb_h_lower_name_out(casadi_int i);
extern "C" const casadi_int* testProb_h_lower_sparsity_in(casadi_int i);
extern "C" const casadi_int* testProb_h_lower_sparsity_out(casadi_int i);
extern "C" int testProb_h_lower_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
extern "C" int testProb_h_lower_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w);
#define testProb_h_lower_SZ_ARG 3
#define testProb_h_lower_SZ_RES 1
#define testProb_h_lower_SZ_IW 0
#define testProb_h_lower_SZ_W 2
