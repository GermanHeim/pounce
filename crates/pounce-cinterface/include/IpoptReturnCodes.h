/* POUNCE drop-in replacement for Ipopt's IpReturnCodes.h /
 * IpReturnCodes_inc.h. Integer values are the C ABI contract — they
 * must match upstream Ipopt 3.14.x byte for byte so PyIpopt / cyipopt
 * / JuMP / Fortran callers see the same constants whether they link
 * against libipopt or libpounce_cinterface.
 *
 * Source of truth: ref/Ipopt/src/Interfaces/IpReturnCodes_inc.h.
 * Pounce mirror: crates/pounce-nlp/src/return_codes.rs (verified by
 * `integer_values_match_upstream` unit test).
 */

#ifndef POUNCE_RETURN_CODES_H
#define POUNCE_RETURN_CODES_H

#ifdef __cplusplus
extern "C" {
#endif

enum ApplicationReturnStatus
{
   Solve_Succeeded                    = 0,
   Solved_To_Acceptable_Level         = 1,
   Infeasible_Problem_Detected        = 2,
   Search_Direction_Becomes_Too_Small = 3,
   Diverging_Iterates                 = 4,
   User_Requested_Stop                = 5,
   Feasible_Point_Found               = 6,

   Maximum_Iterations_Exceeded        = -1,
   Restoration_Failed                 = -2,
   Error_In_Step_Computation          = -3,
   Maximum_CpuTime_Exceeded           = -4,
   Maximum_WallTime_Exceeded          = -5,

   Not_Enough_Degrees_Of_Freedom      = -10,
   Invalid_Problem_Definition         = -11,
   Invalid_Option                     = -12,
   Invalid_Number_Detected            = -13,

   Unrecoverable_Exception            = -100,
   NonIpopt_Exception_Thrown          = -101,
   Insufficient_Memory                = -102,
   Internal_Error                     = -199
};

enum AlgorithmMode
{
   RegularMode          = 0,
   RestorationPhaseMode = 1
};

#ifdef __cplusplus
}
#endif

#endif /* POUNCE_RETURN_CODES_H */
