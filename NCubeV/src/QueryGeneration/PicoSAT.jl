
# Some extensions and additional exports for the PicoSAT interface
module InternalPicoSAT
using PicoSAT
using ....Config
using TimerOutputs
#import PicoSAT : picosat_init, picosat_reset, add_clause, add_clauses, get_solution

const HAS_LIBPICOSAT = isdefined(PicoSAT, :libpicosat)
const HAS_PICOSAT_SET_VERBOSITY = isdefined(PicoSAT, :picosat_set_verbosity)

PicoPtr = isdefined(PicoSAT, :PicoPtr) ? PicoSAT.PicoPtr : Ptr{Cvoid}
const libpicosat = HAS_LIBPICOSAT ? getfield(PicoSAT, :libpicosat) : nothing

function _unsupported_picosat_error()
    error("PicoSAT native library is unavailable on this platform. NCubeV query generation that depends on PicoSAT is not supported here.")
end

picosat_init = isdefined(PicoSAT, :picosat_init) ? PicoSAT.picosat_init : () -> _unsupported_picosat_error()
picosat_reset = isdefined(PicoSAT, :picosat_reset) ? PicoSAT.picosat_reset : (_) -> _unsupported_picosat_error()

function _require_picosat_native()
    HAS_LIBPICOSAT && return
    _unsupported_picosat_error()
end
#add_clause_internal = PicoSAT.add_clause
function add_clause_internal(p::PicoPtr, clause)
    for lit in clause
        v = convert(Cint, lit)
        v == 0 && throw(ErrorException("PicoSAT Error: non zero integer expected"))
        PicoSAT.picosat_add(p, v)
    end
    PicoSAT.picosat_add(p, 0)
    return
end

function add_clause(p::PicoPtr, clause)
    add_clause_internal(p, clause)
    if !isnothing(Config.QUERY_GEN_SAVE_SAT)
        open(Config.QUERY_GEN_SAVE_SAT, "a") do f
            print(f, join(clause, " "), " 0\n")
        end
    end
end

add_clauses = isdefined(PicoSAT, :add_clauses) ? PicoSAT.add_clauses : (_, _) -> _unsupported_picosat_error()
get_solution = isdefined(PicoSAT, :get_solution) ? PicoSAT.get_solution : (_) -> _unsupported_picosat_error()
picosat_set_verbosity = HAS_PICOSAT_SET_VERBOSITY ? PicoSAT.picosat_set_verbosity : (_, _) -> nothing

export PicoPtr, picosat_init, save_original_clauses, picosat_reset, add_clause, add_clauses, get_partial_solution, picosat_set_verbosity

export next_var, push, pop, solve, picosat_set_more_important_lit

function next_var(p::PicoPtr)
    _require_picosat_native()
    return ccall((:picosat_inc_max_var, libpicosat), Cint, (PicoPtr,), p)
end

function push(p::PicoPtr)
    _require_picosat_native()
    return ccall((:picosat_push, libpicosat), Cint, (PicoPtr,), p)
end

function pop(p::PicoPtr)
    _require_picosat_native()
    return ccall((:picosat_pop, libpicosat), Cint, (PicoPtr,), p)
end

function save_original_clauses(p::PicoPtr)
    _require_picosat_native()
    return ccall((:picosat_save_original_clauses, libpicosat), Cvoid, (PicoPtr,), p)
end

picosat_deref_partial(p::PicoPtr, lit::Integer) =
    (_require_picosat_native(); ccall((:picosat_deref_partial, libpicosat), Cint, (PicoPtr,Cint), p, lit))
# void picosat_set_more_important_lit (PicoSAT *, int lit);
function picosat_set_more_important_lit(p::PicoPtr, lit::Int)
    _require_picosat_native()
    return ccall((:picosat_set_more_important_lit, libpicosat), Cvoid, (PicoPtr, Cint), p, lit)
end

function get_partial_solution(p::PicoPtr)
    nvar = PicoSAT.picosat_variables(p)
    if nvar < 0
        PicoSAT.picosat_reset(p)
        throw(ErrorException("number of solution variables < 0"))
    end
    sol = zeros(Int, nvar)
    array_pos = 1
    for i = 1:nvar
        v = picosat_deref_partial(p, i)
        if v!=0
            sol[array_pos] = v * i
            array_pos+=1
        end
    end
    return sol[1:(array_pos-1)]
end

function solve(p::PicoPtr)
    _require_picosat_native()
    @timeit Config.TIMER "PicoSAT_solve" begin
        res =  PicoSAT.picosat_sat(p, -1)
    end
    if res == PicoSAT.SATISFIABLE
        #result = PicoSAT.get_solution(p)
        # Use partial models to improve efficency -> no need to iterate "both sides"
        result = get_partial_solution(p)
    elseif res == PicoSAT.UNSATISFIABLE
        result = :unsatisfiable
    elseif res == PicoSAT.UNKNOWN
        result = :unknown
    else
        throw(ErrorException("PicoSAT Error: return value $res"))
    end
    return result
end

end

using .InternalPicoSAT
