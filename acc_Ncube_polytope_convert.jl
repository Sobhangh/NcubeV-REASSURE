import Pkg
using JLD
using PyCall

#Run this on julia before the script to install polytope package in the python environment used by PyCall
# using Pkg
# Pkg.add("PyCall")
# using PyCall
# println(PyCall.python)   # check which Python PyCall is using
# # install into that exact Python
# run(`$(PyCall.python) -m pip install polytope`)

if length(ARGS) < 1
    error("Usage: julia acc_Ncube_polytope_convert.jl <JLD_FILE_PATH>")
end

input_name = ARGS[1]
if !endswith(lowercase(input_name), ".jld")
    error("Expected a .jld file path, got: $(input_name)")
end

output_name = replace(input_name, r"\.jld$" => ".pkl")

# results-approx-1.jld
# For a run number, you can pass e.g. 0 and this will use the matching filename pattern if present.
data_path = abspath(joinpath(@__DIR__, input_name))
acc_data = load(data_path)
print(acc_data["args"])
acc_result = acc_data["result"]
acc_result = acc_result[1]
acc_stars = acc_result.stars
#acc_stars = map(x->x.stars, acc_result)
println("Number of stars: ", length(acc_stars))
println("Number of certain stars: ", length(filter(x->x.certain,acc_stars)))
#println(propertynames(acc_stars[1]))

function poly_intersect(pc, p1, p2)
    iA = [p1.A; p2.A]
    ib = append!(p1.b,p2.b)

    return pc.Polytope(iA, ib)
end

function get_polytope_list(star_list)
    pc = pyimport("polytope")
    polytope_list = []
    for star in filter(x->x.certain,star_list)
        orig_poly = pc.Polytope(star.constraint_matrix,star.constraint_bias)
        bounds = pc.box2poly(collect(map(x->[x[1],x[2]],star.bounds)))
        orig_poly = poly_intersect(pc, orig_poly, bounds)
        push!(polytope_list,orig_poly)
    end
    return polytope_list
end
py"""
import pickle
def store_polys(name,polys):
    with open(name,"wb") as f:
        pickle.dump(polys,f)
"""
store_polys = py"store_polys"

polytope_list = get_polytope_list(acc_stars)
# polytopes-approx-3-bound.pkl
output_path = abspath(joinpath(@__DIR__, output_name))
store_polys(output_path, polytope_list)
println("Saved: ", output_path)
