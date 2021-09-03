@enum INTEGRALITY MIP CONC LP

function generate_points(G_max,C,d)
    GS = sortperm(C)
    sums = cumsum(G_max[GS])
    duals = C[reverse!(GS)]
    points = Tuple{Float64,Float64}[(0.0,0.0)]
    i = 0
    for e in Iterators.reverse(sums)
        excess_energy = d-e
        if excess_energy > 0
            push!(points,(excess_energy,excess_energy*duals[i]))
        end
        i+=1
    end
    push!(points,(d,d*duals[i]))
    return points
end

function f(e,G_max,C,d)
    GS = [Pair(G_max[i] , C[i]) for i = 1:length(G_max)]
    GS = reverse(sort(collect(GS), by=x->x[2]))
    sums = []
    for i = 1:length(G_max)
        if i == 1
            push!(sums,sum(G_max))
        else
            push!(sums,sum(G_max)-sum([GS[j][1] for j = 1:i-1]))
        end
    end
    def = [sums[i] - d + e for i = 1:length(G_max)]
    z1 = 1
    z2 = length(G_max)
    i = 1
    while def[z1] > 0 && def[z2] < 0
        z_m = convert(Int,floor((z1+z2)/2))
        if def[z_m] < 0
            z2 = z_m
        else
            z1 = z_m
        end
        if z1 == z2 - 1
            break
        end
    end
    if def[z2] >= 0
        i = z2
    else
        i = z1
    end
    return GS[i][2]
end

function get_duals(G_max,C,d)
    GS = sortperm(C)
    sums = cumsum(G_max[GS])
    duals = C[reverse!(GS)]
    eff_duals = []
    i = 0
    for e in Iterators.reverse(sums)
        excess_energy = d-e
        if excess_energy > 0
            push!(eff_duals,duals[i])
        end
        i+=1
    end
    push!(eff_duals,duals[i])
    return eff_duals
end

function cross(o, a, b)
      return (a[1] - o[1]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[1] - o[1])
end

function upper_cvx_hull(points)
    if length(points) <= 1
        return points
    end    
    upper = []
    
    for p in Iterators.reverse(points)
        if length(upper) >= 2
            while cross(upper[end-1], upper[end], p) <= 0
                pop!(upper)
                if length(upper) < 2
                    break
                end
            end
        end
        push!(upper,p)
    end
    return upper
end

function convexify_pi_function(G_max,C,d)
    points = generate_points(G_max,C,d)
    points_cvx = upper_cvx_hull(points)
    points_x = [points_cvx[i][1] for i = 1:length(points_cvx)]
    points_y = [points_cvx[i][2] for i = 1:length(points_cvx)]
    return points_x,points_y
end

function reward_function(G_t,C_t,D,e,model; integrality::INTEGRALITY = MIP)
    
    θ = @variable(model)
    if integrality == MIP || integrality == LP
        duals = get_duals(G_t,C_t,D)
        K = length(duals)
        points = generate_points(G_t,C_t,D)
        points_x = [points[i][1] for i = 1:length(points)]
        Delta = points_x
        if integrality == MIP
            z = @variable(model, [1:K], binary=true)
        elseif integrality == LP
            z = @variable(model, [1:K], lower_bound = 0, upper_bound = 1)
        end
        e_seg = @variable(model, [1:K])
        @constraint(model, θ == sum(duals[k]*e_seg[k] for k in 1:K))
        @constraint(model, sum(e_seg) == e)
        @constraint(model, sum(z) == 1)
        for k in 1:K
            @constraint(model, e_seg[k] <= z[k]*Delta[k+1])
            @constraint(model, e_seg[k] >= z[k]*Delta[k])
        end
    elseif integrality == CONC
        points_x,points_y = convexify_pi_function(G_t,C_t,D)
        obj = conv_func(model, e, "1", points_x, points_y)
        @constraint(model, θ == obj)
    end
    
    return θ
end

function conv_func(m, x, t, points_x, points_y)
    obj = @variable(m, base_name="obj_"*string(t))
    for i in 1:length(points_x)-1
        g_i = (points_y[i+1]-points_y[i])/(points_x[i+1]-points_x[i])
        h_i = (points_y[i]*points_x[i+1]-points_y[i+1]*points_x[i])/(points_x[i+1]-points_x[i])
        @constraint(m, obj <= g_i*x+h_i)
    end
    return obj
end