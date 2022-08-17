# Optimizing parameters of a MTK Standard Library code

## Heat Conduction Model

This example demonstrates the thermal response of two masses connected by a conducting element. 
The two masses have the same heat capacity but different initial temperatures (`T1=100 [°C]`, `T2=0 [°C]`). 
The mass with the higher temperature will cool off while the mass with the lower temperature heats up. 
They will each asymptotically approach the calculated temperature T_final_K that results 
from dividing the total initial energy in the system by the sum of the heat capacities of each element.

## Generating data from the model

```julia
using ModelingToolkitStandardLibrary.Thermal, ModelingToolkit, OrdinaryDiffEq, Plots

@parameters t

C1 = 15
C2 = 15
@named mass1 = HeatCapacitor(C=C1, T_start=373.15)
@named mass2 = HeatCapacitor(C=C2, T_start=273.15)
@named conduction = ThermalConductor(G=10)
@named Tsensor1 = TemperatureSensor() 
@named Tsensor2 = TemperatureSensor()

connections = [
    connect(mass1.port, conduction.port_a),
    connect(conduction.port_b, mass2.port),
    connect(mass1.port, Tsensor1.port),
    connect(mass2.port, Tsensor2.port),
]

@named model = ODESystem(connections, t, systems=[mass1, mass2, conduction, Tsensor1, Tsensor2])
sys = structural_simplify(model)
prob = ODEProblem(sys, Pair[], (0, 5.0))
sol = solve(prob, Tsit5())
```

## Defining the inverse problem and generating a virtual population

```julia
data = DataFrame(sol)

trial = Trial(data, sys, tspan=(0.0, 5.0))

invprob = InverseProblem([trial], sys, [mass1.C => (10, 20), mass2.C => (10, 20)])


vp = vpop(invprob, StochGlobalOpt(maxiters=10), population_size = 50)

@test_nowarn plot(vp, trial)

params = DataFrame(vp)
```

## Comparing parameter estimates with original values

```julia
C1_estimate  = params[:, 1]
C2_estimate = params[:, 2]


l = @layout [a b]

p1 = plot([C1,C1],[0.0, 0.4], ylims = (0, 0.6), lw=3,color=:green,label="True value: C1",linestyle = :dash)
density!(C1_estimate, label = "Estimated value: C1")

p2 = plot([C2,C2],[0.0, 0.4],ylims = (0, 0.6), lw=3,color=:green,label="True value: C2",linestyle = :dash)
density!(C2_estimate, label = "Estimated value: C2")

plot(p1, p2,  layout = l)
```

