using Optimization, OptimizationOptimisers, Zygote, OrdinaryDiffEq, Plots
using Random, ComponentArrays, CSV, Tables
using Lux, LuxCUDA 
import DiffEqFlux: NeuralODE
using DataFrames , JLD2 


function parse_array_string(str)
    try
        ismissing(str) && return missing        
        # Remove brackets and any extra whitespace
        numbers_str = replace(str, r"[\[\]]" => "")
        
        # Split by comma and remove whitespace
        number_strings = strip.(split(numbers_str, ","))
        # Convert to numbers, handling potential errors
        return [tryparse(Float64, n) for n in number_strings]
    catch e
        @warn "Error parsing array string: $str"
        return missing
    end
end



# Initialize CUDA settings
CUDA.allowscalar(false)

# Set up RNG and devices
rng = Xoshiro(0)
const gdev = gpu_device()
const cdev = cpu_device()

# Define the model with explicit types
model = Chain(
    Dense(3 ,128, relu),
    Dense(128 , 128, relu),
        Dense(128 , 3)
) 

# Setup parameters
ps, st = Lux.setup(rng, model)
ps = ps |> ComponentArray 
st = st 





tspan = (0.0f0 , 10.0f0)::Tuple{Float32, Float32}
numsteps = 100 
tsteps = range(tspan[1], tspan[2]; length = numsteps)
neural = NeuralODE(model, tspan, Tsit5(); saveat = tsteps)

function predict_neuralode(u0 , ps)
    Array{Float32}(neural(u0, ps, st)[1])
end


function loss_func(model   , ps , st , (Batch , Time_Batch)...)
    total_loss = 0.0f0 
    for i in 1:Batch_size 
        temp_neural = NeuralODE(model , (Time_Batch[i , 1] , Time_Batch[i , end]) , Tsit5(); saveat = Time_Batch[i , :])
        pred = Array{Float32}(temp_neural(Batch[i , 1 , :], ps, st)[1])
        total_loss += sum( (pred .- Batch[i , : , :]).^2)
    end 

    return total_loss /= Batch_size
end 



const num_epochs = 32
losses = Float32[]

# Create training state 
train_state = Training.TrainState(model, ps, st, Adam(0.01f0))


#### Constants ####
const num_of_samples = 5
const Batch_size = 32
const adtype = Optimization.AutoZygote()


Batch = Array{Float32}(undef,  Batch_size , num_of_samples,3)
Time_Batch = Array{Float32}(undef , Batch_size ,num_of_samples )

for index in 1:101 
    df = CSV.read("Training-data000001000.csv" , DataFrame )
    select!(df , Not([:Column1 , :particle_id]))   
    
	


    # This version processes only columns that start with "hit num"
    for colname in propertynames(df)
        if startswith(String(colname), "hit num")
            df[!, colname] = parse_array_string.(df[!, colname])
        end
    end

    counter = 1


    for track in eachrow(df) 
		#display(track)
        track = track |> Array 
        number_of_hits = length(collect(skipmissing(track)))
		if number_of_hits <= num_of_samples
			continue 
		end 
        distance = zeros(Float32 , numsteps , number_of_hits)
        random_points_from_track = rand(1:number_of_hits , num_of_samples)

        for (ind_samples , sample) in enumerate(random_points_from_track)
            track[sample] = track[sample] |> Array{Float32}
            pred = predict_neuralode( track[sample] , ps) 

            for i in 1:numsteps
                for data_point in 1:number_of_hits
                    distance[i , data_point] = sqrt(sum((pred[ : , i  ] .- track[data_point]).^2) )
                    
                end 
            end 
            
            ti , data_point = argmin(distance).I
            
            
            
            Batch[counter , ind_samples , :] = track[data_point]  

            Time_Batch[counter , ind_samples] = tsteps[ti] 
        end 
		counter += 1 




        # When batch is full, perform training step
        if counter == Batch_size

        
            for epoch in 1:num_epochs 
                
                (_, loss, _, train_state) = Training.single_train_step!(
                    AutoZygote(), loss_func, (Batch, Time_Batch), train_state)
                
                print("goooooooooooooooooooooooooooooooooooooooone right ? ")
                push!(epoch_losses, loss)
                
                # Reset batch counter
                batch_counter = 1
                
                # Print progress
                if length(epoch_losses) % 10 == 0
                    mean_loss = mean(epoch_losses[max(1, end-9):end])
                        @info "Epoch $epoch | Batch $(length(epoch_losses)) | Mean Loss: $mean_loss"  
                end 


            end

        end 






    end 
end 







using StatsBase

counts = countmap(Time_Batch)
sorted_counts = sort(collect(counts), by = x -> x[2], rev = true)
top_n = 10
top_values = [k for (k, v) in sorted_counts[1:min(top_n, end)]]
pushfirst!(top_values , minimum(Time_Batch)) 
push!(top_values  , maximum(Time_Batch))




#saving the model for inference phase 
@save "trained_ode.jdl2" ps st top_values