using Optimization, OptimizationOptimisers, Zygote, OrdinaryDiffEq, Plots
using ComponentArrays, CSV, Tables
using Lux , Random
import DiffEqFlux: NeuralODE
using DataFrames , JLD2 
using LinearAlgebra , Statistics
using Logging

logfile = open("training_log.txt", "a")  # Open the log file in append mode
logger = SimpleLogger(logfile, Logging.Info)  # Create a logger that logs at the Info level
global_logger(logger)

function parse_array_string(str)
    try
        ismissing(str) && return missing        
        # Remove brackets and any extra whitespace
        numbers_str = replace(str, r"[\[\]]" => "")
        
        # Split by comma and remove whitespace
        number_strings = strip.(split(numbers_str, ","))
        # Convert to numbers, handling potential errors
        return [tryparse(Float32, n) for n in number_strings]
    catch e
        @warn "Error parsing array string: $str"
        return missing
    end
end

rng = Xoshiro(0)

# Define the model with explicit types
model = Chain(
    Dense(3 ,128, relu),
    Dense(128 , 128, relu),
    Dense(128 , 128, relu),
        Dense(128 , 3)
) 

# Setup parameters
ps, st = Lux.setup(rng, model)
global ps = ps |> ComponentArray 
st = st 

const tspan = (0.0f0 , 0.1f0)::Tuple{Float32, Float32}
const numsteps = 40
const tsteps = range(tspan[1], tspan[2]; length = numsteps)
neural = NeuralODE(model, tspan, Tsit5(); saveat = tsteps)

function predict_neuralode(u0 , ps)
    temp = neural(u0, ps, st)[1:2]
    return reshape(Array{Float32}(temp[1]), 3, nmu_of_samples, numsteps)
end


function loss_func( ps , tracks)
    local total_loss = 0f0 
    local normalizing::Int16 = 0 
    for track_id in 1:Batch_size
        track = tracks[track_id , : ]
        number_of_hits = count(!ismissing, track)    
        

        if number_of_hits <= nmu_of_samples
            normalizing += 1 
            continue 
        end
        
        track = hcat(track[1:number_of_hits]...)'
        samples = rand(1:number_of_hits - 1 , nmu_of_samples) #take random samples of the track , -1 to assure itsnot the last one 
        input_data = reshape(track[samples, :], 3, nmu_of_samples) 
        pred = predict_neuralode(input_data , ps)


        for (indx , sample)  in enumerate(samples)    

            local distance = 0f0
            local min_dist::Float32 = 1e7

            for step in 1:numsteps
                distance = norm(pred[:,indx, step] .- track[sample + 1  , :])    
                if min_dist > distance 
                    min_dist = distance  
                end 
            end 
        
            total_loss +=  min_dist^2
            

        end 


    end
    return total_loss / ( (Batch_size-normalizing) * nmu_of_samples) 
end 

#### Constants ####
const nmu_of_samples = 5
const Batch_size = 32
const num_epochs = 5

# Define initial and final learning rates
const initial_lr = 0.0001f0
const final_lr = 0.00001f0
const decay_rate = exp(log(final_lr/initial_lr) / 11)

# Function to compute the learning rate based on the file index
function decaying_lr(index)
    return initial_lr * decay_rate^(index - 1)
end

for index in 1:11 
    df = CSV.read("Data/Clean-Data$index.csv" , DataFrame )
    select!(df , Not([:Column1 , :particle_id]))   
    
    # Use the decaying learning rate for each file
    opt = Adam(decaying_lr(index), (0.9, 0.999))  
    opt_state = Optimisers.setup(opt, ps)

    for colname in propertynames(df)
        if startswith(String(colname), "hit num")
            df[!, colname] = parse_array_string.(df[!, colname])
        end
    end


    for batch_counter in Batch_size:Batch_size:size(df)[1]
        data = Array(df[ batch_counter-Batch_size+1 : batch_counter, :])
        ep_losses = Float32[]
        start = time()
        for epoch in 1:num_epochs 
            # Calculate gradients using Zygote
            loss, grads = Zygote.withgradient(ps) do p
                loss_func(p , data )
            end

            # Update parameters using the optimizer
            opt_state, ps = Optimisers.update(opt_state, ps, grads[1])

            # Store loss
            push!(ep_losses, loss[1])
            if epoch == num_epochs
                mean_loss = mean(ep_losses)
                println("Mean Loss:" ,   mean_loss)
                @info "Epoch $epoch | Mean Loss: $mean_loss"  
            end 
        end 
        print(time() - start )

        if batch_counter % 50 == 0 
            @save "trained_ode$index-$batch_counter.jdl2" ps st 
        end
    end 
#saving the model for inference phase 
@save "trained_ode$index.jdl2" ps st 
end 

# Close the logger when done
close(logfile) 