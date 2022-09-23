module HuBMAPapp

using Flux: DataLoader
using CSV, Images, FileIO, DataFrames, FastAI, FastVision, FastAI.Flux, ArgParse, Metalhead

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--debug_csv"
            help = "Bypass predict and write a dummy submission.csv"
            action = :store_true
        "--no_tiles"
            help = "Flag to use for models not needing tiling"
            action = :store_true
        "--tilesize"
            help = "Must match model input size"
            arg_type = Int
            default = 1024
        "--test_data", "-t"
            help = "The directory of the test data including test.csv and test_images/"
            arg_type = String
            default = "./"
        "--model", "-m"
            help = "The location of the inference model"
            arg_type = String
            default = "./model.jld2"
    end
    return parse_args(s)
end


function tileimage(srcimage; stepsize)
    #srcimage = FastAI.load(srcimage)
    srcsz = size(srcimage, 1)
    tiles = 1:stepsize:srcsz
    tmpdims = length(tiles) * stepsize
    tmpimage = zeros(typeof(srcimage[1]), tmpdims, tmpdims)
    tmpimage[1:srcsz, 1:srcsz] .= srcimage

    tiledsrc = []

    ### Threads breaks gauranteed order,
    ### mixing tiles I think
    #Threads.@threads for j in 1:length(tiles)
        #for i in 1:length(tiles)
    ### This is row major, so not efficient
    ### does it matter?
    for i in tiles
        for j in tiles
            tile = @view tmpimage[i:i+stepsize-1, j:j+stepsize-1]
            push!(tiledsrc, tile)
        end
    end
    return tiledsrc
end

function composemask(preds, srcimage, args)
    dims = size(srcimage)
    stepsize = args["tilesize"]
	tiles_per_row = length(1:stepsize:size(srcimage, 1))
    return @view hvcat(tiles_per_row, preds...)[1:dims[1], 1:dims[2]]
end


function runmodel(batch, args)
	taskmodel = args["model"]
	task, model = loadtaskmodel(taskmodel)
	model = gpu(model);
	preds = predictbatch(task, model, batch; device = gpu, context = Inference())
	return preds
end

#FIXME Probably can just use StatsBase by changing Array -> Vector
"""
Transforms FastAI's inference mask of IndirectArray pixel labels
inro a string fit for Kaggle submission.

Largely copied from StatsBase.rle()
"""
function encode_rle(v)
    n = length(v)
    vals = []
    lens = Int[]

    n>0 || return (vals,lens)

    cv = v[1]
    cl = 1

    i = 2
    @inbounds while i <= n
        vi = v[i]
        if isequal(vi, cv)
            cl += 1
        else
            push!(vals, cv)
            push!(lens, cl)
            cv = vi
            cl = 1
        end
        i += 1
    end

    # the last section
    push!(vals, cv)
    push!(lens, cl)

	encoding = string()

	for i in 1:length(vals)
		if vals[i] != "background"
			pixel_idx = i == 1 ? 1 : sum(lens[1:i-1]) + 1
			encoding = string(encoding, pixel_idx, " ", lens[i], " ")
		end
	end
    return encoding
end

function predict_by_part(batch, args)
    taskmodel = args["model"]
    task, model = loadtaskmodel(taskmodel)
    model = gpu(model);
    minibatch = Int(3072 / args["tilesize"])
    predictions = []


    arrayloader = DataLoader(batch; batchsize=minibatch, shuffle=false)
    for x in arrayloader
        tmpred = predictbatch(task, model, x; device = gpu, context = Inference())
        predictions = push!(predictions, tmpred...)
    end
    return predictions
end

function predict(id, args)
    image = Images.load(joinpath(args["test_data"], "test_images", string(id) * ".tiff"))

	#TILEIMAGE
    if args["no_tiles"]
        batch = image
    else
	    batch = tileimage(image; stepsize=args["tilesize"])
    end
	#RUNMODEL
    if args["no_tiles"]
        preds = runmodel(batch, args)
    else
        preds = predict_by_part(batch, args)
    end
	#COMPOSEMASK
    if args["no_tiles"]
        mask = preds
    else
        mask = composemask(preds, image, args)
    end
	#RLE
	return encode_rle(mask)
end

function generate_submission(df::DataFrame, args)
    df_subm = DataFrame()
    if args["debug_csv"]
        for row in 1:nrow(df)
            df_row = DataFrame("id" => df[row, :id], "rle" => "1 1")
            df_subm = vcat(df_subm, df_row)
        end
    else
        for row in 1:nrow(df)
            @info "Predicting $(df[row, :id])"
            df_row = DataFrame("id" => df[row, :id], "rle" => predict(df[row, :id], args))
            df_subm = vcat(df_subm, df_row)
        end
    end
    return df_subm
end

function write_submission(df::DataFrame)
    CSV.write("submission.csv", df; bufsize=2^23)
end

function real_main()
    args = parse_commandline()
    df = DataFrame(CSV.File(joinpath(args["test_data"], "test.csv")))
    df_subm = generate_submission(df, args)
    write_submission(df_subm)
end

function julia_main()::Cint
  # do something based on ARGS?
  try
      real_main()
  catch
      Base.invokelatest(Base.display_error, Base.catch_stack())
      return 1
  end
  return 0 # if things finished successfully
end

end # module

