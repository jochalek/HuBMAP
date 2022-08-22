module HuBMAPapp

using CSV, Images, FileIO, DataFrames, FastAI, ArgParse

function parse_commandline()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--debug_csv"
            help = "Bypass predict and write a dummy submission.csv"
            action = :store_true
        "--test_data", "-t"
            help = "The directory of the test data including test.csv and test_images/"
            arg_type = String
            default = "/kaggle/input/hubmap-organ-segmentation"
        "--model", "-m"
            help = "The location of the inference model"
            arg_type = String
            default = "/kaggle/input/models/initmodel.jld2"
    end
    return parse_args(s)
end


function tileimage(srcimage; stepsize=128)
    #srcimage = FastAI.load(srcimage)
    srcsz = size(srcimage, 1)
    tiles = 1:stepsize:srcsz
    tmpdims = length(tiles) * stepsize
    tmpimage = zeros(typeof(srcimage[1]), tmpdims, tmpdims)
    tmpimage[1:srcsz, 1:srcsz] .= srcimage

    tiledsrc = []

    #Threads.@threads for j in 1:length(tiles)
        #for i in 1:length(tiles)
    for j in tiles
        for i in tiles
            tile = @view tmpimage[i:i+stepsize-1, j:j+stepsize-1]
            push!(tiledsrc, tile)
        end
    end
    return tiledsrc
end

function composemask(preds, srcimage)
    dims = size(srcimage)
	tiles_per_row = length(1:128:size(srcimage, 1))
    return @view hvcat(tiles_per_row, preds...)[1:dims[1], 1:dims[2]] # FIXME tiles_per_row or something better?
end


function runmodel(batch, args)
	taskmodel = args["model"]
	task, model = loadtaskmodel(taskmodel)
	model = gpu(model);
	preds = predictbatch(task, model, batch; device = gpu, context = Inference())
	return preds
end

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
			encoding = string(encoding, sum(lens[1:i]), " ", lens[i], " ")
		end
	end
    #return strip(encoding) #FIXME is this the issue?
    return encoding
end


function predict(id, args)
    image = Images.load(joinpath(args["test_data"], "test_images", string(id) * ".tiff"))

	#TILEIMAGE
	batch = tileimage(image)
	#RUNMODEL
	preds = runmodel(batch, args)
	#COMPOSEMASK
	mask = composemask(preds, image)
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

## FIXME Remove before compile
# HuBMAPapp.real_main(HuBMAPapp.df)
