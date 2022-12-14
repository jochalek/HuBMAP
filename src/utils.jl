using DrWatson, FileIO, ImageIO, DataFrames, Images

##############################
## Mask encoding & decoding
##############################

function decode_rle(rle_seg, mask, class)
    s = split(rle_seg, " ")
    z = zeros(Int, length(s))
    for i in 1:length(s)
        z[i]=parse(Int, s[i])
    end
    for i in range(1, length(z), step=2)
        for k in 0:z[i+1]-1
            mask[z[i]+k] = class
        end
    end
    return mask
end

using ColorTypes, FixedPointNumbers

"""
Looks in a DataFrame for dimensions and labels to generate a mask.
"""
function masks_from_dataframe(df::DataFrame, idx, classes)
	height = df[idx, :img_height]
    width = df[idx, :img_width]
    rle_seg = df[idx, :rle]
    class = classes[df[idx, :organ]]
    # mask = zeros(Gray{N0f8}, height * width)
    mask = zeros(ColorTypes.Gray{FixedPointNumbers.N0f8}, height * width)
    # mask .+= ColorTypes.Gray{FixedPointNumbers.N0f8}(0.004)
    decode_rle(rle_seg, mask, class)
    return mask = reshape(mask, width, :)
end



function gen_masks(df)
    #classes = Dict("prostate"=>Gray{N0f8}(0.03125), "spleen"=>Gray{N0f8}(0.0625), "lung"=>Gray{N0f8}(0.09375), "kidney"=>Gray{N0f8}(0.125), "largeintestine"=>Gray{N0f8}(0.15625))
    classes = Dict("prostate"=>ColorTypes.Gray{FixedPointNumbers.N0f8}(0.004),
                   "spleen"=>ColorTypes.Gray{FixedPointNumbers.N0f8}(0.004),
                   "lung"=>ColorTypes.Gray{FixedPointNumbers.N0f8}(0.004),
                   "kidney"=>ColorTypes.Gray{FixedPointNumbers.N0f8}(0.004),
                   "largeintestine"=>ColorTypes.Gray{FixedPointNumbers.N0f8}(0.004))
    if isdir(datadir("exp_pro", "masks"))
    else mkdir(datadir("exp_pro", "masks"))
    end
    maskdir(args...) = projectdir(datadir("exp_pro", "masks"), args...)

    for i in 1:nrow(df)
        mask = masks_from_dataframe(df, i, classes)
        filename = df[i, :id]
        if isfile(maskdir("$filename" * ".png"))
            println("not overwriting $filename")
            break
        else
            save(File{format"PNG"}(maskdir("$filename" * ".png")), mask)
        end
    end
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
    return strip(encoding)
end


##############################
## Pre/Post processing
##############################

## Chop the images up into smaller ones
# using Augmentor
# img_in = testpattern(RGB, ratio=0.5) ## load the iamge
# img_out = augment(img_in, Crop(20:75,25:120))
# function iter_img_sizes(df::DataFrame)
# 	sizes = unique(df, [:img_height])
#     for i in 1:length(eachrow(sizes[:, 1]))
#         println(sizes[i, [:img_height]][1] / 128)
#     end
# end

# FIXME Iterator oversteps bounds, I think it
# just goes back to 1...
# """
# Returns 'stepsize' sized tiles of a source image. Images should be RBG.
# """
# function tileimage(srcimage; stepsize=128)
#     #srcimage = FastAI.load(srcimage)
#     srcsz = size(srcimage, 1)
#     tiles = 1:stepsize:srcsz
#     tmpdims = length(tiles) * stepsize
#     tmpimage = ones(typeof(srcimage[1]), tmpdims, tmpdims)
#     tmpimage[1:srcsz, 1:srcsz] .= srcimage

#     tiledsrc = []

#     #Threads.@threads for j in 1:length(tiles)
#         #for i in 1:length(tiles)
#     for j in tiles
#         for i in tiles
#             tile = @view tmpimage[i:i+stepsize-1, j:j+stepsize-1]
#             push!(tiledsrc, tile)
#         end
#     end
#     return tiledsrc
# end
# """
# Returns a composite mask from
# tiled predictions.
# """
# function composemask(preds, srcimage)
#     dims = size(srcimage)
#     return @view hvcat(24, preds...)[1:dims[1], 1:dims[2]] # FIXME hardcoded 24 tiles
# end


# FIXME Iterator oversteps bounds, I think it
# just goes back to 1...
using FilePathsBase, FastAI, FastVision, ColorTypes, FixedPointNumbers

DSTDIR = Path(mktempdir())

function presizeimagedir(srcdir, dstdir, sz)
    pathdata = FastAI.MLUtils.filterobs(FastVision.isimagefile, loadfolderdata(srcdir))

    Threads.@threads for i in 1:FastAI.numobs(pathdata)
        srcp = getobs(pathdata, i)
        p = relpath(srcp, srcdir)
        dstp = joinpath(dstdir, p)

        img = loadfile(srcp)
        save_tiles(img, dstp; stepsize=sz)
    end
end

# FIXME filenames including fileextensions
function save_tiles(srcimage, dstdir; stepsize=128)
    srcsz = size(srcimage, 1)
    tiles = 1:stepsize:srcsz
    tmpdims = length(tiles) * stepsize
    if typeof(srcimage[1]) == ColorTypes.Gray{FixedPointNumbers.N0f8}
        tmpimage = zeros(typeof(srcimage[1]), tmpdims, tmpdims)
    else
        tmpimage = ones(typeof(srcimage[1]), tmpdims, tmpdims)
    end
    tmpimage[1:srcsz, 1:srcsz] .= srcimage

    for j in tiles
        for i in tiles
            tile = @view tmpimage[i:i+stepsize-1, j:j+stepsize-1]
            if string(dstdir)[end-3:end] == ".png"
                save(string(dstdir)[begin:end-4] * "_" * string(i) * "_" * string(j) * ".png", tile)
            else
                save(string(dstdir)[begin:end-5] * "_" * string(i) * "_" * string(j) * ".png", tile)
            end
        end
    end
end

##############################
## Misc
##############################
using Dates: now

# Generate unique names for saving models
function runfileid()
    time = now()
    name = replace("$time", ":" => ".")
    return name
end
