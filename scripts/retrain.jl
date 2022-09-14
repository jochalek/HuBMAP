using DrWatson
quickactivate(@__DIR__)

using FileIO, ImageIO, DataFrames, Images, FastAI, Flux, Metalhead, Wandb, Dates, Logging, HuBMAP, Random
using FastVision
using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "--lr"
        arg_type = Float64
        default  = 0.033
    "--epochs"
        arg_type = Int
        default  = 1
    "--imgsize"
        arg_type = Int
        default  = 128
    "--batchsize"
        arg_type = Int
        default  = 2
    "--rng"
        arg_type = Int
        default  = 42
    "--backbone"
        arg_type = String
        default  = "resnet34"
    "--jld2"
        arg_type = String
    "--dataset"
        arg_type = String
        default  = "HuBMAP_HPA128"
    "--prototype"
        action = :store_true
    "--nowandb"
        action = :store_true
end
args = dict2ntuple(parse_args(s))

rng = Random.default_rng()
Random.seed!(rng, args.rng)

## Logging
runname = HuBMAP.runfileid()

lgbackend = WandbBackend(project = "HuBMAP",
                         name = "$runname",
                         config = Dict("learning_rate" => args.lr,
                         "Projective Transforms" => (args.imgsize, args.imgsize),
                         "batchsize" => args.batchsize,
                         "epochs" => args.epochs,
                         "prototype" => args.prototype,
                         "architecture" => "UNet",
                         "backbone" => args.backbone,
                         "dataset" => args.dataset))

function get_backbone(args)
    if args.backbone == "resnet34"
        backbone = Metalhead.ResNet(34).layers[1:end-1]
    else
        backbone = Metalhead.ResNet(34).layers[1:end-1]
    end
    return backbone
end




metricscb = LogMetrics(lgbackend)
hparamscb = LogHyperParams(lgbackend)

callbacks = [
    metricscb,
    hparamscb,
    ToGPU(),
    Metrics(accuracy)
]

## Datasets
datasets = Dict("HuBMAP_HPA" => (("exp_raw", "train_images"), ("exp_pro", "masks")),
                "HuBMAP_HPA128" => (("exp_pro", "t128", "train"), ("exp_pro", "t128", "masks"))
                )

nfsdatadir(args...) = projectdir("../", "data", "HuBMAP", "data", args...)

## Train on 128 presized images
traindir(argz...) = nfsdatadir(datasets[args.dataset][1]..., argz...)
labeldir(argz...) = nfsdatadir(datasets[args.dataset][2]..., argz...)
# traindir(argz...) = datadir(datasets[args.dataset][1]..., argz...)
# labeldir(argz...) = datadir(datasets[args.dataset][2]..., argz...)
modeldir(args...) = nfsdatadir("sims", "models", args...)
classes = readlines(open(nfsdatadir("exp_pro", "codes.txt")))
# classes = ["background", "prostate", "spleen", "lung", "kidney", "largeintestine"]

images = Datasets.loadfolderdata(
    traindir(),
    filterfn=FastVision.isimagefile,
    loadfn=loadfile)

masks = Datasets.loadfolderdata(
    labeldir(),
    filterfn=FastVision.isimagefile,
    loadfn=f -> FastVision.loadmask(f, classes))

data = (images, masks)

if args.prototype
    traindata = [getobs(data, i) for i in rand(rng, 1:numobs(data), 1024)]
else
    traindata = data
end


task, model = loadtaskmodel(modeldir(args.jld2))
model = gpu(model);

lossfn = tasklossfn(task)
batchsize = args.batchsize
traindl, validdl = taskdataloaders(traindata, task, batchsize, buffer=true, parallel=false, collate=true, rng=rng)
optimizer = Adam()
learner = Learner(model, lossfn; data=(traindl, validdl), optimizer, callbacks=callbacks)
epochs = args.epochs
lr = args.lr
fitonecycle!(learner, epochs, lr)

savetaskmodel(modeldir(String(runname * ".jld2")), task, learner.model, force = true)

close(lgbackend)
