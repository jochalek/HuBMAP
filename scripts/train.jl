using DrWatson
quickactivate(@__DIR__)

using FileIO, ImageIO, DataFrames, Images, FastAI, Flux, Metalhead, Wandb, Dates, Logging, HuBMAP
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
    "--backbone"
        arg_type = String
        default  = "resnet34"
    "--dataset"
        arg_type = String
        default  = "HuBMAP_HPA"
    "--prototype"
        action = :store_true
    "--nowandb"
        action = :store_true
end
args = dict2ntuple(parse_args(s))

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
# traindir(args...) = nfsdatadir(datasets[args.dataset][1]..., args...)
# labeldir(args...) = nfsdatadir(datasets[args.dataset][2]..., args...)
traindir(argz...) = datadir(datasets[args.dataset][1]..., argz...)
labeldir(argz...) = datadir(datasets[args.dataset][2]..., argz...)
modeldir(args...) = nfsdatadir("sims", "models", args...)
classes = readlines(open(datadir("exp_pro", "codes.txt")))
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
    traindata = [getobs(data, i) for i in rand(1:numobs(data), 10000)]
else
    traindata = data
end

# task, model = loadtaskmodel(datadir("sims", "models", "initmodel.jld2"))

task = SupervisedTask(
    (Image{2}(), Mask{2}(classes)),
    (
        ProjectiveTransforms((args.imgsize, args.imgsize)),
        ImagePreprocessing(),
        OneHot()
    )
)

backbone = get_backbone(args)
model = taskmodel(task, backbone);

# model = gpu(model);

lossfn = tasklossfn(task)
batchsize = args.batchsize
traindl, validdl = taskdataloaders(traindata, task, batchsize, buffer=true, parallel=false, collate=true)
optimizer = Adam()
learner = Learner(model, lossfn; data=(traindl, validdl), optimizer, callbacks=callbacks)
epochs = args.epochs
lr = args.lr
fitonecycle!(learner, epochs, lr)

savetaskmodel(modeldir(String(runname * ".jld2")), task, learner.model, force = true)

close(lgbackend)
