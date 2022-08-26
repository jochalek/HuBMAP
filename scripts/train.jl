using DrWatson
quickactivate(@__DIR__)

using FileIO, ImageIO, DataFrames, Images, FastAI, Flux, Metalhead, Wandb, Dates, Logging, HuBMAP
using FastVision

## Logging
runname = HuBMAP.runfileid()

lgbackend = WandbBackend(project = "HuBMAP",
                         name = "$runname",
                         config = Dict("learning_rate" => 0.033,
                         "Projective Transforms" => (128, 128),
                         "batchsize" => 2,
                         "epochs" => 10,
                         "prototype" => false,
                         "architecture" => "UNet",
                         "backbone" => "ResNet",
                         "dataset" => "HuBMAP_HPA"))

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

## Train on 128 presized images
traindir(args...) = datadir(datasets[get_config(lgbackend, "dataset")][1]..., args...)
labeldir(args...) = datadir(datasets[get_config(lgbackend, "dataset")][2]..., args...)
modeldir(args...) = datadir("sims", "models", args...)
classes = readlines(open(datadir("exp_pro", "codes.txt")))


images = Datasets.loadfolderdata(
    traindir(),
    filterfn=FastVision.isimagefile,
    loadfn=loadfile)

masks = Datasets.loadfolderdata(
    labeldir(),
    filterfn=FastVision.isimagefile,
    loadfn=f -> FastVision.loadmask(f, classes))

data = (images, masks)
quicksamples = [getobs(data, i) for i in rand(1:numobs(data), 10000)]

if get_config(lgbackend, "prototype")
    traindata = quicksamples
else
    traindata = data
end

# task, model = loadtaskmodel(datadir("sims", "models", "initmodel.jld2"))

task = SupervisedTask(
    (Image{2}(), Mask{2}(classes)),
    (
        ProjectiveTransforms(get_config(lgbackend, "Projective Transforms")),
        ImagePreprocessing(),
        OneHot()
    )
)

backbone = Metalhead.ResNet(34).layers[1:end-1]
model = taskmodel(task, backbone);

# model = gpu(model);

lossfn = tasklossfn(task)
batchsize = get_config(lgbackend, "batchsize")
traindl, validdl = taskdataloaders(traindata, task, batchsize, buffer=true, parallel=false)
optimizer = Adam()
learner = Learner(model, lossfn; data=(traindl, validdl), optimizer, callbacks=callbacks)
epochs = get_config(lgbackend, "epochs")
lr = get_config(lgbackend, "learning_rate")
fitonecycle!(learner, epochs, lr)

savetaskmodel(modeldir(String(runname * ".jld2"), task, learner.model, force = true)

close(lg)
