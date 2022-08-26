using DrWatson
quickactivate(@__DIR__)

using FileIO, ImageIO, DataFrames, Images, FastAI, Flux, Metalhead
using FastVision

## Train on 128 presized images
traindir(args...) = datadir("exp_pro", "t128", "train", args...)
labeldir(args...) = datadir("exp_pro", "t128", "masks", args...)
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

task, model = loadtaskmodel(datadir("sims", "models", "fastai5.jld2"))

# task = SupervisedTask(
#     (Image{2}(), Mask{2}(classes)),
#     (
#         ProjectiveTransforms((128, 128)),
#         ImagePreprocessing(),
#         OneHot()
#     )
# )

# backbone = Metalhead.ResNet(34).layers[1:end-1]
# model = taskmodel(task, backbone);

model = gpu(model);

lossfn = tasklossfn(task)
traindl, validdl = taskdataloaders(data, task, 64, buffer=true, parallel=false)
optimizer = Adam()
learner = Learner(model, lossfn; data=(traindl, validdl), optimizer, callbacks=[ToGPU()])
fitonecycle!(learner, 1, 0.033)
