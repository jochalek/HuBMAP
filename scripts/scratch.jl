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

# ##############################
# ## DrWatson Data Management
# ##############################
# traindir(args...) = datadir("exp_raw", "train_images", args...)
# labeldir(args...) = datadir("exp_pro", "masks", args...)
# classes = readlines(open(datadir("exp_pro", "codes.txt")))

# ##############################
# ## Sample model generation
# ##############################
# using FastAI
# images = Datasets.loadfolderdata(
#     traindir(),
#     filterfn=isimagefile,
#     loadfn=loadfile)

# masks = Datasets.loadfolderdata(
#     labeldir(),
#     filterfn=isimagefile,
#     loadfn=f -> loadmask(f, classes))

# data = (images, masks)

# image, mask = sample = getobs(data, 1);
# task = BlockTask(
#     (Image{2}(), Mask{2}(classes)),
#     (
#         ProjectiveTransforms((128, 128)),
#         ImagePreprocessing(),
#         OneHot()
#     )
# )
# checkblock(task.blocks, sample)
# xs, ys = FastAI.makebatch(task, data, 1:3)
# showbatch(task, (xs, ys))

# savetaskmodel("../tmp/initmodel.jld2", task, learner.model, force = true)
# task, model = loadtaskmodel("catsdogs.jld2")
# model = gpu(model);
# x, y =
# samples = [getobs(data, i) for i in rand(1:100, 2)]
# images = [sample[1] for sample in samples]
# labels = [sample[2] for sample in samples]
# preds = predictbatch(task, model, images; device = gpu, context = Inference())

# begin
# task, model = loadtaskmodel("/home/justin/projects/tmp/initmodel.jld2")
# lossfn = tasklossfn(task)
# traindl, validdl = taskdataloaders(data, task, 16)
# optimizer = FastAI.ADAM()
# learner = Learner(model, (traindl, validdl), optimizer, lossfn, ToGPU())
# fitonecycle!(learner, 1, 0.033)
# showoutputs(task, learner; n = 4)
# end
