# HuBMAP

My first Kaggle submission.

1,090th place.

This project applies [FastAI.jl](https://github.com/FluxML/FastAI.jl)'s image segmentation to [HuBMAP + HPA - Hacking the Human Body](https://www.kaggle.com/competitions/hubmap-organ-segmentation).


This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> HuBMAP

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.
