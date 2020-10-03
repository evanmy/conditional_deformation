# Learning Conditional Deformable ShapeTemplates for Brain Anatomy
<img src="figures/architecture.png" width='600'>

## Abstract
A  brain  template  that  describes  the  anatomical  layout  ofan “average” brain is an essential building block of neuroimage analysis pipelines. However, a single template is often not sufficient to fullycapture the variability in a heterogeneous population. Brain structureshave very different shapes and sizes in different clinical and demographicgroups.  In  this  paper,  we  develop  a  novel  neural  network  model  thatcaptures  this  morphometric  variability.  Our  model  learns  to  computean  attribute-specific  spatial  deformation  that  warps  a  brain  template.We train this model on individual brain MRI segmentations in an end-to-end  fashion,  allowing  for  fast  inference  during  testing.  We  demon-strate the ability of our model to deform a brain template given a widerange of ages, presence of disease and different sexes. Detailed qualita-tive and quantitative experiments are provided in order to demonstratethe  flexibility  of  our  model.  Finally,  we  study  the  surface  of  the  de-formed  template’s  hippocampus  to  show  how  our  model  can  be  usedfor shape analysis

## Requirements
The code was tested on:
- python 3.6
- pytorch 1.1
- torchvision 0.3.0
- scikit-image 0.15.0
- scikit-learn 0.19.1
- matplotlib 3.0.2
- numpy 1.15.4
- tqdm 4.38.0

## Instruction

### Prepocessing 
T1-weighted 3D brain MRI scans was preprocessed using FreeSurfer [1]. This includes skull stripping, bias-field correction, intensity normalization, affine registration to Talairach space, and resampling to 1 mm<sup>3</sup> isotropic resolution. The original images had a shape of 256x256x256, which was futher cut using ((48, 48), (31, 33), (3, 29)) along the sides to elimate empty space. 

Unfortunately, Buckner40 and its manual segmentation are not public dataset. However, we provide some example OASIS [2] volumes in `./data/vols/` and automatic segmentations in `./data/labels/`. The labels legend can be found in the [FreeSurfer webpage](https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT). We also provide the probabilistic atlas that we used in our experiment in the data folder. Check `functions/dataset.py` for further details on how we convert these numpy files into torch tensor. 

### Training
Once your MRI has been preprocessed, the next step is to obtain a good initialization for the encoder. This is accomplish by first mapping your training brain MRI to the probabilistic atlas. As an example, we provide our initialization in `./weights/pretrained_encoder.pth.tar`. However, you should pretrain your own encoder for your dataset in order to obtain the good results.

To train SAE, run `python train.py 0 2 --compute_dice`

Choose which of your gpus you want to use through `args.gpus`. Note that our current model only support `--args.batch_size=1` due to memory and computational constraint.

One important parameters in the script is `args.sigma`. Setting `args.sigma = 2` allows you to estimate the variance σ
in 1/(2σ<sup>2</sup>) ||x-x<sup>'</sup>|| as described in the paper. Setting `args.sigma = 0` allows you to set a fixed weight 
α to the reconstruction term α||x-x<sup>'</sup>||

The parameter `--args.beta` puts weight on the L<sub>mrf</sub>. If this term is not 0, `--args.k` is used to determine the size of the neighboorhood constraint. 

Finally, `--compute_dice` allows you to track the dice score of your prediction agaisnt the ground truth label. This is *not* used for training or model selection, but it can help you debug your code during training. 

### Evaluation
Run `python test.py 0`. It will return a dictionary which can be loaded using `torch.load(result.pth.tar')['stat']`. It has 13 columns. The 12 first column contain the dice loss (1-dice) of 12 region of interest (rois). Namely: pallidum, amygdala, caudate, cerebral cortex, hippocampus, thalamus, putamen, cerebral wm, cerebellum cortex, lateral ventricle, cerebellum wm and brain-stem. The last column is the average of the rois.

## Contact
Feel free to open an issue in github for any problems or questions.

## References
[1] Bruce Fischl. Freesurfer. Neuroimage, 62(2):774–781, 2012

[2] Marcus et al. Open access series of imaging studies (OASIS): cross-sectional MRI data
in young, middle aged, nondemented, and demented older adults. Journal of Cognitive
Neuroscience, 2007.
