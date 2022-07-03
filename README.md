# Downscaled future forcing for Lake Garda

GAN model is taken from Stengel et al (https://github.com/NREL/PhIRE). Modification made to discriminator network was addition of dropout layers after fully connected layers

Data prep is to go from NetCDF files to .tfrecord, this is input format for the GAN. The GAN model is trained in a two steps. First pretraining with only the generator. Second, training in adversarial fashion with generator against a discriminator.

Best to run models on GPU's, for which I used Google Collab (see attached notebooks called temp trainer and wind trainer).

Performance extractor calculates RMSE and SSIM statistics for all epochs of a model suppload. Analysis notebook contains all visualisation scripts. Helper functions contains all functions used in data prep, performance extraction and analysis.
