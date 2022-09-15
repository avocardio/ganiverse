# Ganiverse

The implementation of a General Adversiral Network (GAN) for the creation of new photorealistic celestial bodies. Jump to our website (xxx) to produce and download your own planet(s) in real time.


## Project overview

![This is the pipeline of our project](/media/project_plan.jpg)    


## Data selection

Most of the selected data is taken from NASA. It consists of either original photographs (through telescopes like hubble) or computer simulated / composed images.

## Pre-processing

![Preprocessing](/media/prepro_slide.jpg)

Our pre-processing includes scaling, shuffling, normalising of the original dataset. We take a collection of 31 celestial objects including 13 planets and 18 moons and send it through a mix-up and basic data augmention pipeline to stretch the dataset. The Mix-up data augmentation is inspired by FAIR's preprocessing for their ConvNeXt. We end up with a total of 1023 images to train our GAN.

Raw images (randomxrandom) > Augmenting (1000x1000) > GAN (1000x1000) > Smoothing (1000x1000) > Scaling (2000x2000, etc.)

## Training our GAN

xyz

## Scaling produced images

Using Super Resolution (SR) module of OpenCV  
Pre-trained models  
Data Processing Inequality (cannot produce data that is not already there)  

## Going live / web-application

- random name generator for naming planets (in JS)


## Important links: 

- [ExoplanetKyoto Database (attributes)](http://www.exoplanetkyoto.org/exohtml/JA_All_Exoplanets.html)
- [Exoplanet data for CNN (attributes)](https://www.kaggle.com/datasets/mrisdal/open-exoplanet-catalogue)

- Python Library for mapping textures onto 3D objects: [PyVista](https://docs.pyvista.org/index.html)

__ 

- https://svs.gsfc.nasa.gov/30783
- https://www.nasa.gov/feature/amazing-earth-satellite-images-from-2020/
- https://app.box.com/s/cd51dtkmovwxqlusp9o03l191u0ynrzt
- https://app.box.com/s/q0q6f0audek5932ajhtk45k7whep8jxw
- https://app.box.com/s/7qd4egdv6oktfxwugo92l00uykqw0yd8

## Twitter Account:

@theganiverse
Ganiverse300
