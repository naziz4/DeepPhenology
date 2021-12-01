# DeepPhenology
Abstract

Community science image libraries offer a massive, but largely untapped, source of observational data for phenological research. The iNaturalist platform offers a particularly rich archive, containing more than 49 million verifiable, georeferenced, open access images, encompassing seven continents and over 278,000 species. A critical limitation preventing scientists from taking full advantage of this rich data source is labor. Each image must be manually inspected and categorized by phenophase, which is both time-intensive and costly. Consequently, researchers may only be able to use a subset of the total number of images available in the database. While iNaturalist has the potential to yield enough data for high-resolution and spatially extensive studies, it requires more efficient tools for phenological data extraction. A promising solution is automation of the image annotation process using deep learning. Recent innovations in deep learning have made these open-source tools accessible to a general research audience. However, it is unknown whether deep learning tools can accurately and efficiently annotate phenophases in community science images. Here, we train a convolutional neural network (CNN) to annotate images of Alliaria petiolata into distinct phenophases from iNaturalist and compare the performance of the model with non-expert human annotators. We demonstrate that researchers can successfully employ deep learning techniques to extract phenological information from community science images. A CNN classified two-stage phenology (flowering and non-flowering) with 95.9% accuracy and classified four-stage phenology (vegetative, budding, flowering, and fruiting) with 86.4% accuracy. The overall accuracy of the CNN did not differ from humans (p = 0.383), although performance varied across phenophases. We found that a primary challenge of using deep learning for image annotation was not related to the model itself, but instead in the quality of the community science images. Up to 4% of A. petiolata images in iNaturalist were taken from an improper distance, were physically manipulated, or were digitally altered, which limited both human and machine annotators in accurately classifying phenology. Thus, we provide a list of photography guidelines that could be included in community science platforms to inform community scientists in the best practices for creating images that facilitate phenological analysis.

Models were trained on two systems: 
Personal Laptop: Ryzen 5 3500U
Desktop: Ryzen 5 3600, NVIDIA RTX 3070

Training parameters:
Batch size: 4
Learning rate: 0.01
Optimizer: SGD
Loss: Cross entropy loss

Packages used

Pytorch: https://github.com/pytorch/pytorch
NumPy: https://github.com/numpy/numpy
Pandas: https://github.com/pandas-dev/pandas
Scikit-learn: https://github.com/scikit-learn/scikit-learn
Pillow: https://github.com/python-pillow/Pillow
Matplotlib: https://github.com/matplotlib/matplotlib
