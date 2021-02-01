========
Proteus Models
========

Common interfaces for models.

base.py: Contains shared functionality for models.  Subclasses in theory only need to reimplement preprocesing and postprocessing functions
classifcation.py: Superclass for classification models.  Implementing a new classifiers should be very easy
modelconfigs.py: contains the different configs that can be enabled for your model when deploying to Triton.  Subclass those that work.