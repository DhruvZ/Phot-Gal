# Phot-Gal

``PHOT-GAL`` is a package designed to perform SED modeling for galaxy spectra. However, unlike traditional SED fitting packages, ``PHOT-GAL`` is constructed from a simulation-based inference (SBI) perspective. ``PHOT-GAL`` is trained off a set of > 2 million spectra from ``SIMBA`` and ``IllustrisTNG`` galaxies extracted from a subset of the ``CAMELS-1P`` 25 Mpc/h volumes for integer redshifts 0-6, which varied each simulation astrophysics parameter one at a time from the fiducial values. Because of this design choice, ``PHOT-GAL`` is not tied to the assumptions of galaxy physics in either model and can marginalize over some of the uncertainty in the galaxy physics in simulations. Using 3D radiative transfer on cosmological galaxy simulations allows us to reduce the number of assumptions to derive galaxy properties.

This is an initial, basic release of our fiducial model, with the necessary components to model galaxy spectra in this repository. The ``quick_example_test.py`` file showcases a very simple example for how to run ``PHOT-GAL`` as it currently stands (more QOL features coming). 

Mechanically, ``PHOT-GAL`` takes in inputs for photometric observations (as long as the filters correspond to the list of training filters) in Jy. Unlike for many other ML-based methods, ``PHOT-GAL`` can run on an arbitrary subset of the training filters.  ``PHOT-GAL`` resolves this issue by using a KNN imputer to make reasonable guesses as to what the missing photometry is. It then will then predict the redshift based on this photometry and using a trained ``NGBoost`` regressor (https://github.com/stanfordmlgroup/ngboost). With the imputed photometry and the inferred redshift, there is an ``NGBoost`` model to predict stellar mass, dust mass, and SFR (as a bonus, there are models for stellar age and metallicity included in this repo). Each ``NGBoost`` output is a Gaussian distribution which we randomly sample. Each ``PHOT-GAL`` iteration will randomly sample the photometry uncertainties and then the NGBoost model posteriors. In case of runtime error, ``PHOT-GAL`` saves the output from each iteration locally, so the results are recoverable. Instead of the default settings, there is also an option to use the photo-z code ``EAZY-PY`` (https://github.com/gbrammer/eazy-py) to guide the photo-z estimation. Note: as currently written it doesn't directly take the ``EAZY-PY`` output, but rather this informs the imputation, which is then fed to the photo-z model. Also, if you use ``EAZY-PY``, the code will implicitly assume that all input galaxies have the same filters available. If you would rather estimate photo-z values yourself, ``PHOT-GAL`` does accept user-input redshifts. ``PHOT-GAL`` is trained on photometry run for galaxies 0<z<7, but we would recommend to not run for z~0 galaxies because of their sparsity in the training set.

# Installation

The current installation is pretty simple:

``git clone https://github.com/DhruvZ/Phot-Gal.git``

``cd Phot-Gal``

``pip install .``

The essential piece is the joblib file, which is a dump for the model object we use for the inference of galaxy properties. It was made with specific versions of numpy and scipy, but initial testing has had no issues running them on other versions (though admittedly this has not been rigorously tested for many different versions). Installation has been tested on a fresh Python 3.10 environment, but feel free to ask if there are any issues.

Included in the ``example_scripts`` directory is an example running script ``quick_example_test.py`` with two 'galaxies' with arbitrary photometry for only one filter that demonstrates how to construct inputs to ``PHOT-GAL``. When you call ``PHOT-GAL`` and run this test yourself, you must specify the location of the github directory on your machine. There are several runtime parameters you can specify on call in addition to the ones in the example script that you can check by. By default, ``PHOT-GAL`` will return the median and 16th and 84th percentiles of its predictions for each property, but there is an option to return the full posterior it generates. For a full list of potential runtime parameters, check the ``run_fit.py`` file.

