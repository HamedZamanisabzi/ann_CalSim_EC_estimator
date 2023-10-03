# ann_CalSim_EC_estimator
This code has been written in Python, adapted from an ANN model originally developed in Matlab. A comprehensive version of this code has been developed by Nicky Sandhu is available on GitHub (https://github.com/dwr-psandhu/ann_calsim). This updated version is based on preprocessed input data and incorporates the normalization process directly into the model building phase utilizing the TensorFlow library. As a result, the trained model can be exported in .pb format, enabling seamless integration and utilization in other programming environments, including Java within Eclipse. The implementation of the normalization layer along with calculation of the prediction metrics have been adapted from required delta outflow prediction code developed by Peyman Namadi (http://dwrrhapp0179.ad.water.ca.gov/gitea/peymanhn/CalSimDeltaOutFlowEstimator_CDOE). The reference study for the original model: Jayasundara, N. C., Seneviratne, S. A., Reyes, E., & Chung, F. I. (2020). Artificial neural network for Sacramento–San Joaquin Delta flow–salinity relationship for CalSim 3.0. Journal of Water Resources Planning and Management, 146(4), 04020015. To run and test the code, first, establish the necessary environment by executing "conda env create -f environment.yml" in your Conda prompt. Once set up, you can execute train.ipynb using Jupyter Notebook.


