# ann_CalSim_EC_estimator
This code has been written in Python, adapted from an ANN model originally developed in Matlab. A comprehensive version by Nicky Sandhu is available on GitHub (https://github.com/dwr-psandhu/ann_calsim). 
This updated version is based on preprocessed input data and incorporates the normalization process directly into the model building phase utilizing the TensorFlow library. As a result, the trained model can be exported in .pb format, enabling seamless integration and utilization in other programming environments, including Java within Eclipse. In addition, this code adopts a similar approach to layer normalization as used in the delta outflow estimation required for CalSim, implemented by Peyman Namadi, (http://dwrrhapp0179.ad.water.ca.gov/gitea/peymanhn/CalSimDeltaOutFlowEstimator_CDOE)
CalSim studies are executed within WRIMS through the Eclipse IDE, utilizing version 2.7.4 of the TensorFlow library. First, establish the necessary environment by executing "conda env create -f environment.yml" in your Conda prompt. Once set up, you can execute train.ipynb using Jupyter Notebook.


