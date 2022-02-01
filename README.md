# SAEM
Semi-airborne electromagnetics (SAEM) data processing, visualisation and inversion tools

You will find all functions and classes under the saem folder.
Main purpose is to load, filter, save and post-process semi-airborne data for inversion with custEM or empymod.

It holds three main classes:
* CSEMData (will be renamed to SAEMdata)
    main class for managing SAEM data patches using a single transmitter
* Mare2DEMData
    for reading Mare2DEM data format (.emdata)
    Note that even though it is designed for 2D, the coordinates are 3D 
    and the dipole tranmitters can be overwritten by polygonal tranmitters
* CSEMSurvey
    A subclass for extended CSEM surveys with ground and airborne receivers and multiple transmitters.
    It uses the Mare2DEMData class for reading and writing Mare formats and SAEMdata for the individual patches.