ASBA
====

ASBA refers to "adaptive semirigid body approximation", which is used to construct a physically 
reasonable initial guess of the minimum energy path structures for subsequent NEB/cNEB calculation.

Features
========

The ASBA method extracts the atomic radii from the average bond lengths from the initial and final
states and utilizes a semirigid-body force model and the NEB method to keep the majorly displaced 
atoms in close contact within the intermediate structures.

Citing
======

If you use ASBA in your research, please cite the following work:
          
          
    Hongsheng Cai, Guoyuan Liu, Peiqi Qiu, Guangfu Luo; Structural feature in dynamical processes 
    accelerated transition state calculations. J. Chem. Phys. 21 February 2023; 158 (7): 074105. 
    https://doi.org/10.1063/5.0128376


You should also include the following citations for the pymatgen core package:


    Deng, Z.; Zhu, Z.; Chu, I.-H.; Ong, S. P. Data-Driven First-Principles Methods for the Study 
    and Design of Alkali Superionic Conductors, Chem. Mater., 2016, acs.chemmater.6b02648,
    doi:10.1021/acs.chemmater.6b02648.
    
    
    Shyue Ping Ong, William Davidson Richards, Anubhav Jain, Geoffroy Hautier, Michael Kocher, 
    Shreyas Cholia, Dan Gunter, Vincent Chevrier, Kristin A. Persson, Gerbrand Ceder. *Python 
    Materials Genomics (pymatgen) : A Robust, Open-Source Python Library for Materials Analysis.* 
    Computational Materials Science, 2013, 68, 314-319. doi:10.1016/j.commatsci.2012.10.028.
    
Acknowledgements
================

This work was financially supported by the following agencies:

1. Guangdong Provincial Key Laboratory of Computational Science and Material Design (Grant No. 2019B030301001)
2. The Introduced Innovative R & D Team of Guangdong (Grant No. 2017ZT07C062)
3. The Shenzhen Science and Technology Innovation Commission (Grant No. JCYJ20200109141412308). 
