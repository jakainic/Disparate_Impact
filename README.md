# Disparate_Impact

The aim of this project is to build the repair methods described in the paper [Certifying and removing disparate impact](https://arxiv.org/pdf/1412.3756.pdf) by Feldman, et. al., and to train models on the repaired data.

Ultimately, the authors used an approximate approach to the theoretical motivations described in the paper. Their code for the repairers can be found [here](https://github.com/algofairness/BlackBoxAuditing/tree/master/BlackBoxAuditing/repairers). I have built a few, somewhat similar approximations (one geometric and two combinatorial). The results I get seem to align well with the results reported in their paper. 

Ultimately, however, I think it would be preferrable to use a methodology more definitively grounded in optimal transport theory. Specifically, I would like to build a version in which the repair method maps protected class distributions to their Wasserstein barycenter. To do so, I will be drawing on work on computing barycenters, including work from Sebastian Claici, Charlier Frogner, Gabriel Peyré, and others.

In addition to the barycenter version, I would also like to explore generalizing the algorithm to more general scenarios (where there are multiple protected classes and/or the protected classes are not binary). 
