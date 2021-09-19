# Learning-From-Data-Caltech
Yaser Mostafa Caltech Machine Learning Course Homework Assignments


* Aggregation
  * Overview of ensemble learning (boosting, blending, before and after the fact)
* Bayesian Learning
  * Validity of the Bayesian approach (prior, posterior, unknown versus probabilistic)
* Bias-Variance Tradeoff
  * Basic derivation (overfit and underfit, approximation-generalization tradeoff)
  * Example (sinusoidal target function)
  * Noisy case (Bias-variance-noise decomposition)
* Bin Model
* Hoeffding Inequality (law of large numbers, sample, PAC)
* Relation to learning (from bin to hypothesis, training data)
* Multiple bins (finite hypothesis set, learning: search for green sample)
* Union Bound (uniform inequality, M factor)
* Data Snooping
* Definition and analysis (data contamination, model selection)
* Error Measures
* User-specified error function (pointwise error, CIA, supermarket)
* Gradient Descent
* Basic method (Batch GD) (first-order optimization)
* Discussion (initialization, termination, local minima, second-order methods)
* Stochastic Gradient Descent (the algorithm, SGD in action)
* Initialization - Neural Networks (random weights, perfect symmetry)
* Learning Curves
* Definition and illustration (complex models versus simple models)
* Linear Regression example (learning curves for noisy linear target)
* Learning Diagram
* Components of learning (target function, hypothesis set, learning algorithm)
* Input probability distribution (unknown distribution, bin, Hoeffding)
* Error measure (role in learning algorithm)
* Noisy targets (target distribution)
* Where the VC analysis fits (affected blocks in learning diagram)
* Learning Paradigms
* Types of learning (supervised, reinforcement, unsupervised, clustering)
* Other paradigms (review, active learning, online learning)
* Linear Classification
* The Perceptron (linearly separable data, PLA)
* Pocket algorithm (non-separable data, comparison with PLA)
* Linear Regression
* The algorithm (real-valued function, mean-squared error, pseudo-inverse)
* Generalization behavior (learning curves for linear regression)
* Logistic Regression
* The model (soft threshold, sigmoid, probability estimation)
* Cross entropy error (maximum likelihood)
* The algorithm (gradient descent)
* Netflix Competition
* Movie rating (singular value decomposition, essence of machine learning)
* Applying SGD (stochastic gradient descent, SVD factors)
* Neural Networks
* Biological inspiration (limits of inspiration)
* Multilayer perceptrons (the model and its power and limitations)
* Neural Network model (feedforward layers, soft threshold)
* Backpropagation algorithm (SGD, delta rule)
* Hidden layers (interpretation)
* Regularization (weight decay, weight elimination, early stopping)
* Nonlinear Transformation
* Basic method (linearity in the parameters, Z space)
* Illustration (non-separable data, quadratic transform)
* Generalization behavior (VC dimension of a nonlinear transform)
* Occam's Razor
* Definition and analysis (definition of complexity, why simpler is better)
* Overfitting
* The phenomenon (fitting the noise)
* A detailed experiment (Legendre polynomials, types of noise)
* Deterministic noise (target complexity, stochastic noise)
* Radial Basis Functions
* Basic RBF model (exact interpolation, nearest neighbor)
* K Centers (Lloyd's algorithm, unsupervised learning, pseudo-inverse)
* RBF network (neural networks, local versus global, EM algorithm)
* Relation to other techniques (SVM kernel, regularization)
* Regularization
* Introduction (putting the brakes, function approximation)
* Formal derivation (Legendre polynomials, soft-order constraint, augmented error)
* Weight decay (Tikhonov, smoothness, neural networks)
* Augmented error (proxy for out-of-sample error, choosing a regularizer)
* Regularization parameter (deterministic noise, stochastic noise)
* Sampling Bias
* Definition and analysis (Truman versus Dewey, matching the distributions)
* Support Vector Machines
* SVM basic model (hard margin, constrained optimization)
* The solution (KKT conditions, Lagrange, dual problem, quadratic programming)
* Soft margin (non-separable data, slack variables)
* Nonlinear transform (Z space, support vector pre-images)
* Kernel methods (generalized inner product, Mercer's condition, RBF kernel)
* Validation
* Introduction (validation versus regularization, optimistic bias)
* Model selection (data contamination, validation set versus test set)
* Cross Validation (leave-one-out, 10-fold cross validation)
* VC Dimension
* Growth function (dichotomies, Hoeffding Inequality)
* Examples (growth function for simple hypothesis sets)
* Break points (polynomial growth functions)
* Bounding the growth function (mathematical induction, polynomial bound)
* Definition of VC Dimension (shattering, distribution-free, Vapnik-Chervonenkis)
* VC Dimension of Perceptrons (number of parameters, lower and upper bounds)
* Interpreting the VC Dimension (degrees of freedom, Number of examples)
