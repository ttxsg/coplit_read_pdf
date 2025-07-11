# Solving Graph Flow Problems with Neural Networks: A Lagrangian Duality Approach

# Abstract

The problem of computing minimum cost graph flows has important applications in the management of transportation systems. Existing approaches to solving this problem, however, have high computational complexity and cannot scale to larger graphs. This research presents an entirely new approach to solving the minimum cost flow problem when the cost function is convex. The proposed technique uses message- passing graph neural networks to learn graph flows. This work addresses the question: can graph neural networks compute minimum cost flows faster than existing techniques? The goal of learning minimum cost flows fits within the larger context of using graph neural networks to solve algorithmic problems.

Training a neural network to learn minimum cost graph flows is challenging for two reasons. Firstly, the model must output are flows which constitute a valid circulation. Secondly, ground- truth labels are difficult to compute because existing algorithms are inefficient. These problems are solved by introducing a powerful reparameterization. Instead of directly learning flow values, the neural network learns relative flow proportions. This formulation allows the neural network to always produce valid flows. This maintenance of feasibility gives rise to a novel training technique based on Lagrangian duality. This method, named DualityLoss, uses separate neural networks for the primal and dual optimization problems. The models are then jointly trained with the goal of minimizing the estimated duality gap. Given a strongly dual problem and neural networks which maintain feasibility, this objective encourages the primal and dual models to produce optimal solutions. This training technique uses properties of Lagrangian duality to remove the need for labeled datasets.

This novel approach is used to train a graph neural network on the task of minimum cost graph flows. The graph neural network features an architecture which is more compatible with high- diameter road graphs. This architecture uses ideas of graph attention, gated state updates, and neighborhood expansion. The neural network approach is evaluated against pure optimization techniques on a set of real- world road graphs. These results show that the model learns flows which are close to the optimal cost. Furthermore, the neural network computes these flows significantly faster than the best optimization algorithm. Qualitative examples show how the model learns to coordinate the behavior of vertices within the graph. These results are encouraging because the neural network learns these abilities without ever receiving ground- truth information.

Word Count: 13,702

# Contents

1 Introduction 5

# 2 Related Work 7

2.1 Graph Neural Network Architectures 7

2.2 Graph Neural Networks for Algorithmic Graph Problems 8

2.3 Dueling Neural Networks 9

2.4 Leveraging Duality in Neural Networks 10

2.5 Minimum Cost Flows 11

# 3 Notation and Problem Statement 11

3.1 Mathematical Notation 12

3.2 Graph Notation 12

3.3 The Minimum Cost Flow Problem 13

3.4 Graph Neural Networks 13

# 4 Training Neural Networks Using Lagrangian Duality 14

4.1 Motivating DualityLoss 14

4.2 Defining DualityLoss for Constrained Optimization Problems 15

4.3 DualityLoss for Minimum Cost Flows 17

4.4 Producing Valid Flows with Neural Networks 17

4.5 Theoretical Properties of the Iterative Flow Solver 19

4.5.1 Derivation 19

4.5.2 Properties 20

4.6 The Minimum Cost Flow Dual Problem 20

# 5 Graph Neural Network Design 22

5.1 Motivation and Architecture Overview 22

5.2 End- to- End Model 23

5.3 Implementation 24

5.3.1 Encoder 24

5.3.2 Graph Attention and Neighborhood Lookahead 24

5.3.3 Decoder 26

5.4 Converting Edge Scores into Flow Proportions 26

# 6 Evaluation 27

6.1 An Illustration of Computed Graph Flows 28

6.1.1 Datasets and Training 30

6.2 Comparing the Neural Network Approach to Baseline Techniques: Flow Cost 30

6.3 Comparing the Neural Network Approach to Baseline Techniques: Computation Time 32

6.4 Evaluating the Design Choices of the Graph Neural Network Architecture 33

6.5 Assessing the Impact of the Iterative Flow Solver 34

6.6 Details of the Baseline Solvers 34

6.6.1 Optimization Algorithms 34

6.6.2 Gated GAT Baseline 36

# 7 Exploring the Causes of Suboptimality 36

7.1 Adapting to Changes in Demand 37  7.2 Causes of Suboptimal Flows 38  7.3 Mitigating Flow Cycles with Pre- computed Paths 39

# 8 Future Work 40

8.1 Capacity Constraints 40

8.2 Transfer Learning 40

# 9 Conclusion 40

# 10 Proofs 41

10.1 Correctness 41  10.2 Continuity 42  10.3 Convergence 43

# 11 Appendix 56

11.1 Graphs 56

11.2 Algorithm to Generate Sources and Sinks 56

11.3 Model Hyperparameters 56

11.4 Configurations for Optimization Algorithms 57

# 1 Introduction

This project presents a graph- neural- network- based approach to the minimum cost graph flow problem.

Graph neural networks [18, 36] have been used to solve both classification and prediction problems on graphs. For example, given a social network graph where each vertex is an account and each edge is a friendship, graph neural networks can learn patterns in the social graph which identify fraudulent accounts. Graph neural networks approach such classification problems by (1) defining an initial state  $\mathbf{x}_{u}$  for each vertex  $u$  , (2) repeatedly updating these states using neighboring nodes i.e.  $\mathbf{x}_{u}\leftarrow f(\{\mathbf{x}_{w}\})$  for all neighbors  $w$  of  $u$  , and (3) predicting the final classification labels  $y_{u}\leftarrow g(\mathbf{x}_{u})$  for all vertices. The functions  $f$  and  $g$  are implemented as neural networks which are trained to maximize the classification accuracy.

Optimization problems on graphs, however, are often of a different style. Consider the famous maximum flow problem. This problem involves finding the maximum possible flow between a source  $s$  and a sink  $t$  which both forms a circulation and abides by the given edge capacities. As with many graph optimization problems, the maximum flow problem can be represented within the framework of constrained optimization. The maximum flow problem can be efficiently solved by techniques such as the Ford- Fulkerson method. Not all graph optimization problems, however, can be solved using efficient, specialized algorithms; in many cases, finding solutions requires resorting to general- purpose optimization techniques.

One variation on traditional maximum flow problem is the problem of finding minimum cost flows (MCFs). This problem involves multiple sources and multiple sinks. Each source is assigned an amount of supply, and each sink has a given demand. The goal is to find a flow from sources to sinks which both satisfies all demands and minimizes a cost function. This task has practical applications in both transportation systems and vehicle routing [7]. Despite its close relation to the maximum flow task, the minimum cost flow problem is inefficient to solve when the cost function is nonlinear [20]. This inefficiency even extends to cases where the cost is convex [7, Chapter 9]. As convex cost functions appear in real- world scenarios such as route selection [45], the inefficiencies of existing solvers make them unsuitable for real- time flow computations in practical settings.

This thesis presents a new approach to the MCF problem. This approach uses graph neural networks to learn MCFs when the cost function is nonlinear and convex. Unlike traditional algorithms, neural networks offload a significant amount of computation to the training phase. The information gained during training allows the neural network to efficiently generalize to new problem instances. Furthermore, hardware components, such as GPUs, improve the wall- to- wall runtime of neural network models. For these reasons, graph neural networks can theoretically compute minimum cost flows in near- real- time.

The task of learning MCFs with graph neural networks is difficult for two reasons. Firstly, the desired graph neural network must produce solutions which satisfy the problem's constraints. To ensure the graph neural network outputs valid flows, the neural network learns relative flow proportions instead of actual flow values. These proportions form probability distributions over the outgoing edges of each vertex. The flow proportions are then used to construct a valid flow. As neural networks can easily output distributions by using operations such as softmax, this method enables graph neural networks to always produce valid flows.

![](images/9c2f0cc8b0a3381d3341e956d4e3656cdca51366cc99cc405a4fdbffaff37b4e.jpg)  
Figure 1: High-level architecture of the DualityLoss technique. The variables  $\hat{p}$  and  $\hat{q}$  are the predictions made by the primal and dual models respectively.

The second challenge is that neural networks require large amounts of labeled training data. For algorithmic tasks, this data is often generated by applying an existing algorithm to randomly generated problem instances. For example, to train a neural network for the task of shortest paths, one can create data labels using Dijkstra's shortest path algorithm. This strategy, however, is not feasible when existing solvers have high computational complexities. To address this problem, a novel training method for constrained optimization problems is introduced. This technique, called DualityLoss, uses neural network models for both the primal and the dual optimization problems. These models are then trained with the joint objective of minimizing the difference between the estimated primal and dual costs. Figure 1 depicts this technique. This objective encourages the models to find the optimal solution as long as both models produce feasible solutions and the duality gap is zero. The DualityLoss method is motivated by the fact that, for problems with no duality gap, optimal solutions occur when primal and dual values coincide. DualityLoss removes the need for labeled data, thus allowing the neural network to be trained on large problem instances for which exact solutions are not feasible to compute.

The proposed graph neural network approach is evaluated on a set of real- world road graphs obtained from Open Street Map [32]. The graph neural network uses an architecture based on ideas from graph attention [42], gated state updates [25], and neighborhood expansion [2]. This design addresses some of the problems related to long- range information retention which are induced by high- diameter road graphs. The graph neural network learns flows which are close to the optimal cost, and it computes these flows far faster than the best pure optimization technique. The graph neural network displays the ability to coordinate the behavior of multiple vertices, and this result is encouraging because the neural network learns this strategy without ever receiving ground- truth labels.

This application of graph neural networks to the MCF problem fits within the larger context of using graph neural networks for algorithmic tasks. While graph neural networks have been applied to problems such as boolean satisfiability [37] and path- finding [25], this thesis presents the first graph neural network approach to solving flow problems.

This report proceeds as follows. Section 2 puts forth a selection of related work. Section 3 presents the notation, formal problem statement, and relevant background information.

Section 4 outlines the DualityLoss technique for training neural networks. Section 5 discusses the graph neural network design. Section 6 presents an evaluation of this technique on real- world road graphs, and these results are reflected upon in section 7. Section 8 discusses ideas for future work, and section 9 concludes the report. Section 10 contains a set of proofs for theorems introduced in earlier sections.

# 2 Related Work

The neural network method developed in this thesis uses ideas from graph neural networks, reinforcement learning, adversarial networks, and dual optimization problems. This section places the ideas present in this thesis within the context of related work in these areas.

# 2.1 Graph Neural Network Architectures

Graph neural networks are an extension of recursive neural networks which operate on graph- structured data. Gori et. al. [15] and Scarselli et al. [36] proposed the original graph neural network model, GNN. GNN uses a message- passing framework; at each step, every node updates its state by aggregating the states of its neighbors. An output function maps node states to output values. The output and transition functions are implemented as feedforward neural networks. Training these neural networks using standard backpropagation, however, is memory- inefficient because it requires storing node states for each step of the unrolled network. To address this problem, GNN executes transition steps until the node states reach a fixed point. Given this fixed point, the neural networks can use the Almeida- Pineda algorithm [3, 33] to perform backpropagation without storing the entire history of states. To ensure that the fixed point iteration converges, GNN requires the transition function to be a contraction mapping.

The restriction of the transition model to a contraction mapping limits GNN's ability to retain information across long distances. The gated graph neural network (GG- NN) design addresses these problems [25]. GG- NNs remove the contraction mapping restriction and instead run a fixed number of transition steps. Gradients are then computed using backpropagation through time. This decision allows for a less constricted model at the cost of more memory. GG- NNs use gated recurrent units (GRUs) [11] to update node states during each transition step. The application of GRUs allows the model to better retain information over long distances. This quality is important when applying graph neural networks to high- diameter graphs. For this reason, the model presented in this work makes use of gated node updates.

While recurrent units help address the issue of long- range information retention, they do not completely solve the problem. N- GCN approaches this problem from a different perspective [2]. This model builds on Kipf and Welling's graph convolutional network (GCN) [23]. GCN extends the traditional convolution operation to graph- structured data by using the normalized graph Laplacian. N- GCN augments GCN by applying multiple graph convolution operators to aggregate information from different neighborhoods. For each vertex, these neighborhood levels are formed from nodes located at increasing random walk dis

![](images/d8465f26bfaae5eb57b2334d42485b525e394096984f76e2329e0a03b01691ac.jpg)  
Figure 2: A depiction of the GGS-NN architecture. This diagram is based on figure 2 from Li et al. [25]. The variables  $\mathbf{X}^{(k)}$  and  $\mathbf{Y}^{(k)}$  represent the node states and outputs at sequence step  $k$ . The functions  $F_{x}^{(k)}$  and  $F_{o}^{(k)}$  denote the transition and output GG-NNs at step  $k$ . Each GG-NN executes  $T$  message-passing steps.

tances. This strategy allows N- GCN to collapse multiple GCN transitions into a single step. The inclusion of many neighborhoods reduces the number of message- passing steps required to cover an entire graph, and the model shows the ability to recognize long- range relationships. This neighborhood expansion approach shortens paths within the target graph, and the architecture presented in this thesis leverages this strategy to mitigate problems related to the long paths often seen in high- diameter road graphs.

The models above consider all neighboring states equally when aggregating information. This assumption of equality may not be optimal, and nodes should ideally focus on neighbors which are more relevant to producing the correct output. This idea is at the core of graph attention networks (GATs) [42]. At each transition step, nodes aggregate neighboring states using a weighted average. These weights are formed by applying a softmax operator to the output of a feed- forward neural network. The GAT model represents an extension of sequence attention models [6] to graph- structured domains. The concept of graph attention is compatible with the task of computing graph flows. Flow values can be interpreted as the amount of emphasis placed on a given edge. This interpretation lends itself to the attention model, as attention enables nodes to bias towards neighbors which lead to more optimal solutions. Based on the intuition relating graph attention and graph flows, the architecture in this thesis adopts an attention- based aggregation function.

# 2.2 Graph Neural Networks for Algorithmic Graph Problems

Graph neural networks have seldom been applied to graph- theoretic problems of high computational complexity. NeuroSAT uses a graph neural network to solve the boolean satisfiability (SAT) problem, and it is the most successful graph neural network approach to an NP- Complete problem [37]. This model is trained as a classifier on a randomly generated set of SAT problems with between 10 and 40 variables and roughly 200 causes. As these samples are small, existing SAT solvers can efficiently generate data labels. NeuroSAT generalizes to larger SAT problems by executing more message- passing steps at testing time. Despite being a classifier, the model encodes variable assignments in within intermediate node representations.

Prates, Avelar, and Lemos et al. build on this work by developing a graph neural net

work technique for solving the traveling salesman decision problem (TSP) [34]. The authors train this model on graphs which contain between 20 and 40 nodes. These graphs are small enough to use existing algorithms to compute optimal costs. From each training graph, the authors generate positive and negative samples. These samples have respective costs which are higher and lower than the optimal value. The margin used to create these deviations allows the model to learn the optimal TSP cost within a certain error bound.

Gated graph sequence neural networks (GGS- NNs) are used to address the problem of path- finding [25]. GGS- NN is a graph neural network architecture capable of predicting sequences. The model uses two gated graph neural networks: one to create node states and one to produce outputs. Figure 2 shows an example of this architecture. The authors apply GGS- NNs to the task of general path- finding, shortest paths, and Eulerian circuits. GGS- NN solves these problems by producing sequential outputs until it predicts a special "stop" state. For all tasks, GGS- NN uses training samples with data labels denoting the optimal sequence. Even with limited training examples, GGS- NN achieves 100% accuracy for both shortest paths and Eulerian circuits on small problem instances.

The goal of learning MCFs relates to this previous work in that it also involves using neural networks to solve algorithmic graph problems. The technique presented in this thesis, however, uses a different approach to training. In the related work above, graph neural networks are trained using coarse- grained labels. For tasks such as shortest paths, this strategy is possible due to existing efficient algorithms. For problems where no efficient algorithms exist, training sets are constructed from small instances. In contrast to these approaches, the DualityLoss technique uses no data labels. Instead, it leverages the properties of the primal and dual optimization problems. DualityLoss is useful for the MCF problem because (1) MCFs are inefficient to solve exactly and (2) neural networks can always produce feasible solutions by learning flow proportions. The requirement of feasibility is difficult to satisfy for problems such as TSP.

# 2.3 Dueling Neural Networks

The term dueling neural networks refers to architectures which train a pair of neural network models against one another. Generate Adversarial Networks (GANs) fall under this class of models [14]. GANs are composed of two neural network models: a generator and a discriminator. The generator model uses seed values to generate outputs. The discriminator model evaluates these outputs and assesses whether they belong to the true dataset. The two models engage in a minimax game in which the generator tries to create realistic samples which fool the discriminator. The generator and discriminator are measured against each other, and training occurs by taking turns updating the generator and the discriminator.

AlphaGo Zero is another example of a dueling neural network architecture [40]. AlphaGo Zero is a deep reinforcement learning system which has achieved world- record performance on the games of Go, Shogi, and Chess. The system couples a deep neural network with a Monte- Carlo tree search algorithm to explore game states. Rather than learning by playing against humans, AlphaGo Zero learns by playing against itself. The win/loss outcome of each game decides the reward propagated to each model instance. This technique of self- play has allowed AlphaGo Zero to adopt strategies which fall outside of the conventional human theory surrounding these games.

The DualityLoss technique relates to these ideas in that it also trains a pair of neural networks based on one another. DualityLoss, however, solves a different type of problem than the ones considered by both GANs and AlphaGo Zero. Namely, DualityLoss is a technique to train neural networks for constrained optimization problems. Furthermore, dueling architectures involve competition between opposing neural networks. The primal and dual models of DualityLoss, however, have the collaborative goal of minimizing the estimated duality gap. This goal comes from directly from Lagrangian duality theory.

# 2.4 Leveraging Duality in Neural Networks

To the best of our knowledge, the DualityLoss technique is a novel contribution of this work. The concept of duality, however, has been used elsewhere to aid learning algorithms.

Shalev- Schwarz and Singer use primal and dual optimization problems to train an online linear classifier [38]. Online classifiers are evaluated by the number of classification mistakes made over time. These mistake values are often relative to the optimal prediction algorithm which has the benefit of observing the entire sequence of examples. Shalev- Schwarz and Singer formulate an online linear classifier as a constrained optimization problem subject to a given relative mistake bound. Rather than solving the primal problem, the authors instead use the Lagrangian dual formulation. This substitution makes sense because the optimal dual and primal values coincide due to strong duality. The authors prove that the dual optimization is non- decreasing, and, if the average increase is sufficiently large, then the dual optimization meets the desired relative mistake bound. The authors later apply this same technique to the more general problem of repeated convex games by instead using Fenchel duality [39].

The duality technique for online linear classification leverages the strong duality of the linear classification optimization problem. This idea is similar to the use of strong duality by the DualityLoss technique. DualityLoss, however, directly compares estimated primal and dual solutions. This difference is the sole source of feedback used to train the pair of neural networks. In contrast, the approach introduced by Shalev- Schwarz and Singer uses labeled examples collected at previous timesteps. Furthermore, Shalev- Schwarz and Singer consider online linear classifiers. DualityLoss is instead a technique for training neural networks in offline settings.

Dual optimization problems have also been used to address intractable problems within the context of GANs. GANs often suffer from a phenomenon called mode collapse. Mode collapse occurs when the generator model stops producing certain classes of outputs. Wasserstein Generative Adversarial Networks (WGANs) are an attempt to address this problem [5]. WGANs use the observation that probability distributions are likely to have non- intersecting support when this support is low- dimensional [4]. This quality is problematic when comparing two distributions using KL divergence because the KL divergence becomes either infinite or zero. WGANs substitute the KL divergence with an Earth- Mover's distance function. Using Earth- Mover's distance, however, makes the target optimization intractable. To circumvent this issue, the authors use the technique of Kantorovich- Rubinstein duality to instead optimize the computationally feasible dual problem.

WGANs and DualityLoss are similar in that they use duality to address issues surrounding computational complexity. The context of this complexity, however, differs between the

two approaches. In the case of WGANs, the minimax objective function is itself intractable. WGANs use duality to solve an easier problem which shares the same solution. On the other hand, DualityLoss is motivated by the inability to compute data labels. DualityLoss removes the need for such labels by using the estimated duality gap as a measure for solution quality. Unlike WGANs, DualityLoss does not use the dual problem as a substitute for the primal.

# 2.5 Minimum Cost Flows

The MCF problem and its variants form a class of well- studied problems. The classical formulation consists of a graph equipped with source production values, sink demands, edge capacities, and a linear cost function. The goal is to minimize the flow cost subject to satisfying all demands and abiding by all edge capacities. There are many well- known algorithms, such a cycle canceling [13] and relaxation [8], which solve this problem. Furthermore, simplex methods from linear programming solve the linear MCF variant [12].

The MCF formulation studied by this work involves uncapacitated edges, as well as a nonlinear, convex cost function. Bertsekas presents both gradient- based approaches and auction algorithms to solve the convex minimum- cost flow problem [7]. The auction algorithm derived by Bertsekas is weakly polynomial and has a lower bound runtime of  $\Omega (N^3)$ . This technique is among the most efficient algorithms for the convex MCF problem. This cubic runtime, however, prohibits the auction algorithm from scaling to even larger graphs. Végh [41] describes a strongly polynomial algorithm with runtime of  $O(|E|^4 \log |E|)$  for cases when the cost function is quadratic. For cost functions of polynomial degree at least three, Hochbaum has shown that no strongly polynomial approximation algorithm exists [20].

The MCF problem is also frequently studied within the context of concave cost functions. This variant is more challenging its convex counterpart, and it is proven to be NP- Hard by Guisewite and Padalos [16]. Montiero et al. [30] use an ant- colony optimization algorithm to approximate uncapacitated minimum cost flows with concave cost functions. Their approach uses a heuristic local search algorithm to improve intermediate solutions at every optimization step. This ant- colony method extends to handle edge- specific fixed costs.

The graph neural network technique presented in this work represents a distinct approach to the MCF problem. Instead of relying on hand- designed heuristics, this neural network learns flows by training on problem instances. This technique has the advantage of offloading a significant amount of computation to the pre- processing phase of training. At runtime, the trained neural network requires only a single pass through the neural network to compute a solution. Hardware components, such as GPUs, further increase the empirical efficiency of the neural network approach.

# 3 Notation and Problem Statement

As is the case with many algorithmic problems on graphs, the MCF problem can be concisely stated as a constrained optimization problem. This section presents this problem after defining the notation which will appear in the remainder of this thesis.

# 3.1 Mathematical Notation

Matrices  $\mathbf{B}\in \mathbb{R}^{m\times n}$  and vectors  $\mathbf{x}\in \mathbb{R}^{n}$  are denoted by uppercase and lowercase boldface characters respectively. Scalar values are represented by non- boldface characters. This notation extends to specific elements of both matrices and vectors; the element of  $\mathbf{B}$  in row  $i$  and column  $j$  is represented by  $B_{ij}$  , and the  $i^{t h}$  element of  $\mathbf{x}$  is denoted by  $x_{i}$  . Matrices and vectors are both 1- indexed, meaning that  $B_{11}$  is the upper- left value of matrix  $\mathbf{B}$  and  $x_{1}$  is the first element of vector  $\mathbf{x}$  . The symbol  $[n]$  is defined as  $[n] = \{1,\ldots ,n\}$  , and  $\mathbb{R}_{\geq 0}$  is the set  $[0,\infty)$  . The  $i^{t h}$  row of the matrix  $\mathbf{B}$  is represented by the symbol  $\mathbf{B}_{i}$ $\in \mathbb{R}^{n}$  . Similarly, the  $j^{t h}$  column is written as  $\mathbf{B}_{:j}\in \mathbb{R}^{m}$  . Furthermore, the vectors  $\mathbf{1}_{n}\in \mathbb{R}^{n}$  and  $\mathbf{0}_{n}\in \mathbb{R}^{n}$  are defined as the vectors of all ones and zeros respectively. The matrices  $\mathbf{1}_{n\times m}\in \mathbb{R}^{n\times m}$  and  $\mathbf{0}_{n\times m}\in \mathbb{R}^{n\times m}$  are defined similarly. Finally, the operation  $||:\mathbb{R}^{n}\times \mathbb{R}^{m}\to \mathbb{R}^{n + m}$  represents vector concatenation.

The set  $S = \{s_{i} | i\in [K]\}$  is defined to be a simplex if both  $s_{i}\geq 0$  for all  $s_{i}\in S$  and  $\textstyle \sum_{i = 1}^{K}s_{i} = 1$  . The vector  $\mathbf{v}$  is a simplex if its set of elements constitutes a simplex. The matrix  $\mathbf{B}\in \mathbb{R}^{n\times n}$  is a Markov Chain if every column of  $\mathbf{B}$  is a simplex. A Markov chain  $\mathbf{B}$  is irreducible if, for all  $i,j\in [n]$  , there exists some integer  $m_{ij} > 0$  such that  $B_{ij}^{m_{ij}} > 0$  where  $\mathbf{B}^{m_{ij}}$  is the  $m_{ij}^{t h}$  power of  $\mathbf{B}$

# 3.2 Graph Notation

A graph  $G = (V,E)$  is a collection of vertices and edges. The number of vertices  $|V|$  and number of edges  $|E|$  are denoted by  $n$  and  $m$  respectively. An edge  $e\in E$  is an ordered pair of vertices  $(u,v)$  for  $u,v\in V$  . For compatibility with indexing into vector and matrix variables, each distinct vertex  $u\in V$  is represented by a number in  $\lceil n\rceil$  . The relative ordering of such numerical labels is unimportant, and no specific vertex ordering will ever be assumed. The notation  $u\in V$  and  $u\in [n]$  will be used interchangeably.

The graph  $G$  's adjacency matrix  $\mathbf{A}\in \mathbb{R}^{n\times n}$  is defined to have the following elements for all  $i,j\in [n]$

$$
A_{ij} = \left\{ \begin{array}{ll}1 & (i,j)\in E\\ 0 & (i,j)\notin E \end{array} \right.
$$

The sets  $S_{out}^{(u)}$  and  $S_{in}^{(u)}$  , as shown below, represent the outgoing and incoming neighbors of the node  $u\in [n]$

$$
\begin{array}{r l} & {S_{o u t}^{(u)} = \{v\in [n] | (u,v)\in E\}}\\ & {S_{i n}^{(u)} = \{w\in [n] | (w,u)\in E\}} \end{array}
$$

A simple path  $\gamma \subseteq V$  of length  $K$  is a finite, ordered set of distinct vertices  $\gamma =$ $\{u_{0},u_{1},\ldots u_{K - 1}\}$  such that  $(u_{i - 1},u_{i})\in E$  for all  $i\in [K - 1]$  . A simple cycle is equivalent to a simple path except that its start and end points are equal. The distance between any two nodes  $u,v\in V$  , denoted by  $d(u,v)$  , is the length of the shortest unweighted path between  $u$  and  $\upsilon$  . The diameter of the graph  $G$  is defined as  $\mathrm{diam}(G) = \max_{u,v\in V[n]}d(u,v)$

Finally, graphs are often interchangeably referred to as "networks." Due to the overlap in parlance with neural networks, graphs will never be referred to as "networks" in this report. The word "network" will be used in the context of neural networks.

# 3.3 The Minimum Cost Flow Problem

3.3 The Minimum Cost Flow ProblemThis thesis considers the single commodity, uncapacitated MCF problem with a separable cost function [7]. Formally, let the graph  $G = (V, E)$  be a directed graph satisfying the assumption below.

Assumption 1. The graph  $G$  is strongly connected. That is, for every pair of non- equal vertices  $u, v \in V$ , there is a simple path  $\gamma$  which begins at  $u$  and ends at  $v$ .

Let  $V_{O} \subseteq V$  and  $V_{D} \subseteq V$  be the non- intersecting sets of sources and sinks respectively. The vector  $\mathbf{d} \in \mathbb{R}^{n}$  holds the demand values for every node. By convention,  $d_{i} < 0$  for  $i \in V_{O}$  and  $d_{i} > 0$  for  $i \in V_{D}$ . The demand vector is assumed to meet the following condition.

Assumption 2. The demand vector  $\mathbf{d}$  must sum to zero, i.e.  $\sum_{i = 1}^{n} d_{i} = 0$ .

The function  $c_{ij}: \mathbb{R} \to \mathbb{R}$  represents the cost associated with the flow on edge  $(i, j) \in E$ . For simplicity, the cost function is assumed to be equivalent for all edges, and the global cost function is represented by  $c: \mathbb{R} \to \mathbb{R}$ . The cost function is assumed to have the following properties.

Assumption 3. The cost function  $c: \mathbb{R} \to \mathbb{R}$  is both differentiable and convex on the domain  $\mathbb{R}_{\geq 0}$ . Furthermore,  $c(0) = 0$  and  $c(x) \geq 0$  for all  $x \in \mathbb{R}_{\geq 0}$ .

Finally,  $\mathbf{X} \in \mathbb{R}^{n \times n}$  denotes the flow on each edge  $(i, j)$  in an adjacency- matrix- style. The flow on non- existent edges is defined to be zero. With these quantities, the MCF problem can be concisely stated as the constrained optimization problem below.

$$
\begin{array}{l l}{{\mathrm{min}}}&{{\sum_{i=1}^{n}\sum_{j=1}^{n}c(X_{i j})}}\\ {{\mathrm{over}}}&{{\mathbf{X}\in\mathbb{R}^{n\times n}}}\\ {{\mathrm{subject~to}}}&{{\sum_{k=1}^{n}X_{k i}-\sum_{j=1}^{n}X_{i j}=d_{i}\quad\forall i\in V}}\\ {{}}&{{X_{i j}\geq0\quad\forall(i,j)\in E}}\\ {{}}&{{X_{i j}=0\quad\forall(i,j)\notin E}}\end{array} \tag{1d}
$$

# 3.4 Graph Neural Networks

Message- passing graph neural networks are trainable models which operate on graph structured data [18, 23, 36]. Let  $\mathbf{h}_{i}^{(0)}$  be the input features for node  $i \in [n]$ , and let  $\mathbf{h}_{i}^{(t)}$  be the state of node  $i$  after the  $t^{th}$  message- passing step. The state of node  $i$  after the  $(t + 1)^{st}$  step is computed using the propagation rule below. These operations are executed for all nodes  $i \in [n]$ .

$$
\begin{array}{r l} & {\mathbf{u}_{i}^{(t + 1)} = \mathrm{AGG}_{t + 1}(\mathbf{h}_{i}^{(t)},\{\mathbf{h}_{j}^{(t)}\mid j\in \mathcal{N}(i)\})}\\ & {\mathbf{h}_{i}^{(t + 1)} = \mathrm{UPDATE}_{t + 1}(\mathbf{h}_{i}^{(t)},\mathbf{u}_{i}^{(t + 1)})} \end{array}
$$

The symbol  $\mathcal{N}(i)$  represents any neighborhood of node  $i$ . The functions  $\mathrm{AGG}_{t + 1}$  and  $\mathrm{UPDATE}_{t + 1}$  may contain trainable parameters which are dependent on the current message passing step. The transformations below show a concrete example of this framework. The variables  $\mathbf{W}^{(t + 1)}$  and  $\mathbf{b}^{(t + 1)}$  are trainable parameters.

$$
\begin{array}{l}{{\mathbf{u}_{i}^{(t+1)}=\sum_{j\in S_{o u t}^{(i)}}\mathbf{h}_{j}^{(t)}}}\\ {{\mathbf{h}_{i}^{(t+1)}=\tanh \left(\mathbf{W}^{(t+1)}(\mathbf{h}_{i}^{(t)}||\mathbf{u}_{i}^{(t+1)})+\mathbf{b}^{(t+1)}\right)}}\end{array}
$$

This basic model can be extended to handle features such as multiple edge types, and this framework encompasses many graph neural network architectures [18, 23, 25, 42].

# 4 Training Neural Networks Using Lagrangian Duality

The goal of this work is to create a graph neural network model which learns minimum cost graph flows. This problem is difficult because (1) it is not feasible to generate data labels using existing solvers and (2) the neural network must output valid flows. To address these problems, a novel technique to train neural networks for constrained optimization tasks is developed. From a high- level, this technique, called DualityLoss, uses neural networks to represent the primal and dual optimization problem. The networks then have the joint goal of minimizing the estimated duality gap. This section discusses the motivation and details of this approach.

# 4.1 Motivating DualityLoss

DualityLoss is based on the idea of the duality gaps in constrained optimization. Consider the concrete example of the maximum flow problem. The maximum flow problem aims to find the largest arc flow from a source vertex  $s$  to a sink node  $t$  which abides by the given edge capacities. The dual to the maximum flow problem is that of finding the minimum  $s - t$  graph cut. The maximum flow problem satisfies strong duality. This means that the optimal flow value is equivalent to the weight of the minimum  $s - t$  graph cut. This relationship gives rise to a property useful for verifying solutions. Let  $d$  be the value of some valid  $s - t$  cut and  $p$  be the value of a valid  $s - t$  flow. If  $p = d$ , then strong duality implies that both points are optimal; if the values are unequal, then either the primal or the dual is not optimal. Furthermore, if the difference between  $p$  and  $d$  is large, then one of these points must be far away from the optimum. Intuitively, finding the maximum flow value  $p$  is equivalent to finding the valid flow value  $p$  and valid cut weight  $d$  which minimize  $|p - d|$ .

This relationship between primal and dual problems extends to general constrained optimization problems which exhibit strong duality. Figure (3) visually depicts this idea for the primal problem  $P$  with the variable  $x$  and corresponding dual problem  $D$  with the variable  $\lambda$ . DualityLoss uses neural networks to model the primal and dual problems, and it introduces the goal of minimizing the estimated duality gap. Let  $\mathbf{z}$  be some vector holding features of the target problem. DualityLoss uses two neural networks: the first is a model for the primal problem and the second represents the dual. The primal and dual neural networks are

![](images/3f3ba5b2b1e169549ca0069781fd91d7e227e4870a072cae65494674139f4bb7.jpg)  
Figure 3: For convex optimization problems with no duality gap, the difference in cost between feasible primal and dual values is a proxy for the distance to the optimum. The function  $P$  represents the primal problem, and the function  $D$  represents its dual.

![](images/8363be0e929d222f8badfe15d5ef01ca36a8506d23c1708604ad94975bfe16c7.jpg)  
Figure 4: The primal and dual neural networks,  $g_{\theta}$  and  $h_{\phi}$ , are trained to minimize the estimated duality gap. As shown by the invalid pair  $(g_{\theta_2}(\mathbf{z}), h_{\theta_2}(\mathbf{z}))$ , the difference in cost is only meaningful if points are feasible. The function  $P$  represents the primal cost, and  $D$  is the corresponding dual function.

denoted by  $g_{\theta}$  and  $h_{\phi}$  respectively. The goal of these networks is to use the problem data  $\mathbf{z}$  to predict the optimal primal and dual values. As described within the context of maximum flow, this goal is equivalent to finding model parameters  $\theta$  and  $\phi$  such that  $g_{\theta}(\mathbf{z}) = h_{\phi}(\mathbf{z})$ . Thus, minimizing the absolute distance between these two models will result in both the primal and dual models predicting optimal values.

# 4.2 Defining DualityLoss for Constrained Optimization Problems

The intuition behind the DualityLoss method leads to a loss function which is used to train the primal and dual neural networks. Consider the optimization problem below as an example, and assume that the problem is feasible and bounded for all  $a \in \mathbb{R}$ .

$$
\begin{array}{ll}\min & P(x) \\ \mathrm{over} x \in \mathbb{R} \\ \mathrm{subject} \mathrm{to} & f_1(x) \leq a \end{array}
$$

Let  $x_{a}^{*}$  be the optimal value of the problem for the value  $a\in \mathbb{R}$  , and let  $\mathbf{z}_{a}\in \mathbb{R}^{r}$  be some set of features corresponding to the problem instance defined by  $a$  . The goal of using a neural network  $g_{\theta}:\mathbb{R}^{r}\rightarrow \mathbb{R}$  to solve class of constrained optimization problems involves finding the parameters  $\theta$  which allow  $g_{\theta}$  to map  $\mathbf{z}_{a}$  to  $x_{a}^{*}$  . Obtaining such a model is highly useful because it can solve the optimization problem for any  $a$  using only a forward pass through the neural network.

The DualityLoss method trains this neural network  $g_{\theta}$  by introducing a second neural network,  $h_{\theta}$  , which estimates the optimization problem's dual. The dual function  $D:\mathbb{R}_{\geq 0}\rightarrow$ $\mathbb{R}$  for the example problem is written below.

$$
D(\lambda) = \min_{x\in \mathbb{R}}(P(x) + \lambda (f_{1}(x) - a))
$$

In the context of the example optimization problem, DualityLoss minimizes the following loss function for the training dataset of  $N$  samples,  $\mathcal{X} = \{\mathbf{z}_{a_{i}} | a_{i} \in \mathbb{R}\}_{i = 1}^{N}$ .

$$
L(\pmb {\theta}, \phi) = \frac{1}{N} \sum_{i = 1}^{N} (P(g_{\theta}(\mathbf{z}_{a_{i}})) - D(h_{\phi}(\mathbf{z}_{a_{i}})))
$$

This loss function is non- negative and has a minimum of zero assuming the following two conditions hold.

1. The primal and dual models always produce feasible solutions.

2. The optimization problem has a duality gap of zero for every problem instance (i.e. the duality gap is zero for all  $a \in \mathbb{R}$ ).

The non- negativity of  $L$  comes from the theorem of weak duality. Namely, for any feasible primal point  $x$  and valid dual point  $\lambda$ , the inequality  $D(\lambda) \leq P(x)$  holds. The minimum of zero is a result of strong duality.

To see why the primal and dual models must remain feasible, consider figure (4). For feasible primal and dual values, the duality gap is a proxy for how close the values are to the optimum. This interpretation is possible due to the properties of duality. For values outside the feasible region, however, these properties no longer apply, and the difference in "primal" and "dual" cost has no meaning. This idea is shown by  $g_{\theta_{2}}(\mathbf{z})$  and  $h_{\theta_{2}}(\mathbf{z})$  in figure (4). Furthermore, the feasible regions of primal and dual problems ensure that the loss function has a unique minimum of zero; if arbitrary points are allowed, the loss function can become unbounded and the models will fail to learn anything meaningful.

The condition of strong duality is necessary so that the loss for every training sample is relative to the same value, namely zero. To clarify this idea, consider if the problem above has a nonzero duality gap. For non- equal constants  $a$  and  $a'$ , the duality gap corresponding to the problem with  $a$  is not necessarily equal to the gap corresponding to the value  $a'$ . If these problems have different duality gaps, the two instances will have different optimal loss values. This quality would imply that, during training, higher losses do not imply worse model performance. This idea is incompatible with the optimization of neural networks using stochastic gradient descent.

By defining the loss as the difference between the outputs of trainable models, DualityLoss requires no labeled data samples. The technique instead relies heavily upon the properties of strong duality which state that an optimum will occur when the primal and dual models produce feasible values which are equal.

# 4.3 DualityLoss for Minimum Cost Flows

DualityLoss is useful for the MCF problem because existing algorithms for this problem are inefficient; thus, it is not possible to create a sufficiently large dataset of labeled samples to train a neural network. DualityLoss, however, can train a neural network for the MCF problem without requiring labeled data.

To cast the MCF problem within the context of DualityLoss, let  $G$  be a stronglyconnected, directed graph,  $c:\mathbb{R}\to \mathbb{R}$  be the cost function and,  $\mathbf{d}_k\in \mathbb{R}^n$  be the demands for the  $k^{t h}$  training sample for  $k\in [N]$  . Furthermore, let  $g_{\pmb{\theta}}:(G,\mathbf{d})\rightarrow \mathbb{R}^{n\times n}$  and  $h_{\phi}:(G,\mathbf{d})\rightarrow \mathbb{R}^{n}$  be the primal and dual models respectively. The vectors  $\pmb{\theta}$  and  $\phi$  denote the model parameters. The loss function below defines the DualityLoss training objective for these two models. The function  $H:\mathbb{R}^{n}\to \mathbb{R}$  is the dual function, and section 4.6 discusses the dual problem in detail.

$$
L(\pmb {\theta},\phi) = \frac{1}{N}\sum_{k = 1}^{n}\left(\left(\sum_{i = 1}^{n}\sum_{j = 1}^{n}c(g_{\pmb{\theta}}(G,\mathbf{d}_k))_{ij}\right) - H(h_{\phi}(G,\mathbf{d}_k))\right) \tag{3}
$$

For this loss function to be well- defined, the MCF problem must obey strong duality and the primal and dual models must maintain feasibility. As stated by the proposition below, the MCF problem displays strong duality if the cost function is convex (assumption 3). Maintaining feasibility is the topic of the next section.

Proposition 1. For any graph  $G = (V,E)$  , convex cost function c, and valid demand vector  $\mathbf{d}\in \mathbb{R}^{n}$  , the MCF problem formed from these components has a duality gap of zero.

Proof. By proposition 9.4 of Bertsekas's Network Optimization, the convex MCF problem has a duality gap of zero if there exists a feasible solution to the primal problem [7, Chapter 9]. The existence of such a solution is the direct consequence of theorems 1 and 2 stated in section 4.5.2.

# 4.4 Producing Valid Flows with Neural Networks

DualityLoss only works with the MCF problem if the primal MCF neural network always produces valid flows. The requirement is difficult to meet because the flow constraint is complex; there is no straightforward function which can transform the output of a neural network into a valid flow. There do exist functions, however, that constrain the output of neural networks to a probability distribution. Of these functions, softmax is the most well- known. The ability for neural networks to easily output probability distributions leads to a reparameterization of the MCF problem which enables a neural network to always produce valid flows.

![](images/686d06e39e4fb8ff919222326d058b215b161d36065df435c8d8565c23ac29a1.jpg)  
Figure 5: Example problem which displays how flow proportions (left) correspond to actual flow values (right). Blue nodes denote sources and red nodes denote sinks.

Instead of directly predicting flow values, the primal model instead outputs relative flow proportions. The proportion on each edge  $(i,j)$  represents the fraction of inflow to node  $i$  which should exit on edge  $(i,j)$ . These proportions sum to one across the outgoing edges of each node, and this feature enforces the conservation of flow at every node. Unlike the flow constraint in the MCF problem, the simplex constraint is easily enforced by softmax, so the primal neural network can easily output valid flow proportions.

Flow proportions can be transformed into actual flow values through a transformation which repeatedly pushes flow according until a fixed point is reached. Formally, let  $\mathbf{P} \in \mathbb{R}^{n \times n}$  be a matrix flow proportions produced by the primal model  $g_{\theta}$ . The element  $P_{ij}$  is the amount of inflow to node  $i$  which should exit on edge  $(i,j)$ . Based on the flow constraint in equation (1c), the function  $f: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n}$  below defines a transformation which has a fixed point corresponding to a valid flow. This resulting flow follows the proportions of  $\mathbf{P}$ .

$$
f(\mathbf{X})_{ij} = \max \left(P_{ij} \left(\sum_{k = 1}^{n} X_{ki} - d_i\right), 0\right) \quad \forall i, j \in [n] \tag{4}
$$

This iterative application of  $f$  is fully described in algorithm 1, and figure 5 displays an example problem. The fixed point iteration of  $f$  provably converges as long as the matrix  $\mathbf{P}$  satisfies the conditions of the following remark. This discussion of convergence is left for section 4.5.

Remark 1. A flow proportions matrix  $\mathbf{P}$  satisfies the two conditions below.

1.  $\sum_{j = 1}^{n} P_{ij} = 1 \quad \forall i \in [n]$

2.  $P_{ij} > 0$  if  $(i,j) \in E$ , and  $P_{ij} = 0$  if  $(i,j) \notin E$

The primal model uses both the softmax operator and the graph's adjacency matrix to satisfy these conditions. Therefore, the primal model can construct a flow proportion matrix which results in a valid flow. This reparameterization successfully allows the primal model to meet the feasibility condition of the primal- dual training approach.

# Algorithm 1 Iterative Flow Computation

1: procedure FLOW- SOLVER(P, d, n, e)

2:  $\mathbf{X}^{(0)}\leftarrow \mathbf{0}_{n\times n}$

3:  $\mathbf{X}^{(- 1)}\leftarrow \mathrm{diag}(\epsilon +1)\in \mathbb{R}^{n\times n}$

4:  $\mathbf{D}\leftarrow \mathbf{1}_{n}\otimes \mathbf{d}\in \mathbb{R}^{n\times n}$

5:  $t\gets 1$

6: while  $\| \mathbf{X}^{(t - 1)} - \mathbf{X}^{(t - 2)}\|_{1} > \epsilon$  do > Matrix 1- norm is entry- wise

7:  $\tilde{\mathbf{X}}\leftarrow ((\mathbf{X}^{(t - 1)})^{\top}\cdot \mathbf{1}_{n})\otimes \mathbf{1}_{n}$

8:  $\mathbf{X}^{(t)}\leftarrow \mathbf{P}\odot (\tilde{\mathbf{X}} - \mathbf{D})$

9:  $\mathbf{X}^{(t)}\leftarrow \max (\mathbf{X}^{(t)},0)$  > Maximum operator is element- wise

10:  $t\gets t + 1$

11: return  $\mathbf{X}^{(t)}$

# 4.5 Theoretical Properties of the Iterative Flow Solver

For completeness, this section outlines the derivation of algorithm 1 and states the theorems which ensure the correctness of the iterative process.

# 4.5.1 Derivation

The goal of a transformation from a proportions matrix  $\mathbf{P}$  to a flow matrix  $\mathbf{X}$  is to have  $\mathbf{X}$  satisfy the property below.

$$
X_{i j} = P_{i j}\left(\sum_{k = 1}^{n}X_{k i} - d_{i}\right) \mathrm{and} X_{i j}\geq 0 \quad \forall i,j\in [n] \tag{5}
$$

By ignoring the non- negativity for now, this equation is equivalent to the following.

$$
\mathbf{X}_{i\colon} = \left(\sum_{k = 1}^{n}X_{k i} - d_{i}\right)\cdot \mathbf{P}_{i\colon} \quad \forall i\in [n]
$$

The sum  $\textstyle \sum_{k = 1}^{m}X_{k i}$  is the sum of the  $i$  - th column of  $\mathbf{X}$  . Therefore,  $\begin{array}{r}{\sum_{k = 1}^{m}X_{k i} = \left(\mathbf{X}^{\top}\mathbf{1}_{n}\right)_{i}} \end{array}$  The entire flow matrix  $\mathbf{X}$  can thus be written as follows. The matrix  $\tilde{\mathbf{X}}\in \mathbb{R}^{n\times n}$  is the matrix where each column is  $\mathbf{X}^{\top}\cdot \mathbf{1}_{n}$  , and  $\mathbf{D}\in \mathbb{R}^{n\times n}$  is the demand matrix where each row is  $\mathbf{d}^{\top}$  The symbol  $\odot$  denotes the element- wise (Hadamard) product.

$$
\mathbf{X} = \mathbf{P}\odot (\tilde{\mathbf{X}} -\mathbf{D})
$$

The matrix  $\tilde{\mathbf{X}}$  is equivalent to  $(\mathbf{X}^{\top}\mathbf{1}_{n})\otimes \mathbf{1}_{n}$  where  $\otimes$  denotes the vector outer product. Similarly,  $\mathbf{D} = \mathbf{1}_{n}\otimes \mathbf{d}$  . Therefore, a fixed point of the function  $f$  (as defined in equation (4)), yields the desired matrix  $\mathbf{X}$  . The element- wise maximum in  $f$  ensures that a fixed point meets the non- negativity constraint of equation (5). This derivation shows that a fixed point of  $f$  satisfies the original goal in equation (5).

# 4.5.2 Properties

The reparameterization of the primal problem relies on both the correctness and the convergence of algorithm 1. The theorems guaranteeing these properties are stated in this section. These claims assume the flow proportions matrix  $\mathbf{P}$  satisfies remark 1. All formal proofs are included in section 10.

Theorem 1, stated below, guarantees that a fixed point of  $f$  is indeed a valid flow.

Theorem 1. Let  $\mathbf{X}^{*}\in \mathbb{R}_{\geq 0}^{n\times n}$  be a fixed point of  $f$  . Then,  $\mathbf{X}^{*}$  satisfies the constraints of the MCF problem in (1).

Proof. See section 10.1.

The convergence of algorithm 1 to a fixed point is guaranteed under theorem 2. Before this theorem is stated, the following notation is defined.

Definition 1. The function  $f_{t}$  is function formed by composing  $f$  with itself  $t$  times. More specifically, for some  $\mathbf{X}\in \mathbb{R}_{\geq 0}^{n\times n}$ $f_{t}(\mathbf{X}) = f(f(\ldots f(\mathbf{X})))$

Theorem 2. The sequence  $(f_{t}(\mathbf{0}_{n\times n}))_{t\in \mathbb{N}}$  converges to a fixed point. This statement implies that the sequence  $(\mathbf{X}^{(t)})_{t\in \mathbb{N}}$  converges element- wise to a fixed point, where each  $\mathbf{X}^{(t)}$  is defined by algorithm 1 for  $t\geq 0$

Proof. See section 10.3.

These theorems ensure that the algorithm 1 will always find a valid flow for a given proportions matrix  $\mathbf{P}$  which satisfies remark 1.

The final consideration of this iterative process involves differentiability. The introduced reparameterization allows the primal neural network to produce the flow proportions  $\mathbf{P}$  instead of the actual flows  $\mathbf{X}$  . The cost associated with the matrix  $\mathbf{P}$  , however, is still the cost of the corresponding flow matrix. For the primal model to be trainable using the actual flow cost, algorithm 1 must be differentiable with respect to  $\mathbf{P}$  . This statement amounts to showing that the function below is differentiable with respect to  $P_{i j}$  for all  $i,j\in [n]$

$$
\tilde{f} (P_{i j}) = \max (P_{i j}\sum_{k = 1}^{n}X_{k i} - d_{i},0)
$$

By defining the derivative of  $\max (a,0)$  at  $a = 0$  to be zero, the function  $\tilde{f}$  is differentiable with respect to  $P_{i j}$  . Therefore, loss signals can backpropagate through algorithm 1.

# 4.6 The Minimum Cost Flow Dual Problem

The goal of the primal- dual training algorithm is to minimize the estimated duality gap. Within the context of MCFs, leveraging this technique requires a neural network which

# Algorithm 2 Gradient Descent to solve for Dual Flows

1: procedure DUAL- FLOW- SOLVER  $(\alpha ,n,c,\mathbf{A},\beta ,\gamma ,\epsilon)$  > A is the graph's adjacency matrix 2:  $\mathbf{Y}^{(0)}\leftarrow \mathbf{0}_{n\times n}$  3:  $\mathbf{Y}^{(- 1)}\leftarrow \mathrm{diag}(\epsilon +1)\in \mathbb{R}^{n\times n}$  4:  $u_{ij}^{(0)}\leftarrow 0.1\forall i,j\in [n]$  5:  $t\gets 0$  6: while  $\| \mathbf{Y}^{(t)} - \mathbf{Y}^{(t - 1)}\| _1 > \epsilon$  do > Matrix 1- norm is entry- wise 7: for  $i,j\in [n]$  do 8:  $g_{ij}^{(t)}\leftarrow c^{\prime}(Y_{ij}^{(t)}) + \alpha_{j} - \alpha_{i}$  > Derivative of the minimization in eq 8 9:  $u_{ij}^{(t + 1)}\leftarrow \beta \cdot u_{ij}^{(t)} + (1 - \beta)\cdot (g_{ij}^{(t)})^{2}$  10:  $Y_{ij}^{(t + 1)}\leftarrow \max \left(A_{ij}\cdot \left(Y_{ij}^{(t)} - \frac{\gamma}{\sqrt{u_{ij}^{(t + 1)} + \epsilon}} g_{ij}^{(t)}\right),0\right)$  11:  $t\gets t + 1$  12: return  $\mathbf{Y}^{(t)}$

estimates the MCF dual problem. The function below is the Lagrangian of the MCF problem. The vector  $\alpha \in \mathbb{R}^{n}$  represents the dual variables for each node.

$$
\begin{array}{r l} & {\mathcal{L}(\mathbf{X},\pmb {\alpha}) = \sum_{i = 1}^{n}\sum_{j = 1}^{n}c(X_{i j}) + \sum_{i\in V}\alpha_{i}\left(\sum_{k = 1}^{n}X_{k i} - \sum_{j = 1}^{n}X_{i j} - d_{i}\right)}\\ & {\quad \quad \quad = \sum_{i = 1}^{n}\sum_{j = 1}^{n}(c(X_{i j}) + (\alpha_{j} - \alpha_{i})X_{i j}) + \sum_{i = 1}^{n}\alpha_{i}d_{i}} \end{array} \tag{7}
$$

The dual variable  $\alpha_{i}$  is interpreted as the price of node  $i\in [n]$  [7]. From this Lagrangian, the dual function  $H:\mathbb{R}^{n}\to \mathbb{R}$  is defined below. The matrix  $\mathbf{X}$  is renamed  $\mathbf{Y}$  for clarity.

$$
H(\pmb {\alpha}) = \min_{\mathbf{Y}\succeq 0}\left(\sum_{i = 1}^{n}\sum_{j = 1}^{n}(c(Y_{ij}) + (\alpha_{j} - \alpha_{i})Y_{ij}) + \sum_{i = 1}^{n}\alpha_{i}d_{i}\right) \tag{8}
$$

The dual neural network can directly produce the dual variables  $\alpha \in \mathbb{R}^{n}$  because these quantities are unconstrained. Integrating the dual model within the primal- dual loss function (3) requires evaluating the function  $H$ . The dual cost function  $H$  is evaluated using projected gradient descent on the matrix  $\mathbf{Y}$ . As the cost function  $c$  is both convex and differentiable, this process will converge to the required minimum. Algorithm 2 outlines this gradient descent process. The algorithm is an application of RMSProp [19, 35] to the specific minimization in equation (8). Algorithm 2 is also differentiable with respect to  $\alpha$ . This differentiability enables loss signals to backpropagate to the dual model through this algorithm.

Algorithms 1 and 2 define a differentiable processes to evaluate feasible primal and dual MCF solutions respectively. By always producing feasible solutions, the primal and dual formulations presented in this section can be trained using the DualityLoss technique.

![](images/881bd34b349815b85396399b575e165148c91a36f9d26fbde466c28de01340be.jpg)  
Figure 6: Average validation loss over 100 epochs for graph neural network models which use 1 and 10 message-passing steps. This evaluation is conducted on the CB-500 graph (10)

# 5 Graph Neural Network Design

The MCF problem has practical applications in transportation systems. As such, a graph neural network model to compute MCFs should perform well on urban road graphs. Road topologies present a challenge to graph neural networks because their high diameter [27] requires the neural network to retain information over long distances. To address this problem, the graph neural network model used to solve MCFs employs the concepts of graph attention [42], gated state updates [25], and neighborhood lookahead [2]. The graph neural network is integrated into the DualityLoss framework to enable training without labeled data. The entire model is implemented in Tensorflow [1], and the code can be found in the following repository: https://github.com/tejaskannan/graph- primal- dual.

# 5.1 Motivation and Architecture Overview

The task of learning graph flows can be broken down into two fundamental actions: pathfinding and flow- pushing. To be effective at both actions, nodes must know both the location and the demands of the source and sink vertices. To ensure all nodes receive this information, the graph neural network must execute a number of message- passing steps roughly equal to  $\mathrm{diam}(G)$  [26]. The experiment displayed in figure 6 empirically confirms this assertion. This chart compares the estimated duality gap for two variants of the same model architecture. The variant which uses a single message passing step is unable to make much progress in minimizing the duality gap. This higher loss translates to flow costs which are  $112.5\%$  larger on average. This result shows that nodes require more than just local information for the MCF task.

This observation makes it difficult to apply graph neural networks to high- diameter road

![](images/98ab8ffd284784ade6732b3c976e5f91730ec277b884cdc22984d06ea5fd35b8.jpg)  
Figure 7: The 0th (left), 1st (middle), and 2nd (right) order neighborhoods from the vertex A. Red and blue nodes correspond to outgoing and incoming neighborhoods respectively. Vertices with both colors are members of both neighborhoods. Vertex  $D$  is not included in the 2nd order outgoing neighborhood because it is already a 1st order outgoing neighbor.

graphs [27]. High- diameter graphs cause issues with long- range information retention. This problem can be addressed by a propagation step which combines information from multiple neighborhood levels [2, 43]. An example of such neighborhoods is shown in figure 7. Information for incoming and outgoing neighbors are separately aggregated. The use of many neighborhoods reduces the total number of message- passes required to cover the entire graph. To further enhance the model's ability for long- range information retention, nodes use a GRU [11] update to integrate the features collected from the various neighborhoods [25].

Finally, flow values can be interpreted as the emphasis placed on a path from source to sink. This interpretation lends itself to an attention- based model [6, 42] in which nodes can focus on neighbors which should carry more flow. The model presented in this thesis uses multiple attention levels. The first is used to combine node states within a single neighborhood set. The second level is used to aggregate information between the neighborhood levels.

# 5.2 End-to-End Model

To solve MCF problems, the graph neural network can integrate with the DualityLoss training technique. This integration yields the architecture in figure 8. The input node features are composed of both a trainable and a problem- specific portion. The trainable embeddings allow the model to fit the target graph. The problem- specific information enables the model to adapt to the current flow problem instance. The primal and dual models share the graph neural network component. This decision reduces the number of trainable parameters. Based on the results of Prates et al. [34], the graph neural network component should have the capacity to create useful embedding for both the primal and dual tasks [34]. Separate decoder networks produce specific primal and dual outputs. The loss function is the primal- dual objective established in section 4.3. This function is stated below for completeness. The function  $F$  represents the transformation from flow proportions to flow values (algorithm 1),  $H$  is the dual cost function,  $k$  denotes the training sample index, and

![](images/f3c0dcb6368b9ebc06ec6be438f91ea3f6def2295f35f7eeb8213dfb806f8136.jpg)  
Figure 8: High-level model architecture. Blue node features are trainable embeddings. The red node feature is the demand value for a specific MCF problem instance.

$N$  is the size of the training dataset.

$$
L(\pmb {\theta},\pmb {\phi}) = \frac{1}{N}\sum_{k = 1}^{n}\left(\left(\sum_{i = 1}^{n}\sum_{j = 1}^{n}c(F(g_{\pmb{\theta}}(G,\mathbf{d}_{k})))_{ij}\right) - H(h_{\phi}(G,\mathbf{d}_{k}))\right)
$$

# 5.3 Implementation

This section covers the implementation details of the model motivated in the previous sections. In terms of notation, variables named  $\mathbf{W}$ ,  $\mathbf{a}$ , and  $\mathbf{b}$  are trainable. The vector  $\mathbf{d}$  represents the demand vector of a single problem instance.

# 5.3.1 Encoder

The initial node features are a concatenation of trainable node embeddings with the demand values of a given MCF problem. The initial state of node  $i\in [n]$  is  $\mathbf{y}_i' = (\mathbf{y}_i||d_i)$  where  $\mathbf{y}_i\in \mathbb{R}^{r_1}$  is trainable. The encoder network is a multi- layer perceptron (MLP), and its output is the initial demand- specific embedding  $\mathbf{h}_i^{(0)}\in \mathbb{R}^{r_2}$ . The transformation below shows the operations for a single dense layer.

$$
\mathbf{h}_i^{(0)} = \tanh (\mathbf{W}_1\mathbf{y}_i' + \mathbf{b}_1)
$$

# 5.3.2 Graph Attention and Neighborhood Lookahead

The graph neural network component uses the ideas of neighborhood expansion, graph attention, and gated state updates. Formally, let  $\mathbf{A}\in \mathbb{R}^{n\times n}$  be the graph's adjacency matrix and let  $\mathbf{A}^k$  be the  $k^{th}$  power of  $\mathbf{A}$ . The  $k$ - th order outgoing and incoming neighborhoods of

each node  $i\in [n]$  are defined below for all  $k > 1$  . For  $k = 0$ $\mathcal{N}_{0}^{o u t}(i) = \mathcal{N}_{0}^{i n}(i) = \{i\}$

$$
\begin{array}{r l} & {\mathcal{N}_{k}^{o u t}(i) = \{j\in [n] | A_{i j}^{k} > 0 \mathrm{and} j\notin \mathcal{N}_{\ell}^{o u t}(i) \forall \ell < k\}}\\ & {\mathcal{N}_{k}^{i n}(i) = \{j\in [n] | (A^{\top})_{i j}^{k} > 0 \mathrm{and} j\notin \mathcal{N}_{\ell}^{i n}(i) \forall \ell < k\}} \end{array}
$$

An example of these neighborhoods is shown in figure 7.

Let  $\mathbf{h}_{i}^{(t)}$  be the state of node  $i\in [n]$  after the  $t^{t h}$  message- passing step. Furthermore, let  $T$  be the total number of message passes and  $K$  be highest- order neighborhood considered. During the  $(t + 1)^{st}$  step, nodes use the technique of graph attention [42] to combine information from their various neighborhoods. The equations below define this aggregation for both incoming and outgoing neighborhoods.

$$
\begin{array}{r l r} & {} & {\mathrm{~\alpha_{i j}^{(k)} = \frac{\exp\{\mathbf{a}_{1}^{\top}(\mathbf{W}_{2}\mathbf{h}_{i}^{(t)}\|\mathbf{W}_{2}\mathbf{h}_{j}^{(t)})\}}{\sum_{\ell\in\mathcal{N}_{k}^{o u t}(i)}\exp\{\mathbf{a}_{1}^{T}(\mathbf{W}_{2}\mathbf{h}_{i}^{(t)}\|\mathbf{W}_{2}\mathbf{h}_{\ell}^{(t)})\}\}}\quad \beta_{i j}^{(k)} = \frac{\exp\{\mathbf{a}_{2}^{\top}(\mathbf{W}_{3}\mathbf{h}_{i}^{(t)}\|\mathbf{W}_{3}\mathbf{h}_{j}^{(t)})\}}{\sum_{\ell\in\mathcal{N}_{k}^{o u t}(i)}\exp\{\mathbf{a}_{2}^{T}(\mathbf{W}_{3}\mathbf{h}_{i}^{(t)}\|\mathbf{W}_{3}\mathbf{h}_{\ell}^{(t)})\}\}}\quad \beta_{i j}^{(k)} = \frac{\exp\{\mathbf{a}_{2}^{\top}(\mathbf{W}_{3}\mathbf{h}_{i}^{(t)}\|\mathbf{W}_{3}\mathbf{h}_{j}^{(t)})\}}{\sum_{\ell\in\mathcal{N}^{o u t}(i)}\exp\{\mathbf{a}_{2}^{T}(\mathbf{W}_{3}\mathbf{h}_{i}^{(t)}\|\mathbf{W}_{3}\mathbf{h}_{\ell}^{(t)})\}\}}\quad \beta_{i j}^{(k)} = \frac{\exp\{\mathbf{a}_{2}^{\top}(\mathbf{w}_{3}\mathbf{h}_{i}^{(t)})\}}{\sum_{\ell\in\mathcal{N}^{o u t}(i)}\exp\{\mathbf{a}_{2}^{T}(\mathbf{w}_{3}\mathbf{h}_{i}^{(t)})\}\}} \end{array}
$$

The vectors  $\mathbf{u}_{i}^{(k)}$  and  $\mathbf{v}_{i}^{(k)}$  represent the aggregated states for the  $k^{t h}$  outgoing and incoming neighborhoods. These vectors are in  $\mathbb{R}^{r_{2}}$  . Different trainable weights are applied to the two neighborhood types. This choice follows the common practice of handling different edge types [25]. A second attention level combines the aggregated states for each neighborhood level. This layer is shown below.

$$
\begin{array}{r l r l} & {\gamma_{i}^{(k)} = \frac{\exp\{\mathbf{a}_{3}^{\top}\mathbf{u}_{i}^{(k)}\}}{\sum_{j = 0}^{K}\exp\{\mathbf{a}_{3}^{\top}\mathbf{u}_{i}^{(j)}\}}} & & {\delta_{i}^{(k)} = \frac{\exp\{\mathbf{a}_{4}^{\top}\mathbf{v}_{i}^{(k)}\}}{\sum_{j = 0}^{K}\exp\{\mathbf{a}_{4}^{T}\mathbf{v}_{i}^{(j)}\}}}\\ & {\mathbf{z}_{i}^{o u t} = \tanh \left(\sum_{j = 0}^{K}(\gamma_{i}^{(k)}\mathbf{u}_{i}^{(j)}) + \mathbf{b}_{4}\right)} & & {\mathbf{z}_{i}^{i n} = \tanh \left(\sum_{j = 0}^{K}(\delta_{i}^{(k)}\mathbf{v}_{i}^{(j)}) + \mathbf{b}_{5}\right)} \end{array}
$$

The vectors  $\mathbf{z}_{i}^{o u t},\mathbf{z}_{i}^{i n}$  are also in  $\mathbb{R}^{r_{2}}$  . The incoming and outgoing neighborhood states are combined using a single dense layer.

$$
\mathbf{z}_{i} = \tanh \left(\mathbf{W}_{4}(\mathbf{z}_{i}^{o u t}||\mathbf{z}_{i}^{i n}) + \mathbf{b}_{6}\right)
$$

The vector  $\mathbf{z}_{i}$  represents an aggregated state based on all neighborhood levels at timestep  $t$  . This information is combined with the current node state,  $\mathbf{h}_{i}^{(t)}$  , using the following GRU update [11, 25]. The symbol  $\sigma$  is the sigmoid function.

$$
\begin{array}{r l} & {\mathbf{q}_{i}^{(t + 1)} = \sigma \left(\mathbf{W}_{q}(\mathbf{h}_{i}^{(t)}||\mathbf{z}_{i})\right)}\\ & {\mathbf{r}_{i}^{(t + 1)} = \sigma \left(\mathbf{W}_{r}(\mathbf{h}_{i}^{(t)}||\mathbf{z}_{i})\right)}\\ & {\tilde{\mathbf{h}}_{i}^{(t + 1)} = \tanh \left(\mathbf{W}_{s}((\mathbf{r}_{i}^{(t + 1)}\odot \mathbf{h}_{i}^{(t)})||\mathbf{z}_{i})\right)}\\ & {\mathbf{h}_{i}^{(t + 1)} = (1 - \mathbf{q}_{i}^{(t + 1)})\odot \mathbf{h}_{i}^{(t)} + \mathbf{q}_{i}^{(t + 1)}\odot \tilde{\mathbf{h}}_{i}^{(t + 1)}} \end{array}
$$

![](images/4d9ef58ec1e383a7d34eddb5aad0309f11adf1035abc58954fa05a8ab6a4ebde.jpg)  
Figure 9: Graph Neural Network Architecture. The vector  $\mathbf{v}^{t}$  is the state for a single vertex at timestep  $t$ .

These transformations define how node states are updated between message- passing steps. Figure 9 depicts this entire process. Trainable parameters are shared across all nodes, neighborhood levels, and message- passing steps.

# 5.3.3 Decoder

The graph neural network layer produces a final state  $\mathbf{h}_{i}^{(T)}$  for all nodes  $i \in [n]$ . These states are decoded into primal and dual solutions using distinct MLPs. The transformations for the primal (top) and dual (bottom) networks are shown below for each node  $i \in [n]$ .

$$
\begin{array}{r l} & {s_{j}^{(i)} = \mathbf{a}_{p r i m a l}^{T}\left(\tanh \left(\mathbf{W}_{p r i m a l}(\mathbf{h}_{i}^{(T)}||\mathbf{h}_{j}^{(T)}) + \mathbf{b}_{p r i m a l}\right)\right)\quad \forall j\in [n]}\\ & {\quad q_{i} = \mathbf{a}_{d u a l}^{T}\left(\tanh \left(\mathbf{W}_{d u a l}\mathbf{h}_{i}^{(T)} + \mathbf{b}_{d u a l}\right)\right)} \end{array}
$$

The dual variables  $\mathbf{q}$  are used to compute the dual cost. The edge scores  $s_{j}^{(i)}$  for all  $i, j \in [n]$  are converted to flow proportions as discussed in the next section.

# 5.4 Converting Edge Scores into Flow Proportions

The flow proportions matrix  $\mathbf{P}$  is created by converting each set  $S_{i} = \{s_{j}^{(i)} | (i, j) \in E\} \forall i \in [n]$  into a simplex. The operation can be done by applying the softmax operator to each set  $S_{i}$ . The adjacency matrix  $\mathbf{A}$  masks the scores of non- existent edges. Concretely,

<table><tr><td>Graph Name</td><td>No. Nodes</td><td>No. Edges</td><td>Avg. In Degree</td><td>Avg. Out Degree</td><td>Diameter</td></tr><tr><td>CB-500</td><td>44</td><td>86</td><td>1.95</td><td>1.95</td><td>18</td></tr><tr><td>SF-500</td><td>65</td><td>152</td><td>2.34</td><td>2.34</td><td>13</td></tr><tr><td>London-500</td><td>228</td><td>421</td><td>1.85</td><td>1.85</td><td>42</td></tr><tr><td>NYC-1000</td><td>287</td><td>542</td><td>1.89</td><td>1.89</td><td>32</td></tr></table>

Figure 10: Properties of the graphs used during experimentation. These statistics are computed using the NetworkX library [17].

the resulting proportions matrix  $\mathbf{P}$  has following elements.

$$
P_{ij} = \frac{A_{ij} \cdot \exp\{s_j^{(i)}\}}{\sum_{k = 1}^n A_{ik} \cdot \exp\{s_k^{(i)}\}} \quad i, j \in [n]
$$

Softmax has the downside of producing dense distributions [29]. Within the context of MCFs, this property can be undesirable because sending flow along too many paths can lead to suboptimal solutions.

Sparsemax [29] is an alternative to softmax which addresses the issue of dense distributions. Sparsemax works as follows. Let  $\mathbf{z} \in \mathbb{R}^n$  be the target vector. Then,

$$
\operatorname {sparsemax}(\mathbf{z})_i = \max (z_i - \tau (\mathbf{z}), 0)
$$

where  $\tau (\mathbf{z})$  is the unique threshold such that  $\sum_{i = 1}^{n} \max (z_i - \tau (\mathbf{z}), 0) = 1$ . This threshold can be computed in a differentiable manner [29]. The key feature of sparsemax is that all values of  $z_i$  below this threshold are set to exactly zero. Sparsemax, however, does not guarantee that the proportions matrix  $\mathbf{P}$  satisfies remark 1. This guarantee is made by assigning all edges at least some near- zero proportion. If  $\tilde{\mathbf{z}} = \operatorname {sparsemax}(\mathbf{z})$ , then

$$
\operatorname {clipped - sparsemax}(\tilde{\mathbf{z}})_i = \frac{\max (\tilde{z}_i, \epsilon)}{\|\max (\tilde{\mathbf{z}}, \epsilon)\|_1}
$$

The maximum in the denominator is element- wise and  $\epsilon \approx 10^{- 9}$  is a small, positive constant. This normalization works because  $\tilde{\mathbf{z}}$  is a non- negative vector which sums to one. Applying the clipped version of sparsemax to each  $S_i$  yields a valid flow proportions matrix  $P$ , and the qualities of sparsemax imply that it is more aggressive than softmax at assigning edges a near- zero flow proportion.

These two normalizing transformations yield two variants of the proposed graph neural network: Neighborhood+Softmax and Neighborhood+Sparsemax.

# 6 Evaluation

To evaluate the proposed graph neural network solution to the MCF problem, the constructed models are tested on a set of real- world road graphs from Open Street Map [9, 32]. This evaluation has the following five goals. Unless specified, the graph neural network is trained using the DualityLoss technique.

1. Goal: Compare the proposed graph neural network approach to the MCF problem against pure optimization algorithms in terms of both answer quality and wall-to-wall running time.

Hypothesis: The neural network will produce flows that are sub- optimal because the pure optimization techniques are allowed many optimization steps on each problem instance. The learned model, however, will produce answers in a more timely manner.

2. Goal: Assess the impact of using the DualityLoss method to replace training with ground-truth labels. This comparison encompasses both answer quality and neural network training time, where the training time includes the time to create datasets.

Hypothesis: A neural network trained with DualityLoss will produce slightly higher flow costs than those of the neural network trained with ground truth labels. This hypothesis is based on the fact that the DualityLoss model has no access to ground- truth information. The DualityLoss neural network, however, will be faster to train because it does not require dataset creation.

3. Goal: Evaluate the effects of softmax and sparsemax in terms of answer quality. Hypothesis: Sparsemax will produce lower-costing flows because it can more easily concentrate flow along optimal paths.

4. Goal: Compare the Neighborhood architecture against a design which only aggregates information from direct neighbors.

Hypothesis: On high- diameter graphs, the Neighborhood model will produce better answers than this baseline architecture. This improved performance should come from the benefit of using fewer message- passing steps with neighborhood expansion.

5. Goal: Assess the importance of the flow proportions matrix  $\mathbf{P}$  to the flows computed by the iterative flow solver (algorithm 1). This experiment is important to confirm that the results obtained by the neural networks are not solely due to any ingenuity in the iterative flow solver.

Hypothesis: The flow solver will be sensitive to the flow proportions matrix because it blindly follows any given flow proportions. Therefore, the neural network methods will produce lower- costing flows.

# 6.1 An Illustration of Computed Graph Flows

To show that the proposed neural network method is capable of predicting flows, we first consider an example of flows produced by a trained graph neural network model. Figure 11 shows the flows computed by both the neural network and a baseline trust region optimization algorithm [21] for the cost function  $c(x) = x^2$ . From the figure, it can be seen that the neural network learns the correct overall trend. For example, it transmits flow along the shortest path from source 15 to sink 25. The neural network, however, does face some issues with not fully eradicating flow cycles and overshooting the amount of flow sent to a sink. These issues are explored further in section 7.

The graph neural network used to produce this flow is trained to adapt to changing

![](images/b9e3067edf7695a54c9d0d394027dd5249dca34c8d6c493dd22db9670ee93c31.jpg)

![](images/2035202b0fd18cf9375786f473849d68627d67cd85eecc37b71421a6541919fb.jpg)  
Figure 12: Median flow costs relative to the Trust region optimizer on 1,000 test instances. Error bars denote  $25^{th}$  and  $75^{th}$  percentiles. The ground truth model is not evaluated with the cost of  $e^{x} - 1$  due to the time required to generate data labels. The Neural Network models use the Neighborhood+Sparsemax architecture.

demand patterns for the same set of sources and sinks on the road graph shown in figure 11. More information about the datasets and specifics about model training is included in the section below.

# 6.1.1 Datasets and Training

The graph neural network method is evaluated on four distinct road graphs obtained from Open Street Map (OSM) [9, 32]. Figure 10 lists some relevant statistics of these graphs. The training, validation, and testing datasets are created by randomly generating demand values for a fixed graph, set of sources, and set of sinks. The neural network is tasked with adapting to different demand patterns. Sources and sinks are chosen such that their pairwise distance is maximized. This setup tests the neural network's ability to learn longer paths. Appendix 11.2 contains the exact selection algorithm. Demand values are normalized between 0 and 1. To avoid overfitting, test and validation demands are at a distance of at least 0.1 from any training sample. Appendix 11.3 contains all model hyperparameters. The model with the best validation loss is selected for testing.

# 6.2 Comparing the Neural Network Approach to Baseline Techniques: Flow Cost

The graph neural network approach developed in this work is compared against two pure optimization techniques: sequential least squares quadratic programming (SLSQP) and trust region constrained optimization (Trust Region). The Trust Region algorithm is an advanced optimization technique which produces nearly- optimal values. The Trust Region algorithm is interpreted as the state- of- the- art solver. SLSQP provides lower- quality solutions but can compute such answers more efficiently. Additional background information on these algorithms is included in section 6.6.1.

<table><tr><td rowspan="3">Graph</td><td rowspan="3">Cost</td><td colspan="4">Median Runtime (sec) per Problem Instance</td><td colspan="2">Training Time (hrs)</td></tr><tr><td colspan="2">Pure Optimization</td><td colspan="2">Neural Network</td><td colspan="2">Neural Network</td></tr><tr><td>Trust Region</td><td>SLSQP</td><td>DualityLoss</td><td>Ground Truth</td><td>DualityLoss</td><td>Ground Truth</td></tr><tr><td>CB-500</td><td>x²</td><td>0.235s</td><td>0.024s</td><td>0.011s</td><td>0.013s</td><td>2.15h</td><td>7.85h</td></tr><tr><td>CB-500</td><td>e²-1</td><td>0.609s</td><td>0.018s</td><td>0.010s</td><td>N/A</td><td>2.42h</td><td>N/A</td></tr><tr><td>SF-500</td><td>x²</td><td>1.527s</td><td>0.091s</td><td>0.015s</td><td>0.015s</td><td>2.43h</td><td>6.91h</td></tr><tr><td>SF-500</td><td>e²-1</td><td>1.396s</td><td>0.055s</td><td>0.015s</td><td>N/A</td><td>2.72h</td><td>N/A</td></tr><tr><td>London-500</td><td>x²</td><td>N/A</td><td>1.428s</td><td>0.038s</td><td>N/A</td><td>10.88h</td><td>N/A</td></tr><tr><td>NYC-1000</td><td>x²</td><td>N/A</td><td>7.107s</td><td>0.036s</td><td>N/A</td><td>13.32h</td><td>N/A</td></tr></table>

Figure 13: Execution and training times for both the optimization baselines and the neural network models. The median running times are taken over 1,000 problem instances. Each model is trained for 100 epochs, and the Ground Truth training time includes the time required to generate the training dataset. The neural network uses the Neighborhood  $+$  Sparsemax architecture. Omitted entries are excluded due to the high cost of computation.

The second baseline directly evaluates the DualityLoss technique by training a graph neural network using ground- truth labels. This model uses the same Neighborhood architecture but omits the dual decoder component. Labels are composed of the cost values found by the Trust Region optimizer. The ground truth model is not evaluated on the  $e^{x} - 1$  cost function due to the time required to generate a sufficient number of data labels.

Figure 12 depicts a comparison of computed flow costs between the optimization and neural network approaches. As answer quality is related to how close a solution is to the optimum, these costs are displayed relative to the Trust Region optimizer. The chart shows that optimization approaches consistently produce flows which are of lower cost than those of the learned models. The DualityLoss model produces flows within  $35\%$  of the optimal value. This result supports the hypothesis regarding the answer quality of the neural network model. The neural network's suboptimality is explored further in section 7.

When comparing the DualityLoss ground- truth models, the DualityLoss model appears to produce lower- costing flows. Over the 1,000 test instances, the DualityLoss model predicts flows of lower cost  $80.3\%$  of the time on CB- 500 and  $58.6\%$  of the time on SF- 500. This result is an encouraging validation of the DualityLoss method. One hypothesis for why DualityLoss achieves better performance is because the DualityLoss model explicitly represents the dual problem. This means that it creates vertex embeddings which contain information about both the primal and dual MCF problems. The primal and dual problems are highly related, and obtaining information about the dual problem can aid in solving with the primal. The relationship between primal and dual MCF problems is exemplified by auction algorithms, as these algorithms solve the MCF problem by directly using the dual [7, Chapter 9]. DualityLoss may therefore have the benefit of creating node embeddings which contain more information about the MCF problem. Exploring this hypothesis in detail is left for future work.

# 6.3 Comparing the Neural Network Approach to Baseline Techniques: Computation Time

One of the goals of this project is to create a neural network method which can solve the minimum cost flow problem faster than existing techniques. This goal is addressed by measuring the median wall- to- wall time' required to compute flows for both the pure optimization and the neural network methods. Figure 13 shows these measurements. In concordance with the original hypothesis, the neural network methods consistently produce answers faster than both SLSQP and Trust Region. The impact of this efficiency becomes more evidence on the larger London- 500 and NYC- 1000 graphs. The median runtime of DualityLoss doubles when transitioning from the SF- 500 graph to the London- 500 graph; on the other hand, SLSQP's runtime is over 15 times greater.

The price of the neural network's efficiency at runtime is paid for in its training time. Even for the smaller graphs, the DualityLoss neural network requires over 2 hours of training time. The neural network, however, shows the ability to offload a large amount of the computation to the training phase, and it performs well at testing time.

The final result from this table is the comparison between the DualityLoss and Ground Truth models in terms of training time. When including the time to generate data labels into account, the ground truth model becomes significantly more expensive to train. This high cost is mainly due to the time required to create data labels. The DualityLoss method, on the other hand, requires no labeled datasets, and this benefit is seen in the large discrepancy in training time. Furthermore, the ground truth technique is not computationally feasible on the larger datasets because of the trust region algorithm's inefficiencies. As proven in this experiment, DualityLoss addresses this problem of scalability and allows for the neural network to be trained on the larger graphs.

![](images/ab85047a2f03e0537fa9b91ec29a3d230f3edee1cfaf6d9b2e8c7391d9c7b181.jpg)  
Figure 14: Flow costs for trained Neighborhood models using sparsemax and softmax. These costs are relative to the costs computed by the Trust Region algorithm.

![](images/f0eba35f66111fdf51bf66c599c686e44956554abec21b833167141b4f07a0d9.jpg)  
Figure 15: Flow cost for Neighborhood and Gated GAT models relative to SLSQP on the London-500 and NYC-1000 datasets.

# 6.4 Evaluating the Design Choices of the Graph Neural Network Architecture

The Neighborhood model makes a few design choices based on both the target problem and the properties of the underlying graph. One such choice is the use of either sparsemax or softmax to convert edge scores into flow proportions. The use of sparsemax is motivated by its ability to more aggressively set small flow proportions. This choice is evaluated on the CB- 500 and SF- 500 graphs. Figure 14 displays the flow costs of both the Neighborhood+Sparsemax and Neighborhood+Softmax model relative to the Trust Region costs. In three out of the four experiments, the sparsemax model produces slightly lower flow costs. Furthermore, on the SF- 500 graph, the sparsemax model predicts better flows on  $62.6\%$  test instances for  $x^{2}$  and  $63.6\%$  problems for  $e^{x} - 1$ . On the CB- 500 graph, the sparsemax network predicts lower costing flows on  $55.4\%$  and  $39.2\%$  of instances for  $x^{2}$  and  $e^{x} - 1$  respectively. Although sparsemax generally performs better, the lack of consistency in these values indicates the sparsemax is not conclusively the better model. This finding goes against the original hypothesis regarding these two operators.

To evaluate design choices behind the Neighborhood model's architecture, the Neighborhood model is compared to a baseline model called Gated GAT. Gated GAT is built using only the graph attention and gated state update portions of the Neighborhood mode's propagation step. Section 6.6.2 covers the details of this model. The comparison between the Neighborhood model and Gated GAT has the goal of showing that neighborhood expansion allows the model to better compute flows on high- diameter graphs. This goal is assessed by training both the Neighborhood and Gated GAT models on the larger London- 500 and NYC- 1000 datasets. Figure 15 displays the results of these experiments. On these two datasets, the Neighborhood model outperforms the Gated GAT baseline approach. Over this set of test instances, the Neighborhood model predicts lower- costing flows  $99.5\%$  of the time on London- 500 and  $86.6\%$  of the time on NYC- 1000. This discrepancy in figures is an intriguing result because the London- 500 graph has a larger diameter than that of NYC- 1000. This result indicates that the use of neighborhood expansion allows the neural network to

Figure 16: Flow cost results for fixed baselines when compared against the Neighborhood+Sparsemax model on the SF-500 graph.  

<table><tr><td>Cost Function</td><td>Baseline</td><td>Median Increase</td><td>Median % Increase</td></tr><tr><td rowspan="2">x²</td><td>Random</td><td>726.9</td><td>33,977.5%</td></tr><tr><td>Uniform</td><td>232.3</td><td>10,466.3%</td></tr><tr><td rowspan="2">e²-1</td><td>Random</td><td>4,988.1</td><td>50,118.7%</td></tr><tr><td>Uniform</td><td>518.7</td><td>5,017.5%</td></tr></table>

better adapt to higher- diameter graphs.

# 6.5 Assessing the Impact of the Iterative Flow Solver

The neural network model relies on an iterative flow solver (algorithm 1) to convert learned flow proportions into actual flow values. As this solver plays a critical role in the flow computation process, it is important to ensure that neural network's results are not simply due to some intelligence present in iterative flow solver. In other words, it muse be confirmed that the iterative flow solver does not produce high quality flows for every set of flow proportions.

To assess the abilities of the flow solver, a trained Neighborhood+Sparsemax model is compared to two baselines. The Random baseline feeds a randomly generated valid flow proportions matrix into the flow solver. The Uniform baseline distributes flow across outgoing edges in a uniform manner. Figure 16 shows the flow costs generated by these baselines on the SF- 500 dataset?. These numbers are relative to the Neighborhood+Sparsemax neural network. From this figure, it is evident that flow solver does not always produce low- costing flows. Furthermore, proportions which lead to high- costing flows are easy to find. This analysis confirms the hypothesis that the iterative solver is not capable of turning any set of flow proportions into a low- costing flow. These results give confidence to the notion that the answer quality shown by the neural networks is indeed due to the learned flow proportions.

# 6.6 Details of the Baseline Solvers

For completeness, this section includes details on the baseline approaches which are used during the evaluation.

# 6.6.1 Optimization Algorithms

SLSQP [24] is line search technique to iteratively solve constrained optimization problems using a second- order linear approximation. SLSQP considers optimization problems of the

form below. The functions  $f,g_{i}$  and  $h_{j}$  are from  $\mathbb{R}^{n}$  to  $\mathbb{R}$

$$
\begin{array}{r l} & {\underset {\mathbf{x}\in \mathbb{R}^{n}}{\min}f(\mathbf{x})\quad \mathrm{s.t.}}\\ & {g_{i}(\mathbf{x}) = 0\quad \forall i\in [q]}\\ & {h_{j}(\mathbf{x})\geq 0\quad \forall j\in [r]} \end{array}
$$

At each timestep, SLSQP updates its estimate of  $\mathbf{x}$  using the 2nd- order Taylor approximation as shown below. The matrix  $H_{f}(\mathbf{x}^{(t)})$  is the Hessian of  $f$  evaluated at  $\mathbf{x}^{(t)}$  , and  $\gamma$  is an optional step size parameter.

$$
\begin{array}{r l r} & {} & {\mathbf{x}^{(t + 1)} = \mathbf{x}^{(t)} + \gamma \arg \underset {\mathbf{d}\in \mathbb{R}^{n}}{\min}\frac{1}{2}\mathbf{d}^{\top}H_{f}(\mathbf{x}^{(t)})\mathbf{d} + \nabla f(\mathbf{x}^{(t)}) + f(\mathbf{x}^{(t)})\quad \mathrm{s.t.}}\\ & {} & {g_{i}(\mathbf{x}^{(t)}) + \nabla g_{i}(\mathbf{x}^{(t)})^{\top}\mathbf{d} = 0\quad \forall i\in [q]}\\ & {} & {h_{j}(\mathbf{x}^{(t)}) + \nabla h_{j}(\mathbf{x}^{(t)})^{\top}\mathbf{d}\geq 0\quad \forall j\in [r]} \end{array}
$$

Trust region techniques differ from line- search methods in that they optimize over a neighborhood of the current estimate  $\mathbf{x}^{(t)}$  [44]. If the region minimizes the function at the next iteration, then the neighborhood is expanded; otherwise, its size is decreased. There are many specific types of trust region algorithms. One such technique is an extension of SLSQP presented by Byrd, Schnabel, and Shultz [10]. This algorithm updates estimates  $\mathbf{x}^{(t)}$  as shown below. The parameter  $\theta_{t} \in [0,1]$  is used to handle the possibility the constraints being violated within the trust region, and  $\Delta_{t}$  controls the size of the trust region.

$$
\begin{array}{r l r} & {} & {\mathbf{x}^{(t + 1)} = \mathbf{x}^{(t)} + \gamma \arg \underset {\mathbf{d}\in \mathbb{R}^{n}}{\min}\frac{1}{2}\mathbf{d}^{\top}H_{f}(\mathbf{x}^{(t)})\mathsf{d} + \nabla f(\mathbf{x}^{(t)}) + f(\mathbf{x}^{(t)})\quad \mathrm{s.t.}}\\ & {} & {\theta_{t}g_{i}(\mathbf{x}^{(t)}) + \nabla g_{i}(\mathbf{x}^{(t)})^{\top}\mathsf{d} = 0\quad \forall i\in [q]}\\ & {} & {\theta_{t}h_{j}(\mathbf{x}^{(t)}) + \nabla h_{j}(\mathbf{x}^{(t)})^{\top}\mathsf{d}\geq 0\quad \forall j\in [r]}\\ & {} & {\| \mathbf{d}\| \leq \Delta_{t}} \end{array}
$$

Trust region techniques form a broad class of algorithms, and Yuan [44] provides a good overview of these algorithms and their convergence properties.

The baselines used in this work are from the SciPy optimize library [21]. Each method solves the following optimization problem. This problem is equivalent to the MCF problem in section 3.3.

$$
\begin{array}{l}{\min_{\mathbf{x}\in \mathbb{R}^{m}}\sum_{i = 1}^{m}c(x_{i})\quad \mathrm{such~that}}\\ {\mathbf{B}\mathbf{x} = \mathbf{d}}\\ {x_{i}\geq 0\quad \forall i\in [m]} \end{array}
$$

The matrix  $\mathbf{B} \in \mathbb{R}^{n \times m}$  encodes the flow circulation constraint in (1c) and is defined below. In this notation,  $E_{k}$  refers to the  $k^{th}$  edge for some fixed but arbitrary ordering of  $E$ . The

symbol  $E_{k}[0]$  is the source vertex and  $E_{k}[1]$  the destination vertex for the edge  $E_{k}$ .

$$
b_{ik} = \left\{ \begin{array}{ll}1 & E_{k}[0] = i \\ -1 & E_{k}[1] = i \\ 0 & \text{else} \end{array} \right.
$$

Since the target vector  $\mathbf{x} \in \mathbb{R}^{m}$  only represents valid edges, the existence constraint (1e) can be ignored. The experimental parameters for both optimization algorithms are shown in figure 11.4.

# 6.6.2 Gated GAT Baseline

The Gated Graph Attention Network (Gated GAT) [25, 42] baseline use the same principles of the Neighborhood model except for the neighborhood expansion. Explicitly, Gated GAT uses the propagation step below. In terms of notation,  $S_{O}^{(i)}$  and  $S_{I}^{(i)}$  are the sets  $S_{out}^{(i)} \cup \{i\}$  and  $S_{in}^{(i)} \cup \{i\}$  respectively.

$$
\begin{array}{r l} & {\alpha_{i j} = \frac{\exp\left\{\mathbf{a}_{o u t}^{\top}(\mathbf{W}_{o u t}\mathbf{h}_{i}^{(t)}||\mathbf{W}_{o u t}\mathbf{h}_{j}^{(t)})\right\}}{\sum_{k\in S_{I}^{(i)}}\exp\left\{\mathbf{a}_{o u t}^{\top}(\mathbf{W}_{o u t}\mathbf{h}_{i}^{(t)}||\mathbf{W}_{o u t}\mathbf{h}_{k}^{(t)})\right\}}\qquad \beta_{i j} = \frac{\exp\left\{\mathbf{a}_{i n}^{\top}(\mathbf{W}_{i n}\mathbf{h}_{i}^{(t)}||\mathbf{W}_{i n}\mathbf{h}_{j}^{(t)})\right\}}{\sum_{k\in S_{I}^{(i)}}\exp\left\{\mathbf{a}_{i n}^{\top}(\mathbf{W}_{i n}\mathbf{h}_{i}^{(t)}||\mathbf{W}_{i n}\mathbf{h}_{k}^{(t)})\right\}}}\\ & {\mathbf{u}_{i}^{(t + 1)} = \tanh \left(\sum_{j\in S_{O}^{(i)}}(\alpha_{i j}\mathbf{W}_{o u t}\mathbf{h}_{k}^{(t)}) + \mathbf{b}_{o u t}\right)\qquad \mathbf{v}_{i}^{(t + 1)} = \tanh \left(\sum_{j\in S_{I}^{(i)}}(\beta_{i j}\mathbf{W}_{i n}\mathbf{h}_{k}^{(t)}) + \mathbf{b}_{i n}\right)}\\ & {\mathbf{u}_{i}^{(t + 1)} = \tanh \left(\mathbf{W}_{c o m b}(\mathbf{u}_{i}^{(t + 1)}||\mathbf{v}_{i}^{(t + 1)}) + \mathbf{b}_{c o m b}\right)} \end{array}
$$

The vector  $\mathbf{u}_{i}^{(t + 1)}$  is used to create the updated state  $\mathbf{h}_{i}^{(t + 1)}$  through the GRU [11, 25] layer below.

$$
\begin{array}{r l} & {\mathbf{q}_{i}^{(t + 1)} = \sigma \left(\mathbf{W}_{q}(\mathbf{h}_{i}^{(t)}||\mathbf{u}_{i}^{(t + 1)})\right)}\\ & {\mathbf{r}_{i}^{(t + 1)} = \sigma \left(\mathbf{W}_{r}(\mathbf{h}_{i}^{(t)}||\mathbf{u}_{i}^{(t + 1)})\right)}\\ & {\tilde{\mathbf{h}}_{i}^{(t + 1)} = \tanh \left(\mathbf{W}_{s}((\mathbf{r}_{i}^{(t + 1)}\odot \mathbf{h}_{i}^{(t)})||\mathbf{u}_{i}^{(t + 1)})\right)}\\ & {\mathbf{h}_{i}^{(t + 1)} = (1 - \mathbf{q}_{i}^{(t + 1)})\odot \mathbf{h}_{i}^{(t)} + \mathbf{q}_{i}^{(t + 1)}\odot \tilde{\mathbf{h}}_{i}^{(t + 1)}} \end{array}
$$

This baseline differs from the end- to- end Neighborhood model for only the graph neural network component.

# 7 Exploring the Causes of Suboptimality

The empirical evaluation shows that the graph neural network learns flows which are not optimal. With the goal of understanding this suboptimality, this section presents an exploration into the behavior of the graph neural network method. This behavior highlights some issues which lead to non- optimal solutions.

![](images/c64520b6e4fb585cdfc3d04ba9970d0736ce5c79be10340141b03f393406de19.jpg)  
Figure 17: Flows computed by the Neighborhood+Sparsemax model on two separate CB-500 problem instances.

# 7.1 Adapting to Changes in Demand

The graph neural network has the goal of adapting to different demand patterns. Figure 17 displays a example of this adaptability on the CB- 500 graph. In these two examples, the graph neural network shows the ability to coordinate the actions across nodes. This ability is specifically demonstrated by vertices 4 and 8. In the left graph, the largest sink is node 25, and node 4 has a small production value. To cover the demand for node 25, node 8 sends part of its flow along its downward edge. This action is necessary because node 4 cannot cover node 25 on its own. In the right- hand graph, node 4 is the largest source and can satisfy the demand of node 25. It is counterproductive for node 8 to also send to node 25. Node 8 sending flow along its downward edge is suboptimal for two possible reasons.

1. The flow from node 8 may travel to node 25. Along the way, it will combine with the flow from node 4, and the higher edge flow on these arcs will lead to a higher cost. Furthermore, node 4 can already cover node 25. This action by node 8 is thus incurring cost without helping satisfy demand. 
2. The flow from node 8 may come back to node 8. To meet all demand, Node 4 must send part of its flow to node 20, and node 8 is on node 4's shortest path to node 20. As this scenario involves a cycle, the action necessarily leads to a higher cost. If the flow never comes back, then node 4 must have transmitted to node 20 using a non-optimal path.

Node 8 should thus send flow along its upward edge to help cover the remaining two sinks. This strategy is the behavior adopted by the graph neural network. Node 8 executes

![](images/2aff58b563dc9735bf8d9dd18237b6dac0b33e24d8049cf2a40fe0245c957192.jpg)  
Figure 18: Example flow proportions computed by the neural network on a section of the CB-500 graph. The red path highlights the shortest path from source 15 to sink 20. The blue labels denote the source and sink demands.

contrasting strategies in these examples based on the quantities at nodes 4 and 25. These strategies highlight the graph neural network's ability to both coordinate node actions and adapt to demand patterns. The ability is encouraging because the neural network learns this behavior without ever observing ground- truth labels.

# 7.2 Causes of Suboptimal Flows

The graph neural network produces non- optimal flows for a few key reasons. The first cause of suboptimality is the overallocation of sinks. Consider the graph in figure 18. This graph is the upper region of the left- hand graph in figure 17. The labels on each edge are the flow proportions are computed by the graph neural network. From these proportions, node 15 sends  $39.6\%$  of its production value along its shortest path to the sink 20 (highlighted in red). This set of proportions results in 0.312 units of flow entering 20. This strategy is an over- allocation to the sink. Recovering from this error requires this excess flow to make its way to an unsaturated sink. This recovery generally incurs a high cost because this excess flow is now itself originating from a sink. Since the proportions are static, they are not updated based on a sink's saturation level. Therefore, vertices such as node 21 will continue to feed flow into sink 20. The problem of overallocation also leads to higher penalties is higher on larger graphs because the paths between sinks are longer. In this example, the ideal case calls for node 15 to send at most  $32.9\%$  of its source value to node 20. This figure is in contrast with the  $39.6\%$  instructed by the graph neural network. Techniques for overcoming this  $7\%$  difference are crucial to producing truly optimal flows.

The second main cause of suboptimality involves graph cycles. For an increasing cost

![](images/74e92e5176f4a64451aea50164f6408198b1bbad8056899e0fa7994726673bb7.jpg)  
Figure 19: Example of flow proportions assigned to source-sink paths which do not lead to a valid flow. The quantities on each edge represent flow proportions.

function, sending flow around a cycle leads to a higher total cost. Based on vertices 21 and 17 in figure 18, the neural network fails to completely eradicate flow cycles. During training, the neural network is never explicitly prohibited from sending flow around cycles. Instead, the model must learn to eradicate cycles because they cause higher costs. Cycles, however, play a key role in maintaining feasibility. For example, in the case the sink over- allocation in figure 18, some of the excess flow into node 20 will eventually leak out to node 17. Node 17 will then transmit some of this flow to unsaturated sinks. If cycles are completely eradicated, mistakes such as over- allocation lead to non- feasible outputs. This behavior occurs because a weakly connected flow graph can cause flow to get trapped in areas with no unsaturated sinks. In this manner, cycles are key to maintaining feasibility yet lead to suboptimal answers. This tension likely makes it difficult for the graph neural network to eliminate all cycles.

# 7.3 Mitigating Flow Cycles with Pre-computed Paths

The role occupied by cycles leads to a larger question about the proposed representation of the MCF problem. The graph neural network model proposed in this thesis learns graph flows without any explicit prevention of cycles. If the model were designed to avoid cycles, however, then the learned flows would likely be of lower cost. One possible strategy to prevent cycles is to only transmit flow along precomputed simple paths. Instead of having to learn paths directly, the model can instead focus on determining the optimal quantity of flow to send between each source and sink. Furthermore, since each precomputed path is simple, flows constrained to these paths are less likely to result in unnecessary cycles. This strategy has the further benefit of eliminating the need for an iterative flow solver.

This idea, however, induces problems regarding feasibility because the neural network can still over- allocate flow to a single sink. Figure 7.3 is such a situation. In this example, both sinks receive 0.5 units of flow, and this allocation leaves  $C$  unsaturated. This issue of feasibility can be numerically solved by projecting the neural network's output onto the subspace of feasible solutions. This idea, however, creates a break between the neural network's output and the final solution. Nevertheless, simplifying the learning task to explicitly

remove cycles should help encourage the neural network to never pass flow around a cycle.

# 8 Future Work

The task of learning graph flows presents some interesting avenues for future work. Two such extensions are discussed below.

# 8.1 Capacity Constraints

Adding support for capacity constraints makes the neural network model more applicable to real- world scenarios. One way to add this feature is to introduce a barrier term to the cost function. By incurring a high cost when an edge capacity is violated, the barrier incentivizes the neural network to abide by the capacities. A second method involves introducing slack variables and augmenting the target graph as proposed in section 4.1.2 of Bertsekas's Network Optimization [7, Chapter 4]. This approach allows the neural network model to maintain feasibility using the technique of learning flow proportions. This method, however, comes at the cost of doubling the number of nodes and edges in the target graph.

# 8.2 Transfer Learning

The model presented in this work uses random initial node embeddings. Graph flow problems, however, are highly related to simpler problems such as shortest paths. This fact is exemplified by the correlation between flow and betweenness centrality measures [31]. One approach to potentially accelerate the primal- dual learning process is to use initial embeddings which are pre- trained on tasks such as shortest paths. As algorithms for the shortest- path problem efficient, this pre- training phase can leverage ground- truth labels. The correlation between these tasks may enable the knowledge of shortest paths to transfer to learning graph flows.

# 9 Conclusion

This thesis introduces new a technique to learn uncapacitated MCFs using graph neural networks. This technique leverages a novel training method, called DualityLoss, which removes the need for labeled data and instead relies on principles of Lagrangian duality. This quality is important within the context of MCFs because existing algorithms cannot efficiently produce large labeled datasets required for training. DualityLoss uses neural networks to estimate the primal and dual optimization problems, and these networks are jointly trained with the objective of minimizing the estimated duality gap. This technique encourages the two models to reach their respective optima when the target problem exhibits strong duality and the models produce feasible answers. The end- to- end graph neural network is demonstrated by learning minimum cost flows on a set of real- world road graphs. These results show that, despite never using ground- truth labels during training, the model both learns flows which are close to being optimal and computes graphs flows far faster than existing techniques.

# 10 Proofs

This section contains the formal proofs of theorems 1 and 2. These proofs assume the target graph  $G$  is strongly connected (assumption 1). The function  $f$  is defined in equation (4). The matrix  $\mathbf{P}\in \mathbb{R}^{n\times n}$  refers to a matrix of flow proportions satisfying remark 1. The vector  $\mathbf{d}\in \mathbb{R}^{n}$  is the vector of demands under assumption 2.

# 10.1 Correctness

Theorem 1. Let  $\mathbf{X}^{*}\in \mathbb{R}_{\geq 0}^{n\times n}$  be a fixed point of  $f$  . Then,  $\mathbf{X}^{*}$  satisfies the constraints of the MCF problem in equation (1).

Proof. From the definition of a fixed point, The following relationship holds for all  $i,j\in [n]$

$$
X_{i j}^{*} = \max (P_{i j}(\sum_{k = 1}^{n}X_{k i}^{*} - d_{i}),0)
$$

This matrix is clearly non- negative. It also satisfies  $X_{i j}^{*} = 0$  if  $(i,j)\notin E$  because  $P_{i j} = 0$  if  $(i,j)\notin E$  . It remains to show that the flow constraint in equation (1c) holds for this matrix  $\mathbf{X}^{*}$

For now, let us omit the maximum operator and assume that  $\begin{array}{r}{X_{i j}^{*} = P_{i j}(\sum_{k = 1}^{n}X_{k i}^{*} - d_{i})} \end{array}$  We want to show that  $\begin{array}{r}{\sum_{k = 1}^{n}X_{k i}^{*} - \sum_{j = 1}^{n}X_{i j}^{*} - d_{i} = 0} \end{array}$  for all  $i\in [n]$  . This relationship is shown below. Step 3 relies on the fact that  $\textstyle \sum_{j = 1}^{n}P_{i j} = 1$  (true by remark 1).

$$
\begin{array}{r l}{\sum_{k = 1}^{n}X_{k i}^{*} - \sum_{j = 1}^{n}X_{i j}^{*} - d_{i} = \sum_{k = 1}^{n}X_{k i}^{*} - \left(\sum_{j = 1}^{n}P_{i j}(\sum_{q = 1}^{n}X_{q i}^{*} - d_{i})\right) - d_{i}} & {}\\ {= \sum_{k = 1}^{n}X_{k i}^{*} - \sum_{j = 1}^{n}P_{i j}\left(\sum_{q = 1}^{n}X_{q i}^{*}\right) + \sum_{j = 1}^{n}P_{i j}d_{i} - d_{i}} & {}\\ {= \sum_{k = 1}^{n}X_{k i}^{*} - \sum_{q = 1}^{n}X_{q i}^{*} + d_{i} - d_{i}} & {}\\ {= 0} & {} \end{array}
$$

To justify the exclusion of the maximum operator, we must show that  $\begin{array}{r}{P_{i j}(\sum_{k = 1}^{n}X_{k i}^{*} - d_{i})\geq 0} \end{array}$  for all  $i,j\in [n]$  . For the sake of contradiction, assume there exists some  $s,t\in [n]$  such that  $\begin{array}{r}{P_{s t}(\sum_{k = 1}^{n}X_{k t}^{*} - d_{t})< 0} \end{array}$  . Using the definition of a fixed point, we have the following.

$$
\begin{array}{r l} & {\sum_{i = 1}^{n}\sum_{j = 1}^{n}X_{i j}^{*} = \sum_{i = 1}^{n}\sum_{j = 1}^{n}f(\mathbf{X}^{*})_{i j}}\\ & {\qquad = \sum_{i = 1}^{n}\sum_{j = 1}^{n}\max \left(P_{i j}\left(\sum_{k = 1}^{n}X_{k i}^{*} - d_{i}\right),0\right)}\\ & {\qquad \geq \sum_{i = 1}^{n}\sum_{j = 1}^{n}P_{i j}\left(\sum_{k = 1}^{n}X_{k i}^{*} - d_{i}\right)} \end{array}
$$

From the construction of  $s$  and  $t$  , we have that  $\begin{array}{r}{P_{s t}(\sum_{k = 1}^{n}X_{k t}^{*} - d_{t})< 0} \end{array}$  . This fact implies that the maximum operator is strictly increasing the value of at least one  $(i,j)$  term (namely  $(s,t)$  ). Therefore, the inequality above must be strict. This observation yields the following.

$$
\begin{array}{r l}{\sum_{i = 1}^{n}\sum_{j = 1}^{n}X_{i j}^{*} > \sum_{i = 1}^{n}\sum_{j = 1}^{n}P_{i j}\left(\sum_{k = 1}^{n}X_{k i}^{*} - d_{i}\right)} & {}\\ {= \sum_{i = 1}^{n}\sum_{j = 1}^{n}P_{i j}\cdot \left(\sum_{k = 1}^{n}X_{k i}^{*}\right) - \sum_{i = 1}^{n}\sum_{j = 1}^{n}P_{i j}d_{i}} & {}\\ {= \sum_{i = 1}^{n}\left(\sum_{k = 1}^{n}X_{k i}^{*}\right)\cdot \sum_{j = 1}^{n}P_{i j} - \sum_{i = 1}^{n}d_{i}\sum_{j = 1}^{n}P_{i j}} & {}\\ {= \sum_{i = 1}^{n}\sum_{k = 1}^{n}X_{k i}^{*} - \sum_{i = 1}^{n}d_{i}} & {}\\ {= \sum_{i = 1}^{n}\sum_{k = 1}^{n}X_{k i}^{*}} & {}\\ {= \sum_{i = 1}^{n}\sum_{k = 1}^{n}X_{k i}^{*}} & {} \end{array}
$$

The fourth equality holds because  $\textstyle \sum_{j = 1}^{n}P_{i j} = 1$  (remark 10, and the final equality is true since  $\textstyle \sum_{i = 1}^{n}d_{i} = 0$  (assumption 2). The analysis above shows that  $\textstyle \sum_{i = 1}^{n}\sum_{j = 1}^{n}X_{i j}^{*}>$ $\textstyle \sum_{i = 1}^{n}\sum_{k = 1}^{n}X_{k i}^{*}$  . This statement is a contradiction because both sides are a sum over the same terms. Therefore, no such  $s,t\in [n]$  exist, so  $\begin{array}{r}{P_{i j}(\sum_{k = 1}^{n}X_{k i}^{*} - d_{i})\geq 0} \end{array}$  for all  $i,j\in [n]$  This completes the proof that  $\mathbf{X}^{*}$  satisfies all constraints of the MCF problem.

# 10.2 Continuity

Before we prove the convergence of the sequence  $(f_{t}(\mathbf{X}))_{t\in \mathbb{N}}$  , we will show that  $f$  is continuous. This property is important because, if the sequence  $(f_{t}(\mathbf{X}))_{t\in \mathbb{N}}$  converges, the continuity of  $f$  implies the existence of a fixed point of  $f$  . The continuity of  $f$  is rigorously shown below.

Proposition 2. The function  $f:\mathbb{R}_{\geq 0}^{n\times n}\to \mathbb{R}_{\geq 0}^{n\times n}$  is element- wise continuous.

Proof. Let  $\epsilon >0$  be arbitrary and let  $\mathbf{X},\mathbf{Y}\in \mathbb{R}_{\geq 0}^{n\times n}$  be matrices such that  $\| \mathbf{X} - \mathbf{Y}\|_{1}< \delta = \epsilon$  The norm  $\| \cdot \|_{1}$  is the element- wise 1- norm. We want to show that  $\| f(\mathbf{X}) - f(\mathbf{Y})\|_{1}< \epsilon$  . This fact is shown in the steps below. This analysis uses an alternative formula for the maximum,  $\begin{array}{r}{\max (a,b) = \frac{1}{2} (a + b + |a - b|)} \end{array}$  . The absolute value from  $P_{i j}$  is omitted because these values are non- negative (remark 1).

$$
\begin{array}{r l} & {\| f(\mathbf{X}) - f(\mathbf{Y})\|_{1} = \sum_{i = 1}^{n}\sum_{j = 1}^{n}|\max (P_{i j}(\sum_{k = 1}^{n}X_{k i} - d_{i}),0) - \max (P_{i j}(\sum_{k = 1}^{n}Y_{k i} - d_{i}),0)|}\\ & {\qquad = \sum_{i = 1}^{n}\sum_{j = 1}^{n}\frac{1}{2} P_{i j}\left|\sum_{k = 1}^{n}X_{k i} - d_{i} + |\sum_{k = 1}^{n}X_{k i} - d_{i}| - \sum_{k = 1}^{n}Y_{k i} + d_{i} - |\sum_{k = 1}^{n}Y_{k i} - d_{i}|\right|}\\ & {\qquad = \sum_{i = 1}^{n}\sum_{j = 1}^{n}\frac{1}{2} P_{i j}\left|\sum_{k = 1}^{n}(X_{k i} - Y_{k i}) + |\sum_{k = 1}^{n}X_{k i} - d_{i}| - |\sum_{k = 1}^{n}Y_{k i} - d_{i}|\right|} \end{array}
$$

$$
\begin{array}{r l} & {\leq \sum_{i = 1}^{n}\sum_{j = 1}^{n}\frac{1}{2} P_{i j}\left|\sum_{k = 1}^{n}(X_{k i} - Y_{k i}) + \left|\sum_{k = 1}^{n}X_{k i} - d_{i} - \sum_{k = 1}^{n}Y_{k i} + d_{i}\right|\right|}\\ & {= \sum_{i = 1}^{n}\sum_{j = 1}^{n}\frac{1}{2} P_{i j}\left|\sum_{k = 1}^{n}(X_{k i} - Y_{k i}) + \left|\sum_{k = 1}^{n}(X_{k i} - Y_{k i})\right|\right|}\\ & {\leq \sum_{i = 1}^{n}\sum_{j = 1}^{n}\frac{1}{2} P_{i j}\left|\sum_{k = 1}^{n}(X_{k i} - Y_{k i}) + \sum_{k = 1}^{n}|X_{k i} - Y_{k i}|\right|}\\ & {\leq \sum_{i = 1}^{n}\sum_{j = 1}^{n}\frac{1}{2} P_{i j}\sum_{k = 1}^{n}2|X_{k i} - Y_{k i}|\quad \mathrm{(triangle~ineq.)}}\\ & {= \sum_{i = 1}^{n}\sum_{j = 1}^{n}(X_{k i} - Y_{k i})\cdot \sum_{j = 1}^{n}P_{i j}}\\ & {\leq \sum_{i = 1}^{n}\sum_{j = 1}^{n}|X_{k i} - Y_{k i}|}\\ & {= \| \mathbf{X} - \mathbf{Y}\|_{1}< \delta = \epsilon} \end{array}
$$

Therefore,  $\| f(\mathbf{X}) - f(\mathbf{Y})\|_{1}< \epsilon$  for  $\| \mathbf{X} - \mathbf{Y}\|_{1}< \delta = \epsilon$ . This proves that  $f(\mathbf{X})$  is jointly continuous over the entire matrix  $\mathbf{X}$ . It is then a theorem that  $f$  is also separately (elementwise) continuous.

# 10.3 Convergence

This section has the goal of showing that the fixed point iteration of  $f$  converges when given a starting point of  $\mathbf{X}^{(0)} = \mathbf{0}_{n \times n}$ . We first introduce a lemma which is used in the formal proof of convergence. The matrix  $\mathbf{X}^{(t)}$  refers to  $f_{t}(\mathbf{X}^{(0)}$ .

Lemma 1. For all  $t \geq 0$  and  $i, j \in [n]$ ,  $X_{ij}^{(t + 1)} \geq X_{ij}^{(t)}$ .

Proof. Let  $\mathcal{P}(t)$  be the proposition that  $X_{ij}^{(t + 1)} \geq X_{ij}^{(t)}$  for all  $i, j \in [n]$ . The lemma is equivalent to the statement:  $\mathcal{P}(t)$  holds for all  $t \geq 0$ . We proceed by induction on  $t$ . Base Case: For  $t = 0$ , we have the following for all  $i, j \in [n]$ .

$$
X_{ij}^{(1)} = \max \left(0, P_{ij} \sum_{k = 1}^{n} X_{ki}^{(0)} - P_{ij} d_{i}\right) \geq 0
$$

As  $\mathbf{X}^{(0)} = \mathbf{0}_{n \times n}$ ,  $X_{ij}^{(0)} = 0$ . We have thus shown that  $X_{ij}^{(1)} \geq X_{ij}^{(0)}$ , so  $\mathcal{P}(0)$  holds. Inductive Step: Assume that  $\mathcal{P}(t)$  is true. We want to show  $\mathcal{P}(t + 1)$ . Using the definition

$f$ , we have the following.

$$
\begin{array}{r l} & {X_{i j}^{(t + 1)} = \max \left(0,P_{i j}\sum_{k = 1}^{n}X_{k i}^{(t)} - P_{i j}d_{i}\right)}\\ & {\qquad \geq \max \left(0,P_{i j}\sum_{k = 1}^{n}X_{k i}^{(t - 1)} - P_{i j}d_{i}\right)}\\ & {\qquad = X_{i j}^{(t)}} \end{array}
$$

The second step comes from applying the induction hypothesis. This relationship proves that  $\mathcal{P}(t + 1)$  holds.  $\square$

To prove the convergence of the the sequence  $(f(\mathbf{X}^{(0)}))_{t\in \mathbb{N}}$  , we establish a link between algorithm 1 and Markov chains. We note that algorithm 1 is the implementation of the sequence  $(f(\mathbf{X}^{(0)}))_{t\in \mathbb{N}}$

Consider a variant of algorithm 1 presented in algorithm 3. Algorithm 3 is based on a Markov- chain- like transformation. This property makes it simpler to analyze. Algorithms 1 and 3 are logically equivalent; rather than consider the flow on each edge, algorithm 3 uses the total inflow to each node for only the current timestep. To formally prove the relationship between the two algorithms, we will show that  $\sum_{k = 1}^{n} X_{ki}^{(t)} - \sum_{k = 1}^{n} X_{ki}^{(t - 1)} = y_{i}^{(t)}$  for all nodes  $i \in [n]$ .

# Algorithm 3 Markov-chain-based Flow Computation

1: procedure Markov- Chain- Flow  $(\mathbf{P}, \mathbf{d}, n, \epsilon)$

2:  $\mathbf{y}^{(0)} \leftarrow [- d_{i} \text{for} i \in [n] \text{if} d_{i} < 0 \text{else} 0]$

3:  $\mathbf{r}^{(0)} \leftarrow \mathbf{0}_{n}$

4:  $\mathbf{r}^{(1)} \leftarrow [d_{i} \text{for} i \in [n] \text{if} d_{i} > 0 \text{else} 0]$

$\vartriangleright$  Remaining sink demands

5:  $t \leftarrow 1$

6: while  $\| \mathbf{y}^{(t - 1)} \|_{1} > \epsilon \text{do}$

7:  $\tilde{\mathbf{y}}^{(t - 1)} \leftarrow \max (\mathbf{y}^{(t - 1)} - \mathbf{r}^{(t - 1)}, 0)$

8:  $\mathbf{y}^{(t)} \leftarrow \mathbf{P}^{\top} \cdot \tilde{\mathbf{y}}^{(t - 1)}$

9:  $\mathbf{r}^{(t + 1)} \leftarrow \max (\mathbf{r}^{(t)} - \mathbf{y}^{(t)}, 0)$

10:  $t \leftarrow t + 1$

11: return  $\sum_{s = 0}^{s} \mathbf{y}^{(s)}$

Proposition 3. For all iterations  $t > 0$ ,  $\sum_{k = 1}^{n} X_{ki}^{(t)} - \sum_{k = 1}^{n} X_{ki}^{(t - 1)} = y_{i}^{(t)}$  for all nodes  $i \in [n]$  where  $\mathbf{X}^{(t)}$  is the matrix in algorithm 1 and  $\mathbf{y}^{(t)}$  is the vector in algorithm 3.

Proof. We proceed by strong induction on the number of completed iterations,  $t > 0$ . Base Case: Let  $t = 1$ , and let  $i \in [n]$  be arbitrary. By the definition of  $\mathbf{X}^{(0)}$ , we have

$\sum_{k = 1}^{n}X_{k\ell}^{(0)} = 0$  for all  $\ell \in [n]$ . This statement implies the following.

$$
\begin{array}{rl} & {\sum_{k = 1}^{n}X_{ki}^{(1)} = \sum_{k = 1}^{n}\max (P_{ki}\sum_{j = 1}^{n}X_{jk}^{(0)} - P_{ki}d_k,0)}\\ & {\qquad = \sum_{k = 1}^{n}\max (-P_{ki}d_k,0)} \end{array}
$$

Similarly, from algorithm 3, we have the following.

$$
y_{i}^{(1)} = \sum_{k = 1}^{n}P_{ki}\tilde{y}_{k}^{(0)}
$$

From the initialization of  $\mathbf{y}^{(0)}$  and  $\mathbf{r}^{(0)}$ $\tilde{y}_{k}^{(0)} = \max (- d_{k},0)$ . Thus,  $y_{i}^{(1)} = \sum_{k = 1}^{n}\max (- P_{ki}d_{k},0) = \sum_{k = 1}^{n}X_{ki}^{(1)}$  for all  $i\in [n]$ .

Inductive Step: Assume the proposition holds for all iterations less than or equal to  $t$ . We want to show that it also holds for iteration  $t + 1$ . Let node  $i\in [n]$  be arbitrary. Using the definition of algorithm 1, we have the following. Step 2 uses the fact that  $\max (a - b,0) = a - \min (a,b)$ .

$$
\begin{array}{r l} & {\sum_{k = 1}^{n}X_{k i}^{(t + 1)} - X_{k i}^{(t)} = \sum_{k = 1}^{n}\max (P_{k i}\sum_{j = 1}^{n}X_{j k}^{(t)} - P_{k i}d_{k},0) - \max (P_{k i}\sum_{j = 1}^{n}X_{j k}^{(t - 1)} - P_{k i}d_{k},0)}\\ & {\qquad = \sum_{k = 1}^{n}P_{k i}\left(\sum_{j = 1}^{n}X_{j k}^{(t)} - \min (\sum_{j = 1}^{n}X_{j k}^{(t)},d_{k}) - \sum_{j = 1}^{n}X_{j k}^{(t - 1)} + \min (\sum_{j = 1}^{n}X_{j k}^{(t - 1)},d_{k})\right)}\\ & {\qquad = \sum_{k = 1}^{n}P_{k i}\left(\sum_{j = 1}^{n}(X_{j k}^{(t)} - X_{j k}^{(t - 1)}) - \min (\sum_{j = 1}^{n}X_{j k}^{(t)},d_{k}) + \min (\sum_{j = 1}^{n}X_{j k}^{(t - 1)},d_{k})\right)}\\ & {\qquad = \sum_{k = 1}^{n}P_{k i}\left(y_{k}^{(t)} - (\min (\sum_{j = 1}^{n}X_{j k}^{(t)},d_{k}) - \min (\sum_{j = 1}^{n}X_{j k}^{(t - 1)},d_{k}))\right)} \end{array} \tag{9}
$$

The last step comes from applying the induction hypothesis. We claim the following.

$$
\tilde{y}_{k}^{(t)} = y_{k}^{(t)} - (\min (\sum_{j = 1}^{n}X_{jk}^{(t)},d_{k}) - \min (\sum_{j = 1}^{n}X_{jk}^{(t - 1)},d_{k})) \tag{10}
$$

where  $\tilde{y}_{k}^{(t)} = \max (y_{k}^{(t)} - r_{k}^{(t)},0)$  is defined in algorithm 3. To gain insight into why to this relationship holds, we further specify the residual value  $r_{k}^{(t)}$  below. The first equality comes

from the definition of  $\mathbf{r}^{(t)}$  and the second equality comes from the strong induction hypothesis.

$$
\begin{array}{r l} & {r_{k}^{(t)} = \max \left(d_{k} - \sum_{s = 0}^{t - 1}y_{k}^{(s)},0\right)}\\ & {\quad = \max \left(d_{k} - \left(\sum_{j = 0}^{n}X_{j k}^{(t - 1)} - \sum_{j = 0}^{n}X_{j k}^{(t - 2)} + \sum_{j = 0}^{n}X_{j k}^{(t - 2)} - \ldots +\sum_{j = 0}^{n}X_{j k}^{(1)} - \sum_{j = 0}^{n}X_{j k}^{(0)}\right)\right.}\\ & {\quad = \max \left((d_{k} - \sum_{j = 0}^{n}(X_{j k}^{(t - 1)} + X_{j k}^{(0)}),0\right)}\\ & {\quad = \max \left(d_{k} - \sum_{j = 0}^{n}X_{j k}^{(t - 1)},0\right)} \end{array}
$$

The last step holds because  $\mathbf{X}^{(0)} = \mathbf{0}_{n \times n}$ . We turn our attention back to proving equation 10. There are four possible cases.

1. The first possibility is that the left minimum is greater than the right minimum. This case is impossible as it implies that the flow along an edge decreases with time and thus violates lemma 1.

2. The second scenario is that both minimums result in  $d_{k}$ . Thus,

$$
\min (\sum_{j = 1}^{n} X_{jk}^{(t)}, d_{k}) - \min (\sum_{j = 1}^{n} X_{jk}^{(t - 1)}, d_{k}) = 0
$$

This fact implies that  $\sum_{j = 1}^{n} X_{jk}^{(t - 1)} > d_{k}$ , so  $r_{k}^{(t)} = 0$ . Thus,  $\tilde{y}_{k}^{(t)} = y_{k}^{(t)}$  by the definition of  $\tilde{y}_{k}^{(t)}$ . Using the assumption for this scenario, this equality proves equation (10) in this case.

3. The third case is that the left minimum results in  $d_{k}$  while the right is  $\sum_{j = 1}^{k} X_{jk}^{(t - 1)}$ . Equation (10) results in  $y_{k}^{(t)} - d_{k} + \sum_{j = 1}^{k} X_{jk}^{(t - 1)}$ . This term is exactly  $r_{k}^{(t)}$  because  $d_{k} \geq \sum_{j = 1}^{k} X_{jk}^{(t - 1)}$  by assumption. Furthermore,  $y_{k}^{(t)} \geq r_{k}^{(t)}$  by the following relationship.

$$
\begin{array}{l}{{y_{k}^{(t)}-r_{k}^{(t)}=\sum_{j=1}^{k}X_{j k}^{(t)}-\sum_{j=1}^{k}X_{j k}^{(t-1)}-d_{k}+\sum_{j=1}^{k}X_{j k}^{(t-1)}}}\\ {{=\sum_{j=1}^{k}X_{j k}^{(t)}-d_{k}\geq0\quad(\mathrm{by~assumption~of~this~case})}}\end{array}
$$

Since  $\tilde{y}_{k}^{(t)} = \max (y_{k}^{(t)} - r_{k}^{(t)}, 0) = y_{k}^{(t)} - r_{k}^{(t)}$ , we have shown that  $\tilde{y}_{k}^{(t)} = y_{k}^{(t)} - d_{k} + \sum_{j = 1}^{k} X_{jk}^{(t - 1)}$  because  $r_{k}^{(t)} = d_{k} - \sum_{j = 1}^{k} X_{jk}^{(t - 1)}$  in this scenario. This proves equation (10) for this case.

4. The final scenario occurs when neither minimum equals  $d_{k}$ . The right-hand-side of equation (10) results in the following.

$$
y_{k}^{(t)} - (\sum_{j = 1}^{n}X_{jk}^{(t)} - \sum_{j = 1}^{n}X_{jk}^{(t - 1)}) = 0
$$

This term equals zero due to the induction hypothesis. To prove the claim, it remains to show that  $y_{k}^{(t)} - r_{k}^{(t)} \leq 0$ . This property is displayed below.

$$
\begin{array}{l}{{y_{k}^{(t)}-r_{k}^{(t)}=\sum_{j=1}^{k}X_{j k}^{(t)}-\sum_{j=1}^{k}X_{j k}^{(t-1)}-d_{k}+\sum_{j=1}^{k}X_{j k}^{(t-1)}}}\\ {{=\sum_{j=1}^{k}X_{j k}^{(t)}-d_{k}\leq0\quad(\mathrm{by~the~assumption~of~this~case})}}\end{array}
$$

Thus, equation (10) holds in this final case.

Through these cases, we have proven that equation (10) holds. Therefore, using equation 9, we obtain the following.

$$
\begin{array}{r l r}{{\sum_{k=1}^{n}(X_{k i}^{(t+1)}-X_{k i}^{(t)})=\sum_{k=1}^{n}P_{k i}\tilde{y}^{(t)}}}\\ &{}&{=\mathbf{P}^{\top}\cdot\tilde{\mathbf{y}}^{(t)}}\\ &{}&{=\mathbf{y}^{(t+1)}}\end{array}
$$

The last equality comes from the definition of  $\mathbf{y}^{(t + 1)}$  on line 8 of algorithm 3. This completes the inductive step and proves that  $\mathbf{y}^{(t)} = \sum_{k = 1}^{n} X_{ki}^{(t)} - X_{ki}^{(t - 1)}$  for all  $t > 0$ .

Before we formally prove the termination of algorithm 1, we prove two final lemmas.

Lemma 2. Let  $\mathbf{B} \in \mathbb{R}_{\geq 0}^{n \times n}$  be a Markov chain such that  $\sum_{i = 1}^{n} B_{ij} = 1$  for all  $j \in [n]$ . Then, for any  $\mathbf{x} \in \mathbb{R}_{\geq 0}^{n}$ ,  $\| \mathbf{B} \mathbf{x} \|_{1} = \| \mathbf{x} \|_{1}$ .

Proof. From the definition of matrix multiplication, we have the following.

$$
\begin{array}{r l r}{{\|{\bf B x}\|_{1}=\sum_{i=1}^{n}\sum_{j=1}^{n}B_{i j}|x_{j}|}}\\ &{}&{=\sum_{j=1}^{n}\sum_{i=1}^{n}B_{i j}|x_{j}|}\\ &{}&{=\sum_{j=1}^{n}|x_{j}|\sum_{i=1}^{n}B_{i j}}\\ &{}&{=\sum_{j=1}^{n}|x_{j}|=\|{\bf x}\|_{1}}\end{array}
$$

Lemma 3. For all iterations  $t \geq 1$  of algorithm 3,  $\| \mathbf{y}^{(t)}\| = \| \mathbf{r}^{(t)}\|_{1}$ .

Proof. We proceed by induction on  $t > 0$

Base Case: Let  $t = 1$  .As  $\mathbf{P}^{\top}$  is an irreducible Markov chain (remark 1) and  $\mathbf{r}^{(0)} = \mathbf{0}$  , we know that  $\| \mathbf{y}^{(1)}\|_{1} = \| \mathbf{P}^{\top}\mathbf{y}^{(0)}\|_{1} = \| \mathbf{y}^{(0)}\|_{1}$  by lemma 2. By construction  $\| \mathbf{y}^{(0)}\|_{1} = D$  and  $\| \mathbf{r}^{(1)}\|_{1} = D$  , so  $\| \mathbf{y}^{(1)}\|_{1} = \| \mathbf{r}^{(1)}\|_{1}$  . This completes the proof of the base case.

Inductive Step: Let the lemma hold for iteration  $t$ . Using the updates for both  $\mathbf{y}^{(t + 1)}$  and  $\mathbf{r}^{(t + 1)}$ , we get the following. We use the non- negativity of all terms to omit absolute values for the 1- norm.

$$
\begin{array}{r l}{\| \mathbf{y}^{(t + 1)}\|_{1} = \| \mathbf{P}\mathbf{y}^{(t + 1)}\|_{1}}\\ {= \| \mathbf{y}^{(t + 1)}\|_{1}} & {(\mathrm{Lemma~2})}\\ {= \sum_{i = 1}^{n}\max (y_{i}^{(t)} - r_{i}^{(t)},0)}\\ {= \sum_{i = 1}^{n}\frac{1}{2} (y_{i}^{(t)} - r_{i}^{(t)} + |y_{i}^{(t)} - r_{i}^{(t)}|)} & {(\mathrm{alternative~formula~for~max}(a,b))}\\ {= \sum_{i = 1}^{n}y_{i}^{(t)} - \frac{1}{2} (y_{i}^{(t)} + r_{i}^{(t)} - |y_{i}^{(t)} - r_{i}^{(t)}|)}\\ {= \sum_{i = 1}^{n}y_{i}^{(t)} - \frac{1}{2} (y_{i}^{(t)} + r_{i}^{(t)} - |y_{i}^{(t)} - r_{i}^{(t)}|)}\\ {= \sum_{i = 1}^{n}y_{i}^{(t)} - \sum_{i = 1}^{n}\frac{1}{2} (y_{i}^{(t)} + r_{i}^{(t)} - |y_{i}^{(t)} - r_{i}^{(t)}|)}\\ {= \sum_{i = 1}^{n}r_{i}^{(t)} - \sum_{i = 1}^{n}\frac{1}{2} (y_{i}^{(t)} + r_{i}^{(t)} - |y_{i}^{(t)} - r_{i}^{(t)}|)}\\ {= \sum_{i = 1}^{n}r_{i}^{(t)} - \frac{1}{2} (y_{i}^{(t)} + r_{i}^{(t)} - |y_{i}^{(t)} - r_{i}^{(t)}|)}\\ {= \sum_{i = 1}^{n}\frac{1}{2} (-y_{i}^{(t)} + r_{i}^{(t)} + |y_{i}^{(t)} - r_{i}^{(t)}|)}\\ {= \sum_{i = 1}^{n}\frac{1}{2} (-y_{i}^{(t)} + r_{i}^{(t)} + |y_{i}^{(t)} - r_{i}^{(t)}|)}\\ {= \sum_{i = 1}^{n}\max (r_{i}^{(t)} - y_{i}^{(t)},0)}\\ {= \| \mathbf{r}^{(t + 1)}\|_{1}} \end{array}
$$

The last step holds because all terms are non- negative. This shows that the norm of  $\mathbf{y}$  is equal to the norm of the residual  $\mathbf{r}$  for all iterations.  $\square$

With these results in hand, we prove that both algorithms terminate for any  $\epsilon > 0$ .

Proposition 4. For any  $\epsilon > 0$ , algorithm 3 will terminate. This result implies that algorithm 1 will also terminate for any  $\epsilon > 0$ .

Proof. To prove that algorithm 3 terminates, it suffices to prove that  $\| \mathbf{y}^{(t)}\|_{1} \to 0$  as  $t \to \infty$ . Let  $t_{0}$  be some iteration such that there exists an  $i \in [n]$  where  $y_{i}^{(t_{0})} > 0$ . If no such  $t_{0}$  exits, then the algorithm must have converged because  $\mathbf{y}^{(t)}$  is non- negative. Let  $S$  be the set of

all nodes  $j\in [n]$  such that  $r_{j}^{(t_{0})} > 0$  . By lemma 3 and the non- negativity of  $\mathbf{r}^{(t_{0})}$  , we know that  $S$  is nonempty. Let  $t_{1}\geq 0$  be the smallest value such that  $((\mathbf{P}^{\top})^{(t_{1}}\tilde{\mathbf{y}}^{(t_{0})})_{j} > 0$  for some  $j\in S$  . Such a value  $t_{1}$  must exist because  $\mathbf{P}^{\top}$  is an irreducible Markov chain. We can appeal to this irreducibility because, by construction, the system behaves like a Markov chain for the iterations between  $t_{0}$  and  $t_{1}$  . This appeal holds by the following argument. Since  $y_{k}^{(t)} = 0$  for all  $k\in S$  and  $t\in [t_{0},t_{0} + t_{1}]$  , lines 7 and 9 of algorithm 3 make no changes to either  $\mathbf{y}^{(t)}$  or  $\mathbf{r}^{(t)}$  for all  $t\in [t_{0},t_{0} + t_{1}]$  . This leaves only the transformation  $\mathbf{P}^{\top}$  within this timeframe.

After iteration  $t_{0} + t_{1} + 1$  , we know that  $\| \mathbf{y}^{(t_{0} + t_{1} + 1)}\|_{1}< \| \mathbf{y}^{(t_{0})}\|_{1}$  because both  $\mathbf{y}_{j}^{(t_{0} + t_{1})}$  and  $\mathbf{r}_{j}^{(t_{0} + t_{1})}$  are strictly positive by construction. At this point, we can repeat the above process to further decrease the 1- norm of both the inflow vector  $\mathbf{y}^{(t)}$  and  $\mathbf{r}^{(t)}$  . Now, let  $(s_{k})_{k\in \mathbb{N}}$  be the sequence of time steps such that both  $r_{j}^{(s_{k})}$  and  $y^{(s_{k})}$  are greater than zero for some  $j\in [n]$  We have shown that the sequence  $\| \mathbf{y}^{s_{k}}\|_{1}$  is strictly decreasing. Furthermore, it is bounded below by zero due to the non- negativity of the 1- norm. Therefore,  $(\| \mathbf{y}^{(s_{k})}\|_{1})_{s\in \mathbb{N}}$  is a strictly decreasing sequence which is bounded below, so  $\| \mathbf{y}^{(s_{k})}\|_{1}\to 0$  as  $k\rightarrow \infty$  . Furthermore, we know that  $(\| \mathbf{y}^{t}\|_{1})_{t\in \mathbb{N}}$  is non- increasing since line 7 of algorithm 3 can only decrease the norm of  $\mathbf{y}^{(t)}$  . The norm is preserved by the transformation on line 8 due to lemma 2. Thus, the limit of the convergent subsequence  $(\| \mathbf{y}^{(s_{k})}\|_{1})_{s\in \mathbb{N}}$  implies that  $\| \mathbf{y}^{t}\|_{1}\to 0$  as  $t\to \infty$  as well. This statement proves that algorithm 3 terminates for any  $\epsilon >0$

We now use the termination of algorithm 3 to conclude that algorithm 1 terminates as well. We can use proposition 3 to obtain the following. We can omit the absolute values due to lemma 1.

$$
\begin{array}{r l r}{{\|{\bf X}^{(t)}-{\bf X}^{(t-1)}\|_{1}=\sum_{j=1}^{n}\sum_{j=1}^{n}X_{i j}^{(t)}-X_{i j}^{(t-1)}}}\\ &{}&{=\sum_{j=1}^{n}\sum_{i=1}^{n}X_{i j}^{(t)}-X_{i j}^{(t-1)}}\\ &{}&{=\sum_{j=1}^{n}y_{j}^{(t)}=\|{\bf y}^{(t)}\|_{1}}\end{array}
$$

Using the fact that  $\| y_{j}^{(t)}\|_{1}\to 0$  as  $t\to \infty$  , we can conclude that  $\| \mathbf{X}^{(t)} - \mathbf{X}^{(t - 1)}\|_{1}\to 0$  as  $t\to \infty$  . This statement shows that the loop in algorithm 1 will terminate for any  $\epsilon >0$  .

Proposition 4 proves that algorithm 1 will terminate. It does not, however, guarantee that a limit exists for the sequence of flow matrices. Intuitively, we have only shown that the distance between consecutive elements in the fixed point iteration goes to zero; this fact does not guarantee that a limit exists. The convergence of this sequence is stated and proven in the theorem below.

Theorem 2. The sequence  $(f_{t}(\mathbf{0}_{n\times n}))_{t\in \mathbb{N}}$  converges to a fixed point. This statement implies that the sequence  $(\mathbf{X}^{(t)})_{t\in \mathbb{N}}$  converges element- wise to a fixed point, where each  $\mathbf{X}^{(t)}$  is defined by algorithm 1 for  $t\geq 0$  .

Proof. We will prove the theorem by first showing that the series  $\textstyle \sum_{t = 0}^{\infty}\mathbf{y}^{(t)}$  converges elementwise. The convergence of  $\textstyle \sum_{t = 0}^{\infty}\mathbf{y}^{(t)}$  will be displayed by a comparison to a geometric series.

Let  $t_{0}\in \mathbb{N}$  be the first iteration for which the norm of  $\mathbf{y}^{(t)}$  decreases. This decrease happens when  $\mathbf{y}^{(t_{0})}$  indicates inflow into some sink node which has unmet demand. For all  $t< t_{0}$  each element of  $\mathbf{y}^{(t)}$  is upper bounded by the total flow production,  $D$  , since  $\| \mathbf{y}^{(t)}\|_{1} = D$  and  $\mathbf{y}^{(t)}$  is non- negative. Furthermore,  $t_{0}$  itself is bounded above by  $m$  , the number of edges in the graph. This bound holds because  $t_{0}$  is equal to the (unweighted) shortest path length between any source and sink with unmet demand. Clearly,  $m = |E|$  is the upper bound for a simple path in the graph  $G$  . We claim that, every  $m$  iterations, the norm of  $\mathbf{y}^{(t)}$  decreases by at least a factor of  $(1 - r)$  where  $\begin{array}{r}{r = \prod_{(i,j)\in E}P_{i j}} \end{array}$  . In other words,  $\| \mathbf{y}^{(k m)}\|_{1}\leq D(1 - r)^{k}$  for  $k\in \mathbb{Z}_{\geq 0}$  . As a note  $(1 - r)< 0$  because each  $P_{i j} > 0$  for  $(i,j)\in E$  by remark 1. We prove this claim by induction on  $k\geq 0$  .

Base Case: Let  $k = 0$  . By definition,  $\| \mathbf{y}^{(0)}\|_{1} = D$  . Thus, the claim holds for  $k = 0$  . Inductive Step: Assume the claim holds for  $k$  . We want to show that it also holds for  $k + 1$  . Let  $S = \{i\in [n] \mid d_{i} > 0\}$  be the set of all sinks. Consider the behavior of the algorithm between iterations  $m k$  and  $m(k + 1)$  . Let node  $j \in [n]$  be such that  $j$  has more incoming flow than outgoing flow at iteration  $m$  . Formally,  $y_{j}^{(k m)} > 0$  . Then,  $j$  has a path  $\gamma$  of length at most  $m$  to some sink  $i\in S$  with unmet demand. Such a path is guaranteed by the irreducibility of  $\mathbf{P}$  . The proportion of flow which is transferred to  $i\in S$  along  $\gamma$  is  $y_{j}^{(k m)}\cdot \prod_{(u,v)\in \gamma}B_{u v}$  . The proportion is nonzero by remark 1. This process occurs for all  $j\in [n]$  such that  $y_{j}^{(k m)} > 0$  . As  $\mathbf{y}_{(k m)}$  is non- negative, the set of all such  $j$  is correspond to a total flow of at most  $D(1 - r)^{k}$  by the induction hypothesis. Thus, in the worst case, all of the flow is concentrated on a single node which is  $m$  hops away from the nearest sink. This upper- bound “path” retains exactly  $\textstyle \prod_{(i,j)\in E}P_{i j} = r$  fraction of the given flow. Thus, by iteration  $m(k + 1)$  , the total unmet demand must have decreased by at least  $r D(1 - r)^{k}$  . Therefore,  $\| \mathbf{y}^{(m(k + 1))}\|_{1}\leq D(1 - r)^{k} - r D(1 - r)^{k} = D(1 - r)^{k + 1}$  . This completes the inductive proof.

We have shown that  $\| \mathbf{y}^{(k m)}\|_{1}\leq D(1 - r)^{k}$  for all  $k\geq 0$  . As the vector  $\mathbf{y}^{(t)}$  is non- negative, we have that each element of  $\mathbf{y}^{(k m)}$  is at most  $D(1 - r)^{k}$  as well. The sequence  $(y_{i}^{(t)})_{t\in \mathbb{N}}$  for  $i\in [n]$  is then bounded above by the sequence  $(a_{k})_{k\in \mathbb{N}}$  shown below.

$$
D,D,\ldots ,D,D(1 - r),D(1 - r),\ldots ,D(1 - r),D(1 - r)^{2},D(1 - r)^{2},\ldots
$$

Each term is repeated  $m$  times. We claim that the series  $\textstyle \sum_{k = 0}^{\infty}a_{k}$  converges. We can redefine the series as  $\begin{array}{r}{\lim_{N\to \infty}\sum_{k = 0}^{N}a_{k} = \lim_{N\to \infty}\sum_{k = 0}^{N}m D(1 - r)^{k}} \end{array}$  . Combining terms in this manner is possible because the number of repeated terms is finite. The limit  $\begin{array}{r}{\lim_{N\to \infty}\sum_{k = 0}^{N}m D(1 - } \end{array}$ $r)^{k}$  exists because it is a geometric series with ratio  $(1 - r)< 1$  . Therefore,  $\textstyle \sum_{k = 0}^{\infty}y_{i}^{(t)}$  converges for all  $i\in [n]$  by comparison with  $\textstyle \sum_{k = 0}^{\infty}a_{k}$  . Note that since each term is non- negative, this series converges absolutely.

Using the proof of proposition 3, we know that  $\begin{array}{r}{\sum_{t = 0}^{N}y_{i}^{(t)} = \sum_{k = 1}^{n}X_{k i}^{(N)}} \end{array}$  . Therefore, the existence of  $\begin{array}{r}{\lim_{N\to \infty}\sum_{t = 0}^{N}y_{i}^{(t)}} \end{array}$  means that  $\begin{array}{r}{\lim_{N\to \infty}\sum_{k = 1}^{n}X_{k i}^{(N)}} \end{array}$  must exist as well. We will call this limit  $L_{i}$  . Since each  $\mathbf{X}^{(t)}$  is non- negative, the summation over  $k$  is finite, and the sequence of these matrices is element- wise non- decreasing by lemma 1, we have that each  $X_{k i}^{(t)}\leq L_{i}$  for all  $t\geq 0$  and  $i,k\in [n]$  . Thus, each  $(X_{k i}^{(t)})_{t\in \mathbb{N}}$  is a monotone sequence which is bounded above.

This fact implies that each  $(X_{ki}^{(t)})_{t\in \mathbb{N}}$  converges, so the sequence of these matrices converges element- wise. We have thus shown that the sequence  $(f_{t}(\mathbf{0}_{n\times n}))_{t\in \mathbb{N}}$  converges element- wise to some matrix  $\mathbf{X}^{*}$ . Since  $f$  is continuous by proposition 2, the matrix  $\mathbf{X}^{*}$  must be a fixed point of  $f$ . This statement means that the sequence of matrices in algorithm 1 converges element- wise to a fixed point.  $\square$

In summary, these proofs have shown that the sequence formed by repeatedly applying  $f$  is guaranteed to converge to a non- negative fixed point which is a valid flow. Therefore, algorithm 1 converges to a valid flow.

# References

[1] Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. Tensorflow: A system for large- scale machine learning. In 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI 16), pages 265–283, 2016. [2] Sami Abu- El- Haija, Amol Kapoor, Bryan Perozzi, and Joonseok Lee. N- GCN: Multi- scale graph convolution for semi- supervised node classification. arXiv preprint arXiv:1802.08888, 2018. [3] Luis B Almeida. A learning rule for asynchronous perceptrons with feedback in a combinatorial environment. In Proceedings, 1st First International Conference on Neural Networks, volume 2, pages 609–618. IEEE, 1987. [4] Martin Arjovsky and Léon Bottou. Towards principled methods for training generative adversarial networks. arXiv preprint arXiv:1701.04862, 2017. [5] Martin Arjovsky, Soumith Chintala, and Léon Bottou. Wasserstein generative adversarial networks. In International Conference on Machine Learning, pages 214–223, 2017. [6] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014. [7] Dimitri P Bertsekas. Network optimization: continuous and discrete models. Athena Scientific Belmont, 1998. [8] Dimitri P Bertsekas and Paul Tseng. Relaxation methods for minimum cost ordinary and generalized network flow problems. Operations Research, 36(1):93–114, 1988. [9] Geoff Boeing. OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. Computers, Environment and Urban Systems, 65:126–139, 2017. [10] Richard H Byrd, Robert B Schnabel, and Gerald A Shultz. A trust region algorithm for nonlinearly constrained optimization. SIAM Journal on Numerical Analysis, 24(5):1152–1170, 1987. [11] Kyunghyun Cho, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. Learning phrase representations using RNN encoder- decoder for statistical machine translation. arXiv preprint arXiv:1406.1078, 2014. [12] William H Cunningham. A network simplex method. Mathematical Programming, 11(1):105–116, 1976. [13] Andrew V Goldberg. An efficient implementation of a scaling minimum- cost flow algorithm. Journal of algorithms, 22(1):1–29, 1997.

[14] Ian Goodfellow, Jean Pouget- Abadie, Mehdi Mirza, Bing Xu, David Warde- Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, pages 2672- 2680, 2014. [15] Marco Gori, Gabriele Monfardini, and Franco Scarselli. A new model for learning in graph domains. In Proceedings. 2005 IEEE International Joint Conference on Neural Networks, 2005. , volume 2, pages 729- 734. IEEE, 2005. [16] Geoffrey M Guisewite and Panos M Pardalos. Minimum concave- cost network flow problems: Applications, complexity, and algorithms. Annals of Operations Research, 25(1):75- 99, 1990. [17] Aric Hagberg, Pieter Swart, and Daniel S Chult. Exploring network structure, dynamics, and function using NetworkX. Technical report, Los Alamos National Lab.(LANL), Los Alamos, NM (United States), 2008. [18] Will Hamilton, Zhitao Ying, and Jure Leskovec. Inductive representation learning on large graphs. In Advances in Neural Information Processing Systems, pages 1024- 1034, 2017. [19] Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky. Neural networks for machine learning lecture 6a overview of mini- batch gradient descent.[20] Dorit S Hochbaum. Lower and upper bounds for the allocation problem and other nonlinear optimization problems. Mathematics of Operations Research, 19(2):390- 409, 1994. [21] Eric Jones, Travis Oliphant, Pearu Peterson, et al. SciPy: Open source scientific tools for Python, 2001- .[22] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014. [23] Thomas N Kipf and Max Welling. Semi- supervised classification with graph convolutional networks. arXiv preprint arXiv:1609.02907, 2016. [24] Dieter Kraft. A software package for sequential quadratic programming. Forschungsbericht- Deutsche Forschungs- und Versuchsanstalt fur Luft- und Raumfahrt, 1988. [25] Yujia Li, Daniel Tarlow, Marc Brockschmidt, and Richard Zemel. Gated graph sequence neural networks. arXiv preprint arXiv:1511.05493, 2015. [26] Renjie Liao, Marc Brockschmidt, Daniel Tarlow, Alexander L Gaunt, Raquel Urtasun, and Richard Zemel. Graph partition neural networks for semi- supervised classification. arXiv preprint arXiv:1803.06272, 2018. [27] Bing Liu. Route finding by using knowledge about the road network. IEEE transactions on systems, man, and cybernetics- part A: Systems and humans, 27(4):436- 448, 1997.

[28] Ilya Loshchilov and Frank Hutter. Online batch selection for faster training of neural networks. arXiv preprint arXiv:1511.06343, 2015. [29] Andre Martins and Ramon Astudillo. From softmax to sparsemax: A sparse model of attention and multi- label classification. In International Conference on Machine Learning, pages 1614–1623, 2016. [30] Marta SR Monteiro, Dalila BMM Fontes, and Fernando ACC Fontes. An ant colony optimization algorithm to solve the minimum cost network flow problem with concave cost functions. In Proceedings of the 13th annual conference on Genetic and evolutionary computation, pages 139–146. ACM, 2011. [31] Mark EJ Newman. A measure of betweenness centrality based on random walks. Social networks, 27(1):39–54, 2005. [32] OpenStreetMap contributors. Planet dump retrieved from https://planet.osm.org .[33] Fernando J Pineda. Generalization of back- propagation to recurrent neural networks. Physical review letters, 59(19):2229, 1987. [34] Marcelo OR Prates, Pedro HC Avelar, Henrique Lemos, Luis Lamb, and Moshe Vardi. Learning to solve NP- Complete problems- A graph neural network for the decision TSP. arXiv preprint arXiv:1809.02721, 2018. [35] Sebastian Ruder. An overview of gradient descent optimization algorithms. arXiv preprint arXiv:1609.04747, 2016. [36] Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, and Gabriele Monfardini. The graph neural network model. IEEE Transactions on Neural Networks, 20(1):61–80, 2009. [37] Daniel Selsam, Matthew Lamm, Benedikt B¨unz, Percy Liang, Leonardo de Moura, and David L Dill. Learning a SAT solver from single- bit supervision. arXiv preprint arXiv:1802.03685, 2018. [38] Shai Shalev- Shwartz and Yoram Singer. Online learning meets optimization in the dual. In International Conference on Computational Learning Theory, pages 423–437. Springer, 2006. [39] Shai Shalev- Shwartz and Yoram Singer. Convex repeated games and Fenchel duality. In Advances in neural information processing systems, pages 1265–1272, 2007. [40] David Silver, Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Matthew Lai, Arthur Guez, Marc Lanctot, Laurent Sifre, Dharshan Kumaran, Thore Graepel, et al. Mastering chess and shogi by self- play with a general reinforcement learning algorithm. arXiv preprint arXiv:1712.01815, 2017. [41] László A V´egh. Strongly polynomial algorithm for a class of minimum- cost flow problems with separable convex objectives. In Proceedings of the forty- fourth annual ACM symposium on Theory of computing, pages 27–40. ACM, 2012.

[42] Petar Velickovic, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, and Yoshua Bengio. Graph Attention Networks. *International Conference on Learning Representations*, 2018. accepted as poster.[43] Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken- ichi Kawarabayashi, and Stefanie Jegelka. Representation learning on graphs with jumping knowledge networks. *arXiv preprint arXiv:1806.03536*, 2018. [44] Ya- xiang Yuan. Trust region algorithms for constrained optimization. 1994. [45] Jing Zhang, Sepideh Pourazarm, Christos G Cassandras, and Ioannis Ch Paschalidis. The price of anarchy in transportation networks by estimating user cost functions from actual traffic data. In *2016 IEEE 55th Conference on Decision and Control (CDC)*, pages 789–794. IEEE, 2016.

# 11 Appendix

# 11.1 Graphs

The table below shows the Open Street Map (OSM) [9, 32] queries used to obtain the graphs used during evaluation.

<table><tr><td>Graph Name</td><td>Target Address</td><td>Distance (Meters)</td></tr><tr><td>CB-500</td><td>Pembroke College, Cambridge, UK</td><td>500</td></tr><tr><td>SF-500</td><td>555 Market Street, San Francisco, USA</td><td>500</td></tr><tr><td>London-500</td><td>Leicester Square, London, UK</td><td>500</td></tr><tr><td>NYC-1000</td><td>Empire State Building, New York City, USA</td><td>1000</td></tr></table>

# 11.2 Algorithm to Generate Sources and Sinks

Algorithm 4 Algorithm to generate sources and sinks.

1: procedure GENERATE- SOURCES- SINKS  $(G = (V,E)$  , nSources, nSinks)

2:  $v_{0}\gets \mathrm{RANDOM - VERTEX}(G)$

3: nodes  $\leftarrow \{v_{0}\}$ $\vartriangleright$  nodes is a set initialized to contain  $v_{0}$  4: while LENGTH(nodes)  $<$  (nSources  $^+$  nSinks) do

5:  $u^{*}\gets \mathrm{argmin}_{u\in V - \mathrm{nodes}}(\mathrm{min}_{v\in \mathrm{nodes}}(\mathrm{min}(\mathrm{DIST}(G,u,v),\mathrm{DIST}(G,v,u))))$  6: nodes.add  $(u^{*})$

7: sources  $\leftarrow$  nodes[0:nSources]

8: sinks  $\leftarrow$  nodes[nSources:nSinks]

9: return sources, sinks

# 11.3 Model Hyperparameters

The table below show the hyperparameters which are held constant over all datasets. All models are trained with the Adam [22] optimizer and use an online batch selection [28] technique.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Step Size</td><td>0.001</td></tr><tr><td>Gradient Clip</td><td>1</td></tr><tr><td>Batch Size</td><td>100</td></tr><tr><td>Initial Embedding Dimensions</td><td>16</td></tr><tr><td>Hidden State Dimensions</td><td>32</td></tr><tr><td>Encoder Units per Layer</td><td>(8, 32, 32)</td></tr><tr><td>Decoder Units per Layer</td><td>(32, 32, 32, 1)</td></tr><tr><td>Max Dual Iters</td><td>500</td></tr><tr><td>Dual Step Size</td><td>0.01</td></tr><tr><td>Dual Momentum</td><td>0.9</td></tr></table>

The hyperparameters below refer to those used for the Neighborhood models on each dataset. The Gated- GAT baseline uses the same hyperparameters apart from both the maximum neighborhood level and the number of message- passing steps. For a given graph, the Gated- GAT model is configured to use the following. The variable MPS refers to the number of message- passing steps.

$$
\begin{array}{r}{\mathrm{MPS}_{\mathrm{GATED - GAT}} = \mathrm{MPS}_{\mathrm{NBRHOOD}}\times \mathrm{MAX - NBRHOOD - LEVEL}_{\mathrm{NBRHOOD}}}\\ {\mathrm{ATTN - HEADS}_{\mathrm{GATED - GAT}} = \mathrm{MAX - NBRHOOD - LEVEL}_{\mathrm{NBRHOOD}} + 1} \end{array}
$$

<table><tr><td>Parameter</td><td>Cambridge-500</td><td>SF-500</td><td>London-500</td><td>NYC-1000</td></tr><tr><td>Max Flow Iters</td><td>1,000</td><td>1,000</td><td>1,000</td><td>1,000</td></tr><tr><td>Message-Passing Steps</td><td>10</td><td>10</td><td>11</td><td>12</td></tr><tr><td>Max Neighborhood Level</td><td>2</td><td>2</td><td>4</td><td>3</td></tr><tr><td>Training Samples</td><td>10,000</td><td>10,000</td><td>20,000</td><td>25,000</td></tr><tr><td>Validation Samples</td><td>1,000</td><td>1,000</td><td>2,000</td><td>2,500</td></tr><tr><td>Test Samples</td><td>1,000</td><td>1,000</td><td>2,000</td><td>2,500</td></tr><tr><td>No. Sources</td><td>3</td><td>3</td><td>4</td><td>4</td></tr><tr><td>No. Sinks</td><td>3</td><td>3</td><td>4</td><td>4</td></tr></table>

# 11.4 Configurations for Optimization Algorithms

The configurations for SLSQP (left) and Trust Region Optimization (right) are listed below.

{ "max_iter": 1000, "max_iter": 1000, "tolerance": 1e- 7, "tolerance": 1e- 7, "hessian": "BFGS", "jacobian": "2- point" "jacobian": "2- point", "factorization": "SVD" }
